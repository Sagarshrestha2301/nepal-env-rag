"""
embed_chunks.py — Step 6 of nepal-env-rag.

Embeds every chunk's `embed_text` with BAAI/bge-m3 on the GPU and saves
normalized vectors to disk in resumable shards.

Input:  data/chunks/<collection>/<category>.jsonl
Output: data/embeddings/<collection>/<category>/vectors_00000.npy  (float16, N x 1024)
        data/embeddings/<collection>/<category>/ids_00000.txt       (chunk_id per row)
        data/embeddings/<collection>/<category>/hashes_00000.txt    (hash of embed_text per row)
        data/embeddings/info.json                                   (model settings)

Re-embedding is incremental: every vector is stored with a hash of the exact
text it came from. After re-chunking (Preeti conversion, OCR, label fixes),
only chunks whose embed_text actually changed are sent to the GPU; unchanged
chunks reuse their saved vectors, even if their position in the file moved.

Usage:
    python embed_chunks.py --category 01_Floods_GLOFs     # pilot: measures speed
    python embed_chunks.py                                # everything (resumable)
    python embed_chunks.py --batch-size 8                 # if GPU memory runs out
    python embed_chunks.py --query "GLOF risk Imja lake" --category 01_Floods_GLOFs
                                                          # quick search sanity check

Stop anytime with Ctrl+C; rerun the same command to continue.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from typing import Iterator

import numpy as np

# --- Configuration --------------------------------------------------------

CHUNK_DIR = os.path.join("data", "chunks")
EMBED_DIR = os.path.join("data", "embeddings")
INFO_FILE = os.path.join(EMBED_DIR, "info.json")

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
MAX_SEQ_LENGTH = 1024      # longest chunk + header is ~830 tokens
DEFAULT_BATCH_SIZE = 16    # fits a 4 GB GPU in fp16; auto-halves on OOM
DEFAULT_SHARD_SIZE = 2048  # chunks per saved file (resume granularity)

# Settings that change what a vector means. If any differ from a previous run,
# old vectors must not be mixed with new ones.
VECTOR_SETTINGS = {
    "model": MODEL_NAME,
    "dim": EMBED_DIM,
    "dtype": "float16",
    "normalized": True,
    "max_seq_length": MAX_SEQ_LENGTH,
    "text_field": "embed_text",
}


# --- Model ----------------------------------------------------------------

class Encoder:
    """bge-m3 dense encoder in fp16 with automatic batch-size backoff."""

    def __init__(self, model_name: str, batch_size: int):
        import torch
        from sentence_transformers import SentenceTransformer

        if not torch.cuda.is_available():
            sys.exit("CUDA GPU not available. Install the CUDA build of PyTorch.")

        self.torch = torch
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cuda")
        self.model.half()                      # fp16: half the memory, ~2x faster
        self.model.max_seq_length = MAX_SEQ_LENGTH

    def encode(self, texts: list[str]) -> np.ndarray:
        while True:
            try:
                vectors = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,   # cosine similarity = dot product
                    convert_to_numpy=True,
                    show_progress_bar=False,     # sorts by length internally
                )
                return vectors.astype(np.float16)
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size //= 2
                print(f"    GPU out of memory: batch size reduced to {self.batch_size}")


class LazyEncoder:
    """Loads the model only if something actually needs embedding."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self._encoder: Encoder | None = None

    def ensure_loaded(self) -> None:
        if self._encoder is None:
            print(f"Loading {MODEL_NAME} (first run downloads ~2.3 GB)...")
            t0 = time.time()
            self._encoder = Encoder(MODEL_NAME, self.batch_size)
            print(f"  model loaded in {time.time() - t0:.0f} s")

    def encode(self, texts: list[str]) -> np.ndarray:
        self.ensure_loaded()
        return self._encoder.encode(texts)

    @property
    def current_batch_size(self) -> int:
        return self._encoder.batch_size if self._encoder else self.batch_size


def check_vectors(vectors: np.ndarray, where: str) -> None:
    """Stop immediately on NaN/inf or non-unit vectors instead of saving bad data."""
    if not np.isfinite(vectors).all():
        bad = int((~np.isfinite(vectors).all(axis=1)).sum())
        sys.exit(f"\n{where}: {bad} vectors contain NaN/inf (fp16 problem). Nothing was saved "
                 f"for this shard. Report this before continuing.")
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    if np.abs(norms - 1.0).max() > 0.01:
        sys.exit(f"\n{where}: vectors are not unit length (min {norms.min():.3f}, "
                 f"max {norms.max():.3f}). Nothing was saved for this shard.")


# --- Chunk files ----------------------------------------------------------

def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def chunk_files(category: str | None) -> list[str]:
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, "*", "*.jsonl")))
    if category:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] == category]
    if not files:
        sys.exit(f"No chunk files found in {CHUNK_DIR}"
                 + (f" for category '{category}'" if category else "") + ".")
    return files


def output_dir(chunk_file: str) -> str:
    collection = os.path.basename(os.path.dirname(chunk_file))
    category = os.path.splitext(os.path.basename(chunk_file))[0]
    return os.path.join(EMBED_DIR, collection, category)


def iter_shards(path: str, shard_size: int
                ) -> Iterator[tuple[int, list[str], list[str], list[int]]]:
    """Yield (shard_index, chunk_ids, texts, tokens_per_chunk) without loading the whole file."""
    ids, texts, tokens, index = [], [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["chunk_id"])
            texts.append(rec["embed_text"])
            tokens.append(rec.get("embed_tokens", 0))
            if len(ids) == shard_size:
                yield index, ids, texts, tokens
                ids, texts, tokens, index = [], [], [], index + 1
    if ids:
        yield index, ids, texts, tokens


def shard_paths(out_dir: str, index: int) -> tuple[str, str, str]:
    return (os.path.join(out_dir, f"vectors_{index:05d}.npy"),
            os.path.join(out_dir, f"ids_{index:05d}.txt"),
            os.path.join(out_dir, f"hashes_{index:05d}.txt"))


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def shard_done(out_dir: str, index: int, ids: list[str], hashes: list[str]) -> bool:
    """Done only if the saved shard holds exactly these chunks with exactly these texts."""
    vec_path, ids_path, hash_path = shard_paths(out_dir, index)
    if not all(os.path.exists(p) for p in (vec_path, ids_path, hash_path)):
        return False
    return read_lines(ids_path) == ids and read_lines(hash_path) == hashes


def load_cache(out_dir: str) -> dict[str, np.ndarray]:
    """Map text hash -> saved vector, from every complete shard of this category."""
    cache: dict[str, np.ndarray] = {}
    for vec_path in sorted(glob.glob(os.path.join(out_dir, "vectors_*.npy"))):
        hash_path = vec_path.replace("vectors_", "hashes_").replace(".npy", ".txt")
        if not os.path.exists(hash_path):
            continue                              # old shard without hashes: not reusable
        hashes = read_lines(hash_path)
        vectors = np.load(vec_path)
        if len(hashes) != len(vectors) or vectors.shape[1:] != (EMBED_DIM,):
            continue
        for h, v in zip(hashes, vectors):
            cache[h] = v
    return cache


def save_shard(out_dir: str, index: int, ids: list[str], hashes: list[str],
               vectors: np.ndarray) -> None:
    """Write files atomically so a crash never leaves a bad shard."""
    os.makedirs(out_dir, exist_ok=True)
    vec_path, ids_path, hash_path = shard_paths(out_dir, index)
    with open(vec_path + ".tmp", "wb") as f:
        np.save(f, vectors)
    with open(hash_path + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(hashes) + "\n")
    with open(ids_path + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    os.replace(vec_path + ".tmp", vec_path)
    os.replace(hash_path + ".tmp", hash_path)
    os.replace(ids_path + ".tmp", ids_path)  # ids last: marks the shard complete


def remove_stale_shards(out_dir: str, last_index: int) -> int:
    """Delete shards beyond the current end (left over when a category shrinks)."""
    removed = 0
    for path in glob.glob(os.path.join(out_dir, "*_*.*")):
        m = re.search(r"(?:vectors|ids|hashes)_(\d{5})\.(?:npy|txt)(?:\.tmp)?$", path)
        if m and int(m.group(1)) > last_index:
            os.remove(path)
            removed += 1
    return removed


TOKENS_RE = re.compile(r'"embed_tokens":\s*(\d+)')


def count_work(files: list[str]) -> tuple[int, int, int]:
    """Total chunks and tokens, for progress and ETA; also counts over-length chunks.
    Reads only the token count from each line (no full JSON parse), so it is fast."""
    print(f"Counting chunks in {len(files)} files (takes a moment)...", flush=True)
    chunks = tokens = too_long = 0
    for i, path in enumerate(files, 1):
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = TOKENS_RE.search(line)
                n = int(m.group(1)) if m else 0
                chunks += 1
                tokens += n
                too_long += n > MAX_SEQ_LENGTH
        print(f"  counted {i}/{len(files)}: {os.path.relpath(path, CHUNK_DIR)} "
              f"({chunks:,} chunks so far)", flush=True)
    return chunks, tokens, too_long


# --- Embedding run --------------------------------------------------------

def check_and_write_info(batch_size: int) -> None:
    """Refuse to mix vectors made with different model settings."""
    os.makedirs(EMBED_DIR, exist_ok=True)
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE, encoding="utf-8") as f:
            old = json.load(f)
        changed = {k: (old.get(k), v) for k, v in VECTOR_SETTINGS.items() if old.get(k) != v}
        if changed:
            sys.exit(f"Existing vectors in {EMBED_DIR} were made with different settings: "
                     f"{changed}. Move that folder away before embedding again.")
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        json.dump({**VECTOR_SETTINGS, "batch_size": batch_size}, f, indent=2)


def run(files: list[str], batch_size: int, shard_size: int) -> None:
    total_chunks, total_tokens, too_long = count_work(files)
    print(f"{len(files)} files, {total_chunks:,} chunks, {total_tokens:,} tokens")
    if too_long:
        print(f"  Warning: {too_long:,} chunks exceed {MAX_SEQ_LENGTH} tokens and will be truncated.")

    check_and_write_info(batch_size)
    encoder = LazyEncoder(batch_size)

    done_chunks = embedded_chunks = done_tokens = skipped_tokens = reused = 0
    start = time.time()
    embed_seconds = 0.0

    for path in files:
        out_dir = output_dir(path)
        name = os.path.relpath(path, CHUNK_DIR)
        cache = load_cache(out_dir)
        last_index = -1

        for index, ids, texts, tokens in iter_shards(path, shard_size):
            last_index = index
            hashes = [text_hash(t) for t in texts]
            if shard_done(out_dir, index, ids, hashes):
                skipped_tokens += sum(tokens)
                done_chunks += len(ids)
                continue

            vectors = np.empty((len(texts), EMBED_DIM), dtype=np.float16)
            missing = []
            for i, h in enumerate(hashes):
                if h in cache:
                    vectors[i] = cache[h]
                else:
                    missing.append(i)
            missing_set = set(missing)
            reused += len(ids) - len(missing)
            skipped_tokens += sum(t for i, t in enumerate(tokens) if i not in missing_set)

            if missing:
                encoder.ensure_loaded()          # load time is not embedding time
                t0 = time.time()
                new = encoder.encode([texts[i] for i in missing])
                embed_seconds += time.time() - t0
                check_vectors(new, f"{name} shard {index}")
                vectors[missing] = new
                embedded_chunks += len(missing)
                done_tokens += sum(tokens[i] for i in missing)

            save_shard(out_dir, index, ids, hashes, vectors)
            done_chunks += len(ids)

            speed = done_tokens / embed_seconds if embed_seconds else 0
            remaining = total_tokens - done_tokens - skipped_tokens
            eta_h = remaining / speed / 3600 if speed else 0
            print(f"  {name} shard {index}: {done_chunks:,}/{total_chunks:,} chunks | "
                  f"embedded {len(missing):,}, reused {len(ids) - len(missing):,} | "
                  f"{speed:,.0f} tok/s | ~{eta_h:.1f} h left | batch {encoder.current_batch_size}")

        removed = remove_stale_shards(out_dir, last_index)
        if removed:
            print(f"  {name}: removed {removed} stale files from an older, longer run")

    minutes = (time.time() - start) / 60
    print(f"\nDone in {minutes:.1f} min. Embedded {embedded_chunks:,} chunks, "
          f"reused {reused:,} saved vectors. Vectors in {EMBED_DIR}")
    if done_tokens and embed_seconds:
        speed = done_tokens / embed_seconds
        print(f"Speed: {speed:,.0f} tokens/s -> all {total_tokens:,} tokens from scratch "
              f"would take ~{total_tokens / speed / 3600:.1f} h")


# --- Quick search check ---------------------------------------------------

def load_vectors(files: list[str]) -> tuple[np.ndarray, list[str]]:
    vectors, ids = [], []
    for path in files:
        out_dir = output_dir(path)
        for vec_path in sorted(glob.glob(os.path.join(out_dir, "vectors_*.npy"))):
            ids_path = vec_path.replace("vectors_", "ids_").replace(".npy", ".txt")
            vectors.append(np.load(vec_path))
            ids.extend(read_lines(ids_path))
    if not vectors:
        sys.exit("No vectors found. Run the embedding first.")
    return np.vstack(vectors).astype(np.float32), ids


def search(query: str, files: list[str], top_k: int = 5) -> None:
    """Embed one query and print the closest chunks (brute force)."""
    matrix, ids = load_vectors(files)
    encoder = Encoder(MODEL_NAME, batch_size=1)
    q = encoder.encode([query]).astype(np.float32)[0]
    scores = matrix @ q
    best = np.argsort(-scores)[:top_k]

    wanted = {ids[i]: float(scores[i]) for i in best}
    found: dict[str, dict] = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["chunk_id"] in wanted:
                    found[rec["chunk_id"]] = rec

    print(f"\nQuery: {query}\nSearched {len(ids):,} chunks\n")
    for rank, cid in enumerate(sorted(wanted, key=wanted.get, reverse=True), 1):
        rec = found[cid]
        pages = f"p{rec['page_start']}" + (f"-{rec['page_end']}" if rec["page_end"] != rec["page_start"] else "")
        print(f"{rank}. score {wanted[cid]:.3f} | {rec['title'][:70]} ({rec['year'] or 'n.d.'}) {pages}")
        print(f"   {rec['text'][:300].replace(chr(10), ' ')}...\n")


# --- Entry point ----------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)   # show progress live, even when piped to a log
    parser = argparse.ArgumentParser(description="Embed chunks with bge-m3 on GPU.")
    parser.add_argument("--category", help="only one folder, e.g. 01_Floods_GLOFs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--query", help="search the embedded chunks instead of embedding")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    files = chunk_files(args.category)
    if args.query:
        search(args.query, files, args.top_k)
    else:
        run(files, args.batch_size, args.shard_size)


if __name__ == "__main__":
    main()