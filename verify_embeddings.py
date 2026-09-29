"""
verify_embeddings.py — check the output of embed_chunks.py before building the index.

Checks, for every chunk file:
  * every chunk has exactly one vector, in the same order (chunk_id matches)
  * each vector was made from the chunk's CURRENT embed_text (hash matches)
  * vectors are float16, 1024-dim, finite (no NaN/inf) and unit length
  * no leftover .tmp files, no extra shards, no embedding folders without chunks
  * no chunk_id appears twice across the corpus
  * info.json has the expected model settings

Uses no GPU and loads one shard at a time, so it is safe to run on the laptop.

Usage:
    python verify_embeddings.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from embed_chunks import (CHUNK_DIR, EMBED_DIR, EMBED_DIM, INFO_FILE, VECTOR_SETTINGS,
                          chunk_files, output_dir, read_lines, text_hash)

NORM_TOLERANCE = 0.01


def check_category(path: str, seen_ids: set[str], problems: list[str]) -> tuple[int, int]:
    """Return (chunks, vectors) for one chunk file; append any problems found."""
    name = os.path.relpath(path, CHUNK_DIR)
    out_dir = output_dir(path)

    ids, hashes = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["chunk_id"])
            hashes.append(text_hash(rec["embed_text"]))

    dupes = [cid for cid in ids if cid in seen_ids]
    if dupes:
        problems.append(f"{name}: {len(dupes)} chunk_ids also appear in another file, e.g. {dupes[0]}")
    seen_ids.update(ids)

    if not os.path.isdir(out_dir):
        problems.append(f"{name}: no embeddings folder ({out_dir})")
        return len(ids), 0

    tmp = glob.glob(os.path.join(out_dir, "*.tmp"))
    if tmp:
        problems.append(f"{name}: {len(tmp)} leftover .tmp files (a save was interrupted)")

    saved_ids, saved_hashes, n_vectors = [], [], 0
    bad_nan = bad_norm = 0
    min_norm, max_norm = 2.0, 0.0
    for vec_path in sorted(glob.glob(os.path.join(out_dir, "vectors_*.npy"))):
        ids_path = vec_path.replace("vectors_", "ids_").replace(".npy", ".txt")
        hash_path = vec_path.replace("vectors_", "hashes_").replace(".npy", ".txt")
        shard = os.path.basename(vec_path)
        if not (os.path.exists(ids_path) and os.path.exists(hash_path)):
            problems.append(f"{name}: {shard} is missing its ids or hashes file")
            continue

        vectors = np.load(vec_path)
        shard_ids, shard_hashes = read_lines(ids_path), read_lines(hash_path)
        if vectors.dtype != np.float16 or vectors.ndim != 2 or vectors.shape[1] != EMBED_DIM:
            problems.append(f"{name}: {shard} has shape {vectors.shape} dtype {vectors.dtype}")
            continue
        if not (len(vectors) == len(shard_ids) == len(shard_hashes)):
            problems.append(f"{name}: {shard} has {len(vectors)} vectors, "
                            f"{len(shard_ids)} ids, {len(shard_hashes)} hashes")
            continue

        finite = np.isfinite(vectors).all(axis=1)
        bad_nan += int((~finite).sum())
        norms = np.linalg.norm(vectors[finite].astype(np.float32), axis=1)
        if len(norms):
            bad_norm += int((np.abs(norms - 1.0) > NORM_TOLERANCE).sum())
            min_norm, max_norm = min(min_norm, norms.min()), max(max_norm, norms.max())

        saved_ids.extend(shard_ids)
        saved_hashes.extend(shard_hashes)
        n_vectors += len(vectors)

    if bad_nan:
        problems.append(f"{name}: {bad_nan} vectors contain NaN/inf")
    if bad_norm:
        problems.append(f"{name}: {bad_norm} vectors are not unit length "
                        f"(norms {min_norm:.3f}–{max_norm:.3f})")

    if saved_ids != ids:
        missing = len(set(ids) - set(saved_ids))
        extra = len(set(saved_ids) - set(ids))
        problems.append(f"{name}: chunk_ids differ from the chunk file "
                        f"({len(ids):,} chunks, {len(saved_ids):,} vectors; "
                        f"{missing:,} missing, {extra:,} extra"
                        + ("; same set but different order" if not missing and not extra else "")
                        + ")")
    elif saved_hashes != hashes:
        stale = sum(a != b for a, b in zip(saved_hashes, hashes))
        problems.append(f"{name}: {stale:,} vectors were made from older text "
                        f"(chunks changed after embedding); rerun embed_chunks.py")

    status = "ok" if not any(p.startswith(name + ":") for p in problems) else "PROBLEM"
    print(f"  {status:7} {name}: {len(ids):,} chunks, {n_vectors:,} vectors")
    return len(ids), n_vectors


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    problems: list[str] = []

    if not os.path.exists(INFO_FILE):
        problems.append(f"{INFO_FILE} is missing")
    else:
        with open(INFO_FILE, encoding="utf-8") as f:
            info = json.load(f)
        for key, value in VECTOR_SETTINGS.items():
            if info.get(key) != value:
                problems.append(f"info.json: {key} is {info.get(key)!r}, expected {value!r}")

    files = chunk_files(None)
    print(f"Checking {len(files)} chunk files against {EMBED_DIR} ...")
    seen_ids: set[str] = set()
    total_chunks = total_vectors = 0
    for path in files:
        c, v = check_category(path, seen_ids, problems)
        total_chunks += c
        total_vectors += v

    expected_dirs = {os.path.normpath(output_dir(p)) for p in files}
    for d in glob.glob(os.path.join(EMBED_DIR, "*", "*")):
        if os.path.isdir(d) and os.path.normpath(d) not in expected_dirs:
            problems.append(f"{d}: embeddings folder with no matching chunk file "
                            f"(build_index.py may load it by mistake)")

    size_gb = sum(os.path.getsize(p) for p in glob.glob(
        os.path.join(EMBED_DIR, "*", "*", "vectors_*.npy"))) / 1e9
    print(f"\nTotal: {total_chunks:,} chunks, {total_vectors:,} vectors, {size_gb:.2f} GB on disk")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S) FOUND:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nALL CHECKS PASSED: every chunk has one correct, current, unit-length vector.")


if __name__ == "__main__":
    main()