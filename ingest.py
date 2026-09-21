import os
import hashlib
import pickle
from pathlib import Path
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = Path("data/raw_pdfs")
CHROMA_DIR = Path("index/chroma")
BM25_PATH = Path("index/bm25.pkl")
LOG_PATH = Path("index/ingest_log.txt")

EMBED_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def guess_metadata(pdf_path):
    name = pdf_path.stem
    year = None
    for token in name.replace("_", " ").replace("-", " ").split():
        if token.isdigit() and 1900 < int(token) < 2100:
            year = int(token)
            break
    return {"title": name, "year": year}


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("nepal_env")
    except Exception:
        pass
    collection = client.create_collection("nepal_env")

    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    all_ids, all_chunks, all_metas = [], [], []
    seen_hashes = {}
    log = []

    pdfs = list(DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")

    for pdf in tqdm(pdfs):
        h = file_hash(pdf)
        if h in seen_hashes:
            log.append(f"SKIP duplicate: {pdf.name} == {seen_hashes[h]}")
            continue
        seen_hashes[h] = pdf.name

        paper_id = pdf.stem
        meta_base = guess_metadata(pdf)

        try:
            pages = extract_pages(pdf)
        except Exception as e:
            log.append(f"ERROR {pdf.name}: {e}")
            continue

        if not pages:
            log.append(f"NO TEXT (scanned?): {pdf.name}")
            continue

        for p in pages:
            for ci, ch in enumerate(chunk_text(p["text"])):
                cid = f"{paper_id}_p{p['page']}_c{ci}"
                all_ids.append(cid)
                all_chunks.append(ch)
                all_metas.append({
                    "paper_id": paper_id,
                    "file_name": pdf.name,
                    "title": meta_base["title"],
                    "year": meta_base["year"] or 0,
                    "page": p["page"],
                    "chunk_index": ci,
                })

    print(f"Total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("Nothing to index. Check PDFs.")
        return

    passages = [f"passage: {c}" for c in all_chunks]
    embeddings = model.encode(
        passages, batch_size=16, show_progress_bar=True,
        normalize_embeddings=True,
    )

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metas,
        embeddings=embeddings.tolist(),
    )

    tokenized = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25, "ids": all_ids,
            "chunks": all_chunks, "metadatas": all_metas,
        }, f)

    LOG_PATH.write_text("\n".join(log), encoding="utf-8")
    print(f"Ingestion done. Log: {LOG_PATH}")


if __name__ == "__main__":
    main()