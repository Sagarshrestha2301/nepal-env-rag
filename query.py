import os
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHROMA_DIR = "index/chroma"
BM25_PATH = "index/bm25.pkl"
EMBED_MODEL = "intfloat/multilingual-e5-small"

# Switch based on speed vs quality:
LLM_MODEL = "qwen2.5:3b"        # fast, fits VRAM
# LLM_MODEL = "deepseek-r1:8b"  # slow but deeper reasoning
# LLM_MODEL = "qwen3.5:latest"  # your existing, quality model

_model = _data = _collection = None


def load():
    global _model, _data, _collection
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")
    if _data is None:
        with open(BM25_PATH, "rb") as f:
            _data = pickle.load(f)
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection("nepal_env")
    return _model, _data, _collection


def rrf(rank_lists, k=60):
    scores = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve(query, top_k=5):
    model, data, collection = load()
    bm25 = data["bm25"]
    ids = data["ids"]
    chunks = data["chunks"]
    metas = data["metadatas"]

    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    dense = collection.query(query_embeddings=[q_emb], n_results=min(20, collection.count()))
    dense_ids = dense["ids"][0]

    scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
    bm25_ids = [ids[i] for i in top_idx]

    merged = rrf([dense_ids, bm25_ids])
    idx_map = {d: i for i, d in enumerate(ids)}

    return [{
        "id": d, "text": chunks[idx_map[d]],
        "metadata": metas[idx_map[d]], "score": s,
    } for d, s in merged[:top_k]]


def answer(query):
    contexts = retrieve(query)
    if not contexts:
        return "No indexed papers found.", []

    context_text = "\n\n".join(
        f"[{c['metadata']['paper_id']} p.{c['metadata']['page']}]\n{c['text']}"
        for c in contexts
    )

    prompt = f"""You are a Nepal environmental research assistant.
Answer ONLY using the context below.
Cite every fact as [paper_id p.page].
If the answer is not in the context, reply exactly: "Not found in indexed papers."

Context:
{context_text}

Question: {query}
Answer:"""

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 4096, "temperature": 0.2},
    )
    return resp["message"]["content"], contexts