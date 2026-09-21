# =============================================================
# Nepal Environmental Studies RAG — Query Pipeline
# =============================================================
# Yo script le:
#   1. User ko prashna lai vector ma badalcha
#   2. ChromaDB (vector search) ra BM25 (keyword search) dubai ma khojcha
#   3. Dui list lai RRF le jodcha
#   4. Top 6 chunk chhanera LLM (Ollama) lai dincha
#   5. Jawaf + citation return garcha
# =============================================================

# ---------- Standard library imports ----------
import os        # environment variable set garna
import pickle    # BM25 index load garna

# ---------- Third-party imports ----------
import chromadb                            # vector DB client
from sentence_transformers import SentenceTransformer   # text → vector
import ollama                              # local LLM client

# Tokenizer warning off — multi-thread ma error aauna sakcha
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================
# 1. Paths ra constants
# =============================================================
# ingest.py ma jastai same paths — ingestion le yetikai save gareko thau

CHROMA_DIR  = "index/chroma"                        # vector DB
BM25_PATH   = "index/bm25.pkl"                      # keyword index
EMBED_MODEL = "intfloat/multilingual-e5-small"      # same model as ingest
LLM_MODEL   = "qwen2.5:3b"                          # fast, 4GB VRAM ma fit


# =============================================================
# 2. Global cache variables
# =============================================================
# Yo variables le model, index, DB lai memory ma hold garcha.
# Kin? Har query ma model load garnu slow hunxa. Ek patak load → reuse.
# Value None = ajai load bhaeko chhaina.

_model      = None
_data       = None
_collection = None


def load():
    """
    Lazy loading — pahilo patak call garda matra load garcha.
    Arko patak call garda cache bata turuntai return garcha.

    Return: (model, bm25_data, chroma_collection)
    """
    global _model, _data, _collection

    # (a) Embedding model load — CPU ma
    # Kin CPU? VRAM LLM ko lagi bachaaunu
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")

    # (b) BM25 pickle load — keyword index
    if _data is None:
        with open(BM25_PATH, "rb") as f:
            _data = pickle.load(f)

    # (c) ChromaDB collection load
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection("nepal_env")

    return _model, _data, _collection


def list_categories():
    """
    Indexed PDF haru kun kun category ma chan list garcha.
    Streamlit UI ma dropdown banauna use huncha.
    Example return: ['01_Floods_GLOFS', '02_Landslides_Earthquakes', ...]
    """
    _, data, _ = load()
    # Metadatas bata category nikalera unique set banau
    cats = sorted({m["category"] for m in data["metadatas"]})
    return cats


def rrf(rank_lists, k=60):
    """
    Reciprocal Rank Fusion — dui rank list lai euta list ma jodcha.

    Kasari kaam garcha?
      Har document ko score = Σ 1/(k + rank)
      Dui list ma top ma bhayeko document ko score dherai hunxa.
      Kunai document ek list ma matra bhayepani score linxa.

    Example:
      dense list:  [A, B, C]
      bm25 list:   [B, C, D]
      Result: B (dubai ma top) > C > A (dense only) > D (bm25 only)

    k=60 default — research paper bata aayeko standard value.
    """
    scores = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    # Score descending order ma sort
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve(query, top_k=6, category=None, theme=None,
             basin=None, year_min=None):
    """
    Hybrid retrieval — Chroma (semantic) + BM25 (keyword) merge.

    Parameters:
      query     : user ko prashna
      top_k     : kati chunk LLM lai pathaune (default 6)
      category  : filter — "01_Floods_GLOFS" jastai
      theme     : filter — "glof", "flood", ...
      basin     : filter — "koshi", "gandaki", ...
      year_min  : filter — yetti saal paxi ko matra

    Return: list of dicts — [{"id", "text", "metadata", "score"}, ...]
    """
    # Model, BM25 data, Chroma collection load
    model, data, collection = load()

    bm25     = data["bm25"]        # BM25 object
    ids      = data["ids"]         # chunk IDs (BM25 ko index order ma)
    chunks   = data["chunks"]      # chunk text haru
    metas    = data["metadatas"]   # chunk metadata haru

    # ---------- 2.1 Query lai embed ----------
    # "query: " prefix — e5 model le passage vs query distinguish garna
    # (ingestion ma "passage:" use gareko thiyo)
    q_emb = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    )[0].tolist()

    # ---------- 2.2 Chroma filter banau ----------
    # Jati filter chhan teti clause banaune, ani $and le jodne
    where_clauses = []
    if category:
        where_clauses.append({"category": category})
    if year_min:
        where_clauses.append({"year": {"$gte": year_min}})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]                # euta matra filter
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}         # multiple filter

    # ---------- 2.3 Dense search (Chroma) ----------
    # Top 30 candidate linxa (top_k 6 bhaye pani, reranking ko lagi 30)
    n_dense = min(30, collection.count())
    dense = collection.query(
        query_embeddings=[q_emb],
        n_results=n_dense,
        where=where,
    )
    dense_ids = dense["ids"][0]     # list of chunk IDs

    # ---------- 2.4 Sparse search (BM25) ----------
    # Query lai lowercase garera word-by-word todne
    scores = bm25.get_scores(query.lower().split())
    # Top 30 index linxa (score descending)
    top_idx = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:30]
    bm25_ids = [ids[i] for i in top_idx]    # IDs ma convert

    # ---------- 2.5 RRF le merge ----------
    merged = rrf([dense_ids, bm25_ids])

    # ---------- 2.6 Post-filter ra top_k chhanne ----------
    idx_map = {d: i for i, d in enumerate(ids)}  # ID → index map

    results = []
    for d, s in merged:
        if d not in idx_map:
            continue                            # BM25 only — safety check

        m = metas[idx_map[d]]                   # chunk ko metadata

        # Post-filter — BM25 le where clause bypass garcha
        # tesaile manual check garna parxa
        if category and m.get("category") != category:
            continue
        if theme and theme not in m.get("themes", ""):
            continue
        if basin and basin not in m.get("basins", ""):
            continue
        if year_min and m.get("year", 0) < year_min:
            continue

        results.append({
            "id":       d,
            "text":     chunks[idx_map[d]],
            "metadata": m,
            "score":    s,
        })

        if len(results) >= top_k:
            break                               # top_k pura bhayo

    return results


def answer(query, category=None, theme=None, basin=None, year_min=None):
    """
    Full RAG pipeline — retrieve + LLM generate.

    Return: (jawaf_text, context_list)
    """
    # ---------- 3.1 Retrieve top 6 chunk ----------
    contexts = retrieve(
        query,
        top_k=6,
        category=category,
        theme=theme,
        basin=basin,
        year_min=year_min,
    )

    # Kunai relevant chunk bhetiyena bhane early return
    if not contexts:
        return "No relevant documents found.", []

    # ---------- 3.2 Context text banau ----------
    # Har chunk ko format:
    #   [paper_id p.page]
    #   <text>
    # Paper_id ra page — LLM le citation garna sajilo
    context_text = "\n\n".join(
        f"[{c['metadata']['paper_id']} p.{c['metadata']['page']}]\n{c['text']}"
        for c in contexts
    )

    # ---------- 3.3 Prompt banau ----------
    # Rules:
    #   - sirf context bata jawaf
    #   - har fact ma citation
    #   - bhetiyena bhane "Not found" bhanne
    #   - numbers/locations priority
    #   - user ko language ma jawaf
    prompt = f"""You are a Nepal environmental research assistant.

Answer ONLY using the context below, drawn from peer-reviewed studies and
reports about Nepal's environment (floods, GLOFs, climate, forests,
biodiversity, water, agriculture, air quality, policy, geology, health).

Rules:
- Cite every factual claim as [paper_id p.page].
- If multiple papers agree, cite all of them.
- If the answer is not in the context, reply exactly:
  "Not found in indexed papers."
- Prefer specific numbers, dates, locations, and basin names.
- Answer in the same language as the question (English or Nepali).

Context:
{context_text}

Question: {query}

Answer:"""

    # ---------- 3.4 LLM call ----------
    # num_ctx   = 4096 token context window (4 GB VRAM ma fit)
    # temperature = 0.2 → factual, less creative
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 4096, "temperature": 0.2},
    )

    # ---------- 3.5 Return ----------
    # Jawaf ra context dubai dinxa (UI ma source dekhauna)
    return resp["message"]["content"], contexts