# =============================================================
# Nepal Environmental Studies RAG — Ingest Pipeline
# =============================================================
# Yo script le:
#   1. data/raw_pdfs/ bhitra ko sabai PDF (subfolder sahit) khojcha
#   2. Har PDF bata text nikalcha
#   3. Duplicate check garcha
#   4. Metadata (title, year, theme, basin, district) nikalcha
#   5. Text lai sano chunk ma todcha
#   6. Chunk lai embedding (vector) ma badalcha
#   7. ChromaDB ra BM25 ma save garcha
# =============================================================

# ---------- Standard library imports ----------
import os        # environment variables set garna
import re        # regular expression — year khojna
import hashlib   # file ko unique fingerprint (SHA256) banauna
import pickle    # BM25 object disk ma save garna

from pathlib import Path   # file path cross-platform handle garna

# ---------- Third-party imports ----------
import fitz                                # PyMuPDF — PDF bata text nikalna
import chromadb                            # vector database (embedding save)
from sentence_transformers import SentenceTransformer   # text → vector
from rank_bm25 import BM25Okapi            # keyword based search
from tqdm import tqdm                      # progress bar

# Tokenizer warning off — multi-thread ma error aauna sakcha
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================
# 1. Paths ra constants
# =============================================================

DATA_DIR    = Path("data/raw_pdfs")         # PDF haru kaha chan
CHROMA_DIR  = Path("index/chroma")          # vector DB save hune thau
BM25_PATH   = Path("index/bm25.pkl")        # keyword index file
LOG_PATH    = Path("index/ingest_log.txt")  # error + summary log

# Embedding model — multilingual (Nepali + English dubai support)
EMBED_MODEL = "intfloat/multilingual-e5-small"

# Chunking parameters
CHUNK_SIZE    = 2000   # ek chunk ma 2000 characters
CHUNK_OVERLAP = 200    # consecutive chunk bich ma 200 char overlap


# =============================================================
# 2. Nepal domain keyword dictionaries
# =============================================================
# Yo keywords le text scan garcha. Bhetiyo bhane tag linxa.
# Example: paper ma "GLOF" chha bhane themes ma "glof" add hunxa.

THEME_KEYWORDS = {
    "glof":        ["glof", "glacial lake outburst", "moraine dam", "supraglacial"],
    "flood":       ["flood", "inundation", "flash flood", "debris flow", "discharge"],
    "landslide":   ["landslide", "slope failure", "mass movement", "rockfall"],
    "earthquake":  ["earthquake", "seismic", "tectonic", "richter"],
    "glacier":     ["glacier", "ice melt", "snow cover", "permafrost", "cryosphere"],
    "climate":     ["climate change", "warming", "precipitation trend", "monsoon"],
    "forest":      ["forest", "community forest", "afforestation", "deforestation"],
    "biodiversity":["biodiversity", "wildlife", "species", "habitat", "conservation"],
    "forest_fire": ["forest fire", "wildfire", "burn", "fire risk"],
    "water":       ["water quality", "hydrology", "river", "streamflow", "watershed"],
    "hydropower":  ["hydropower", "hydroelectric", "dam", "reservoir", "turbine"],
    "agriculture": ["agriculture", "crop", "farming", "yield", "irrigation"],
    "livelihood":  ["livelihood", "income", "poverty", "adaptation", "vulnerability"],
    "air":         ["air quality", "pm2.5", "pm10", "pollution", "emission"],
    "urban_heat":  ["urban heat", "heat island", "temperature rise", "land surface"],
    "land_use":    ["land use", "land cover", "remote sensing", "gis", "satellite"],
    "policy":      ["policy", "governance", "regulation", "eia", "institution"],
    "geology":     ["geology", "stratigraphy", "rock", "mineral", "fault"],
    "health":      ["health", "microbiology", "food safety", "nutrition", "disease"],
    "disaster":    ["disaster", "hazard", "risk", "vulnerability", "early warning"],
}

# Nepal ko major river basin haru
RIVER_BASINS = [
    "koshi", "saptakoshi", "gandaki", "narayani", "karnali", "mahakali",
    "bagmati", "rapti", "kamala", "trishuli", "sun koshi", "tamakoshi",
    "bhotekoshi", "dudhkoshi", "marsyangdi", "bheri", "seti", "modi",
    "madi", "tinau", "west rapti", "east rapti",
]

# Nepal ko 77 district (English spelling)
DISTRICTS = [
    "kathmandu", "lalitpur", "bhaktapur", "chitwan", "makwanpur",
    "sindhupalchok", "dolakha", "solukhumbu", "taplejung", "sankhuwasabha",
    "rasuwa", "dhading", "nuwakot", "gorkha", "manang", "mustang",
    "kaski", "lamjung", "syangja", "parbat", "myagdi", "baglung",
    "dolpa", "mugu", "humla", "jumla", "kalikot", "jajarkot",
    "bajhang", "bajura", "darchula", "baitadi", "dadeldhura",
    "kanchanpur", "kailali", "bardiya", "banke", "dang", "salyan",
    "rukum", "rolpa", "pyuthan", "gulmi", "arghakhanchi", "palpa",
    "kapilvastu", "rupandehi", "nawalparasi", "parsa", "bara",
    "rautahat", "sarlahi", "mahottari", "dhanusha", "siraha",
    "saptari", "udayapur", "okhaldhunga", "khotang", "bhojpur",
    "dhankuta", "terhathum", "panchthar", "ilam", "jhapa",
    "morang", "sunsari",
]


# =============================================================
# 3. Helper functions
# =============================================================

def file_hash(path):
    """
    File ko unique fingerprint (SHA256) nikalcha.
    Same PDF dui patak bhetiyo bhane hash same hunxa → skip garna help.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # 8KB chunk ma padhchha — dherai memory linna
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_pages(pdf_path):
    """
    PDF kholera page-by-page text nikalcha.
    Return: list of dicts — [{"page": 1, "text": "..."}, ...]
    Khali page (image-only) skip garcha.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():                 # khali text skip
            pages.append({"page": i + 1, "text": text})
    return pages


def detect_terms(text_lower, vocab):
    """
    Text bhitra kun kun vocab word chha hercha.
    Example: vocab = RIVER_BASINS → text ma "koshi", "gandaki" chan ki.
    """
    return [term for term in vocab if term in text_lower]


def detect_themes(text_lower):
    """
    THEME_KEYWORDS scan garera kun theme haru chhan detect garcha.
    Example: "glof" keyword bhetiyo bhane themes ma "glof" add.
    """
    themes = []
    for theme, keywords in THEME_KEYWORDS.items():
        # kunai ek keyword bhetiyo bhane tyo theme add
        if any(kw in text_lower for kw in keywords):
            themes.append(theme)
    return themes


def guess_year(text, filename):
    """
    Text ra filename bata saal (year) nikalcha.
    - Pahilo 2000 char ma 4-digit number khojcha
    - 1970–2030 bich ko matra linxa
    - Sabai bhanda thulo linxa (recent mention = publication year)
    """
    candidates = re.findall(
        r"\b((?:19|20)\d{2})\b",
        text[:2000] + " " + filename
    )
    valid = [int(y) for y in candidates if 1970 <= int(y) <= 2030]
    return max(valid) if valid else None


def guess_title(pages, filename):
    """
    Pahilo page ko top 10 line herera title nikalcha.
    Long line (>20 char) jo "abstract/keywords/introduction" bata
    start hudaina, tyo title ho bhanera linxa.
    Bhetiena bhane filename nai title.
    """
    if not pages:
        return filename

    # Pahilo page ko line haru
    lines = [l.strip() for l in pages[0]["text"].split("\n") if l.strip()]

    for line in lines[:10]:
        # Long + boilerplate hudaina
        if len(line) > 20 and not line.lower().startswith(
            ("abstract", "keywords", "introduction")
        ):
            return line[:200]        # 200 char samma matra

    return filename                  # fallback


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Text lai fixed-size chunk ma todcha, overlap sahit.

    Example: 5000 char text, size 2000, overlap 200
      Chunk 1: char 0    → 2000
      Chunk 2: char 1800 → 3800   (200 char overlap)
      Chunk 3: char 3600 → 5000

    Kin overlap? Border ma haraeko idea dubai chunk ma rakhna.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap         # overlap chhoddai agadi badhne
    return chunks


# =============================================================
# 4. Main pipeline
# =============================================================

def main():
    # ---------- 4.1 ChromaDB collection ready ----------
    # Purano index delete → naya banaunu
    # NOTE: yesle full rebuild garcha. 100–500 PDF samma thik.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("nepal_env")
    except Exception:
        pass                          # pehilo patak chha bhane ignore
    collection = client.create_collection("nepal_env")

    # ---------- 4.2 Embedding model load ----------
    # CPU ma load — VRAM LLM ko lagi bachaaunu
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    # ---------- 4.3 Accumulator variables ----------
    all_ids         = []   # har chunk ko unique ID
    all_chunks      = []   # chunk ko asli text
    all_metas       = []   # chunk ko metadata
    seen_hashes     = {}   # duplicate detection ko lagi
    log             = []   # error + summary log
    category_counts = {}   # kun category ma kati PDF

    # ---------- 4.4 Sabai PDF khojne (recursive) ----------
    # rglob("*.pdf") → subfolder bhitra pani search garcha
    pdfs = sorted(DATA_DIR.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs across all categories")

    # ---------- 4.5 Har PDF process ----------
    for pdf in tqdm(pdfs):

        # (a) Category = immediate parent folder name
        # Example: 01_Floods_GLOFS/paper.pdf → category = "01_Floods_GLOFS"
        category = pdf.parent.name

        # (b) Duplicate check
        h = file_hash(pdf)
        if h in seen_hashes:
            log.append(f"SKIP duplicate: {pdf.name} == {seen_hashes[h]}")
            continue
        seen_hashes[h] = pdf.name

        # (c) Unique paper ID banau
        paper_id = f"{category}/{pdf.stem}"

        # (d) PDF bata text nikaal
        try:
            pages = extract_pages(pdf)
        except Exception as e:
            log.append(f"ERROR {paper_id}: {e}")
            continue

        if not pages:
            log.append(f"NO TEXT (scanned?): {paper_id}")
            continue

        # (e) Metadata extract
        full_text  = "\n".join(p["text"] for p in pages)
        text_lower = full_text.lower()

        title     = guess_title(pages, pdf.stem)
        year      = guess_year(full_text, pdf.stem)
        themes    = detect_themes(text_lower)
        basins    = detect_terms(text_lower, RIVER_BASINS)
        districts = detect_terms(text_lower, DISTRICTS)

        # Category count badhau
        category_counts[category] = category_counts.get(category, 0) + 1

        # (f) Page by page chunk banaune
        for p in pages:
            page_chunks = chunk_text(p["text"])
            for ci, ch in enumerate(page_chunks):
                # Chunk ID example: 01_Floods_GLOFS/Marahatta_2009_p3_c1
                cid = f"{paper_id}_p{p['page']}_c{ci}"

                all_ids.append(cid)
                all_chunks.append(ch)
                all_metas.append({
                    "paper_id":    paper_id,
                    "file_name":   pdf.name,
                    "category":    category,
                    "title":       title,
                    "year":        year or 0,
                    "page":        p["page"],
                    "chunk_index": ci,
                    "themes":      ",".join(themes),
                    "basins":      ",".join(basins),
                    "districts":   ",".join(districts),
                })

    print(f"Total chunks: {len(all_chunks)}")

    # ---------- 4.6 Empty guard ----------
    if not all_chunks:
        print("Nothing to index.")
        LOG_PATH.write_text("\n".join(log), encoding="utf-8")
        return

    # ---------- 4.7 Embedding banaune ----------
    # "passage: " prefix kina? e5 model le passage vs query distinguish garna
    # Ingestion ma "passage:", query.py ma "query:"
    passages = [f"passage: {c}" for c in all_chunks]

    embeddings = model.encode(
        passages,
        batch_size=16,                 # RAM kam chha bhane 8 garne
        show_progress_bar=True,
        normalize_embeddings=True,     # cosine similarity ko lagi
    )

    # ---------- 4.8 ChromaDB ma save (batch ma) ----------
    # Chroma ko payload limit bachauna 5000 chunk ek batch
    BATCH = 5000
    for i in range(0, len(all_ids), BATCH):
        collection.add(
            ids        = all_ids[i:i+BATCH],
            documents  = all_chunks[i:i+BATCH],
            metadatas  = all_metas[i:i+BATCH],
            embeddings = embeddings[i:i+BATCH].tolist(),
        )

    # ---------- 4.9 BM25 index banaune ra save ----------
    # Lowercase + whitespace split → word list
    tokenized = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25":      bm25,
            "ids":       all_ids,
            "chunks":    all_chunks,
            "metadatas": all_metas,
        }, f)

    # ---------- 4.10 Summary log ----------
    log.append("\n--- SUMMARY ---")
    log.append(f"Total PDFs: {len(seen_hashes)}")
    log.append(f"Total chunks: {len(all_chunks)}")
    log.append("PDFs per category:")
    for cat in sorted(category_counts):
        log.append(f"  {cat}: {category_counts[cat]}")

    LOG_PATH.write_text("\n".join(log), encoding="utf-8")
    print(f"Done. Log: {LOG_PATH}")


# =============================================================
# 5. Entry point
# =============================================================
# Yo file directly run garey matra main() call hunxa.
# Import garey chai hudaina — safe.
if __name__ == "__main__":
    main()