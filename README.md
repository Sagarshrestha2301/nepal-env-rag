# Nepal Environmental Studies RAG

A fully offline, local Retrieval-Augmented Generation (RAG) system for querying
Nepal-focused environmental research papers. Answers questions with **page-level
citations** from your own PDF library. No cloud, no API keys, no telemetry.

Built for low-spec machines: Windows 11, RTX 3050 (4 GB VRAM), 16 GB RAM.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [Hardware requirements](#hardware-requirements)
4. [Installation](#installation)
5. [Project structure](#project-structure)
6. [Usage](#usage)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Known limitations](#known-limitations)
10. [Roadmap: scaling to 20,000 papers](#roadmap-scaling-to-20000-papers)
11. [Licenses](#licenses)

---

## What it does

- Reads a folder of PDF research papers (English / Nepali / mixed).
- Extracts text, splits into chunks, and stores them in a local vector database.
- Answers natural-language questions using only the content of those papers.
- Every answer cites the paper and page number.
- Runs 100% offline after initial model download.

Example questions it can answer:

- "What are PM2.5 trends in Kathmandu Valley?"
- "Which studies discuss EIA implementation in Nepal?"
- "Impacts of hydropower on Koshi River biodiversity?"
- "What does the 1993 flood literature say about specific discharge in Bagmati?"

---

## How it works

```
PDFs
  ↓
PyMuPDF text extraction
  ↓
Chunking (2000 chars, 200 overlap)
  ↓
sentence-transformers (multilingual-e5-small, CPU)
  ↓
ChromaDB (dense vectors) + BM25 (sparse keywords)
  ↓
Hybrid retrieval (Reciprocal Rank Fusion)
  ↓
Ollama local LLM (qwen2.5:3b)
  ↓
Answer with page-level citations
```

**Why hybrid?** Dense embeddings catch semantic similarity; BM25 catches exact
terms like "Bagmati" or "PM2.5". Merging both gives better recall for scientific
papers than either alone.

**Why CPU embeddings?** Your 4 GB VRAM is better spent on the LLM. Small
multilingual embeddings run fast enough on CPU for a 100–500 paper corpus.

---

## Hardware requirements

### Tested on
- Windows 11
- NVIDIA RTX 3050 (4 GB VRAM)
- 16 GB RAM
- ~20 GB free disk for 100 papers

### Minimum
- Windows 10/11, macOS, or Linux
- 8 GB RAM (16 GB recommended)
- 10 GB free disk (100 papers)
- Any GPU with 4 GB VRAM, or CPU-only (slower)

### Recommended for 500+ papers
- 16 GB RAM
- 6 GB+ VRAM
- 50 GB free disk

### Not supported by this build
- 20,000+ paper corpora (see [Roadmap](#roadmap-scaling-to-20000-papers))
- Fine-tuning
- Reranking models (`bge-reranker-v2-m3` needs ~2 GB extra VRAM)

---

## Installation

### 1. Install Python 3.11
Download from [python.org](https://www.python.org/downloads/).
**Check "Add Python to PATH" during installation.**

### 2. Install Ollama
Download from [ollama.com/download](https://ollama.com/download).

### 3. Clone or create the project
```powershell
mkdir nepal-env-rag
cd nepal-env-rag
```

### 4. Create and activate a virtual environment
```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 5. Install Python dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install pymupdf sentence-transformers chromadb rank_bm25 ollama streamlit pandas tqdm numpy
```

> **Note:** we install CPU-only PyTorch. If you have a bigger GPU later,
> replace the torch line with the CUDA wheel from pytorch.org.

### 6. Verify installation
```powershell
python -c "import fitz, chromadb, sentence_transformers, ollama, streamlit; print('all OK')"
```
Should print `all OK`.

### 7. Pull the LLM (one model per command)
```powershell
ollama pull qwen2.5:3b
```

Optional slower/bigger models:
```powershell
ollama pull deepseek-r1:8b
ollama pull qwen3.5:latest
```

### 8. Create project folders
```powershell
mkdir data\raw_pdfs
mkdir index
```

### 9. Add your PDFs
Copy your PDFs into `data\raw_pdfs\`. Start with **100 papers** for testing.

---

## Project structure

```text
nepal-env-rag/
├── data/
│   └── raw_pdfs/           # Your PDF files live here
├── index/
│   ├── chroma/             # ChromaDB persistent store (auto-created)
│   ├── bm25.pkl            # Sparse index (auto-created)
│   └── ingest_log.txt      # Errors and skipped files (auto-created)
├── ingest.py               # Extract → chunk → embed → index
├── query.py                # Retrieval + LLM answer
├── eval.py                 # Batch test questions
├── app.py                  # Streamlit UI
├── requirements.txt
├── README.md
└── .venv/                  # Python virtual environment
```

---

## Usage

### Step 1 — Ingest PDFs
```powershell
python ingest.py
```

Expected output:
```
Found 100 PDFs
100%|██████████████████| 100/100 [10:24<00:00]
Total chunks: 8432
Ingestion done. Log: index/ingest_log.txt
```

Check `index/ingest_log.txt` for skipped or failed files:
```powershell
type index\ingest_log.txt
```

### Step 2 — Test with sample questions
```powershell
python eval.py
```

### Step 3 — Launch the UI
```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Step 4 — Re-ingest after adding papers
Any time you add new PDFs to `data\raw_pdfs\`:
```powershell
python ingest.py
```

The index is rebuilt from scratch. This is fine for ≤ 500 papers.

---

## Configuration

Edit these constants at the top of the scripts.

### `ingest.py`
```python
EMBED_MODEL = "intfloat/multilingual-e5-small"  # CPU, multilingual
CHUNK_SIZE = 2000                                # characters
CHUNK_OVERLAP = 200
```

### `query.py`
```python
LLM_MODEL = "qwen2.5:3b"      # fast, fits 4 GB VRAM
# LLM_MODEL = "deepseek-r1:8b"  # slower, deeper reasoning
# LLM_MODEL = "qwen3.5:latest"  # slower, high quality
```

Retrieval options:
```python
top_k=5                # number of chunks passed to LLM
num_ctx=4096           # LLM context window
temperature=0.2        # lower = more factual
```

### Model performance on RTX 3050 (4 GB VRAM)

| Model | Size | Speed | Quality |
|---|---|---|---|
| `qwen2.5:1.5b` | 1 GB | ~30 tok/s | Basic |
| `qwen2.5:3b` | 2 GB | ~15–25 tok/s | **Recommended** |
| `deepseek-r1:8b` | 5.2 GB | 3–8 tok/s | Deep reasoning |
| `qwen3.5:latest` | 6.6 GB | 5–15 tok/s | High quality |

Larger models spill from VRAM to RAM → much slower. Use `qwen2.5:3b` for
interactive work; use the bigger ones for batch analysis.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'fitz'`
You are running the wrong Python. Activate the venv first:
```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```
The output must end in `.venv\Scripts\python.exe`.

### `Error: accepts 1 arg(s), received 3`
Ollama pulls **one model per command**. Do not chain:
```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### `Connection refused` from Ollama
Start the Ollama service in another terminal:
```powershell
ollama serve
```
Or restart the Ollama desktop app.

### Streamlit opens but crashes on query
Same root cause as `fitz` error — Streamlit is using a different Python.
Stop it, activate the venv, and run `streamlit run app.py` again.

### PDFs are skipped with "NO TEXT (scanned?)"
Those PDFs are image-only scans. Options:
1. Skip them for now (fine for 100-paper MVP).
2. Add OCR later:
   ```powershell
   python -m pip install pytesseract pdf2image pillow
   ```
   You also need the Tesseract binary with `nep` language data:
   [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

### Ingestion is very slow
- Reduce `batch_size` in `ingest.py` (default 16) if RAM is tight.
- Or switch to a smaller embedding model:
  ```python
  EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  ```
  Slightly lower quality, ~2× faster.

### Answers contain no citations
Increase `top_k` in `query.py` from 5 to 8. Also confirm the reranked chunks
include `page` metadata — check `index/ingest_log.txt`.

### Year filter returns wrong results
The MVP uses filename-based year detection. Filenames like
`1970_Marahatta_..._W2121362616.pdf` may be a **Zotero accession year**, not the
publication year. This is a known limitation. See [Roadmap](#roadmap-scaling-to-20000-papers).

---

## Known limitations

- **No OCR** — scanned PDFs are skipped.
- **Naive metadata** — title/year come from filename, not extracted text.
- **No reranker** — top-k is chosen by hybrid score only.
- **Rebuilds on every ingest** — not incremental.
- **ChromaDB local** — single process, no concurrent writes.
- **No conversation memory** — each query is independent.
- **No feedback loop** — user corrections are not stored.

---

## Roadmap: scaling to 20,000 papers

When you're ready to move past the 100-paper MVP:

| Change | Reason |
|---|---|
| Migrate to **Qdrant** | Handles millions of vectors, filtered search |
| Add **OpenSearch / Tantivy** | Better BM25 than in-memory `rank_bm25` |
| Use **`bge-m3`** embeddings | Stronger multilingual retrieval |
| Add **`bge-reranker-v2-m3`** | Big precision boost (needs 24 GB GPU) |
| Extract **real metadata** | Title, authors, year, DOI from PDF text |
| Add **OCR pipeline** | Tesseract `nep+eng` for scanned reports |
| **Incremental ingestion** | Hash-based diffing, no full rebuild |
| **PostgreSQL** for metadata | Faster filters, versioning |
| **FastAPI** backend | Decouple UI from processing |
| **Langfuse** self-hosted | Trace queries, measure quality |
| **Docker Compose** | Reproducible multi-service deployment |

Long-term: fine-tune a Nepali-capable LLM (LoRA on Qwen2.5) for domain tone.
Keep **facts in RAG**, use fine-tuning only for style and language quality.

---

## Extending the project

### Add a new document type
Edit `ingest.py` → extend `extract_pages()` to handle `.docx`, `.txt`, `.html`.

### Add metadata filters
1. Extract fields in `guess_metadata()`.
2. Store them in `all_metas`.
3. Pass `where={"year": {"$gte": 2015}}` to `collection.query()` in `query.py`.

### Swap the LLM
Change `LLM_MODEL` in `query.py`. Any Ollama model works.

### Use a bigger GPU
Reinstall PyTorch with CUDA, and change:
```python
model = SentenceTransformer(EMBED_MODEL, device="cuda")
```

### Add a reranker
```powershell
python -m pip install sentence-transformers
```
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
```
Requires ~1 GB VRAM. Only feasible on 6 GB+ GPUs.

---

## Licenses

All components are open source and free for research and commercial use:

| Component | License |
|---|---|
| PyMuPDF | AGPL-3.0 / commercial |
| sentence-transformers | Apache-2.0 |
| multilingual-e5-small | MIT |
| ChromaDB | Apache-2.0 |
| rank-bm25 | Apache-2.0 |
| Ollama | MIT |
| Qwen2.5 | Apache-2.0 |
| DeepSeek-R1 | MIT |
| Streamlit | Apache-2.0 |

**Everything runs offline** after the first model download. No API keys,
no telemetry, no cloud calls.

---

## Acknowledgements

Built for the Nepal environmental research community. Contributions welcome:
- OCR pipelines for Nepali documents
- Metadata extraction for South Asian journals
- Domain-specific evaluation sets
- Province / district / river-basin filter ontologies

---

## Quick reference card

```powershell
# Setup
cd nepal-env-rag
.\.venv\Scripts\Activate.ps1

# Add PDFs to data\raw_pdfs\

# Ingest
python ingest.py

# Test
python eval.py

# UI
streamlit run app.py

# Add new papers → re-run ingest
python ingest.py
```

**One-liner health check:**
```powershell
python -c "import fitz, chromadb, sentence_transformers, ollama, streamlit; print('OK')"; ollama list
```
