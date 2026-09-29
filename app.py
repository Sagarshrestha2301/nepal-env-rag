"""
app.py: offline chat app for nepal-env-rag.

A local research assistant over the Nepal environment corpus, laid out like a
chat app: streaming answers, citation chips, source cards with the exact
passage, an "Open PDF" button, citation checks, filters and follow-ups.

Run:
    streamlit run app.py

Everything runs on this computer: LanceDB (search), bge-m3 on CPU (query
embedding) and Ollama (answers). No data leaves the machine.
"""

from __future__ import annotations

import os

# Offline mode: must come before anything that loads the embedding model.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import html
import re
from datetime import datetime

import streamlit as st

from answer import DEFAULT_K, DEFAULT_MODEL, AnswerResult, answer_question, apa_author, apa_pages
from retriever import DB_DIR, Retriever

RAW_DIR = os.path.join("data", "raw")
WINDOWS_BAD_CHARS = re.compile(r'[<>:"\\|?*\x00-\x1f]')
FOLLOW_UP_MAX_WORDS = 8   # short questions after a previous one are treated as follow-ups
ASSISTANT_AVATAR = "🏔️"

# (button label, question sent)
EXAMPLES = [
    ("🌊  What caused the 2021 Melamchi flood?",
     "What caused the Melamchi flood in 2021 and what damage did it cause?"),
    ("🧊  Which glacial lakes are dangerous?",
     "Which glacial lakes in Nepal are considered dangerous?"),
    ("🌲  Has community forestry grown forest cover?",
     "How has community forestry affected forest cover in Nepal?"),
    ("🔥  नेपालमा वन डढेलोका कारणहरू के हुन्?",
     "नेपालमा वन डढेलोका मुख्य कारणहरू के हुन्?"),
]

# Citations in the answer: [Kshetri, 2024, pp. 3–4], [~Kshetri, 2024] or [S2]
CITE_RE = re.compile(
    r"\[(~?[^\[\]\n]{1,120}?(?:\d{4}|n\.d\.)[^\[\]\n]{0,60})\]"
    r"|\[(S\d+(?:\s*[,–-]\s*S?\d+)*)\]"
)

RIDGE_SVG = """
<svg class="ridge" width="220" height="56" viewBox="0 0 220 56" aria-hidden="true">
  <defs>
    <linearGradient id="ridge-grad" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0" stop-color="#0071e3"/>
      <stop offset="1" stop-color="#5ac8fa"/>
    </linearGradient>
  </defs>
  <path d="M2 52 L28 38 L42 44 L68 17 L82 29 L100 6 L118 27 L134 21 L158 39 L178 31 L218 52"
        fill="none" stroke="url(#ridge-grad)" stroke-width="2.5"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

CSS = """
<style>
  :root {
    --ink: #1d1d1f;
    --ink-2: #6e6e73;
    --ink-3: #86868b;
    --line: #e5e5ea;
    --fill: #f5f5f7;
    --bubble: #f2f2f7;
    --accent: #0071e3;
    --accent-soft: #e8f1fc;
    --warn: #a65200;
    --warn-soft: #fff5e6;
    --ok: #34c759;
    --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable Text",
            "Segoe UI", system-ui, "Nirmala UI", "Noto Sans Devanagari", sans-serif;
  }

  /* Base */
  .stApp { font-family: var(--font); color: var(--ink); background: #fff; }
  .stApp textarea, .stApp input, .stApp button { font-family: var(--font); }
  .block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 8rem; }
  header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
  }
  footer { visibility: hidden; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: var(--fill); border-right: 1px solid var(--line); }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
  .brand { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem; }
  .brand svg { flex: none; }
  .brand-name { font-weight: 600; font-size: 1.02rem; letter-spacing: -0.01em; line-height: 1.2; }
  .brand-sub { color: var(--ink-2); font-size: 0.8rem; }
  .side-label { font-size: 0.8rem; font-weight: 600; color: var(--ink-2); margin: 1.4rem 0 0.25rem; }
  .offline {
    display: flex; gap: 0.55rem; align-items: flex-start; margin-top: 1.5rem;
    font-size: 0.8rem; line-height: 1.4; color: var(--ink-2);
    background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 0.65rem 0.8rem;
  }
  .offline .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok);
                  margin-top: 0.35rem; flex: none; box-shadow: 0 0 0 3px rgba(52, 199, 89, 0.18); }

  /* Buttons */
  .stButton button, .stDownloadButton button {
    border-radius: 12px; border: 1px solid var(--line); background: #fff; color: var(--ink);
    font-weight: 500; box-shadow: none; transition: background 0.15s, border-color 0.15s;
  }
  .stButton button:hover, .stDownloadButton button:hover {
    background: var(--fill); border-color: #d2d2d7; color: var(--ink);
  }
  .stButton button:active { transform: scale(0.98); }
  .stButton button:focus-visible, .stDownloadButton button:focus-visible {
    outline: none; box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.25);
  }
  .stButton button[kind="primary"] {
    background: var(--accent); border: none; color: #fff; border-radius: 999px;
  }
  .stButton button[kind="primary"]:hover { background: #0077ed; color: #fff; }

  /* Welcome screen */
  .hero { text-align: center; margin: 10vh 0 2.25rem; }
  .hero h1 {
    font-size: 2.2rem; font-weight: 600; letter-spacing: -0.025em; line-height: 1.15;
    color: var(--ink); margin: 0.9rem 0 0.5rem; padding: 0;
  }
  .hero h1 a { display: none; }
  .hero p { color: var(--ink-2); font-size: 1.05rem; line-height: 1.5; max-width: 30rem; margin: 0 auto; }
  .ridge path { stroke-dasharray: 420; stroke-dashoffset: 420;
                animation: draw 1.6s cubic-bezier(0.2, 0.7, 0.2, 1) 0.1s forwards; }
  @keyframes draw { to { stroke-dashoffset: 0; } }
  @media (prefers-reduced-motion: reduce) { .ridge path { animation: none; stroke-dashoffset: 0; } }

  /* Suggestion cards (the only side-by-side buttons in the main area) */
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] .stButton button,
  section.main [data-testid="stHorizontalBlock"] .stButton button {
    justify-content: flex-start; text-align: left; min-height: 60px;
    padding: 0.8rem 1rem; border-radius: 16px; background: #fff;
  }
  [data-testid="stHorizontalBlock"] .stButton button p { text-align: left; font-size: 0.93rem; }

  /* Messages */
  .user-row { display: flex; justify-content: flex-end; margin: 1.5rem 0 0.75rem; }
  .user-bubble {
    background: var(--bubble); color: var(--ink); border-radius: 20px 20px 6px 20px;
    padding: 0.65rem 1rem; max-width: 80%; font-size: 1rem; line-height: 1.5;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  [data-testid="stChatMessage"] { background: transparent; padding: 0.25rem 0; gap: 0.85rem; }
  [data-testid="stChatMessage"] .stMarkdown p,
  [data-testid="stChatMessage"] .stMarkdown li { font-size: 1.02rem; line-height: 1.68; }
  .cite {
    display: inline-block; white-space: nowrap; font-size: 0.76rem; font-weight: 500;
    line-height: 1.5; color: var(--accent); background: var(--accent-soft);
    border: 1px solid transparent; border-radius: 999px; padding: 0 0.5rem; margin: 0 0.1rem;
    vertical-align: 1px;
  }
  .cite.auto { border: 1px dashed rgba(0, 113, 227, 0.5); }
  .cursor { display: inline-block; width: 0.5em; height: 1.05em; margin-left: 2px;
            background: var(--ink); border-radius: 2px; vertical-align: -2px;
            animation: blink 1s steps(2, start) infinite; }
  @keyframes blink { to { visibility: hidden; } }
  .stats { display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; color: var(--ink-3);
           font-size: 0.8rem; margin: 0.25rem 0 0.9rem; }
  .auto-note { color: var(--ink-3); font-size: 0.8rem; margin: -0.5rem 0 0.9rem; }

  /* Expanders and status */
  [data-testid="stExpander"] details {
    border: 1px solid var(--line); border-radius: 14px; background: #fff; overflow: hidden;
  }
  [data-testid="stExpander"] summary { font-weight: 500; font-size: 0.92rem; }
  [data-testid="stExpander"] summary:hover { color: var(--accent); }
  .check { background: var(--warn-soft); color: var(--warn); border-radius: 10px;
           padding: 0.55rem 0.75rem; font-size: 0.88rem; line-height: 1.45; margin-bottom: 0.5rem; }

  /* Source cards */
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; border-color: var(--line); }
  .src-title { font-weight: 600; font-size: 0.95rem; line-height: 1.35; }
  .src-author { color: var(--ink-2); font-size: 0.85rem; margin-top: 0.15rem; }
  .tags { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.55rem 0 0.6rem; }
  .tag { font-size: 0.74rem; color: var(--ink-2); background: var(--fill);
         border-radius: 6px; padding: 0.1rem 0.45rem; }
  .src-text { font-size: 0.88rem; line-height: 1.55; color: #3a3a3c;
              border-left: 2px solid var(--line); padding-left: 0.75rem; }

  /* Chat input */
  [data-testid="stBottom"] > div { background: linear-gradient(to top, #fff 75%, rgba(255, 255, 255, 0)); }
  [data-testid="stChatInput"] {
    border-radius: 26px; border: 1px solid var(--line); background: #fff;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 8px 28px rgba(0, 0, 0, 0.07);
  }
  [data-testid="stChatInput"]:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.14), 0 8px 28px rgba(0, 0, 0, 0.07);
  }
  [data-testid="stChatInput"] textarea { font-size: 1rem; }

  @media (max-width: 640px) {
    .block-container { padding-top: 1.5rem; }
    .hero { margin-top: 5vh; }
    .hero h1 { font-size: 1.75rem; }
    .user-bubble { max-width: 90%; }
  }
</style>
"""


# --- Helpers ----------------------------------------------------------------

def local_pdf_path(drive_path: str) -> str:
    """Where download_drive.py saved this file (same sanitizing rules)."""
    parts = [p for p in drive_path.strip("/").split("/") if p]
    safe = [WINDOWS_BAD_CHARS.sub("_", p).rstrip(" .") or "_" for p in parts]
    return os.path.abspath(os.path.join(RAW_DIR, *safe))


def search_query_for(question: str, history: list[dict]) -> str:
    """Short follow-ups ('what about deaths?') borrow the previous question's topic."""
    previous = [m["content"] for m in history if m["role"] == "user"]
    if previous and len(question.split()) <= FOLLOW_UP_MAX_WORDS:
        return f"{previous[-1]} {question}"
    return question


def list_ollama_models() -> list[str]:
    try:
        import ollama
        response = ollama.list()
        models = getattr(response, "models", None) or response.get("models", [])
        names = [getattr(m, "model", None) or m.get("name") or m.get("model") for m in models]
        return sorted(n for n in names if n)
    except Exception:
        return []


def open_file(path: str) -> None:
    if not os.path.exists(path):
        st.toast(f"PDF not found at {path}")
        return
    if hasattr(os, "startfile"):   # Windows
        os.startfile(path)
    else:
        import subprocess
        import sys
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


def greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning."
    if hour < 17:
        return "Good afternoon."
    return "Good evening."


def format_answer(text: str) -> str:
    """Escape HTML, stop '$' turning into math, and draw citations as chips."""
    safe = html.escape(text, quote=False).replace("$", "\\$")

    def chip(match: re.Match) -> str:
        label = match.group(1) or match.group(2)
        if label.startswith("~"):
            return (f'<span class="cite auto" title="Citation set automatically">'
                    f'{label[1:].strip()}</span>')
        return f'<span class="cite">{label}</span>'

    return CITE_RE.sub(chip, safe)


def clean_snippet(text: str, limit: int = 600) -> str:
    """PDF text has hard line breaks mid-sentence; join them and escape for HTML."""
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) > limit:
        flat = flat[:limit].rsplit(" ", 1)[0] + "…"
    return html.escape(flat)


def answer_markdown(question: str, result: AnswerResult) -> str:
    lines = [f"# {question}", "", result.apa]
    if result.references:
        lines += ["", "## References", ""] + [f"- {r}" for r in result.references]
    return "\n".join(lines)


# --- Cached resources -------------------------------------------------------

@st.cache_resource(show_spinner="Loading the library for the first time. This takes a minute…")
def load_retriever() -> Retriever:
    return Retriever(device="cpu")  # GPU stays free for the LLM


@st.cache_data(ttl=60, show_spinner=False)
def cached_models() -> list[str]:
    return list_ollama_models()


# --- Rendering --------------------------------------------------------------

def render_user(text: str) -> None:
    st.markdown(f"<div class='user-row'><div class='user-bubble'>{html.escape(text)}</div></div>",
                unsafe_allow_html=True)


def render_source(hit: dict, key: str) -> None:
    with st.container(border=True):
        tags = [apa_pages(hit["page_start"], hit["page_end"]), hit["collection"], hit["category"]]
        if hit.get("section"):
            tags.insert(1, hit["section"][:60])
        tag_html = "".join(f"<span class='tag'>{html.escape(str(t))}</span>" for t in tags if t)
        st.markdown(
            f"<div class='src-title'>{html.escape(str(hit['title']))}</div>"
            f"<div class='src-author'>{html.escape(apa_author(hit))}, {hit['year'] or 'n.d.'}</div>"
            f"<div class='tags'>{tag_html}</div>"
            f"<div class='src-text'>{clean_snippet(hit['text'])}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Open PDF", key=key):
            open_file(local_pdf_path(hit["path"]))


def render_result(result: AnswerResult, msg_index: int, question: str) -> None:
    if result.error:
        st.error(result.error)
        return

    st.markdown(format_answer(result.apa), unsafe_allow_html=True)

    st.markdown(
        "<div class='stats'>"
        f"<span>{len(result.used)} sources</span>"
        f"<span>Search {result.search_s:.1f} s</span>"
        f"<span>Answer {result.gen_s:.1f} s</span>"
        f"<span>{html.escape(result.model)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    if result.added or result.fixed:
        parts = []
        if result.added:
            parts.append(f"{result.added} added")
        if result.fixed:
            parts.append(f"{result.fixed} moved to the source that contains the numbers")
        st.markdown(f"<div class='auto-note'>Dashed citations were set automatically "
                    f"({', '.join(parts)}).</div>", unsafe_allow_html=True)

    checks = []
    if result.invalid:
        checks.append(f"The answer cited sources that don't exist "
                      f"({', '.join(f'S{n}' for n in result.invalid)}).")
    if result.uncited_warning:
        checks.append("The answer has no citations. Verify it against the sources below.")
    for number, sentence in result.problems:
        checks.append(f"“{number}” doesn't appear in the cited source: “{sentence[:160]}”")
    if checks:
        label = "1 thing to double-check" if len(checks) == 1 else f"{len(checks)} things to double-check"
        with st.expander(f"⚠️ {label}", expanded=True):
            for c in checks:
                st.markdown(f"<div class='check'>{html.escape(c)}</div>", unsafe_allow_html=True)

    if result.cited:
        cited_hits, seen = [], set()
        for n in result.cited:
            hit = result.used[n - 1]
            if hit["chunk_id"] not in seen:
                seen.add(hit["chunk_id"])
                cited_hits.append((n, hit))
        with st.expander(f"Sources cited ({len(cited_hits)})"):
            for n, hit in cited_hits:
                render_source(hit, key=f"open_{msg_index}_{n}")

    if result.uncited:
        with st.expander(f"Also found ({len(result.uncited)})"):
            for i, hit in enumerate(result.uncited):
                render_source(hit, key=f"open_u_{msg_index}_{i}")

    st.download_button("Save answer", data=answer_markdown(question, result),
                       file_name="answer.md", mime="text/markdown", key=f"dl_{msg_index}")


def render_history() -> None:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            render_user(msg["content"])
        else:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                render_result(msg["result"], i, msg.get("question", ""))


def render_welcome() -> None:
    st.markdown(
        f"<div class='hero'>{RIDGE_SVG}<h1>{greeting()}</h1>"
        "<p>Ask about floods, glaciers, forests, climate or air quality in Nepal. "
        "Every answer cites the papers and reports it comes from.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (label, question) in enumerate(EXAMPLES):
        if cols[i % 2].button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = question
            st.rerun()


# --- Sidebar ----------------------------------------------------------------

def sidebar() -> tuple[str, int, dict]:
    with st.sidebar:
        small_ridge = RIDGE_SVG.replace('width="220" height="56"', 'width="34" height="22"') \
                               .replace('class="ridge"', 'class="mark"') \
                               .replace("ridge-grad", "mark-grad") \
                               .replace('stroke-width="2.5"', 'stroke-width="12"')
        st.markdown(
            f"<div class='brand'>{small_ridge}<div>"
            "<div class='brand-name'>Nepal Environment</div>"
            "<div class='brand-sub'>Research assistant</div></div></div>",
            unsafe_allow_html=True,
        )

        if st.button("New chat", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("<div class='side-label'>Answering</div>", unsafe_allow_html=True)
        models = cached_models()
        if not models:
            st.warning("Can't reach Ollama. Open the Ollama app, then reload this page.")
            models = [DEFAULT_MODEL]
        default = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
        model = st.selectbox("Model", models, index=default)
        k = st.slider("Sources per answer", 3, 10, DEFAULT_K,
                      help="More sources give broader answers but take longer.")

        st.markdown("<div class='side-label'>Search in</div>", unsafe_allow_html=True)
        collection = st.radio("Documents", ["Everything", "Papers", "Reports"], horizontal=True)
        lang = st.radio("Language", ["Any", "English", "Nepali"], horizontal=True)
        year_min = year_max = None
        if st.toggle("Only certain years"):
            year_min, year_max = st.slider("Years", 1950, 2026, (2010, 2026))

        st.markdown(
            "<div class='offline'><span class='dot'></span>"
            "<span>Runs on this computer. Your questions and documents never leave it.</span></div>",
            unsafe_allow_html=True,
        )

    filters = {
        "collection": {"Everything": None, "Papers": "papers", "Reports": "reports"}[collection],
        "category": None,
        "year_min": year_min,
        "year_max": year_max,
        "lang": {"Any": None, "English": "en", "Nepali": "ne"}[lang],
    }
    return model, k, filters


# --- Main -------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Nepal Environment Research",
                       page_icon="🏔️", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    if not os.path.isdir(DB_DIR):
        st.error("The search index is missing. Run `python build_index.py`, then reload this page.")
        st.stop()

    st.session_state.setdefault("messages", [])
    model, k, filters = sidebar()
    retriever = load_retriever()

    # chat_input always sits at the bottom, so reading it first is safe.
    question = st.chat_input("Ask about Nepal's environment, in English or नेपाली") \
        or st.session_state.pop("pending", None)

    if not st.session_state.messages and not question:
        render_welcome()
    render_history()
    if not question:
        return

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    render_user(question)

    msg_index = len(st.session_state.messages)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        status = st.status("Searching the library…", expanded=False)
        live = st.empty()
        pieces: list[str] = []

        def on_text(text: str) -> None:
            if not pieces:
                status.update(label="Writing the answer…")
            pieces.append(text)
            live.markdown(format_answer("".join(pieces)) + "<span class='cursor'></span>",
                          unsafe_allow_html=True)

        result = answer_question(retriever, question, model, k, filters, on_text=on_text,
                                 search_query=search_query_for(question, history))
        live.empty()
        if result.error:
            status.update(label="Couldn't answer", state="error")
        else:
            status.update(label=f"Read {len(result.used)} sources", state="complete")
        render_result(result, msg_index, question)

    st.session_state.messages.append({"role": "assistant", "result": result, "question": question})


if __name__ == "__main__":
    main()