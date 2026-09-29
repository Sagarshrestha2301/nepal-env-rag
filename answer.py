"""
answer.py — Step 8 of nepal-env-rag: answer questions with citations, offline.

Pipeline per question:
  1. Hybrid search (retriever.py) finds the best chunks
  2. Chunks become numbered sources [1]..[n] with title, year, pages
  3. A local LLM (Ollama) answers ONLY from those sources, citing [n]
  4. Citations are checked and the source list is printed

Usage:
    python answer.py "What caused the Melamchi flood in 2021?"
    python answer.py                                  # interactive: models load once
    python answer.py "GLOF risk of Imja lake" --year-min 2015 --collection papers
    python answer.py "..." --model qwen3.5 --k 8      # compare another model

Requires: Ollama running (ollama.com) with the model pulled, e.g. `ollama pull qwen2.5:3b`.
"""

from __future__ import annotations

import os

# Everything is cached locally; never touch the network for Hugging Face models.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import re
import sys
import time
from dataclasses import dataclass, field

from retriever import Retriever, format_pages

# --- Configuration --------------------------------------------------------

DEFAULT_MODEL = "qwen2.5:3b"   # fits a 4 GB GPU fully
DEFAULT_K = 6                  # sources given to the LLM
CONTEXT_CHAR_BUDGET = 14_000   # ~3,500 tokens of sources; leaves room to answer
NUM_CTX = 8192                 # Ollama's default window is small; set it explicitly
TEMPERATURE = 0.1              # low: stick to the sources, don't improvise

SYSTEM_PROMPT = """You are a research assistant for Nepal's environmental research literature.

Rules:
- Answer ONLY with information from the sources provided. Each source is labelled S1, S2, S3...
- Cite every factual statement with the label of the source it comes from, in square brackets.
  Example: "The flood damaged 50 houses [S2] and blocked the highway [S1][S3]."
- Only use labels that appear in the sources. Check that the fact really is in the source you cite.
- If the sources do not contain the answer, say: "I could not find this in the documents." Do not guess.
- Never invent numbers, dates, names or places that are not in the sources.
- Be concise: a short paragraph or a few bullet points. State each fact only once.
- Answer in the same language as the question."""

# Our citations: S1..Sn in any style the model uses: [S3], [S2, S5], (S3, p4-5), bare S3.
# Papers' own "[12]" references are stripped from source text so they can't be confused.
SOURCE_LABEL = re.compile(r"\bS\s?(\d{1,2})\b")
CITE_GROUP = re.compile(r"[\[(][^\])]{0,40}?\bS\s?\d{1,2}\b[^\])]{0,40}[\])]")
PAPER_REF = re.compile(r"\s?\[\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\]")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
SENTENCE = re.compile(r"(?<=[.!?।)\]])\s+(?=[A-Z0-9\u0900-\u097F\"(])")


# --- Prompt building ------------------------------------------------------

def clean_source_text(text: str) -> str:
    """Remove the paper's own numeric citations like [12] or [15, 16]."""
    return PAPER_REF.sub("", text).strip()


def build_sources(hits: list[dict], budget: int = CONTEXT_CHAR_BUDGET) -> tuple[str, list[dict]]:
    """Labelled source blocks (S1..Sn) within a character budget. Returns (text, used_hits)."""
    blocks, used, spent = [], [], 0
    for hit in hits:
        label = f"S{len(used) + 1}"
        header = f"[{label}] {hit['title']} ({hit['year'] or 'n.d.'}), {format_pages(hit)}"
        if hit.get("section"):
            header += f", section: {hit['section']}"
        block = f"{header}\n{clean_source_text(hit['text'])}\n[end of {label}]"
        if used and spent + len(block) > budget:
            break
        blocks.append(block)
        used.append(hit)
        spent += len(block)
    return "\n\n".join(blocks), used


def build_messages(question: str, sources_text: str) -> list[dict]:
    # Question comes after the sources, and the citation rule is repeated last:
    # small models follow the most recent instruction best.
    user = (f"Sources:\n\n{sources_text}\n\nQuestion: {question}\n\n"
            "Answer using only the sources above. End every sentence with its "
            "source label in square brackets, like [S1] or [S2][S3].")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# --- Output checks --------------------------------------------------------

def cited_labels(text: str) -> list[int]:
    """Source numbers cited anywhere in the text (S3, [S3], (S2, p4-5), ...)."""
    return [int(n) for n in SOURCE_LABEL.findall(text)]


def check_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """(valid cited source numbers, invalid numbers that point to no source)."""
    cited = sorted(set(cited_labels(answer)))
    valid = [n for n in cited if 1 <= n <= n_sources]
    invalid = [n for n in cited if n not in valid]
    return valid, invalid


def normalize_number(n: str) -> str:
    return n.replace(",", "").rstrip(".")


# --- Automatic citations ----------------------------------------------------

WORD = re.compile(r"[a-z\u0900-\u097F]{4,}|\d+(?:[.,]\d+)*")
STOPWORDS = {
    "that", "this", "with", "from", "were", "have", "been", "which", "their", "there",
    "also", "into", "such", "than", "these", "those", "about", "after", "over", "including",
    "caused", "resulted", "flood", "floods", "nepal", "combination", "factors", "significant",
}
AUTO_CITE_MIN_SCORE = 0.35   # share of a sentence's key words found in the source


def key_terms(text: str) -> set[str]:
    return {normalize_number(t) for t in WORD.findall(text.lower()) if t not in STOPWORDS}


def best_source(sentence: str, source_terms: list[set[str]]) -> tuple[int, float]:
    """(1-based source number, score) for the source that best supports a sentence.

    If the sentence has numbers, prefer sources containing ALL of them: numbers
    are the most specific evidence. Among candidates, pick by shared key words
    (numbers count double)."""
    terms = key_terms(sentence)
    if not terms:
        return 0, 0.0
    numbers = {t for t in terms if t[0].isdigit()}
    candidates = [i for i, src in enumerate(source_terms) if numbers <= src] if numbers else []
    if not candidates:
        candidates = list(range(len(source_terms)))

    weight = {t: (2.0 if t[0].isdigit() else 1.0) for t in terms}
    total = sum(weight.values())
    scores = {i: sum(weight[t] for t in terms & source_terms[i]) / total for i in candidates}
    best = max(scores, key=scores.get)
    return best + 1, scores[best]


def auto_cite(answer: str, used: list[dict]) -> tuple[str, int]:
    """Add [~Sn] to sentences the model left uncited, keeping paragraph breaks.
    Returns (text, sentences_added)."""
    source_terms = [key_terms(clean_source_text(h["text"])) for h in used]
    paragraphs, added = [], 0
    for paragraph in re.split(r"\n\s*\n", answer):
        out = []
        for sentence in SENTENCE.split(paragraph.strip()):
            if not sentence or cited_labels(sentence) or len(sentence.split()) < 4:
                out.append(sentence)
                continue
            n, score = best_source(sentence, source_terms)
            if n and score >= AUTO_CITE_MIN_SCORE:
                stripped = sentence.rstrip()
                end = stripped[-1] if stripped[-1] in ".!?।" else ""
                body = stripped[:-1] if end else stripped
                out.append(f"{body} [~S{n}]{end}")
                added += 1
            else:
                out.append(sentence)
        paragraphs.append(" ".join(out))
    return "\n\n".join(p for p in paragraphs if p), added


def repair_citations(answer: str, used: list[dict]) -> tuple[str, int]:
    """Move a sentence's citation to the source that actually contains its numbers.

    If a sentence's numbers aren't in the source(s) it cites but ALL of them are in
    another retrieved source, re-cite it as [~Sn], preferring another chunk of the
    same document (usually just a page correction). Returns (text, sentences_fixed).
    """
    texts = [clean_source_text(h["text"]) for h in used]
    source_numbers = [{normalize_number(n) for n in NUMBER.findall(t)} for t in texts]
    source_terms = [key_terms(t) for t in texts]

    paragraphs, fixed = [], 0
    for paragraph in re.split(r"\n\s*\n", answer):
        out = []
        for sentence in SENTENCE.split(paragraph.strip()):
            labels = [n for n in cited_labels(sentence) if 1 <= n <= len(used)]
            body = SOURCE_LABEL.sub("", CITE_GROUP.sub("", sentence))
            numbers = {normalize_number(n) for n in NUMBER.findall(body)}
            cited_nums = set().union(*(source_numbers[n - 1] for n in labels)) if labels else set()
            if not labels or not numbers or numbers <= cited_nums:
                out.append(sentence)
                continue

            candidates = [i for i, nums in enumerate(source_numbers) if numbers <= nums]
            if not candidates:
                out.append(sentence)  # nowhere to move it; the CHECK warning stays
                continue
            cited_docs = {used[n - 1]["doc_id"] for n in labels}
            same_doc = [i for i in candidates if used[i]["doc_id"] in cited_docs]
            pool = same_doc or candidates
            terms = key_terms(body)
            best = max(pool, key=lambda i: len(terms & source_terms[i]))

            stripped = CITE_RUN.sub("", sentence).rstrip()
            end = stripped[-1] if stripped and stripped[-1] in ".!?।" else ""
            core = (stripped[:-1] if end else stripped).rstrip()
            out.append(f"{core} [~S{best + 1}]{end}")
            fixed += 1
        paragraphs.append(" ".join(s for s in out if s))
    return "\n\n".join(p for p in paragraphs if p), fixed


def ungrounded_numbers(answer: str, used: list[dict]) -> list[tuple[str, str]]:
    """Numbers in the answer that don't appear in the source(s) the sentence cites.

    Returns (number, sentence) pairs. Sentences without citations are checked
    against all sources. Small models often attach a true fact to the wrong source;
    this catches the numeric cases cheaply.
    """
    texts = [clean_source_text(h["text"]) for h in used]
    source_numbers = [{normalize_number(n) for n in NUMBER.findall(t)} for t in texts]
    everywhere = set().union(*source_numbers) if source_numbers else set()

    problems = []
    for sentence in SENTENCE.split(answer):
        labels = [n for n in cited_labels(sentence) if 1 <= n <= len(used)]
        allowed = set().union(*(source_numbers[n - 1] for n in labels)) if labels else everywhere
        # Drop citation groups like "(S2, p4-5)" and bare labels before reading numbers.
        body = SOURCE_LABEL.sub("", CITE_GROUP.sub("", sentence))
        for n in NUMBER.findall(body):
            if normalize_number(n) not in allowed:
                problems.append((n, sentence.strip()))
    return problems


class ThinkFilter:
    """Hides <think>...</think> reasoning that some models stream before answering."""

    def __init__(self):
        self.buffer = ""
        self.inside = False

    def feed(self, piece: str) -> str:
        self.buffer += piece
        out = ""
        while self.buffer:
            if self.inside:
                end = self.buffer.find("</think>")
                if end == -1:
                    self.buffer = self.buffer[-8:]  # keep a tail in case the tag is split
                    return out
                self.buffer = self.buffer[end + len("</think>"):]
                self.inside = False
            else:
                start = self.buffer.find("<think>")
                if start == -1:
                    safe = len(self.buffer) - 7  # hold back a possible partial "<think>"
                    if safe <= 0:
                        return out
                    out += self.buffer[:safe]
                    self.buffer = self.buffer[safe:]
                    return out
                out += self.buffer[:start]
                self.buffer = self.buffer[start + len("<think>"):]
                self.inside = True
        return out

    def flush(self) -> str:
        rest = "" if self.inside else self.buffer
        self.buffer = ""
        return rest


# --- LLM ------------------------------------------------------------------

def _print_text(text: str) -> None:
    print(text, end="", flush=True)


def generate(model: str, messages: list[dict], on_text=_print_text) -> str:
    """Stream the answer through on_text(piece) and return the full text."""
    import ollama

    options = {"temperature": TEMPERATURE, "num_ctx": NUM_CTX}
    try:
        stream = ollama.chat(model=model, messages=messages, options=options,
                             stream=True, think=False)
    except TypeError:  # older ollama client without the `think` parameter
        stream = ollama.chat(model=model, messages=messages, options=options, stream=True)

    think = ThinkFilter()
    parts = []
    for part in stream:
        text = think.feed(part["message"]["content"])
        if text:
            on_text(text)
        parts.append(text)
    tail = think.flush()
    if tail:
        on_text(tail)
    parts.append(tail)
    return THINK_BLOCK.sub("", "".join(parts)).strip()


# --- Cleanup and APA rendering ----------------------------------------------

# A trailing run of 2+ labels after the last sentence, e.g. "... [S5]. [S1][S2][S3]"
LABEL_DUMP = re.compile(r"(?<=[.!?।])\s*(?:\[\s*~?\s*S\s?\d{1,2}[^\]]*\]\s*){2,}$")
LABEL_ONLY = re.compile(r"^\s*(?:\[\s*~?\s*S\s?\d{1,2}[^\]]*\]\s*)+$")
# One citation "run": consecutive [..S..] groups, or "(S3, p4-5)" styles.
CITE_RUN = re.compile(r"(?:\[\s*~?[^\]]{0,40}?\bS\s?\d{1,2}\b[^\]]{0,40}\]\s?)+"
                      r"|\(\s*~?\s*S\s?\d{1,2}\b[^)]{0,40}\)")

APA_BRACKETS = "[]"   # APA standard is "()"; square brackets as requested


def remove_label_dumps(answer: str) -> str:
    """Drop citation lists that aren't attached to a statement."""
    paragraphs = []
    for p in re.split(r"\n\s*\n", answer.strip()):
        if LABEL_ONLY.match(p):
            continue
        paragraphs.append(LABEL_DUMP.sub("", p.rstrip()))
    return "\n\n".join(paragraphs)


def apa_author(hit: dict) -> str:
    if hit.get("author"):
        return hit["author"]
    if hit.get("collection") == "reports" and hit.get("category"):
        return hit["category"].replace("_", " ")   # organization as author
    return "Unknown"


def apa_pages(start: int, end: int) -> str:
    return f"p. {start}" if start == end else f"pp. {start}–{end}"


def to_apa(answer: str, used: list[dict]) -> str:
    """Replace S-labels with APA in-text citations, merging chunks of the same document."""
    open_b, close_b = APA_BRACKETS

    def render(match: re.Match) -> str:
        run = match.group(0)
        auto = "~" in run
        labels = [n for n in cited_labels(run) if 1 <= n <= len(used)]
        if not labels:
            return run
        docs: dict[str, dict] = {}
        for n in labels:
            h = used[n - 1]
            d = docs.setdefault(h["doc_id"], {"hit": h, "start": h["page_start"], "end": h["page_end"]})
            d["start"] = min(d["start"], h["page_start"])
            d["end"] = max(d["end"], h["page_end"])
        parts = [f"{apa_author(d['hit'])}, {d['hit']['year'] or 'n.d.'}, {apa_pages(d['start'], d['end'])}"
                 for d in docs.values()]
        trailing = " " if run.endswith(" ") else ""
        return f"{open_b}{'~' if auto else ''}{'; '.join(parts)}{close_b}{trailing}"

    return CITE_RUN.sub(render, answer)


def apa_references(used: list[dict], cited: list[int]) -> list[str]:
    """APA-style reference entries for cited documents, in order of first citation."""
    seen, refs = set(), []
    for n in cited:
        h = used[n - 1]
        if h["doc_id"] in seen:
            continue
        seen.add(h["doc_id"])
        refs.append(f"{apa_author(h)}. ({h['year'] or 'n.d.'}). {h['title']}. {h['path']}")
    return refs


# --- One question ---------------------------------------------------------

@dataclass
class AnswerResult:
    question: str
    model: str
    used: list[dict] = field(default_factory=list)   # sources given to the LLM (S1..Sn)
    draft: str = ""                                   # raw model output
    answer: str = ""                                  # cleaned, S-labels (for checks)
    apa: str = ""                                     # display text with APA citations
    references: list[str] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)    # S numbers, order of first citation
    uncited: list[dict] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    problems: list[tuple[str, str]] = field(default_factory=list)
    added: int = 0
    fixed: int = 0
    search_s: float = 0.0
    gen_s: float = 0.0
    error: str = ""

    @property
    def uncited_warning(self) -> bool:
        return not self.cited and "could not find" not in self.answer.lower()


def answer_question(retriever: Retriever, question: str, model: str, k: int, filters: dict,
                    on_text=_print_text, search_query: str | None = None) -> AnswerResult:
    """Full pipeline: search -> generate -> clean -> cite -> check -> APA.

    search_query lets a caller search with extra context (e.g. a follow-up
    question combined with the previous one) while the LLM sees the question as asked.
    """
    result = AnswerResult(question=question, model=model)

    start = time.time()
    hits = retriever.search(search_query or question, k=k, **filters)
    result.search_s = time.time() - start
    if not hits:
        result.error = "No matching documents found. Try different words or remove filters."
        return result

    sources_text, result.used = build_sources(hits)
    start = time.time()
    try:
        result.draft = generate(model, build_messages(question, sources_text), on_text)
    except Exception as e:
        result.error = (f"LLM call failed: {e}. Is Ollama running, and is '{model}' pulled? "
                        f"Try: ollama pull {model}")
        return result
    result.gen_s = time.time() - start

    # Checks run on S-labels (unambiguous); APA is only for display.
    answer = remove_label_dumps(result.draft)
    answer, result.added = auto_cite(answer, result.used)
    answer, result.fixed = repair_citations(answer, result.used)
    result.answer = answer

    valid, result.invalid = check_citations(answer, len(result.used))
    result.problems = [(n, to_apa(s, result.used)) for n, s in ungrounded_numbers(answer, result.used)]
    result.cited = list(dict.fromkeys(n for n in cited_labels(answer) if 1 <= n <= len(result.used)))
    result.apa = to_apa(answer, result.used)
    result.references = apa_references(result.used, result.cited)
    result.uncited = [h for i, h in enumerate(result.used, 1) if i not in valid]
    return result


def ask(retriever: Retriever, question: str, model: str, k: int, filters: dict) -> None:
    """Command-line version: stream the draft, then print the checked answer."""
    print("\nDraft (live):\n")
    r = answer_question(retriever, question, model, k, filters)
    if r.error:
        print(f"\n{r.error}\n")
        return
    print(f"\n\n({len(r.used)} sources, search {r.search_s * 1000:.0f} ms)")

    notes = []
    if r.added:
        notes.append(f"{r.added} added")
    if r.fixed:
        notes.append(f"{r.fixed} corrected to the source containing the numbers")
    note = f" (citations marked ~: {', '.join(notes)})" if notes else ""
    print(f"\n{'=' * 70}\nAnswer with APA citations{note}:\n")
    print(r.apa)

    print("\nReferences:")
    for ref in r.references:
        print(f"  {ref}")
    if r.uncited:
        print("\nAlso retrieved (not cited):")
        for h in r.uncited:
            print(f"  {apa_author(h)} ({h['year'] or 'n.d.'}). {h['title']}, "
                  f"{apa_pages(h['page_start'], h['page_end'])}")

    if r.invalid:
        print(f"\nWARNING: answer cites {', '.join(f'S{n}' for n in r.invalid)}, "
              "which don't match any source.")
    if r.uncited_warning:
        print("\nWARNING: answer has no citations; treat it with caution.")
    for number, sentence in r.problems:
        print(f"\nCHECK: '{number}' is not in the cited source(s): \"{sentence[:140]}\"")
    print(f"\n[{r.gen_s:.1f} s to answer with {r.model}]\n")


# --- Entry point ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Nepal environment RAG.")
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--collection", choices=["papers", "reports"])
    parser.add_argument("--category")
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--lang", choices=["en", "ne"])
    args = parser.parse_args()

    filters = {"collection": args.collection, "category": args.category,
               "year_min": args.year_min, "year_max": args.year_max, "lang": args.lang}

    print("Loading search models...")
    retriever = Retriever(device="cpu")  # GPU stays free for the LLM

    if args.question:
        ask(retriever, args.question, args.model, args.k, filters)
        return

    print(f"Ready. Model: {args.model}. Type a question (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(retriever, question, args.model, args.k, filters)


if __name__ == "__main__":
    main()