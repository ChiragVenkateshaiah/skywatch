"""Knowledge about SkyWatch for the chat (roadmap CHAT-2, docs/CHAT_KNOWLEDGE_PLAN.md).

The repo's docs are cut into passages at their headings, embedded with `databricks-gte-large-en`
by scripts/index_docs.py into `skywatch.chat.doc_chunks`, loaded once by the App and searched in
memory by the `search_docs` tool. Each passage carries its source as `FILE.md §Section`, which the
answer must cite (lib.chat checks it).

The search embeds the model's query *and* the user's question (one embedding call) and
interleaves the two rankings (query #1, question #1, query #2, ...): the model tends to shorten a
question to a vague query ("model issues"), which lost the right passage for doc-05
(docs/TRACING_PLAN.md §5, 2026-10-23); the question's own words found it at rank 1. Not the max
of the two scores: a short query scores higher across the board (0.61 vs 0.58), so it would win.

Pure: no network — the embedding calls live in the callers. Unit-tested in tests/test_docs_index.py.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

import numpy as np

# README + the design docs. Left out: DEMO_SCRIPT.md (the chat would recite the demo) and
# SKYWATCH_MLOPS_ARTICLE.md (a publication text that repeats MLOPS_PLAN / MLOPS_ARCHITECTURE).
CORPUS = ["README.md", "docs/MLOPS_ARCHITECTURE.md", "docs/MODEL_CARDS.md", "docs/RUNBOOKS.md",
          "docs/LESSONS.md", "docs/ML_ROADMAP.md", "docs/MLOPS_PLAN.md", "docs/CHAT_PLAN.md",
          "docs/OPS_BRIEFING_PLAN.md", "docs/DEMO_REFRESH_PLAN.md", "docs/M1_TRANSFORMER_PLAN.md",
          "docs/CAPABILITIES_ROADMAP.md"]
EMBED_ENDPOINT = "databricks-gte-large-en"
TABLE = "skywatch.chat.doc_chunks"   # chunk_id, source, text, embedding (JSON array), docs_hash, indexed_at
LOAD_SQL = f"SELECT source, text, embedding FROM {TABLE} ORDER BY chunk_id"
MAX_WORDS = 250          # gte-large-en reads 512 tokens; 250 words stays well inside with the title
MAX_CHARS = 1500         # ... but code and tables cost more tokens per word (a 256-word README
                         # passage measured 550 tokens): ~4 chars/token keeps a passage < ~400
MIN_WORDS = 12           # headings with no body ("## Models" before its subsections) are dropped
TOP_K = 3                # 3 x MAX_CHARS fits lib.chat.RESULT_CHARS (6000); a 4th passage was cut mid-JSON

_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _clean_heading(h: str) -> str:
    return re.sub(r"[`*]", "", h).strip()


def _fits(text: str, max_words: int, max_chars: int) -> bool:
    return len(text.split()) <= max_words and len(text) <= max_chars


def chunk_markdown(name: str, text: str, max_words: int = MAX_WORDS,
                   max_chars: int = MAX_CHARS) -> list[dict[str, str]]:
    """One passage per section (heading levels 1-4), split at blank lines when longer than
    `max_words` (and a single over-long paragraph at its line ends). Headings inside code fences are not headings. Source = `name §Heading`, or
    just `name` for text before the first heading."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        m = None if fenced else _HEADING.match(line)
        if m:
            sections.append((_clean_heading(m.group(2)), []))
        else:
            sections[-1][1].append(line)

    out = []
    for heading, lines in sections:
        source = f"{name} §{heading}" if heading else name
        paras = []
        for p in re.split(r"\n\s*\n", "\n".join(lines)):
            p = p.strip()
            if _fits(p, max_words, max_chars):
                paras += [p] if p else []
            else:   # a long table or list with no blank lines: split at line ends (a single
                    # over-long line is cut, so no passage passes the model's input limit)
                paras += [ln.strip()[:max_chars] for ln in p.splitlines() if ln.strip()]
        part: list[str] = []
        for p in paras:
            if part and not _fits("\n\n".join(part + [p]), max_words, max_chars):
                out.append({"source": source, "text": "\n\n".join(part)})
                part = []
            part.append(p)
        if part:
            out.append({"source": source, "text": "\n\n".join(part)})
    return [c for c in out if len(c["text"].split()) >= MIN_WORDS]


def embed_input(chunk: dict[str, str]) -> str:
    """What gets embedded: the source line gives a short passage its topic."""
    return f"{chunk['source']}\n{chunk['text']}"


def docs_hash(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + files[name].encode() + b"\0")
    return h.hexdigest()[:16]


def normalise(vectors: list[list[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def search(query_vecs: list[list[float]], matrix: np.ndarray, chunks: list[dict[str, str]],
           k: int = TOP_K) -> dict[str, Any]:
    """Cosine top-k over the normalised matrix, the rankings of several queries interleaved ->
    a tool result shaped like lib.chat.compact."""
    ranked = np.argsort(-(matrix @ normalise(query_vecs).T), axis=0)    # one column per query
    order = list(dict.fromkeys(int(i) for i in ranked.ravel()))[:k]     # row by row = interleaved
    rows = [[chunks[i]["source"], chunks[i]["text"]] for i in order]
    return {"columns": ["source", "text"], "rows": rows, "row_count": len(rows)}


class DocIndex:
    """The loaded index: passages + their normalised embeddings, searched in memory."""

    def __init__(self, chunks: list[dict[str, str]], vectors: list[list[float]]):
        self.chunks, self.matrix = chunks, normalise(vectors)

    @classmethod
    def from_rows(cls, rows: list[list[Any]]) -> "DocIndex":
        """Rows of LOAD_SQL (source, text, embedding as a JSON string)."""
        if not rows:
            raise RuntimeError(f"{TABLE} is empty — run scripts/index_docs.py")
        return cls([{"source": r[0], "text": r[1]} for r in rows], [json.loads(r[2]) for r in rows])

    def search(self, query: str, embed: Callable[[list[str]], list[list[float]]], k: int = TOP_K,
               question: str = "") -> dict[str, Any]:
        """`question` = the user's own question, searched together with the model's query."""
        query, question = (" ".join(str(s or "").split())[:300] for s in (query, question))
        if not query:
            raise ValueError("search_docs needs a query")
        texts = [query] + ([question] if question and question.lower() != query.lower() else [])
        return search(embed(texts), self.matrix, self.chunks, k)


def embed_response(resp: dict[str, Any]) -> list[list[float]]:
    """`llm/v1/embeddings` response -> vectors in input order."""
    return [d["embedding"] for d in sorted(resp["data"], key=lambda d: d.get("index", 0))]


_CITE = re.compile(r"([A-Za-z0-9_]+\.md)(?:\s*§\s*([^)\];,\n]+))?")


def citation_problems(answer: str, results: list[dict[str, Any]]) -> list[str]:
    """When search_docs returned passages, the answer must cite at least one of them as
    `FILE.md §Section`, and every file it cites must be among the returned sources."""
    sources = [row[0] for r in results for row in r.get("rows") or []]
    if not sources:
        return []
    files = {s.split(" §")[0].split("/")[-1] for s in sources}
    cited = [m.group(1) for m in _CITE.finditer(answer)]
    if not cited:
        return [f"cite the passage you used as FILE.md §Section, e.g. ({sources[0].split('/')[-1]})"]
    bad = sorted({c for c in cited if c not in files})
    return [f"cited {c}, which search_docs did not return — cite one of: "
            + "; ".join(dict.fromkeys(s.split("/")[-1] for s in sources)) for c in bad]
