"""Tests for src/lib/docs_index.py — chunking, search and citations for the chat's search_docs
tool (roadmap CHAT-2) — and the citation guard in the turn loop. Nothing touches the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib import docs_index
from lib.chat import APP_TOOLS, run_turn
from lib.docs_index import DocIndex, chunk_markdown, citation_problems, docs_hash, embed_response

ROOT = Path(__file__).resolve().parents[1]

DOC = """# SkyWatch — Model Cards

Intro text that is long enough to be kept as its own passage before any section heading.

## Model 1 — Time-to-touchdown (`skywatch.ml.eta_touchdown`)

LightGBM predicts each inbound aircraft's minutes to touchdown from distance, speed and altitude.

```bash
# not a heading, a shell comment inside a fence
databricks bundle deploy
```

## Empty

## Model 3 — Irregularity early-warning (rules, not a learned model)

Rules flag emergencies, go-arounds and holding; there is no learned model behind them yet.
"""


class TestChunking:
    def test_sections_sources_and_fences(self):
        chunks = chunk_markdown("MODEL_CARDS.md", DOC)
        assert [c["source"] for c in chunks] == [
            "MODEL_CARDS.md §SkyWatch — Model Cards",
            "MODEL_CARDS.md §Model 1 — Time-to-touchdown (skywatch.ml.eta_touchdown)",
            "MODEL_CARDS.md §Model 3 — Irregularity early-warning (rules, not a learned model)"]
        assert "# not a heading" in chunks[1]["text"]          # the fence stayed in its section

    def test_long_sections_split_at_paragraphs_and_lines(self):
        para = " ".join(["word"] * 40)
        table = "\n".join(f"| row {i} | " + " ".join(["cell"] * 10) + " |" for i in range(40))
        chunks = chunk_markdown("X.md", "## S\n\n" + "\n\n".join([para] * 10) + "\n\n" + table, max_words=100)
        assert len(chunks) > 5
        assert all(len(c["text"].split()) <= 100 for c in chunks)
        assert {c["source"] for c in chunks} == {"X.md §S"}

    def test_real_corpus_chunks_fit_the_model(self):
        files = {f: (ROOT / f).read_text() for f in docs_index.CORPUS}
        chunks = [c for f, t in files.items() for c in chunk_markdown(Path(f).name, t)]
        assert len(chunks) > 100
        assert max(len(docs_index.embed_input(c).split()) for c in chunks) <= docs_index.MAX_WORDS + 40
        assert max(len(c["text"]) for c in chunks) <= docs_index.MAX_CHARS
        assert "DEMO_SCRIPT.md" not in {c["source"].split(" §")[0] for c in chunks}

    def test_docs_hash_changes_with_content(self):
        assert docs_hash({"a.md": "x"}) == docs_hash({"a.md": "x"}) != docs_hash({"a.md": "y"})


def _index():
    rows = [["A.md §Alpha", "about alpha", json.dumps([1.0, 0.0, 0.0])],
            ["B.md §Beta", "about beta", json.dumps([0.0, 1.0, 0.0])],
            ["C.md §Gamma", "about gamma", json.dumps([0.6, 0.8, 0.0])]]
    return DocIndex.from_rows(rows)


class TestSearch:
    def test_top_k_by_cosine(self):
        r = _index().search("beta please", lambda texts: [[0.0, 2.0, 0.0]], k=2)
        assert r["columns"] == ["source", "text"] and r["row_count"] == 2
        assert [row[0] for row in r["rows"]] == ["B.md §Beta", "C.md §Gamma"]

    def test_question_searched_with_the_query_rankings_interleaved(self):
        vecs = {"vague": [0.0, 1.0, 0.0], "about alpha?": [1.0, 0.0, 0.0]}
        seen = []

        def embed(texts):
            seen.append(texts)
            return [vecs[t] for t in texts]
        r = _index().search("vague", embed, k=2, question="about alpha?")
        assert seen == [["vague", "about alpha?"]]
        assert [row[0] for row in r["rows"]] == ["B.md §Beta", "A.md §Alpha"]   # each one's best
        r = _index().search("vague", embed, k=3, question="about alpha?")
        assert [row[0] for row in r["rows"]] == ["B.md §Beta", "A.md §Alpha", "C.md §Gamma"]   # no repeats

    def test_question_same_as_query_embeds_once(self):
        seen = []
        _index().search("Beta", lambda t: seen.append(t) or [[0.0, 1.0, 0.0]] * len(t), question=" beta ")
        assert seen == [["Beta"]]

    def test_empty_query_and_empty_table(self):
        with pytest.raises(ValueError):
            _index().search("  ", lambda t: [[1.0, 0, 0]])
        with pytest.raises(RuntimeError, match="index_docs"):
            DocIndex.from_rows([])

    def test_embed_response_keeps_input_order(self):
        resp = {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
        assert embed_response(resp) == [[1.0], [2.0]]


RESULT = {"columns": ["source", "text"], "rows": [["MODEL_CARDS.md §Model 1 — Time-to-touchdown", "LightGBM ..."],
                                                 ["LESSONS.md §Models", "..."]], "row_count": 2}


class TestCitations:
    def test_no_search_no_rule(self):
        assert citation_problems("113 aircraft are inbound.", []) == []

    def test_cited_returned_source_passes(self):
        assert citation_problems("It uses LightGBM (MODEL_CARDS.md §Model 1 — Time-to-touchdown).", [RESULT]) == []
        assert citation_problems("See LESSONS.md.", [RESULT]) == []

    def test_missing_or_foreign_citation_fails(self):
        assert "cite the passage" in citation_problems("It uses LightGBM.", [RESULT])[0]
        [msg] = citation_problems("It uses LightGBM (RUNBOOKS.md §Deploy).", [RESULT])
        assert msg.startswith("cited RUNBOOKS.md, which search_docs did not return — cite one of: ")
        assert "MODEL_CARDS.md §Model 1 — Time-to-touchdown; LESSONS.md §Models" in msg


def _tool_msg(name, args):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def _text_msg(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class TestTurnWithDocs:
    def _run(self, *answers):
        responses = [_tool_msg("search_docs", {"query": "Model 1"})] + [_text_msg(a) for a in answers]
        return run_turn("How does Model 1 work?", llm=lambda m, s: responses.pop(0),
                        execute=lambda n, a: RESULT, tools=APP_TOOLS)

    def test_cited_answer_passes(self):
        t = self._run("Model 1 is a LightGBM model (MODEL_CARDS.md §Model 1 — Time-to-touchdown).")
        assert t.source == "llm" and t.check.passed, t.check.reasons

    def test_uncited_answer_is_repaired(self):
        t = self._run("Model 1 is a LightGBM model.", "Model 1 is a LightGBM model (MODEL_CARDS.md §Model 1 — Time-to-touchdown).")
        assert t.repaired and t.source == "llm"
        assert any("cite the passage" in r for r in t.repair_reasons)

    def test_still_uncited_falls_back(self):
        t = self._run("Model 1 is a LightGBM model.", "It is LightGBM.")
        assert t.source == "fallback"
