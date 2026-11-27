# SkyWatch — knowledge for the chat (roadmap CHAT-2, App v2.1)

## 1. Why

CHAT-1 answers *what* questions from live tools. Viewers of the demo also ask *how* and *why*:
how Model 1 works, why Model 3 is rules, why the Transformer didn't ship, how a model is
promoted. Those answers are in the repo's docs. CHAT-2 lets the chat search them and cite the
passage, under the same rule as every other answer: facts only from tool results, checked.

Entry gate (roadmap): CHAT-1 golden set ≥ 90% on three consecutive runs — met 2026-10-19
(38/38, 37/38, 38/38, `docs/CHAT_PLAN.md` §6).

## 2. Measured before designing (2026-10-21)

| Check | Result |
|---|---|
| `databricks-gte-large-en` on Free Edition | READY; 1024 dims; 10 tokens for a short query; 1.7–3.3 s per call through the CLI (CLI start-up and token included) |
| Corpus (README + 12 docs) | 30.5k words → **207 passages** of ≤ 250 words and ≤ 1,500 chars (cut at headings, long sections at paragraphs, long tables at lines). A 256-word passage measured 550 tokens (gte reads 512) → the char cap |
| Embedding QPS on Free Edition | a batch of 16 → `REQUEST_LIMIT_EXCEEDED` (workspace QPS limit) even after 5 retries; batches of 4 with a 1 s pause pass: 207 passages in 148 s (52 calls), table write 52 s |
| Bundle sync | `*.md` is excluded — a bundle job can't see the docs |
| Serverless job start | waited 645–705 s on 2026-10-19 (DATA-1) |

## 3. Design (decided 2026-10-21)

- **Index: a local script, not a job.** `scripts/index_docs.py` chunks the corpus
  (`lib.docs_index.CORPUS`), embeds it in batches of 4 (QPS limit, §2) and replaces
  `skywatch.chat.doc_chunks` (`chunk_id, source, text, embedding` as a JSON array, `docs_hash`,
  `indexed_at`). If the passages' hash matches the table, it stops after one query (OPT-4 spirit).
  The roadmap said "a job"; changed because of the two facts above.
- **Corpus:** README + 12 docs. Left out: `DEMO_SCRIPT.md` (the chat would recite the demo) and
  `SKYWATCH_MLOPS_ARTICLE.md` (a publication text that repeats MLOPS_PLAN / ARCHITECTURE).
- **Search in memory.** The App loads the table once (1 SQL, `st.cache_resource`, 1 h) and runs
  cosine top-3 with numpy (3 × 1,500 chars fits the 6,000-char tool-result cap; a 4th passage
  was cut); one embedding call per search. ~200 × 1024 floats is < 1 MB.
- **`search_docs(query)`** is an App-side tool (`lib.chat.APP_TOOLS`). Result rows:
  `source` (`FILE.md §Section`) and `text`.
- **Citations are checked**, like numbers: when search_docs returned passages, the answer must
  cite one as `FILE.md §Section`, and may cite only files it returned
  (`lib.docs_index.citation_problems`) — one repair round, then the fallback. Numbers in a
  passage are evidence for the existing grounding check.
- Prompt `chat-v2.2.1`: one rule — for how / why / where-to-look questions about SkyWatch, call
  search_docs (even when a data tool also applies) and cite the source.
- Same code everywhere: `lib.docs_index` is copied into `app/lib/` (`scripts/sync_app_lib.py`);
  `scripts/eval_chat.py` runs search_docs against the same table and endpoint.

## 4. Permissions (App service principal)

- `GRANT SELECT ON TABLE skywatch.chat.doc_chunks` (listed in `resources/skywatch.app.yml`).
- App resource `embedding-endpoint`: `databricks-gte-large-en`, `CAN_QUERY`.
- The self-check gains a `docs` line (table count + one embedding) → **15/15** expected.

## 5. Order of work

1. `lib.docs_index` + tests; `search_docs` + citation check in `lib.chat`; eval support. ✅
2. Golden set: 6 `doc-*` items (44 total). ✅
3. `python scripts/index_docs.py --warehouse-id …` — live. ✅
4. `eval_chat.py --snapshot --only doc-01,…,doc-06` — the new items first. ✅ (§6)
5. Ship rule: full golden set ≥ 90% on three consecutive runs. ✅ (§6)
6. Grant + deploy; SELFCHECK 15/15; one doc question in the App.

## 6. Progress log

- **2026-10-21 — steps 1–4** on branch `chat/knowledge`. Index built (207 passages).
  `doc-*` items only (`--snapshot`):

  | Run | Pass | Tool sel. | Grounded | Repairs | p50 / p95 s | What changed before it |
  |---|---|---|---|---|---|---|
  | 1 | 3/6 | 5/6 | 5/6 | 2 | 14.3 / 17.6 | baseline (top-4) |
  | 2 | 3/6 | 4/6 | **6/6** | 0 | 6.7 / 10.9 | top-3; a wrong citation's repair lists the sources it may cite; doc-02 / doc-05 expectations widened (the answers were right) |

  Findings: (1) the citation guard works — run 1's two repairs were "cited MODEL_CARDS.md,
  which search_docs did not return"; without the list of allowed sources the model repeated the
  mistake. (2) The remaining misses are **tool choice**: the model answers "why didn't the
  Transformer ship" from `model_status` (accepted — the gate verdict is the source of truth),
  "where should I look first" from `model_health`, and "how does the chat check numbers" from
  its own instructions (both kept as failures).

- **2026-10-21 — ship rule met.** `GRANT SELECT ON TABLE skywatch.chat.doc_chunks` to the App SP.
  Prompt `chat-v2.2.1` (the search_docs rule also covers where-to-look questions and applies
  "even when a data tool also applies"). Full golden set, 44 items, `--snapshot`:

  | Run | Pass | Tool sel. | Grounded | Refusal | Repairs | p50 / p95 s | Model calls |
  |---|---|---|---|---|---|---|---|
  | 1 | 43/44 (44/44 rescored) | 44/44 | 42/42 | 5/5 | 4 | 4.2 / 16.5 | 72 |
  | 2 | 43/44 | 44/44 | 41/42 | 5/5 | 4 | 3.4 / 12.5 | 72 |
  | 3 | 42/44 | 43/44 | 41/42 | 5/5 | 2 | 3.7 / 13.0 | 67 |

  Run 1's only failure was a golden-set defect: doc-05 required the word `model_health`, which
  the grounding check forbids (no raw key names) — the answer said "monitoring checks" and cited
  `RUNBOOKS.md §Incident`; the item now accepts "health" / "monitoring". Run 3 needed `--force`:
  OPT-4 skipped it as unchanged, as designed. The weak spot is the same two items (doc-05,
  doc-06): the model sometimes skips search_docs or doesn't cite after the repair — the guard
  then shows the tool rows, never an unchecked answer. The CHAT-1 items stayed at 38/38.
- **2026-10-23 — doc-05 diagnosed with traces (OPT-2), search fixed (`chat-v2.2.2`).** Traced
  eval run `f6e11c46db0a` (doc-05, doc-06): both called `search_docs` — tool choice was fine —
  but for doc-05 the model searched `"model issues"`, and no monitoring passage came in the top
  3 (first at rank 6). The user's own question found `RUNBOOKS.md §Incident` at rank 1. Now
  `run_turn` hands `search_docs` the question too, and `DocIndex.search` embeds both in one call
  and interleaves the two rankings (max of the two scores failed: the short query scores higher
  across the board, 0.61 vs 0.58). On the cached passages doc-05 now gets
  `ML_ROADMAP.md §Model 2`, `RUNBOOKS.md §Incident`, `CHAT_PLAN.md §6`. The `tool search_docs`
  span now records the returned sources.
- **2026-10-24 — `chat-v2.2.2` ship check: 43/44 (97.7 %) on three consecutive full runs**
  (eval_runs `f1ac36f8f2e3`, `6448621fa30a`, `f69466720821`; 67–73 model calls and 1 SQL each,
  tool cache reused). doc-06 passed all three. doc-05 failed all three, never on retrieval: run 1
  skipped `search_docs`; runs 2–3 called `search_docs` + `model_health` but didn't cite a passage
  after the repair (fallback shown). doc-01–06 alone, the day before: 4/6 (doc-05 a text tool call
  twice, doc-06 no tool call). Next for doc-05: a prompt rule or a lighter citation rule when a
  live tool also answers — not a retrieval change.
