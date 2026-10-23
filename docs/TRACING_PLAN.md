# SkyWatch — tracing (roadmap OPT-2)

## 1. Why

`chat_audit` records *what* a chat turn did (tools, answer, check, totals). It can't show *where
the time went* inside a turn, or which round made a wrong choice — e.g. why doc-05 / doc-06
sometimes skip `search_docs` (`docs/CHAT_KNOWLEDGE_PLAN.md` §6). The refresh has the same gap:
`refresh_audit` has per-task totals, but the 645–705 s serverless *setup* wait of 2026-10-19 was
found by reading run JSON by hand. One MLflow trace per chat turn and per refresh run makes both
visible in the experiment UI.

## 2. Measured before designing (2026-10-21)

| Check | Result |
|---|---|
| MLflow in the App image / local venvs | not installed — the App needs a dependency |
| `mlflow-tracing` 3.17.2 (tracing-only package) | 59 MB with its deps; resolves with the App's pins (`streamlit==1.38.0`, protobuf 5.29.6) — but see §5: the App uses `mlflow-skinny` |
| Trace from a laptop to a Free Edition experiment | works; read back with all spans |
| Cost on the request path | **first trace in a process ~3.3 s** (client setup; plus ~20 s the first time an experiment is created), **every later trace ~0.00 s** (async export) — same with async logging forced on |
| App resource for an experiment | `experiment` resource, GA, permission `CAN_EDIT` — no manual grant |
| Refresh run → trace with real timestamps | `start_span_no_context(start_time_ns=…)` + `end(end_time_ns=…)`: the 2026-10-19 run became one 616.8 s trace, 27 spans, no compute |

## 3. Design

- **Experiment:** bundle resource `skywatch_traces` → `/Users/<me>/skywatch-traces`
  (`resources/skywatch.traces.experiment.yml`). The App gets `CAN_EDIT` through its
  `traces-experiment` resource and the id as `SKYWATCH_TRACE_EXPERIMENT_ID`.
- **Chat turn** — one trace, tagged `turn_id` (the `chat_audit` key) and `prompt_version`:
  `chat turn` (AGENT: question, answer, source, check, rounds, endpoint, snapshot tools) →
  `model round N` (CHAT_MODEL: messages, tokens, tool calls requested) · `tool <name>` (TOOL:
  args, row count, error) · `check` (PARSER: passed, reasons, repair round).
- **`lib.chat` stays MLflow-free:** `run_turn(..., span=...)` takes a context-manager factory;
  the default is a no-op. `lib.tracing` holds the only MLflow code, imported lazily.
- **Never breaks a turn:** a span that can't be opened or exported becomes a no-op; tracing is
  set up once per App process in a background thread, so no visitor waits ~3 s for it; turns
  before it is ready run untraced.
- **Refresh run** — one trace per finished run, written by `scripts/demo_ready.py` (also with
  `--report-only`, so a refresh started from the App's button is traced the next time the report
  runs): `refresh run <id>` → `task <key>` (state, attempt, setup_s, exec_s) → `setup` ·
  `execution`, at the run's real timestamps (`lib.refresh.trace_spans`). Tagged `kind=refresh`,
  `run_id`; a run that already has a trace is skipped. Not traced inside the job: that would add
  an import and a dependency to a serverless task to record what the Jobs API already knows.

## 4. Order of work

1. `lib.chat` span hook, `lib.refresh.trace_spans`, `lib.tracing` + tests. ✅
2. Live check without warehouse or model compute: a fake-model turn and the real 2026-10-19
   refresh run written as traces and read back. ✅
3. Bundle experiment + App resource + env; `mlflow-tracing` in `app/requirements.txt` and
   `requirements-dev.txt`. ✅ (`bundle validate` OK)
4. Deploy; one chat turn in the App → its trace in `skywatch-traces`, tagged with its
   `turn_id`; compare the root span's duration with the turn's `total_seconds`.
5. Then use it: trace doc-05 / doc-06 to see which round skips `search_docs`.

## 5. Progress log

- **2026-10-21 — steps 1–3** on branch `opt/tracing`. Not deployed: the App is stopped to save
  quota until the next demo.
- **2026-10-22 — first deploy: no traces.** The App logged `tracing off: cannot import name
  'EvaluationDataset' from 'mlflow.entities'` — the fallback worked (the turn ran untraced). The
  Apps base image ships its own `mlflow-skinny`; `mlflow-tracing` 3.17.2 on top mixed two
  versions in one `mlflow` package folder. Reproduced locally (`mlflow-skinny` 2.21.3 or 3.1.0 +
  `mlflow-tracing` 3.17.2 → the same ImportError) and fixed by pinning `mlflow-skinny==3.17.2`
  in `app/requirements.txt`, which replaces the image's copy whole. The self-check gained a
  `trace` line (MLflow versions + experiment access as the App SP), so this shows at start-up.
- **2026-10-22 — second deploy: self-check 16/16** (`trace  experiment 4236095730365180 ·
  mlflow-skinny 3.17.2, mlflow-tracing 3.17.2`; the build log confirmed the image's
  `mlflow-skinny 2.16.2` was replaced). The next question was still untraced: the page called
  `tracing_state()` only when a question arrived, so the set-up started with the first question
  (set-up at 05:21:37, turn 05:21:36–05:21:40). Fixed: the page starts it when it renders.
- **2026-10-23 — third deploy: the trace arrives, its spans don't.** Turn `3f95e50d…` ("Which
  aircraft lands next?") made trace `tr-33ec2cfd…` in `skywatch-traces`: tagged `turn_id` +
  `prompt_version`, question, answer, 2.68 s, 3 spans in its size stats. But the span file never
  uploaded: MLflow writes the trace *record* through the workspace host, then PUTs `traces.json` to
  a signed URL on `us-east-2.storage.cloud.databricks.com`, and the App's network refuses that host
  (`Connection refused`, App log 09:54:21). A laptop reaches it, so laptop traces (eval runs,
  `demo_ready.py` refresh traces) are complete. App traces show only the root's inputs, outputs,
  duration and tags.
