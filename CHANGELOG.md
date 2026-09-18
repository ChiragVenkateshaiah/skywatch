# Changelog

All notable changes to SkyWatch are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/) — MAJOR for a new model or an architecture change
(e.g. the dev/staging split), MINOR for a new capability (a Track from `docs/MLOPS_PLAN.md`,
a new App panel), PATCH for fixes.

Project history before this file existed isn't itemized retroactively — `git log` and the
README's "Origins" section cover it. This changelog starts the discipline at 1.0.0, not before.

## [Unreleased]

## [1.1.0] - 2026-09-18

The MLOps arc completed — all eight tracks of `docs/MLOPS_PLAN.md` done.

### Added
- Retraining automation (Track 6): `skywatch_retrain` chains a full Gold rebuild → training →
  the promotion gate for both models in one workflow, deployed paused, demonstrated live end
  to end (~9 min).
- Release management (Track 7): this changelog, a tag-triggered GitHub Release workflow, and a
  rollback procedure practiced live (both a model-alias rollback and the process for a code
  rollback).
- Governance (Track 8): `docs/RUNBOOKS.md`, `docs/MODEL_CARDS.md`, `docs/MLOPS_ARCHITECTURE.md`,
  and a real `<catalog>.ml.promotion_audit` table recording every promotion decision — model,
  challenger, prior champion, verdict, and whether it was actually applied.

### Changed
- `eta_touchdown@champion` v9 refreshed into live serving — `skywatch_score_eta` re-run so
  predictions and click-to-predict's Volume export both reflect it.

## [1.0.0] - 2026-09-18

The project at "portfolio-complete" — three models, a live App and AI/BI dashboard, and a full
practiced MLOps arc (`docs/MLOPS_PLAN.md` Tracks 1-6) on top of Databricks Free Edition.

### Added
- **Model 1** — time-to-touchdown (LightGBM + Hyperopt), MAE 1.14 min, ~80% better than a
  naive baseline. Powers the arrival sequence, the live map, and click-to-predict.
- **Model 2** — arrival demand forecast; climatological mean beat a fine-tuned foundation
  model (Chronos-Bolt) on the available data — MASE 0.79.
- **Model 3** — rule-based irregularity flags (emergency / go-around / holding), a deliberate
  decision over a learned classifier once the label volume proved 10-40x too thin.
- Databricks App (Streamlit) + an AI/BI dashboard, both reading batch-scored Delta tables
  (Free Edition has no model-serving endpoints).
- **MLOps arc**: a pytest suite + `ruff` in CI on every PR (Track 1/2); a second, isolated
  `staging` environment with its own catalog and a dedicated `skywatch-ci` service principal
  (Track 3); a promotion gate for both models — a challenger is scored against the current
  champion and only a human's explicit `apply=true` run ever flips `@champion` (Track 4);
  DIY drift/performance/freshness monitoring, since Lakehouse Monitoring isn't available on
  Free Edition (Track 5); a retraining workflow chaining gold-rebuild → train → gate for both
  models, demonstrated live end to end (Track 6).

### Changed
- `eta_touchdown@champion` promoted v7 → v9 (MAE 1.144 → 1.139) — the first real promotion
  through the Track 4 gate, not a test artifact.
