# SkyWatch — Model Cards

Following the standard [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
shape. This is a governance artifact (`docs/MLOPS_PLAN.md` Track 8) — it summarizes decisions
already made and justified at length elsewhere (`pitch.md`, `docs/ML_ROADMAP.md`) rather than
re-deriving them; follow the links for the reasoning, not just the conclusion.

---

## Model 1 — Time-to-touchdown (`skywatch.ml.eta_touchdown`)

**Model details**
LightGBM gradient-boosted regressor, tuned with Hyperopt (24 trials over `num_leaves`,
`learning_rate`, `feature_fraction`, `bagging_fraction`, `min_child_samples`, `lambda_l2`).
Current `@champion`: v9. Registered in Unity Catalog with `@champion` / `@challenger` aliases;
promotion is gated (`docs/MLOPS_PLAN.md` Track 4), not automatic.

**Intended use**
Given one live ADS-B position report of an aircraft inbound to KATL, predict minutes until
touchdown. Powers the App's predicted arrival sequence, the live map's ETA-band coloring, and
click-to-predict. **Not** intended for separation assurance, collision avoidance, or any
safety-of-flight decision — it's a planning aid for ground/gate/hub-control staffing, built on
a hobbyist ADS-B feed with no certification behind it.

**Training data**
Self-supervised labels: complete historical trajectories, touchdown detected by geofence
(altitude/speed/distance thresholds — see `docs/ML_ROADMAP.md` §5), every earlier report on
that flight labelled with time-to-that-landing. ~480k reports / ~25,000 arrivals across 24
collection days (mostly the 1st-of-each-month `readsb-hist` archive, 15 months, plus live
poller bursts). Time-based train/test split — the model has never seen the test day. Trains
only on `touchdown_confidence = 'confirmed'` rows (~99% of dense 60s days, ~60% of sparse 180s
days) — widening to include `inferred` touchdowns measurably hurt accuracy (documented in
`train_eta.py`'s own comments; test MAE regressed 1.232 → 1.326 when tried).

**Evaluation**
MAE by distance band against a `dist_to_apt_nm / gs_kt * 60` ("fly straight in at current
speed") baseline — the honest naive comparison, not a strawman. Current champion (v9):
**MAE 1.139 min**, ~80% better than the ~5.85 min baseline. Sharpest close to the airport
(0-20 nm), loosest at 70-100 nm (documented limitation, not chased further — see
`docs/ML_ROADMAP.md` §7). The promotion gate re-evaluates every challenger on this same metric
before it can replace `@champion`.

**Limitations**
- Archive data is entirely **clear-weather days** (the monthly snapshots that exist) — no
  evidence this generalizes to weather-driven approach patterns (extended vectoring, different
  runway configurations).
- Crowd-sourced ADS-B coverage is denser near the airport and thins with distance/altitude —
  the accuracy gradient by distance band partly reflects this, not just task difficulty.
- No adversarial or out-of-distribution testing performed; a wildly anomalous flight profile
  (emergency, unusual aircraft type) gets a prediction with the same confidence as a routine
  one — there is no uncertainty quantification on this model (unlike Model 2, which does
  produce quantile bands).

**Ethical considerations**
Non-commercial, non-safety-critical use only, consistent with the ADS-B Exchange data license
(CC-BY-NC on the historical archive). No personal data — aircraft registration/callsign are
public transponder broadcasts, not linked to any individual.

---

## Model 2 — Arrival demand forecast (`skywatch.ml.demand_forecast`)

**Model details**
A backtested-and-selected baseline, not a single fixed algorithm — `forecast_demand.py` bakes
off seasonal-naive, several statistical baselines, a blended climatology/context model, and a
zero-shot time-series foundation model (Chronos-Bolt), then registers whichever wins.
**Climatological mean has won every retrain so far** (v1 through the current v3 — see
`docs/MLOPS_PLAN.md` §6a for the full retrain history), including after the training window
grew from 9 to 24 days. Wrapped as an MLflow `pyfunc` model (`DemandForecastModel` in
`src/demand_lib.py`) so serving is alias-shaped the same way as Model 1, despite not being a
single trained network.

**Intended use**
Forecast arrival counts for each of the next twelve 15-minute bins (3 hours), with a
10th-90th percentile band, from a partial day's arrivals-so-far. Powers the App's demand
curve and surge alerts against the Airport Acceptance Rate. Meant to complement Model 1, not
replace it: Model 1 covers 0-45 minutes out (real airborne aircraft, precise); Model 2 covers
45 minutes to 3 hours (aircraft not yet airborne or in ADS-B range).

**Training data**
Same touchdown detection as Model 1, aggregated into 96 fifteen-minute bins per day. Each
collection day is an **independent series** — the `readsb-hist` archive only has the 1st of
each month, so there's no day-to-day continuity to exploit, which is the central constraint
this model's whole design works around (see `pitch.md`'s "honest note").

**Evaluation**
Leave-one-day-out backtest × several forecast-start cut points (32/44/56/68/80 of 96 bins),
scored on MASE against seasonal-naive (< 1.0 = beats it). Current champion: **MASE 0.79** —
beats seasonal-naive by 21%, and zero-shot Chronos-Bolt by ~44-50% depending on the retrain.
**The most useful result in this project's whole modeling story isn't the number — it's that
the "sophisticated" approach lost, honestly, on real backtests, twice** (at 9 days and again
at 24 days), and the simple one shipped instead.

**Limitations**
- No continuous multi-day time series exists in this archive — a fine-tuned foundation model
  might behave completely differently on a real continuous feed; this result is specific to
  the within-day, partial-context forecasting task this data actually supports.
- Climatology assumes future days resemble past days' *shape* — a real disruption (weather,
  ground stop, major diversion event) that changes the arrival pattern intraday has no
  mechanism to be detected or reacted to by this model family.
- Fine-tuning Chronos was deferred as "unlikely to pay off" at 24 days/120 backtest windows,
  not proven impossible — revisit if the data volume changes materially.

**Ethical considerations**
Same non-commercial data-license basis as Model 1. No individual-level data; forecasts are
airport-aggregate counts.

---

## Model 3 — Irregularity early-warning (rules, not a learned model)

**Model details**
Deterministic rules, not a trained model — geometry (racetrack-pattern detection: heading
spread, bounding-box size, altitude/distance band) plus squawk-code/emergency-field matching.
Historical counterpart (`gold_irregularities`) and live serving (`irregularity_flags`) are the
same rule set, run in batch vs. on the current picture respectively. `train_hold_risk.py`
exists, ready, and permanently dormant.

**Intended use**
Flag currently-airborne aircraft as `emergency` / `go_around` / `holding` for the App's
irregularity panel. An early-warning surface, not a diagnostic — a flag means "this pattern
looks like X," not a confirmed classification.

**Training data**
None — this is the point. A learned classifier was seriously investigated: the 24-day archive
was specifically expanded (9 → 24 days at finer cadence) to try to get enough positive labels.
**Confirmed not viable**: of 159 detected "holding" segments, geometrically real racetracks,
**zero** were followed by that same aircraft landing at KATL within 90 minutes — every one was
a persistent loiterer (survey, traffic-watch, law-enforcement, airwork), not a flow-controlled
arrival hold. KATL's archive is entirely clear-weather days, and KATL doesn't hold arrivals in
clear weather — it absorbs demand with in-trail spacing and speed control instead. The
phenomenon a hold-risk classifier would predict essentially isn't present in this data.

**Evaluation**
Not applicable in the ML sense — there's no held-out test set for a rule set. The rules
themselves were checked against real live data throughout this project (e.g. the racetrack
rule correctly flagged real circling aircraft during live poller bursts, visible in the App's
Model 3 map panel).

**Limitations**
- The `go_around` rule is coded but essentially unvalidated — the archive contains ~0 clean
  examples at 180-second cadence (a go-around's low point is 0-1 reports; departures climb out
  through the same geometric funnel as a go-around, which the rule doesn't yet fully
  disambiguate).
- Every threshold (heading spread ≥ 0.5, bounding box ≤ 15 nm, altitude/distance bands) is a
  hand-picked cutoff, not a fitted one — reasonable given the data, but not optimized against
  a labeled objective the way Models 1-2 are.
- This finding is specific to KATL's archived clear-weather days. A hold-risk classifier might
  be genuinely viable at an airport with more weather-driven holding, or on a continuous
  (not monthly-snapshot) live feed that eventually captures a real disruption event.

**Ethical considerations**
Deciding "rules, not a weak model" *on evidence* — rather than shipping a classifier that
looks sophisticated but has no real predictive signal behind it — is itself the governance
decision worth recording here, not just a technical footnote.
