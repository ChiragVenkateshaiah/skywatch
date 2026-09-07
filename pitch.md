# SkyWatch — the pitch

The conceptual version of what this project does. No code — the kind of thing you can say out
loud when someone asks "what are the models actually doing?"

For the full build plan see [`docs/ML_ROADMAP.md`](docs/ML_ROADMAP.md).

---

## The problem both models serve

At a busy hub like Atlanta, **arrival demand is spiky**. Sometimes 25 aircraft want to land in
the same 15 minutes when the runways can only take ~22. Controllers absorb the overflow with
holding, vectoring, and speed control — which burns fuel and pushes delay downstream, and it's
hard to see coming. Real systems that manage this (Eurocontrol's AMAN, the FAA's TBFM) are
expensive infrastructure. But the raw signal they run on — live aircraft position and speed — is
exactly what **ADS-B transponder data gives us for free**.

So the project turns that free data stream into an **Arrival Manager**: it answers two questions
that together tell an arrival coordinator everything they need.

---

## Model 1 — Time-to-touchdown

**The question:** *For this specific aircraft, right now, how many minutes until it lands at
KATL?*

| | |
|---|---|
| **ML type** | Tabular **regression** — features in, one number out |
| **One row =** | one position report of one inbound flight (a plane 120 nm out generates dozens of rows as it approaches) |
| **Inputs** | distance to the airport, groundspeed, altitude, vertical speed, how directly it's pointed at the field, how fast the gap is closing, flight phase (climb/descent/cruise), time of day, and how many other aircraft are inbound (congestion) |
| **What it predicts** | `minutes_to_touchdown` |
| **How it learned** | **Self-supervised.** We watch complete historical trajectories, detect the actual landing (aircraft drops below ~3 nm, near field elevation, low speed), then label every earlier report on that flight with "time from here to that landing." No human labeling. |
| **Algorithm** | LightGBM (gradient-boosted trees), tuned with Hyperopt, trained on ~111k reports / ~8,800 arrivals over 9 days |
| **How good** | **MAE ≈ 1.3 minutes.** The naive "distance ÷ speed" estimate is off by ~5.9 min — so the model is **~79% better**. Sharpest close in (~1 min inside 20 nm), looser far out. |

**What it powers in the app:** the **predicted arrival sequence** — rank every inbound aircraft
by its predicted touchdown time and you have the landing order and timing. Ground handlers, gate
planners, and airline hub control all work off that ranked list.

**Why trees, not deep learning:** there's no sequence to model here, it's just *current state →
an outcome*. Gradient boosting is the right, fast, well-understood tool, and it trains inside
Free Edition compute limits.

---

## Model 2 — Arrival demand forecast

**The question:** *How many aircraft will land in each future 15-minute bin, out to +3 hours?*

| | |
|---|---|
| **ML type** | Univariate **time-series forecasting** — history of a quantity in, its future out |
| **One value =** | count of landings in a 15-minute bin (96 bins per day) |
| **Inputs** | today's arrival counts so far, plus the *typical* shape of a day learned from all collected days (the overnight lull, the morning bank, the afternoon push) |
| **What it predicts** | arrivals in each of the next 12 bins, with an **uncertainty band** (10th–90th percentile), not just a point estimate |
| **How it learned** | Same touchdown detection as Model 1, just **aggregated into bins** instead of per flight |
| **Algorithm** | We baked off several: seasonal-naive, statistical models (AutoETS / AutoARIMA), and a **time-series foundation model** (Chronos-Bolt, zero-shot). The winner was **climatological mean** — the historical average for each time-of-day slot. It beat seasonal-naive by 21% and the foundation model by 44%. |
| **How good** | **MASE 0.79** (below 1.0 means it beats the standard seasonal benchmark) |

**Honest note for the story:** the *intended* centerpiece was fine-tuning the Chronos foundation
model. With only 9 days of data, plain climatology won — a fine-tune needs more history to be
worth it. That's a documented decision to revisit once we've collected ~2+ weeks. It's a good
talking point: *you try the fancy thing, you measure, and you ship the simple thing that
actually wins.*

**What it powers in the app:** the **demand curve** and **surge alerts** — overlay the forecast
against the **Airport Acceptance Rate** (how many landings the runway configuration can absorb).
Any bin where predicted demand crosses the AAR line is a surge window, flagged 30+ minutes ahead
so a coordinator can start metering before holding becomes necessary.

---

## Model 3 — Irregularity early-warning (rules, not a learned model — on purpose)

**The question:** *Is this flight holding, going around, or squawking an emergency right now?*

The plan was a third trained classifier. Before building it I measured the labelled event
volume in the data: **4 emergency aircraft, ~25 real holding patterns, and effectively zero
cleanly-labelled go-arounds** across the 9 collected days. A classifier needs hundreds to
thousands of examples per class. Forcing a model onto 25 examples produces something that looks
impressive in a README and fails in practice.

So Model 3 ships as **geometry and squawk rules** — racetrack detection (circular variance of
heading), emergency transponder codes, descend-then-climb-out patterns — feeding both a
historical register (`gold_irregularities`) and a live per-aircraft flag stream
(`irregularity_flags`). The learned version is a documented next step, gated on collecting
enough continuous live data to support it.

That decision — *try the sophisticated approach, measure, ship the honest one* — is the same
judgement call as choosing plain climatology over the fine-tuned foundation model in Model 2.
It's a feature of the project, not a gap.

## How they work together

They cover different parts of the same timeline:

- **0–45 minutes out:** aggregate Model 1's per-flight ETAs. These aircraft are airborne and in
  range, so the prediction is grounded in real objects — precise.
- **45 minutes – 3 hours out:** Model 2. Those aircraft aren't airborne yet or aren't in ADS-B
  range, so there's no per-flight signal — only the statistical pattern.

Stitch them and you get **one continuous "arrivals expected" curve** vs the capacity line.

---

## The one-line analogy

> **Model 1 is Google Maps ETA for each individual car heading toward a bridge. Model 2 is the
> rush-hour traffic forecast for that bridge over the next 3 hours.** Together you know both *who
> arrives when* and *whether the total will exceed what the bridge can handle.*

And the portfolio point: these are **two genuinely different ML problems** — "predict an outcome
from an object's current state" vs. "predict a quantity's future from its history" — with
different data shapes, different algorithms, and different evaluation metrics. That's deliberate,
not redundant.
