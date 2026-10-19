# SkyWatch — lessons, with the evidence

Roadmap DOC-1. The engineering stories from the arcs after the MLOps arc
(`docs/SKYWATCH_MLOPS_ARTICLE.md` covers that one), each with the measurement that settled it and
where it is recorded. Written to be answerable in an interview: *what happened, how we knew,
what changed.*

The pattern behind almost all of them: **measure it live before designing around it, and check
model output against evidence instead of trusting it.**

## Data and pipelines

**A 2-hour "slow model" was a pandas loop.** The M1 Transformer job timed out at 2 h — and an
earlier attempt had died at ~100 min, logged as a platform error. The only stdout line was the
one *before* the sequence-frame build, so training had never started. `build_sequence_frame`
assigned 456 columns onto a pandas chunk per segment (~11 M single-column inserts). Built in numpy
over contiguous segments instead: **~157 min → 42 s** at real volume, identical output.
(`docs/M1_TRANSFORMER_PLAN.md` §6, attempt #3)

**Parallel isn't faster when the platform queues it.** Three scoring jobs started together each
spent **645–705 s in setup** for 43–68 s of work; the same job alone: **4 s setup**. The
pre-interview refresh runs all nine tasks strictly in sequence and takes **8.5–10 min instead of
~25**. (`docs/DEMO_REFRESH_PLAN.md` §2)

**"No rows" can mean "nothing happened" or "nothing ran".** The irregularity job writes no rows
on a quiet run, so the newest flags could be days older than the newest predictions and still
look current — in the App and in the LLM briefing. Rule, now in three places (briefing facts,
App panel, chat tool): flags older than 15 min before the current Model 1 run are not current.
(`docs/OPS_BRIEFING_PLAN.md` §6)

**A correct number can still mislead.** "Arrivals so far today: 1" at 15:39 UTC was true — for
what SkyWatch had *observed* after one short burst. Relabelled "Arrivals seen today", with a
help text, and the same distinction is pushed into the briefing and chat prompts and a chat rule.
(`docs/DEMO_REFRESH_PLAN.md` §4)

**Timestamps lie at the edges.** `scored_at` has microseconds; the SQL Statement API's JSON has
milliseconds — an equality filter built from the parsed value matched nothing, so every
historical snapshot looked empty. Filter on the value cast to string.
(`docs/OPS_BRIEFING_PLAN.md` §4)

## Models

**LightGBM forgives what PyTorch doesn't.** Transformer training was `nan` from epoch 1: ~5% of
`closure_kt` / `turn_rate_dps` are null, LightGBM splits on missing values natively, and one NaN
batch poisons every PyTorch weight on the first optimiser step. Fix: standardise on the train
split only, map NaN to the train mean, and **fail in minutes** on a non-finite loss instead of
after the full budget. (`docs/M1_TRANSFORMER_PLAN.md` §6, attempt #4)

**"It only trained one epoch" was the wrong excuse.** The first Transformer HOLD came from a
1-epoch model (a 61-row validation day picked epoch 1). With a 10-epoch floor, train MAE fell
1.69 → 1.09 and **test MAE rose 1.19 → 1.27**: overfitting, not under-training. Same registered
model, same promotion gate, HOLD twice — the gate kept the better model, and a HOLD is a
complete result. (`docs/M1_TRANSFORMER_PLAN.md` §6, attempts #5–#6)

**A reference is not a snapshot.** `best_state = model.state_dict()` kept live references, so the
"best" weights silently tracked every later step. Harmless where it was used, wrong if ever
relied on — `copy.deepcopy`. (`docs/M1_TRANSFORMER_PLAN.md` §6)

**A missing dependency can break a feature silently.** Click-to-predict raised `'super' object
has no attribute 'get_params'`: the App never installed scikit-learn, and a pickled
`LGBMRegressor` loads with a stub base class without it. Pinned to the champion's own training
env (read from its MLflow `requirements.txt`). (`docs/OPS_BRIEFING_PLAN.md` §6)

## LLMs and agents

**Check the output, not just the prompt.** On a one-prompt probe, one open model wrote "42
aircraft **arrived**" for 42 inbound. Every LLM text in SkyWatch now passes a deterministic check
— every number, `HH:MM` time and callsign must appear in the evidence — with a template or
tool-rows fallback. Model choice by evaluation, not by name: briefing 42/42 checks on Maverick;
the reasoning models (gpt-oss) failed by spending the whole token budget before answering.
(`docs/OPS_BRIEFING_PLAN.md` §2, §4)

**An agent with tools invented its evidence.** Asked to diagnose a real failed run (`ValueError:
Input contains NaN`), the ops agent wrote a *fake tool call as plain text* and then **invented a
`java.io.FileNotFoundException`** that was nowhere in the run. Fixes: evidence goes straight into
the prompt; the diagnosis must quote an error line that really is in it, or the raw error is shown;
and the retry decision is code (a transient/auth/code/data classifier), never the model's.
(`docs/DEMO_REFRESH_PLAN.md` §4)

**Most early "model errors" were the checker being too strict.** Golden-set run 1 passed 32/38;
reading every failure showed values that *were* evidence but sat inside free-text cells
(`squawk=7700`), a returned briefing paragraph, or the system prompt ("the next 3 hours").
Widening what counts as evidence — not loosening the check — took it to 36/38.
(`docs/CHAT_PLAN.md` §6)

**A value check can't see meaning, so add a few rules.** "6 flights have landed today" passes any
number check. The handful of misreadings that matter get a deterministic rule with a repair
instruction; it fired and repaired correctly in every later run. (`docs/CHAT_PLAN.md` §6)

**Give the tool the number instead of letting the model compute it.** Asked about "the next
hour", the model added four bins itself; the check rightly blocked the invented sum. The tool now
returns `next_hour_total`. (`docs/CHAT_PLAN.md` §6)

**Latency is rounds, not tokens.** A model round cost the same ~1.7 s with 360 or 2,128 prompt
tokens, so shortening the prompt would only have risked accuracy. Prefetching the four tools that
answer most questions — handed to the model as results it already has — cut the chat's p50 from
**5.2 s to 2.1–2.6 s** (rounds per question 2.05 → 1.42) at 38/38. It also surfaced a new failure:
asked to print its prompt, the model dumped the prefetched data; one prompt rule fixed it.
(`docs/CHAT_PLAN.md` §6)

**Temperature 0 is not deterministic.** The same 38 questions scored 32, 36, 33, 38, 37, 38 across
runs — about three items of noise. Ship rule: ≥ 90% on **three consecutive** runs.
(`docs/CHAT_PLAN.md` §6)

## Platform (Databricks Free Edition)

- **Managed MCP works, but it costs a round trip:** the same UC function was 3.3–4.3 s through
  MCP vs 1.9–2.3 s by direct SQL (first MCP call 37 s, cold). Tools are written once as UC
  functions; the App calls them directly, MCP serves them to other agents. (`docs/CHAT_PLAN.md` §2)
- **UC SQL functions read tables with the owner's rights** — the App's identity needs only
  `EXECUTE`, so the tools are the chat's only door to the data. Verified by a self-check that runs
  *as the App's service principal* at every start. (`docs/CHAT_PLAN.md` §6)
- **`bundle deploy` doesn't upload files behind a symlink** — shared modules are copied into the
  App with a drift test. (`docs/CHAT_PLAN.md` §6)
- **The CLI's OAuth refresh token goes stale after ~12 idle days** — log in the day before a demo.
- **Agent hosting (Agent Framework endpoints) isn't available; agents run as our own code** on
  the Foundation Model API, with tools as UC functions. (`docs/CAPABILITIES_ROADMAP.md` §2)
