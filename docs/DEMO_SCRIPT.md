# SkyWatch — demo script (roadmap DEMO-1)

A fixed path for a recorded interview demo, about 8 minutes. The point to land is not "it has a
chatbot" but **"every answer is checked, and you can see the evidence"**. Live data changes, so
each step names what to show and a fallback if today's sky doesn't have it.

## Before recording

| When | Do | Why |
|---|---|---|
| Day before | `databricks auth login --profile skywatch` | the CLI's OAuth refresh token goes stale after ~12 idle days (`docs/M1_TRANSFORMER_PLAN.md` §6) |
| T−20 min | `python scripts/demo_ready.py` | today's live picture in ≤ 10 min (DATA-1), verdict, then the warm-up: every chat tool, managed MCP, a Genie query, both endpoints — no cold start on camera (the first MCP call was 37 s cold) |
| T−20 min, if `NOT_READY` | read the diagnosis it prints; run it once more if it says transient | otherwise demo on the last good data: the App's stale-data banner is itself a talking point ("it tells you when it's old") |
| T−10 min | open the App → **Chat with SkyWatch** → ask "Why does the model predict that DAL… lands when it does?" for any inbound callsign → **New conversation** | warms the App's own caches (tool list, Model 1 for `explain_eta`: 8.5 s the first time) |
| T−5 min | sidebar → **Tool status**: all ✅ | the startup self-check, run as the App's service principal |

Do **not** press **Get SkyWatch ready** during the recording: it clears the live picture and
takes ~10 minutes.

## Recording

| # | Show | Say (one line) | If today's data doesn't have it |
|---|---|---|---|
| 1 | Dashboard header: "Model 1 scored N min ago"; KPI tiles | "This is today's ingestion, as of HH:MM UTC — refreshed by one command before this call." | — |
| 2 | Dashboard **Ops briefing** panel and its caption | "An LLM writes this paragraph from the three models' numbers; a deterministic check rejects any number, time or callsign that isn't in the facts, and a template takes over if it fails." | — |
| 3 | Chat: **How busy is KATL right now?** | "The model never sees the database — it calls tools, which are Unity Catalog functions, and the badge says the answer was checked against what they returned." | — |
| 4 | Chat: **Is there any emergency or holding right now?** → open **How I answered** | "Here are the exact rows behind the answer." | "Nothing flagged" is a fine answer — say Model 3 is rules, and why (too few labels) |
| 5 | Chat: **Why does the model predict that _callsign_ lands when it does?** (a callsign from step 3/4) | "That's the in-process LightGBM champion's per-feature contributions, in plain words." | — |
| 6 | Chat: **How many flights have landed at Atlanta today?** | "Trap question: the number is right but easy to misread — SkyWatch only counts what it saw. A meaning rule forces it to say so; a correct number in a wrong sentence passes any value check." | — |
| 7 | Chat: **Did the Transformer beat LightGBM?** | "No — and that's the story: same registered model, same promotion gate, HOLD twice, and training longer made it worse. The gate kept the better model." | — |
| 8 | Chat: **What's the weather in Paris?** | "Out of scope: it declines without calling a tool." | — |
| 9 | Optional, ~20 s: **Which hour of the day is busiest on average?** | "Open questions go to Genie, which writes its own SQL — shown under the answer." | skip if short on time |
| 10 | Sidebar **Tool status**, **Get SkyWatch ready** | "A self-check runs as the App's identity at every start; refreshes are a human action with a confirmation — the model can read the refresh status but can't start one." | — |

## Numbers worth having ready

- Golden set: 38 questions, ≥ 90% on three consecutive runs before shipping (`docs/CHAT_PLAN.md`).
- Refresh: 8.5–10.1 min end to end; nine tasks in sequence because parallel serverless starts
  waited 645–705 s each (`docs/DEMO_REFRESH_PLAN.md`).
- Briefing model chosen by evaluation: Maverick 42/42 checks (`docs/OPS_BRIEFING_PLAN.md` §4).
- M1: LightGBM champion MAE 1.14 min; Transformer 1.19 (1 epoch) → 1.27 (10 epochs), both HOLD.

## If something breaks on camera

- **⚠️ fallback badge:** say what it means — the answer failed its check, so the App shows the
  tool rows instead of prose. That *is* the design.
- **Model endpoint errors:** the chat switches to the fallback endpoint by itself (shown in the
  answer's caption).
- **App down:** `databricks bundle run skywatch_arrival_manager -t dev --profile skywatch`
  (≈ 1 min), then check `databricks apps logs skywatch-arrival-manager | grep SELFCHECK`.
