"""The LLM ops briefing (docs/OPS_BRIEFING_PLAN.md) — pure Python + pandas, no Spark/SDK, so it
has a real pytest suite (tests/test_briefing.py). `brief_ops.py` reads the latest scoring rows,
calls the Foundation Model API, and uses these helpers for everything around that call:

  build_facts       latest predictions / demand_forecast / irregularity_flags -> one small dict,
                    numbers rounded once here so the check can compare them exactly
  build_messages    the chat prompt (facts as JSON, data age left out on purpose)
  extract_text      the answer text from an `llm/v1/chat` response (string or content parts)
  check_briefing    deterministic grounding check: every number, HH:MM time and callsign in the
                    text must come from the facts, and the text must be short
  template_briefing the same picture as plain f-strings — the fallback when the endpoint fails or
                    the check rejects its answer. It always passes check_briefing (tested).

The check catches invented or recomputed numbers, times and callsigns. It does not catch a
wrong word next to a correct number ("42 arrived" for 42 inbound); the evaluation
(scripts/eval_briefing.py) measures that, it isn't claimed solved here.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

import pandas as pd

from lib import grounding

WINDOWS_MIN = (10, 30)       # "predicted to land within N minutes" buckets
BIN_MINUTES = 15
MAX_WORDS = 120
MAX_FLAGGED = 5
MAX_NEXT = 3


# ----------------------------------------------------------------------------- facts
def _as_of(df: pd.DataFrame, now: _dt.datetime) -> dict[str, Any] | None:
    if df is None or df.empty or "scored_at" not in df:
        return None
    ts = pd.to_datetime(df["scored_at"], utc=True).max()
    if pd.isna(ts):
        return None
    age = (now - ts.to_pydatetime()).total_seconds() / 60.0
    return {"as_of": ts.strftime("%Y-%m-%d %H:%M"), "age_min": max(0, round(age))}


def _inbound(preds: pd.DataFrame | None) -> dict[str, Any] | None:
    if preds is None:
        return None
    eta = pd.to_numeric(preds["predicted_eta_min"], errors="coerce")
    order = preds.assign(eta_=eta).sort_values("eta_").head(MAX_NEXT)
    out: dict[str, Any] = {"count": int(len(preds))}
    for w in WINDOWS_MIN:
        out[f"within_{w}_min"] = int((eta <= w).sum())
    out["next"] = [
        {"callsign": str(r.callsign).strip() or str(r.icao), "eta_min": round(float(r.eta_), 1)}
        for r in order.itertuples()
        if pd.notna(r.eta_)
    ]
    return out


def _demand(fc: pd.DataFrame | None, aar_per_hour: float) -> dict[str, Any] | None:
    if fc is None or fc.empty:
        return None
    fc = fc.sort_values("horizon_step")
    aar_bin = aar_per_hour * BIN_MINUTES / 60.0
    mean = pd.to_numeric(fc["predicted_mean"], errors="coerce")
    upper = pd.to_numeric(fc["predicted_q90"], errors="coerce")
    peak = int(mean.to_numpy().argmax())
    peak_mean = round(float(mean.iloc[peak]), 1)
    return {
        # landings SkyWatch *observed* today: slots it wasn't collecting count as 0 in
        # score_demand.py, so after a short burst this is far below the airport's real count
        "arrivals_seen_by_skywatch_today": int(round(float(fc["arrivals_so_far"].iloc[0]))),
        "predicted_next_hour": int(round(float(mean.head(60 // BIN_MINUTES).sum()))),
        "horizon_hours": round(len(fc) * BIN_MINUTES / 60),
        "peak_15min_start_utc": pd.to_datetime(fc["bin_start_ts"].iloc[peak], utc=True).strftime("%H:%M"),
        "peak_15min_predicted": peak_mean,
        "peak_15min_upper": round(float(upper.iloc[peak]), 1),
        "peak_15min_headroom": round(aar_bin - peak_mean, 1),
        "bins_over_aar": int((mean > aar_bin).sum()),
        "bins_upper_over_aar": int((upper > aar_bin).sum()),
    }


def _irregularities(flags: pd.DataFrame | None) -> dict[str, Any] | None:
    if flags is None:
        return None
    counts = {k: 0 for k in ("emergency", "go_around", "holding")}
    flagged = []
    if not flags.empty:
        for k, n in flags["kind"].value_counts().items():
            counts[str(k)] = int(n)
        top = flags.sort_values(["severity", "dist_to_apt_nm"], ascending=[False, True])
        for r in top.head(MAX_FLAGGED).itertuples():
            flagged.append({
                "callsign": str(r.callsign).strip() or str(r.icao),
                "kind": str(r.kind),
                "dist_nm": int(round(float(r.dist_to_apt_nm))),
            })
    return {**counts, "flagged": flagged}


def build_facts(
    preds: pd.DataFrame | None,
    fc: pd.DataFrame | None,
    flags: pd.DataFrame | None,
    *,
    apt_icao: str,
    aar_per_hour: float,
    now: _dt.datetime,
    stale_after_min: int = 30,
    flags_window_min: int = 15,
) -> dict[str, Any]:
    """`None` means "that source is unavailable" (table missing / unreadable) and becomes a
    `null` section. An empty `preds` or `flags` frame is a real zero — nothing inbound, nothing
    flagged. An empty `fc` is `null`: a forecast run always writes its full horizon.

    `score_irregularities.py` writes no rows on a run that flags nothing, so the newest flags
    can be from long before the newest predictions. Flags older than `flags_window_min` before
    the predictions run are dropped (= nothing flagged now) — both jobs run on the same 10-min
    schedule, so a fresh run would have re-flagged them."""
    if flags is not None and not flags.empty and preds is not None and not preds.empty:
        p_ts = pd.to_datetime(preds["scored_at"], utc=True).max()
        f_ts = pd.to_datetime(flags["scored_at"], utc=True).max()
        if f_ts < p_ts - pd.Timedelta(minutes=flags_window_min):
            flags = flags.iloc[0:0]
    data = {
        "arrival_predictions": _as_of(preds, now),
        "demand_forecast": _as_of(fc, now),
        "irregularity_flags": _as_of(flags, now),
    }
    ages = [d["age_min"] for d in data.values() if d]
    return {
        "airport": apt_icao,
        "aar_per_hour": int(round(aar_per_hour)),
        "aar_per_15min": round(aar_per_hour * BIN_MINUTES / 60.0, 1),
        "bin_minutes": BIN_MINUTES,
        "inbound": _inbound(preds),
        "demand": _demand(fc, aar_per_hour),
        "irregularities": _irregularities(flags),
        "data": data,
        "max_age_min": max(ages) if ages else None,
        "source_skew_min": (max(ages) - min(ages)) if len(ages) > 1 else 0,
        "stale": (not ages) or max(ages) > stale_after_min,
    }


def prompt_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """What the model sees: everything except data age. The App renders age from the row's own
    columns; the model is not trusted to state it, and can't if it never sees it."""
    return {k: v for k, v in facts.items() if k not in ("data", "max_age_min", "source_skew_min")}


# ---------------------------------------------------------------------------- prompt
SYSTEM_PROMPT = """You write the ops briefing for an airport arrival coordinator.
Terms:
- AAR (airport acceptance rate) is capacity: the most arrivals the airport can take per hour (aar_per_hour) or per 15 minutes (aar_per_15min). It is not a forecast and not a count.
- "inbound" aircraft are still in the air.
- "arrivals_seen_by_skywatch_today" counts only landings SkyWatch observed while it was collecting data; it is NOT the airport's total for the day. If you mention it, say "SkyWatch has seen N landings today", never "N arrivals so far today".
- "predicted" values are the forecast; "upper" values are the high end of its range, not the expected value.
- "headroom" is capacity minus the predicted peak; negative means over capacity.
- "bins" are 15-minute periods.
Rules:
- Use only the facts in the JSON. Every number, time and callsign you write must appear in the facts exactly as given.
- Do not calculate new numbers: no differences, sums, percentages or ratios.
- If a section is null, say that its data is unavailable.
- Do not mention data age or timestamps other than peak_15min_start_utc; the app shows them.
- Lead with the most important point: an emergency or a predicted surge over the AAR first, then holding, then the normal picture.
- Write for a person: plain words, never the JSON key names.
- Write 3 to 5 plain sentences, under 100 words. No lists, no headings, no markdown."""


def build_messages(facts: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Facts:\n" + json.dumps(prompt_facts(facts), indent=1)},
    ]


def extract_text(response: dict[str, Any]) -> str:
    """`choices[0].message.content` is a string for most endpoints; reasoning models
    (gpt-oss) return a list of parts — keep the `text` parts, drop `reasoning`."""
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return str(content or "").strip()


# ----------------------------------------------------------------------------- check
# The check itself lives in lib/grounding.py, shared with the chat (lib/chat.py).
CheckResult = grounding.CheckResult


def check_briefing(text: str, facts: dict[str, Any], max_words: int = MAX_WORDS) -> CheckResult:
    # the window sizes are in the facts as key names ("within_10_min"), not values
    allowed = grounding.harvest(prompt_facts(facts), extra_numbers=WINDOWS_MIN)
    return grounding.check_text(text, allowed, max_words, source="facts")


# -------------------------------------------------------------------------- template
def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def template_briefing(facts: dict[str, Any]) -> str:
    apt = facts["airport"]
    out = []

    irr = facts["irregularities"]
    if irr and irr["emergency"]:
        cs = [f["callsign"] for f in irr["flagged"] if f["kind"] == "emergency"]
        out.append(f"Emergency: {', '.join(cs) or str(irr['emergency']) + ' aircraft'} squawking emergency.")

    dem = facts["demand"]
    if dem is None:
        out.append("The demand forecast is unavailable.")
    else:
        lead = (f"Surge: {dem['bins_over_aar']} {_plural(dem['bins_over_aar'], 'bin', 'bins')} "
                f"of 15 minutes {_plural(dem['bins_over_aar'], 'is', 'are')} predicted over the AAR. "
                if dem["bins_over_aar"] else "")
        out.append(
            f"{lead}{dem['predicted_next_hour']} arrivals are predicted in the next hour against an "
            f"AAR of {facts['aar_per_hour']} per hour. The busiest 15 minutes starts at "
            f"{dem['peak_15min_start_utc']} UTC with {dem['peak_15min_predicted']} predicted "
            f"(upper {dem['peak_15min_upper']}) against {facts['aar_per_15min']}."
        )
        if not dem["bins_over_aar"] and dem["bins_upper_over_aar"]:
            out.append(f"No bin is predicted over the AAR, but {dem['bins_upper_over_aar']} "
                       f"could be at the upper estimate.")

    inb = facts["inbound"]
    if inb is None:
        out.append("Arrival predictions are unavailable.")
    elif inb["count"] == 0:
        out.append(f"No aircraft are inbound to {apt}.")
    else:
        nxt = inb["next"][0] if inb["next"] else None
        first = f" First is {nxt['callsign']} in {nxt['eta_min']} minutes." if nxt else ""
        out.append(
            f"{inb['count']} {_plural(inb['count'], 'aircraft is', 'aircraft are')} inbound to {apt}: "
            f"{inb['within_10_min']} predicted to land within 10 minutes and "
            f"{inb['within_30_min']} within 30.{first}"
        )

    if irr is None:
        out.append("Irregularity flags are unavailable.")
    elif irr["holding"] or irr["go_around"]:
        parts = [f"{f['callsign']} ({f['kind'].replace('_', '-')}, {f['dist_nm']} nm)"
                 for f in irr["flagged"] if f["kind"] != "emergency"]
        out.append(f"Flagged: {', '.join(parts)}.")
    elif not irr["emergency"]:
        out.append("No aircraft are flagged for holding, go-around or emergency.")
    return " ".join(out)
