"""The promotion-gate decision (docs/MLOPS_PLAN.md Track 4) — pure Python, no Spark/mlflow,
so it has a real pytest suite (tests/test_promotion.py) instead of only ever being exercised
by a live job run. `promote_eta.py` scores the challenger/champion (needs Spark + the loaded
models) and calls `evaluate_gate` with the resulting per-band MAE dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_gate(
    challenger_bands: dict[str, float],
    champion_bands: dict[str, float] | None,
    min_improvement_pct: float,
    noise_band_pct: float,
    max_band_regression_min: float,
) -> GateResult:
    """`challenger_bands` / `champion_bands`: {band_label: MAE}, both must include "ALL".
    `champion_bands=None` means no `@champion` exists yet — bootstrap promotion, auto-pass.

    Passes if the overall MAE improves by at least `min_improvement_pct`, OR is within
    `noise_band_pct` of the champion (not meaningfully worse) — AND no individual band's MAE
    is worse than the champion's by more than `max_band_regression_min` minutes. A band
    improving is never a problem; only regression beyond the tolerance blocks promotion.
    """
    if champion_bands is None:
        return GateResult(True, ["no @champion exists yet — bootstrap promotion"])

    champ_all, chal_all = champion_bands["ALL"], challenger_bands["ALL"]
    improvement_pct = 100 * (champ_all - chal_all) / champ_all
    overall_ok = improvement_pct >= min_improvement_pct or improvement_pct >= -noise_band_pct

    band_regressions = {
        band: chal_mae - champion_bands[band]
        for band, chal_mae in challenger_bands.items()
        if band != "ALL"
        and band in champion_bands
        and chal_mae - champion_bands[band] > max_band_regression_min
    }

    passed = overall_ok and not band_regressions
    reasons = [f"overall MAE {chal_all:.3f} vs champion {champ_all:.3f} ({improvement_pct:+.1f}%)"]
    if not overall_ok:
        reasons.append(
            f"FAILS: worse than champion by more than the {noise_band_pct}% noise band "
            f"and short of the {min_improvement_pct}% improvement bar"
        )
    if band_regressions:
        reasons.append(
            f"FAILS: band regression > {max_band_regression_min} min in "
            f"{ {k: round(v, 3) for k, v in band_regressions.items()} }"
        )
    return GateResult(passed, reasons)
