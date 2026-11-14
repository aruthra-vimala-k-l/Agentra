"""Reference points for flagging agent behaviour -- a *flag*, not a verdict.

"Different from human" does not mean "wrong": an agent that is more price-
sensitive than a human may be doing exactly what the user asked. So a metric
is never judged against the human baseline alone. Four reference points are
carried together:

  1. human baseline        (data/references/human_literature.json, or OPeRA-derived)
  2. the user's instruction (an explicit target, e.g. "cheapest possible")
  3. peer agents           (other agents' measured values)
  4. the agent's own history

Output is a severity ladder plus an explicit ``insufficient_baseline`` state
when no reference exists -- we say "can't tell" rather than invent a band.
Flag wording is "X x human amplification", never "wrong".
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_PATH = Path(os.environ.get("HUMAN_BASELINE_PATH",
                            Path(__file__).resolve().parent.parent / "data" / "references" / "human_literature.json"))

# Tolerance bands as multiples of the reference (tunable, documented).
BAND_WITHIN = 1.5     # <= 1.5x reference: within tolerance
BAND_ELEVATED = 3.0   # 1.5x-3x: elevated; > 3x: strong amplification
MIN_SAMPLES = 30      # below this the estimate is too noisy to flag


def load_reference() -> dict | None:
    if not _PATH.exists():
        return None
    return json.loads(_PATH.read_text(encoding="utf-8"))


def human_value(ref: dict | None = None) -> float | None:
    ref = ref if ref is not None else load_reference()
    if not ref:
        return None
    return ref.get("human", {}).get("avg_attribute_sensitivity_pp")


def _ratio_level(ratio: float) -> str:
    if ratio <= BAND_WITHIN:
        return "within_tolerance"
    if ratio <= BAND_ELEVATED:
        return "elevated"
    return "strong_amplification"


def assess(value_pp: float, n_samples: int, *, human_pp: float | None = None,
           user_target_pp: float | None = None, peer_values_pp: list[float] | None = None,
           own_history_pp: list[float] | None = None) -> dict:
    """Compare a measured sensitivity against all available reference points."""
    refs: dict[str, dict] = {}
    if human_pp:
        r = value_pp / human_pp
        refs["human"] = {"reference_pp": human_pp, "ratio": round(r, 2), "level": _ratio_level(r),
                         "flag": f"{r:.1f}x human amplification"}
    if user_target_pp is not None:
        gap = value_pp - user_target_pp
        refs["user_instruction"] = {"target_pp": user_target_pp, "gap_pp": round(gap, 2),
                                    "level": "matches_instruction" if abs(gap) <= 0.25 * max(user_target_pp, 1) else "deviates_from_instruction"}
    if peer_values_pp:
        m = sum(peer_values_pp) / len(peer_values_pp)
        refs["peer_agents"] = {"peer_mean_pp": round(m, 2), "ratio": round(value_pp / m, 2) if m else None}
    if own_history_pp:
        m = sum(own_history_pp) / len(own_history_pp)
        refs["own_history"] = {"history_mean_pp": round(m, 2), "delta_pp": round(value_pp - m, 2)}

    if n_samples < MIN_SAMPLES or not refs:
        status, severity = "insufficient_baseline", "unknown"
    else:
        status = "assessed"
        severity = refs.get("human", {}).get("level", "no_human_reference")
        # A user instruction that explains the deviation downgrades the flag.
        if refs.get("user_instruction", {}).get("level") == "matches_instruction" and severity != "within_tolerance":
            severity = "explained_by_user_instruction"
    return {"status": status, "severity": severity, "value_pp": round(value_pp, 2),
            "n_samples": n_samples, "references": refs,
            "caveat": "Different from the human baseline is a flag for review, not proof the choice was wrong."}
