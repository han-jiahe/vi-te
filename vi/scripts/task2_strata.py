"""Task 2 E1 -- context stratification of the pooled substitution effect.

Loads the existing per-match segment/substitution tables (no tracking data
needed) plus the ``match.json`` metadata for player roles, then re-estimates
the pooled TMLE ATE within each stratum:

  * score state      W4 < 0 (trailing) / == 0 (level) / > 0 (leading)
  * match phase      k in {0,1} early / {2,3} mid / {4,5} late (5-min blocks)
  * position of the replaced player (defender / midfielder / forward / mixed)
  * timing of the substitution (2nd-half <15 / 15-30 / >30 min / mixed)

Position and timing are attributes of the *treated* segments, so those two
analyses compare each exposure group against the shared control (A = 0) with
the same confounders W1..W7 -- a standard effect-modification reading.
Each stratum runs its own min-max normalisation and its own permutation test.

Usage:
    python scripts/task2_strata.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression

from scripts.task2_tmle import glm_ate, tmle      # noqa: E402
from scripts.task2_tmle import q_library           # noqa: E402

WCOLS = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]


# --------------------------------------------------------------------- loading
def load_segments(out_root: Path) -> pd.DataFrame:
    seg = pd.concat(
        [pd.read_csv(f) for f in sorted(out_root.glob("*/task2_segments.csv"))],
        ignore_index=True)
    seg["match_id"] = seg["match_id"].astype(str)
    seg["W7"] = pd.factorize(seg["match_id"])[0]
    return seg


def load_subs(out_root: Path) -> pd.DataFrame:
    subs = pd.concat(
        [pd.read_csv(f) for f in sorted(out_root.glob("*/task2_substitutions.csv"))],
        ignore_index=True)
    subs["match_id"] = subs["match_id"].astype(str)
    return subs


def load_roles(raw_root: Path) -> dict:
    """{match_id: {short_name: role_name}} from the match metadata."""
    out: dict[str, dict[str, str]] = {}
    for f in sorted(raw_root.glob("*/*_match.json")):
        mid = f.parent.name
        meta = json.loads(f.read_text(encoding="utf-8"))
        out[mid] = {
            p.get("short_name"): (p.get("player_role") or {}).get("name")
            for p in meta.get("players", []) if p.get("short_name")}
    return out


def broad_role(role: str | None) -> str | None:
    if not role or not isinstance(role, str):
        return None
    r = role.lower()
    if "back" in r:
        return "defender"
    if "midfield" in r:
        return "midfielder"
    if "forward" in r or "winger" in r:
        return "forward"
    return None          # Goalkeeper / "Substitute" / unknown


def timing_bucket(t0: float) -> str:
    if t0 < 900:
        return "early_lt15m"
    if t0 <= 1800:
        return "mid_15_30m"
    return "late_gt30m"


# ----------------------------------------------------------- treatment mapping
def map_treatment(seg: pd.DataFrame, subs: pd.DataFrame,
                  roles: dict) -> pd.DataFrame:
    """Add exposure-group columns for the treated rows.

    ``t_start`` of the team's previous segment bounds the interval in which a
    substitution triggers A = 1 for the current segment, so a row is joined to
    exactly the substitutions that caused its treatment.
    """
    seg = seg.sort_values(["match_id", "side", "k"]).reset_index(drop=True)
    sub_by_match_side = {
        (mid, side): g for (mid, side), g in
        subs.groupby(["match_id", "side"])}
    prev = seg.groupby(["match_id", "side"])["t_start"].shift(1)
    seg["t_prev_start"] = prev

    pos_group, tim_group = [], []
    for row in seg.itertuples():
        if row.A != 1 or np.isnan(row.t_prev_start):
            pos_group.append("")
            tim_group.append("")
            continue
        evs = sub_by_match_side.get((row.match_id, row.side))
        mask = (evs["t0"] >= row.t_prev_start) & (evs["t0"] < row.t_start)
        trig = evs[mask] if evs is not None else evs.iloc[0:0]
        rmap = roles.get(row.match_id, {})
        roles_seen = {broad_role(rmap.get(n)) for n in trig["player_out"]}
        times_seen = {timing_bucket(t) for t in trig["t0"]}
        roles_seen.discard(None)
        if not roles_seen:
            pos_group.append("unknown")
        elif len(roles_seen) == 1:
            pos_group.append(next(iter(roles_seen)))
        else:
            pos_group.append("mixed_positions")
        if len(times_seen) == 1:
            tim_group.append(next(iter(times_seen)))
        else:
            tim_group.append("mixed_timing")
    seg["sub_pos"] = pos_group
    seg["sub_time"] = tim_group
    return seg


# ---------------------------------------------------------------------- TMLE
def tmle_ate_fast(A: np.ndarray, W: np.ndarray, y: np.ndarray,
                  model, clip: float = 1e-6) -> float:
    """Logit-fluctuation TMLE ATE with a pre-fitted Q model (no CV).

    The Q model is refit by the caller on the (possibly permuted) data; only
    the fluctuation step and the ATE averaging run here.  The learner choice
    is fixed to the observed-data winner, so this is a conditional permutation
    scheme (documented approximation; the CI is the primary inference).
    """
    from scipy.special import expit, logit

    from scripts.task2_tmle import _epsilon_fit        # noqa: F401

    n = A.size
    X = np.column_stack([A, W])
    q = np.clip(model.predict(X), clip, 1 - clip)
    q1 = np.clip(model.predict(np.column_stack([np.ones(n), W])), clip, 1 - clip)
    q0 = np.clip(model.predict(np.column_stack([np.zeros(n), W])), clip, 1 - clip)
    gmodel = LogisticRegression(C=1.0, max_iter=2000).fit(W, A.astype(int))
    g = np.clip(gmodel.predict_proba(W)[:, 1], clip, 1 - clip)
    h = A / g - (1 - A) / (1 - g)
    eps = _epsilon_fit(y, logit(q), h)
    qs1 = np.clip(expit(logit(q1) + eps * 1.0 / g), clip, 1 - clip)
    qs0 = np.clip(expit(logit(q0) - eps * 1.0 / (1 - g)), clip, 1 - clip)
    return float(np.mean(qs1 - qs0))


def run_stratum(sub: pd.DataFrame, label: str, n_perm: int = 100,
                seed: int = 0, perm: str = "fast") -> dict | None:
    d = sub.dropna(subset=["Y_nats", "W6"]).reset_index(drop=True)
    if len(d) < 6 or d["A"].nunique() < 2:
        return None
    A = d["A"].to_numpy(float)
    W = d[WCOLS].to_numpy(float)
    y = d["Y_nats"].to_numpy(float)
    lo, hi = float(y.min()), float(y.max())
    scale = hi - lo if hi > lo else 1.0
    yn = (y - lo) / scale if scale > 0 else np.zeros_like(y)

    r = tmle(A, W, yn, seed=seed)
    scheme = perm
    rng = np.random.default_rng(seed)
    perm_vals = []
    for _ in range(n_perm):
        try:
            if scheme == "full":
                perm_vals.append(tmle(rng.permutation(A), W, yn, seed=seed)["ate"])
            else:
                ap = rng.permutation(A)
                m = q_library(seed)[r["learner"]].fit(np.column_stack([ap, W]), yn)
                perm_vals.append(tmle_ate_fast(ap, W, yn, m))
        except Exception:                                      # noqa: BLE001
            continue
    perm = np.asarray(perm_vals)
    p = float((np.abs(perm) >= abs(r["ate"])).mean()) if perm.size else np.nan

    return dict(
        stratum=label,
        n=int(len(d)), n_treated=int(A.sum()),
        ate_nats=float(r["ate"] * scale),
        ci_low_nats=float(r["ci_low"] * scale),
        ci_high_nats=float(r["ci_high"] * scale),
        ate_norm=float(r["ate"]),
        se_norm=float(r["se"]),
        scale=float(scale),
        y_range_nats=[lo, hi],
        super_learner=r["learner"],
        g_range=[float(r["g_min"]), float(r["g_max"])],
        glm_nats=float(glm_ate(A, W, yn) * scale),
        perm_p=p,
        perm=scheme,
        n_perm=int(perm.size),
    )


def exposure_stratum(seg: pd.DataFrame, col: str, value: str,
                     label: str) -> pd.DataFrame | None:
    """A = 0 controls + the treated rows whose exposure attribute is ``value``."""
    treated = (seg["A"] == 1) & (seg[col] == value)
    if treated.sum() == 0:
        return None
    return seg[(seg["A"] == 0) | treated]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / "output" / "vi"))
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root, raw_root = Path(args.out_root), Path(args.raw_root)
    seg = load_segments(out_root)
    subs = load_subs(out_root)
    roles = load_roles(raw_root)
    seg = map_treatment(seg, subs, roles)

    seg["score"] = np.select([seg["W4"] < 0, seg["W4"] == 0],
                             ["trailing", "level"], default="leading")
    seg["phase"] = np.select([seg["k"].isin([0, 1]), seg["k"].isin([2, 3])],
                             ["early_k01", "mid_k23"], default="late_k45")

    results, rows = [], []

    def add(r: dict | None) -> None:
        if r is not None:
            results.append(r)
            rows.append(r)

    # sanity: pooled baseline, comparable to 报告 §3.5 (full 200-perm to
    # reproduce the reported p = 0.975; the strata below use the fast scheme)
    base = run_stratum(seg, "pooled_all", seed=args.seed, n_perm=200, perm="full")
    add(base)

    for v, lab in [("trailing", "score_trailing"), ("level", "score_level"),
                   ("leading", "score_leading")]:
        add(run_stratum(seg[seg["score"] == v], lab, seed=args.seed))
    for v, lab in [("early_k01", "phase_early_k01"), ("mid_k23", "phase_mid_k23"),
                   ("late_k45", "phase_late_k45")]:
        add(run_stratum(seg[seg["phase"] == v], lab, seed=args.seed))
    for v, lab in [("defender", "pos_defender"), ("midfielder", "pos_midfielder"),
                   ("forward", "pos_forward"), ("mixed_positions", "pos_mixed"),
                   ("unknown", "pos_unknown")]:
        sub = exposure_stratum(seg, "sub_pos", v, lab)
        if sub is not None:
            add(run_stratum(sub, lab, seed=args.seed))
    for v, lab in [("early_lt15m", "time_early_lt15m"),
                   ("mid_15_30m", "time_mid_15_30m"),
                   ("late_gt30m", "time_late_gt30m"),
                   ("mixed_timing", "time_mixed")]:
        sub = exposure_stratum(seg, "sub_time", v, lab)
        if sub is not None:
            add(run_stratum(sub, lab, seed=args.seed))

    out = out_root / "task2_strata_summary.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    pd.DataFrame(results).to_csv(out_root / "task2_strata_ate.csv",
                                 index=False)

    # readable table
    cols = ["stratum", "n", "n_treated", "ate_nats", "ci_low_nats",
            "ci_high_nats", "glm_nats", "perm_p", "super_learner", "g_range"]
    print(pd.DataFrame(results)[cols].round(4).to_string(index=False))
    print("\ntreated-segment exposure groups:")
    print(seg[seg["A"] == 1].groupby(["sub_pos", "sub_time"]).size().to_string())
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
