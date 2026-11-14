"""Task 2 E4 -- bridge the substitution TMLE to performance outcomes.

The main Task 2 outcome is Psi_VI (coordination/emergence).  Paper B measures
*running performance* instead, so the two results are not directly comparable.
This script reuses the exact same design (5-min segments, A = substitution in
the previous segment, W1..W6 confounders, TMLE) with performance outcomes:

  * Y_dist   mean team speed (sum of the 10 outfielders' |dp/dt|, m/s)
  * Y_poss   fraction of the segment in which the team had possession
  * Y_psi    mean Psi_VI in the segment (the original outcome, for reference)

Per match:  python scripts/task2_perf_outcome.py --match-id 1886347
Pooled:     python scripts/task2_perf_outcome.py --pool

The pooled table lets "no effect on coordination" be read against
"effect on running/possession" directly (报告 §3.4's open question).
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

from src.ce_scalar import psi_series
from src.vi_pipeline import TEAMS, ViMatch
from scripts.task2_tmle import glm_ate, tmle

WCOLS = ["W1", "W2", "W3", "W4", "W5", "W6"]
SEG = 300
WIN = 60


# ---------------------------------------------------------------- per match
def build_segments(match_id: str, out_root: Path) -> pd.DataFrame:
    v = ViMatch(match_id, ROOT / "data" / "raw" / match_id)
    names = {s: v.match.teams[s].short_name for s in TEAMS}
    subs = v.substitutions()

    # Psi_VI per side (identical to task2_tmle.py)
    psi_vi = {}
    for side in TEAMS:
        macro = v.team_vi[:, TEAMS.index(side)]
        micro = np.full((v.second.size, 10), np.nan)
        for t in np.where(v.valid)[0]:
            ids = v.outfield_order(side, t)
            if ids.size == 10:
                micro[t] = v.player_vi[t, ids]
        valid = (v.valid & np.isfinite(macro) & np.isfinite(micro).all(axis=1))
        psi_vi[side] = psi_series(macro, micro, valid, win=WIN)["psi"]

    # team speed per second: sum of |dp| over the 10 outfielders, only when
    # the on-pitch 10 at t and t+1 are identical (same rule as task1)
    speed = {}
    for side in TEAMS:
        sp = np.full(v.second.size, np.nan)
        for t in range(v.second.size - 1):
            a = v.outfield_order(side, t)
            b = v.outfield_order(side, t + 1)
            if a.size != 10 or b.size != 10 or not np.array_equal(a, b):
                continue
            d = v.pos[t + 1, a] - v.pos[t, a]
            if np.isfinite(d).all():
                sp[t] = float(np.linalg.norm(d, axis=1).sum())
        speed[side] = sp

    poss = {s: (v.poss_group == s) for s in TEAMS}

    valid2_idx = np.where((v.period == 2) & v.valid)[0]
    n_seg = valid2_idx.size // SEG
    sec_valid2 = v.second[valid2_idx]

    def seg_of(t0: float) -> int:
        return int(np.searchsorted(sec_valid2, t0)) // SEG

    ev_path = (ROOT / "data/raw" / match_id
               / f"{match_id}_dynamic_events.csv")
    if ev_path.exists():
        ev = pd.read_csv(ev_path, low_memory=False)
        ev = ev[ev["team_score"].notna()]
    else:
        print(f"  WARNING {ev_path.name} missing: W4 falls back to 0")
        ev = None
    team_id = {s: v.match.teams[s].team_id for s in TEAMS}
    p2_frame0 = v.match.period_bounds[2][0]

    def goals_at(frame: int) -> dict:
        if ev is None:
            return {s: 0 for s in TEAMS}
        out = {}
        for s in TEAMS:
            sub = ev[(ev["team_id"] == team_id[s]) & (ev["frame_start"] <= frame)]
            out[s] = int(sub["team_score"].max()) if not sub.empty else 0
        return out

    poss_all = v.poss_group[v.valid]
    poss_all = poss_all[(poss_all == "home") | (poss_all == "away")]

    rows = []
    for side in TEAMS:
        other = "away" if side == "home" else "home"
        opp_poss = float((poss_all == other).mean()) if poss_all.size else np.nan
        for k in range(n_seg):
            idxs = valid2_idx[k * SEG:(k + 1) * SEG]
            y_psi = np.nanmean(psi_vi[side][idxs])
            y_dist = np.nanmean(speed[side][idxs])
            y_poss = float((poss[side][idxs]).mean())
            a_prev = any(e["team"] == side and seg_of(e["t0"]) == k - 1
                         for e in subs)
            a_now = any(e["team"] == side and seg_of(e["t0"]) == k
                        for e in subs)
            n_used = sum(1 for e in subs if e["team"] == side
                         and seg_of(e["t0"]) < k)
            g = goals_at(p2_frame0 + int(sec_valid2[k * SEG]) * 10)
            rows.append(dict(
                match_id=match_id, team=names[side], side=side, k=k,
                A=int(a_prev),
                Y_psi=y_psi, Y_dist=y_dist, Y_poss=y_poss,
                W1=k, W2=n_used, W3=int(a_now),
                W4=int(g[side] - g[other]),
                W5=int(side == "home"), W6=opp_poss,
            ))
    seg = pd.DataFrame(rows)
    seg.to_csv(out_root / match_id / "task2_perf_segments.csv", index=False)
    return seg


# ---------------------------------------------------------------------- TMLE
def run_outcome(sub: pd.DataFrame, ycol: str, label: str,
                n_perm: int = 100, seed: int = 0) -> dict | None:
    from scipy.special import expit, logit
    from sklearn.linear_model import LogisticRegression

    from scripts.task2_tmle import _epsilon_fit, q_library

    d = sub.dropna(subset=[ycol, "W6"]).reset_index(drop=True)
    if len(d) < 6 or d["A"].nunique() < 2:
        return None
    A = d["A"].to_numpy(float)
    W = d[WCOLS].to_numpy(float)
    y = d[ycol].to_numpy(float)
    lo, hi = float(y.min()), float(y.max())
    scale = hi - lo if hi > lo else 1.0
    yn = (y - lo) / scale if scale > 0 else np.zeros_like(y)

    r = tmle(A, W, yn, seed=seed)

    def ate_with_model(model):
        n = A.size
        X = np.column_stack([A, W])
        q = np.clip(model.predict(X), 1e-6, 1 - 1e-6)
        q1 = np.clip(model.predict(np.column_stack([np.ones(n), W])), 1e-6, 1 - 1e-6)
        q0 = np.clip(model.predict(np.column_stack([np.zeros(n), W])), 1e-6, 1 - 1e-6)
        gmodel = LogisticRegression(C=1.0, max_iter=2000).fit(W, A.astype(int))
        g = np.clip(gmodel.predict_proba(W)[:, 1], 1e-6, 1 - 1e-6)
        h = A / g - (1 - A) / (1 - g)
        eps = _epsilon_fit(yn, logit(q), h)
        qs1 = np.clip(expit(logit(q1) + eps * 1.0 / g), 1e-6, 1 - 1e-6)
        qs0 = np.clip(expit(logit(q0) - eps * 1.0 / (1 - g)), 1e-6, 1 - 1e-6)
        return float(np.mean(qs1 - qs0))

    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        ap = rng.permutation(A)
        m = q_library(seed)[r["learner"]].fit(np.column_stack([ap, W]), yn)
        try:
            perm.append(ate_with_model(m))
        except Exception:                                      # noqa: BLE001
            continue
    perm = np.asarray(perm)
    p = float((np.abs(perm) >= abs(r["ate"])).mean()) if perm.size else np.nan

    return dict(
        outcome=ycol, group=label,
        n=int(len(d)), n_treated=int(A.sum()),
        ate_abs=float(r["ate"] * scale),
        ci_low=float(r["ci_low"] * scale),
        ci_high=float(r["ci_high"] * scale),
        scale=float(scale), units="nats" if ycol == "Y_psi" else
        ("m/s" if ycol == "Y_dist" else "fraction"),
        super_learner=r["learner"], g_range=[float(r["g_min"]),
                                             float(r["g_max"])],
        glm_abs=float(glm_ate(A, W, yn) * scale),
        perm_p=p,
    )


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", default=None)
    ap.add_argument("--pool", action="store_true",
                    help="run the pooled analysis over every match that has "
                         "task2_perf_segments.csv")
    ap.add_argument("--out-root", default=str(ROOT / "output" / "vi"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not args.pool:
        if args.match_id is None:
            raise SystemExit("give --match-id or --pool")
        seg = build_segments(args.match_id, out_root)
        frames = [seg]
    else:
        files = sorted(out_root.glob("*/task2_perf_segments.csv"))
        if not files:
            raise SystemExit("no task2_perf_segments.csv found -- run per "
                             "match first")
        frames = [pd.read_csv(f) for f in files]

    seg = pd.concat(frames, ignore_index=True)
    seg["match_id"] = seg["match_id"].astype(str)
    seg["W7"] = pd.factorize(seg["match_id"])[0]
    if args.pool:
        WCOLS.append("W7")

    rows = []
    for ycol, label in [("Y_psi", "psi_vi"), ("Y_dist", "team_speed"),
                        ("Y_poss", "possession")]:
        # per-match ATE for the pooled view
        for mid, g in seg.groupby("match_id"):
            r = run_outcome(g, ycol, f"{label}_match_{mid}", seed=args.seed)
            if r:
                rows.append(r)
        if args.pool:
            r = run_outcome(seg, ycol, f"{label}_pooled", seed=args.seed)
            if r:
                rows.append(r)

    out = out_root / "task2_perf_tmle_summary.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_root / "task2_perf_tmle_ate.csv",
                              index=False)
    cols = ["outcome", "group", "n", "n_treated", "ate_abs", "ci_low",
            "ci_high", "glm_abs", "perm_p", "super_learner"]
    print(pd.DataFrame(rows)[cols].round(5).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
