"""E2 audit -- usable Psi_VI windows around substitution points.

报告 §3.1 flagged that the 60 s Psi_VI rolling window needs 60 consecutive
valid seconds, so only 4-13 usable windows exist in the +/-300 s around a
substitution.  This script counts, for every substitution in every match,
the finite Psi_VI windows under four definitions without running the TMLE:

  w60_g0    baseline: 60 s window, strict contiguity
  w30_g0    shorter window
  w60_g90   60 s window, unusable gaps <= 90 s linearly interpolated
  w60_gall  60 s window, gaps of any length interpolated

Output: output/vi/e2_window_audit.csv (one row per substitution per variant)
and a per-variant summary printed to stdout.

Usage:  python scripts/e2_window_audit.py [match-id ...]   (default: all
        matches that have a tracking file in data/raw)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_scalar import psi_series                       # noqa: E402
from src.vi_pipeline import TEAMS, ViMatch                 # noqa: E402

WINDOW = 300          # seconds before/after the substitution second (task2)
VARIANTS = [          # (name, win, max_gap)
    ("w60_g0", 60, 0),
    ("w30_g0", 30, 0),
    ("w60_g90", 60, 90),
    ("w60_gall", 60, None),
]


def psi_variants(v: ViMatch, side: str) -> dict[str, np.ndarray]:
    macro = v.team_vi[:, TEAMS.index(side)]
    micro = np.full((v.second.size, 10), np.nan)
    for t in np.where(v.valid)[0]:
        ids = v.outfield_order(side, t)
        if ids.size == 10:
            micro[t] = v.player_vi[t, ids]
    valid = (v.valid & np.isfinite(macro) & np.isfinite(micro).all(axis=1))
    return {name: psi_series(macro, micro, valid, win=win,
                             max_gap=max_gap)["psi"]
            for name, win, max_gap in VARIANTS}


def main() -> int:
    mids = (sys.argv[1:] if len(sys.argv) > 1
            else sorted(p.name for p in (ROOT / "data" / "raw").iterdir()
                        if p.is_dir()
                        and (p / f"{p.name}_tracking_extrapolated.jsonl").exists()))
    rows = []
    for mid in mids:
        raw = ROOT / "data" / "raw" / mid
        v = ViMatch(mid, raw, verbose=False)
        subs = v.substitutions()
        if not subs:
            print(f"{mid}: no substitutions, skipped")
            continue
        per_side = {s: psi_variants(v, s) for s in TEAMS}
        for e in subs:
            side = e["team"]
            idx = int(np.where((v.period == 2) & (v.second == e["t0"]))[0][0]) \
                if np.any((v.period == 2) & (v.second == e["t0"])) else np.nan
            for name, _, _ in VARIANTS:
                psi = per_side[side][name]
                n_pre = n_post = np.nan
                if np.isfinite(idx):
                    pre = psi[max(0, idx - WINDOW):idx]
                    post = psi[idx + 1:min(psi.size, idx + WINDOW + 1)]
                    n_pre = int(np.isfinite(pre).sum())
                    n_post = int(np.isfinite(post).sum())
                rows.append(dict(match_id=mid, team=side, t0=int(e["t0"]),
                                 variant=name, n_pre=n_pre, n_post=n_post))
    df = pd.DataFrame(rows)
    out = ROOT / "output" / "vi" / "e2_window_audit.csv"
    df.to_csv(out, index=False)

    summary = df.groupby("variant")[["n_pre", "n_post"]].agg(
        ["median", "min", "max", "mean"])
    n_subs = df[df["variant"] == "w60_g0"].dropna().shape[0]
    # share of substitution sides with >= 30 usable windows on each side
    for name, _, _ in VARIANTS:
        sub = df[df["variant"] == name]
        ge30 = int(((sub["n_pre"] >= 30) & (sub["n_post"] >= 30)).sum())
        print(f"{name:10s} n={int(sub.dropna().shape[0]):3d} "
              f"pre  med={sub['n_pre'].median():5.0f} "
              f"min={sub['n_pre'].min():3.0f} max={sub['n_pre'].max():3.0f} | "
              f"post med={sub['n_post'].median():5.0f} "
              f"min={sub['n_post'].min():3.0f} max={sub['n_post'].max():3.0f} | "
              f"both>=30: {ge30}/{n_subs}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
