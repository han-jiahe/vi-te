"""Regenerate only the spec-named `figure_task2_psi_prepost.png` for a match.

The full `task2_tmle.py` run takes ~10 min because of the 200-permutation test;
this script recomputes just Psi_VI and the +/-5 min windows, so the figure can
be (re)built in seconds without touching the saved TMLE results.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_scalar import psi_series                 # noqa: E402
from src.vi_pipeline import TEAMS, ViMatch           # noqa: E402

WIN = 60
WINDOW = 300


def main() -> int:
    for mid in (sys.argv[1:] or ["1886347"]):
        outdir = ROOT / "output" / "vi" / mid
        outdir.mkdir(parents=True, exist_ok=True)
        v = ViMatch(mid, ROOT / "data" / "raw" / mid, verbose=False)

        psi = {}
        for side in TEAMS:
            macro = v.team_vi[:, TEAMS.index(side)]
            micro = np.full((v.second.size, 10), np.nan)
            for t in np.where(v.valid)[0]:
                ids = v.outfield_order(side, t)
                if ids.size == 10:
                    micro[t] = v.player_vi[t, ids]
            valid = v.valid & np.isfinite(macro) & np.isfinite(micro).all(axis=1)
            psi[side] = psi_series(macro, micro, valid, win=WIN)["psi"]

        p2_start = int(np.where(v.period == 2)[0][0])
        rows = []
        for e in v.substitutions():
            idx = p2_start + e["t0"]
            tr = np.full(2 * WINDOW + 1, np.nan)
            pre = np.arange(idx - WINDOW, idx)
            post = np.arange(idx + 1, idx + WINDOW + 1)
            ok = (pre >= 0) & (pre < v.second.size)
            tr[np.where(ok)[0]] = psi[e["team"]][pre[ok]]
            ok = post < v.second.size
            tr[WINDOW + 1 + np.where(ok)[0]] = psi[e["team"]][post[ok]]
            rows.append(tr)
        if not rows:
            print(f"{mid}: no substitution events")
            continue

        mat = np.vstack(rows)
        x = np.arange(-WINDOW, WINDOW + 1) / 60.0
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / np.sqrt(
                np.maximum(np.sum(np.isfinite(mat), axis=0), 1))

        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        for row in mat:
            ax.plot(x, row, color="#bbbbbb", lw=0.6, alpha=0.5)
        ax.plot(x, m, color="#1f77b4", lw=2.2,
                label=f"mean over {len(mat)} events (both teams)")
        ax.fill_between(x, m - sem, m + sem, color="#1f77b4", alpha=0.2, lw=0)
        ax.axvline(0, color="k", lw=0.9, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("time relative to substitution t0 (min)")
        ax.set_ylabel("Psi_VI (nats)")
        ax.set_title("Task 2: Psi_VI in the +/-5 min around every substitution",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = outdir / "figure_task2_psi_prepost.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"{mid}: wrote {out}  ({len(mat)} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
