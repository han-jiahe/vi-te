"""Task 1 — causal emergence of the 50 micro features onto a team's VI.

Follows 实现方案_VI因果涌现与换人TMLE §4:

    V(t)      = team VI (scalar, sum of the 11 on-pitch players' Voronoi VI)
    X_k(t)    = 10 outfield players' (x, y) [20], (vx, vy) [20], d [10]
    Psi       = I(V_t; V_{t+1}) - sum_k I(X_k(t); V_{t+1})

Global estimate per half plus a 60 s sliding window (step 1 s) assigned to the
window's last second.
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

WIN = 60


def team_features(v: ViMatch, side: str):
    """Return (V, sources, valid) for one team over the whole match.

    ``sources`` has 50 columns: x1..x10, y1..y10, vx1..vx10, vy1..vy10, d1..d10.
    """
    n = v.second.size
    k = 10
    x = np.full((n, k), np.nan)
    y = np.full((n, k), np.nan)
    d = np.full((n, k), np.nan)
    centre = np.array([v.length / 2.0, v.width / 2.0])

    order = [None] * n
    for t in range(n):
        if not v.valid[t]:
            continue
        ids = v.outfield_order(side, t)
        if ids.size != k:
            continue
        order[t] = ids
        p = v.pos[t, ids]
        x[t] = p[:, 0]
        y[t] = p[:, 1]
        d[t] = np.linalg.norm(p - centre, axis=1)

    vx = np.full((n, k), np.nan)
    vy = np.full((n, k), np.nan)
    same = np.zeros(n, dtype=bool)
    for t in range(n - 1):
        if order[t] is None or order[t + 1] is None:
            continue
        if order[t].size != k or not np.array_equal(order[t], order[t + 1]):
            continue
        p0 = v.pos[t, order[t]]
        p1 = v.pos[t + 1, order[t]]
        vx[t] = p1[:, 0] - p0[:, 0]
        vy[t] = p1[:, 1] - p0[:, 1]
        same[t] = True

    macro = v.team_vi[:, TEAMS.index(side)]
    sources = np.hstack([x, y, vx, vy, d])
    valid = v.valid & np.isfinite(macro) & np.isfinite(sources).all(axis=1) & same
    return macro, sources, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", default="1886347")
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "vi"))
    ap.add_argument("--hz", type=int, default=1, choices=(1, 10),
                    help="VI rate grid: 1 = dt 1 s (pipeline legacy), "
                         "10 = dt 0.1 s at 10 fps downsampled to 1 Hz "
                         "(paper arXiv:2102.11182 spec)")
    args = ap.parse_args()

    # 10 Hz runs go to a separate folder so the baseline outputs are kept
    outdir = Path(args.out_dir) / (args.match_id
                                   + ("" if args.hz == 1 else "_h10"))
    outdir.mkdir(parents=True, exist_ok=True)

    v = ViMatch(args.match_id, ROOT / "data" / "raw" / args.match_id, hz=args.hz)
    sides = {"home": v.match.teams["home"].short_name,
             "away": v.match.teams["away"].short_name}

    rows, per_second, per_player_vi = [], [], []
    psi_store = {}
    for side in TEAMS:
        macro, sources, valid = team_features(v, side)
        per_player_vi.append(pd.DataFrame({
            "second": v.second, "period": v.period, "team": side,
            "team_vi": macro,
            "player_vi_mean": np.array([
                np.nanmean(v.player_vi[t, v.outfield[side]])
                if v.valid[t] else np.nan
                for t in range(v.second.size)]),
            "player_vi_sd": np.array([
                np.nanstd(v.player_vi[t, v.outfield[side]])
                if v.valid[t] else np.nan
                for t in range(v.second.size)]),
        }))
        for per in (1, 2):
            sel = v.period == per
            if not sel.any():
                continue
            g = psi_series(macro, sources, valid & sel, win=None)
            r = psi_series(macro, sources, valid & sel, win=WIN)
            psi_store[(side, per)] = r
            poss = v.poss_group[sel & v.valid]
            poss = poss[(poss == "home") | (poss == "away")]
            rows.append(dict(
                team=sides[side], side=side, half=per,
                possession_rate=float((poss == side).mean()) if poss.size else np.nan,
                i_macro=g["i_macro_mean"],
                i_micro_sum=g["i_micro_sum_mean"],
                psi_global=g["psi_mean"],
                psi_rolling_mean=r["psi_mean"],
                i_macro_rolling_mean=r["i_macro_mean"],
                i_micro_rolling_mean=r["i_micro_sum_mean"],
                n_seconds=int((valid & sel).sum()),
                n_windows=r["n_windows"],
            ))
            for t in np.where(np.isfinite(r["psi"]))[0]:
                per_second.append(dict(second=int(v.second[t]), period=int(v.period[t]),
                                       half=per, side=side, team=sides[side],
                                       psi_rolling=float(r["psi"][t])))

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "task1_ce_summary.csv", index=False)
    pd.DataFrame(per_second).to_csv(outdir / "task1_per_second_psi.csv", index=False)
    pd.concat(per_player_vi, ignore_index=True).to_csv(
        outdir / "task1_vi_timeseries.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))

    # ------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6]})
    colours = {"home": "#1f77b4", "away": "#d62728"}
    p1_len = float(v.second[v.period == 1].max()) / 60.0 + 0.5

    ax = axes[0]
    for side in TEAMS:
        for per in (1, 2):
            sel = (v.period == per) & v.valid
            if not sel.any():
                continue
            off = 0.0 if per == 1 else p1_len
            ax.plot(off + v.second[sel] / 60.0, v.team_vi[sel, TEAMS.index(side)],
                    color=colours[side], lw=1.2, alpha=0.9,
                    label=f"{sides[side]}" if per == 1 else None)
    ax.set_ylabel("team VI")
    ax.set_title("Team Voronoi VI (V(t))", fontsize=10, loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    for side in TEAMS:
        for per in (1, 2):
            sel = (v.period == per)
            if not sel.any():
                continue
            off = 0.0 if per == 1 else p1_len
            ps = psi_store[(side, per)]["psi"]
            m = np.isfinite(ps)
            ax.plot(off + v.second[m] / 60.0, ps[m], color=colours[side], lw=1.0,
                    alpha=0.9, label=f"{sides[side]} ({side})" if per == 1 else None)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("match time (min, halves concatenated)")
    ax.set_ylabel("Psi (nats)")
    ax.set_title("Causal emergence Psi(V = team VI), 60 s sliding window",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "figure_task1_psi_timeseries.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------- optional extension (§8.2)
    # Ψ in the 60 s before each shot, mirroring report ② §4.3
    ev_path = (ROOT / "data/raw" / args.match_id
               / f"{args.match_id}_dynamic_events.csv")
    shot_rows = []
    if ev_path.exists():
        from scipy import stats as _st
        ev = pd.read_csv(ev_path, low_memory=False)
        shots = ev[ev["end_type"] == "shot"].copy()
        side_map = {v.match.teams["home"].team_id: "home",
                    v.match.teams["away"].team_id: "away"}
        shots["side"] = shots["team_id"].map(side_map)
        starts = {p: v.match.period_bounds[p][0] for p in v.match.period_bounds}
        shots = shots[shots["period"].isin([1, 2]) & shots["side"].notna()]
        shots["t0"] = ((shots["frame_end"] - shots["period"].map(starts))
                       / 10.0).round().astype(int)

        other = {"home": "away", "away": "home"}
        rng = np.random.default_rng(0)
        panels = {"attacking": [], "defending": [], "random": []}
        for sh in shots.itertuples():
            per = int(sh.period)
            idx_sel = np.where(v.period == per)[0]
            base = idx_sel[0]
            for role, team in (("attacking", sh.side), ("defending", other[sh.side])):
                ps = psi_store[(team, per)]["psi"]
                idx = base + int(sh.t0)
                if idx - WIN < base:
                    continue
                window = ps[idx - WIN:idx]
                if np.isfinite(window).sum() >= WIN // 2:
                    panels[role].append(window)
            ps = psi_store[(sh.side, per)]["psi"]
            seg = ps[base:base + (v.period == per).sum()]
            if np.isfinite(seg).sum() > WIN:
                start = int(rng.integers(0, seg.size - WIN))
                window = seg[start:start + WIN]
                if np.isfinite(window).sum() >= WIN // 2:
                    panels["random"].append(window)

        for role, mats in panels.items():
            if not mats:
                continue
            mat = np.vstack(mats)
            y = np.nanmean(mat, axis=0)
            x = np.arange(y.size)
            ok = np.isfinite(y)
            if ok.sum() < 5:
                continue
            lr = _st.linregress(x[ok], y[ok])
            shot_rows.append(dict(macro="V=team VI", role=role, n_events=len(mat),
                                  mean_psi=float(np.nanmean(y)),
                                  beta=lr.slope, r2=lr.rvalue ** 2, p=lr.pvalue,
                                  psi_at_t_minus_10s=float(y[-11]) if y.size > 10 else np.nan,
                                  psi_at_shot=float(y[-1])))
        if shot_rows:
            shot_df = pd.DataFrame(shot_rows)
            shot_df.to_csv(outdir / "task1_shot_regressions.csv", index=False)
            print("\nPsi before shots (optional extension §8.2)")
            print(shot_df.round(4).to_string(index=False))

            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for role, col in (("attacking", "#1f77b4"), ("defending", "#e07b39"),
                              ("random", "#999999")):
                if not panels[role]:
                    continue
                mat = np.vstack(panels[role])
                with np.errstate(invalid="ignore"):
                    m = np.nanmean(mat, axis=0)
                    cnt = np.sum(np.isfinite(mat), axis=0)
                    sem = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(cnt, 1))
                xs = np.arange(m.size) - m.size + 1
                ax.plot(xs, m, color=col, lw=1.6, label=role)
                ax.fill_between(xs, m - sem, m + sem, color=col, alpha=0.18, lw=0)
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.set_xlabel("time relative to shot (s)")
            ax.set_ylabel("Psi (nats)")
            ax.set_title("Psi(team VI) in the 60 s before shots", fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(outdir / "figure_task1_psi_before_shot.png", dpi=150)
            plt.close(fig)

    (outdir / "task1_meta.json").write_text(json.dumps({
        "match_id": args.match_id,
        "valid_seconds": int(v.valid.sum()),
        "window": WIN,
        "n_micro_features": 50,
        "hz": args.hz,
    }, indent=2), encoding="utf-8")
    print("\nwrote", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
