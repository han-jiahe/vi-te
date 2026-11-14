"""Pool the 5-min segments of several matches and re-run the Task 2 TMLE.

Implements 实现方案_VI因果涌现与换人TMLE §6: every match contributes its
segments, ``match_id`` is added as an extra confounder, and the ATE is
re-estimated.  Also reports the per-match ATE so the pooled number can be read
against its between-match spread.

Run ``scripts/task1_vi_ce.py`` and ``scripts/task2_tmle.py`` for each match
first (or just ``scripts/run_all.py``).
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

from scripts.task2_tmle import glm_ate, tmle      # noqa: E402

WCOLS = ["W1", "W2", "W3", "W4", "W5", "W6"]


def fit(A, W, y, normalise="global", groups=None):
    """TMLE + GLM on min-max normalised Y; returns a small dict.

    ``normalise='global'``   one min-max over all segments (spec §5.5 as written).
    ``normalise='per_match'`` min-max inside each match, so a match with a wide
                             Y range cannot dominate the pooled estimate.
    """
    y = np.asarray(y, dtype=float)
    if normalise == "per_match" and groups is not None:
        yn = np.empty_like(y)
        for g in np.unique(groups):
            m = groups == g
            lo, hi = float(y[m].min()), float(y[m].max())
            rng = hi - lo if hi > lo else 1.0
            yn[m] = (y[m] - lo) / rng
        scale = float(np.mean([y[groups == g].max() - y[groups == g].min()
                               for g in np.unique(groups)]))
        ymin, ymax = float(y.min()), float(y.max())
    else:
        ymin, ymax = float(y.min()), float(y.max())
        scale = ymax - ymin if ymax > ymin else 1.0
        yn = (y - ymin) / scale if ymax > ymin else np.zeros_like(y)
    res = tmle(A, W, yn)
    g = glm_ate(A, W, yn)
    return dict(ate_norm=res["ate"], ate_nats=res["ate"] * scale,
                ci_norm=[res["ci_low"], res["ci_high"]],
                ci_nats=[res["ci_low"] * scale, res["ci_high"] * scale],
                se_norm=res["se"], eps=res["eps"], learner=res["learner"],
                g_range=[res["g_min"], res["g_max"]],
                ate_glm_norm=g, ate_glm_nats=g * scale,
                n=int(len(A)), n_treated=int(A.sum()),
                ymin=ymin, ymax=ymax, scale=scale,
                normalise=normalise)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / "output" / "vi"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalise", choices=["global", "per_match"],
                    default="global",
                    help="global (spec §5.5 as written) or per_match, so that a "
                         "match with a wide Y range cannot dominate the pool")
    ap.add_argument("--pattern", default="*/task2_segments.csv",
                    help="glob of the segment tables to pool (defaults to the "
                         "baseline; e.g. '*/task2_segments_*_h10.csv' pools a "
                         "10 Hz variant)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files when pooling a variant "
                         "(e.g. '_h10')")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="permutations for the pooled test")
    args = ap.parse_args()
    root = Path(args.out_root)
    norm = args.normalise

    frames = [pd.read_csv(f) for f in sorted(root.glob(args.pattern))]
    if not frames:
        raise SystemExit(f"no files match {args.pattern}")
    if not args.tag and args.pattern != "*/task2_segments.csv":
        raise SystemExit("non-default --pattern needs an explicit --tag "
                         "so the baseline outputs are not overwritten")
    seg = pd.concat(frames, ignore_index=True)
    seg = seg.dropna(subset=["Y_nats", "W6"]).reset_index(drop=True)
    seg["match_id"] = seg["match_id"].astype(str)

    # ---- per-match ATE
    per_match = []
    for mid, g in seg.groupby("match_id"):
        if len(g) < 6 or g["A"].nunique() < 2:
            per_match.append(dict(match_id=mid, n=len(g), ate_nats=np.nan,
                                  ci_low=np.nan, ci_high=np.nan,
                                  note="too few segments / no variation"))
            continue
        r = fit(g["A"].to_numpy(float), g[WCOLS].to_numpy(float),
                g["Y_nats"].to_numpy(float))
        per_match.append(dict(match_id=mid, n=r["n"], ate_nats=r["ate_nats"],
                              ate_norm=r["ate_norm"],
                              ci_low=r["ci_nats"][0], ci_high=r["ci_nats"][1],
                              ate_glm_nats=r["ate_glm_nats"], note=""))
    pm = pd.DataFrame(per_match)

    # ---- pooled, match_id as an extra confounder (spec §6)
    seg["W7"] = pd.factorize(seg["match_id"])[0]
    A = seg["A"].to_numpy(float)
    W = seg[["W1", "W2", "W3", "W4", "W5", "W6", "W7"]].to_numpy(float)
    Y = seg["Y_nats"].to_numpy(float)
    pooled = fit(A, W, Y, normalise=norm, groups=seg["match_id"].to_numpy())
    # robustness: the same pool with per-match min-max, so a match with a wide
    # Y range cannot dominate (see 报告 §3.5.3)
    other = fit(A, W, Y, normalise="per_match", groups=seg["match_id"].to_numpy())

    # ---- permutation test on the pooled estimate
    rng = np.random.default_rng(args.seed)
    if norm == "per_match":
        yn = np.empty_like(Y)
        for g in np.unique(seg["match_id"].to_numpy()):
            m = seg["match_id"].to_numpy() == g
            lo, hi = float(Y[m].min()), float(Y[m].max())
            yn[m] = (Y[m] - lo) / (hi - lo if hi > lo else 1.0)
    else:
        yn = (Y - pooled["ymin"]) / pooled["scale"]
    perm = []
    for _ in range(args.n_perm):
        try:
            perm.append(tmle(rng.permutation(A), W, yn)["ate"])
        except Exception:                                      # noqa: BLE001
            continue
    perm = np.asarray(perm)
    p_perm = float((np.abs(perm) >= abs(pooled["ate_norm"])).mean()) if perm.size else np.nan

    summary = dict(
        variant=args.tag or "baseline",
        pattern=args.pattern,
        n_matches=int(seg["match_id"].nunique()),
        N_segments=int(len(seg)),
        n_treated=int(A.sum()),
        treat_fraction=float(A.mean()),
        normalise=norm,
        Y_range_nats=[pooled["ymin"], pooled["ymax"]],
        Y_scale=pooled["scale"],
        ate_norm=pooled["ate_norm"],
        ate_nats=pooled["ate_nats"],
        ci95_norm=pooled["ci_norm"],
        ci95_nats=pooled["ci_nats"],
        se_norm=pooled["se_norm"],
        epsilon=pooled["eps"],
        super_learner=pooled["learner"],
        g_range=pooled["g_range"],
        ate_glm_norm=pooled["ate_glm_norm"],
        ate_glm_nats=pooled["ate_glm_nats"],
        perm_p_two_sided=p_perm,
        sensitivity_per_match_normalisation=dict(
            ate_norm=other["ate_norm"], ate_nats=other["ate_nats"],
            ci95_norm=other["ci_norm"], ci95_nats=other["ci_nats"],
            ate_glm_nats=other["ate_glm_nats"], Y_scale=other["scale"],
        ),
        per_match=pm.to_dict(orient="records"),
    )
    out = root / f"task2_pool_summary{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pm.to_csv(root / f"task2_pool_per_match{args.tag}.csv", index=False)

    pd.set_option("display.width", 200)
    print(pm.round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_match"},
                     indent=2, ensure_ascii=False))
    print("wrote", out)

    # ------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                          # noqa: BLE001
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    rng2 = np.random.default_rng(0)
    mids = list(seg["match_id"].unique())
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(mids), 1)))
    for k, mid in enumerate(mids):
        g = seg[seg.match_id == mid]
        for a, marker in ((0, "o"), (1, "s")):
            v = g.loc[g.A == a, "Y_nats"].to_numpy()
            if v.size:
                ax.scatter(np.full(v.size, a) + rng2.normal(0, 0.05, v.size), v,
                           color=colours[k], marker=marker, s=38, alpha=0.75,
                           label=f"match {mid}" if a == 0 else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A = 0", "A = 1"])
    ax.set_ylabel("segment mean Psi_VI (nats)")
    ax.set_title(f"Segments by treatment (N={len(seg)}, {len(mids)} matches)", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    rows = pm.dropna(subset=["ate_nats"])
    ypos = np.arange(len(rows))
    ax.errorbar(rows.ate_nats, ypos,
                xerr=[rows.ate_nats - rows.ci_low, rows.ci_high - rows.ate_nats],
                fmt="o", color="#1f77b4", ms=5, lw=1.1,
                label="per-match TMLE")
    ax.axvline(pooled["ate_nats"], color="#d62728", lw=1.6,
               label=f"pooled {pooled['ate_nats']:+.3f}")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_yticks(ypos)
    ax.set_yticklabels(rows.match_id, fontsize=7)
    ax.set_xlabel("ATE on Psi_VI (nats), 95% CI")
    ax.set_title("Per-match and pooled effect", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / f"figure_task2_pool{args.tag}.png", dpi=150)
    print("wrote", root / f"figure_task2_pool{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
