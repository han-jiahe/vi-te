"""Task 2 — substitution -> team coordination, with Psi(VI) as the outcome.

Implements 实现方案_VI因果涌现与换人TMLE §5:

  * macro  V(t)  = team VI
  * micro  X(t)  = the 10 outfield players' individual VI
  * Psi_VI(t)    = I(V_t; V_{t+1}) - sum_k I(VI_k(t); V_{t+1})   (60 s rolling)
  * 5-min segments of the second half, A = substitution in the previous
    segment, Y = mean Psi_VI in the segment, W1..W6 as specified
  * discrete Super Learner + logit-fluctuation TMLE, with a GLM baseline and a
    misspecified-model (drop W3) comparison
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_scalar import psi_series
from src.vi_pipeline import TEAMS, ViMatch

WIN = 60
SEG = 300
WINDOW = 300


# --------------------------------------------------------------------- TMLE
def q_library(seed: int) -> dict:
    return {
        "ols": LinearRegression(),
        "lasso": LassoCV(cv=3, max_iter=5000, random_state=seed),
        "rf": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                    random_state=seed),
        "gbr": GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                         learning_rate=0.05, random_state=seed),
    }


def discrete_super_learner(X, y, seed=0, cv=5):
    """Pick the learner with the lowest cross-validated MSE (spec §5.5 step 1)."""
    k = min(cv, max(2, X.shape[0] // 2))
    folds = KFold(n_splits=k, shuffle=True, random_state=seed)
    table = {}
    best_name, best_score, best_pred = None, np.inf, None
    for name, est in q_library(seed).items():
        try:
            pred = cross_val_predict(est, X, y, cv=folds)
            mse = float(mean_squared_error(y, pred))
        except Exception:                                  # noqa: BLE001
            continue
        table[name] = mse
        if mse < best_score:
            best_name, best_score, best_pred = name, mse, pred
    model = q_library(seed)[best_name].fit(X, y)
    return model, best_name, table


def _epsilon_fit(y, offset, h):
    def nll(eps):
        p = np.clip(expit(offset + eps * h), 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(-20, 20), method="bounded")
    return float(res.x)


def tmle(A, W, Y, seed=0, clip=1e-6):
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    n = A.size

    X = np.column_stack([A, W])
    model, best, table = discrete_super_learner(X, Y, seed=seed)
    q = np.clip(model.predict(X), clip, 1 - clip)
    q1 = np.clip(model.predict(np.column_stack([np.ones(n), W])), clip, 1 - clip)
    q0 = np.clip(model.predict(np.column_stack([np.zeros(n), W])), clip, 1 - clip)

    gmodel = LogisticRegression(C=1.0, max_iter=2000).fit(W, A.astype(int))
    g = np.clip(gmodel.predict_proba(W)[:, 1], clip, 1 - clip)
    h = A / g - (1 - A) / (1 - g)

    eps = _epsilon_fit(Y, logit(q), h)
    qs = np.clip(expit(logit(q) + eps * h), clip, 1 - clip)
    qs1 = np.clip(expit(logit(q1) + eps * 1.0 / g), clip, 1 - clip)
    qs0 = np.clip(expit(logit(q0) - eps * 1.0 / (1 - g)), clip, 1 - clip)

    ate = float(np.mean(qs1 - qs0))
    ic = h * (Y - qs) + qs1 - qs0 - ate
    se = float(np.std(ic, ddof=1) / np.sqrt(n))
    return dict(
        ate=ate, se=se, ci_low=ate - 1.96 * se, ci_high=ate + 1.96 * se,
        eps=eps, learner=best, cv_mse=table,
        g_min=float(g.min()), g_max=float(g.max()),
        n=int(n), n_treated=int(A.sum()),
    )


def glm_ate(A, W, Y):
    X = np.column_stack([A, W])
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(A)), X]), Y, rcond=None)
    return float(beta[1])


# --------------------------------------------------------------- main pipeline
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", default="1886347")
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "vi"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--win", type=int, default=WIN,
                    help="rolling window for Psi_VI in seconds (60 = spec)")
    ap.add_argument("--max-gap", default="0",
                    help="interpolate unusable stretches up to this many "
                         "seconds before windowing ('none' = any length). "
                         "0 = the strict contiguous-window behaviour.")
    ap.add_argument("--hz", type=int, default=1, choices=(1, 10),
                    help="VI rate grid: 1 = dt 1 s (pipeline legacy), "
                         "10 = dt 0.1 s at 10 fps downsampled to 1 Hz "
                         "(paper arXiv:2102.11182 spec)")
    args = ap.parse_args()

    max_gap: int | None
    if str(args.max_gap).strip().lower() == "none":
        max_gap = None
    else:
        max_gap = int(args.max_gap)

    # variant files carry a tag so experimental settings never overwrite the
    # baseline outputs and are invisible to task2_pool.py's default glob
    tag = "" if (args.win == WIN and max_gap == 0 and args.hz == 1) else \
        f"_w{args.win}_g{max_gap if max_gap is not None else 'all'}" \
        f"{'_h10' if args.hz == 10 else ''}"

    outdir = Path(args.out_dir) / args.match_id
    outdir.mkdir(parents=True, exist_ok=True)

    v = ViMatch(args.match_id, ROOT / "data" / "raw" / args.match_id, hz=args.hz)
    names = {s: v.match.teams[s].short_name for s in TEAMS}

    # ------------------------------------------------ Psi_VI per team
    psi_vi, macro_vi, valid_vi = {}, {}, {}
    for side in TEAMS:
        macro = v.team_vi[:, TEAMS.index(side)]
        micro = np.full((v.second.size, 10), np.nan)
        for t in np.where(v.valid)[0]:
            ids = v.outfield_order(side, t)
            if ids.size == 10:
                micro[t] = v.player_vi[t, ids]
        valid = (v.valid & np.isfinite(macro) & np.isfinite(micro).all(axis=1))
        psi_vi[side] = psi_series(macro, micro, valid, win=args.win,
                                  max_gap=max_gap)
        macro_vi[side] = macro
        valid_vi[side] = valid

    # ------------------------------------------------ substitution events
    subs = v.substitutions()
    p2_start_idx = int(np.where(v.period == 2)[0][0]) if (v.period == 2).any() else 0
    for e in subs:
        e["side"] = e["team"]

    rows_sub = []
    pre_post = {s: [] for s in TEAMS}
    for e in subs:
        side = e["side"]
        t0 = e["t0"]
        # global index of second t0 of period 2
        idx = p2_start_idx + t0
        pre = slice(max(0, idx - WINDOW), idx)
        post = slice(idx + 1, min(v.second.size, idx + WINDOW + 1))
        p_out, p_in = e["p_out"], e["p_in"]
        pre_vi = v.player_vi[pre, p_out]
        post_vi = v.player_vi[post, p_in]
        pre_mean = float(np.nanmean(pre_vi)) if np.isfinite(pre_vi).any() else np.nan
        post_mean = float(np.nanmean(post_vi)) if np.isfinite(post_vi).any() else np.nan
        psi_pre = psi_vi[side]["psi"][pre]
        psi_post = psi_vi[side]["psi"][post]
        rows_sub.append(dict(
            match_id=args.match_id, team=names[side], side=side, t0=t0,
            minute=round(t0 / 60.0, 2),
            player_out=e["p_out_name"], player_in=e["p_in_name"],
            pre_VI_mean=pre_mean, post_VI_mean=post_mean,
            delta_VI=post_mean - pre_mean,
            pre_VI_sd=float(np.nanstd(pre_vi)) if np.isfinite(pre_vi).any() else np.nan,
            post_VI_sd=float(np.nanstd(post_vi)) if np.isfinite(post_vi).any() else np.nan,
            n_valid_pre=int(np.isfinite(pre_vi).sum()),
            n_valid_post=int(np.isfinite(post_vi).sum()),
            psi_pre_mean=float(np.nanmean(psi_pre)) if np.isfinite(psi_pre).any() else np.nan,
            psi_post_mean=float(np.nanmean(psi_post)) if np.isfinite(psi_post).any() else np.nan,
            n_psi_pre=int(np.isfinite(psi_pre).sum()),
            n_psi_post=int(np.isfinite(psi_post).sum()),
            delta_psi=((float(np.nanmean(psi_post)) - float(np.nanmean(psi_pre)))
                       if np.isfinite(psi_post).any() and np.isfinite(psi_pre).any()
                       else np.nan),
        ))
        pre_post[side].append(dict(
            event=e["t0"], rel=np.arange(-WINDOW, WINDOW + 1),
            psi=np.concatenate([np.full(WINDOW, np.nan), [np.nan],
                                np.full(WINDOW, np.nan)]),
            vi_out=np.full(WINDOW, np.nan), vi_in=np.full(WINDOW, np.nan),
        ))
        tr = pre_post[side][-1]
        pre_idx = np.arange(idx - WINDOW, idx)
        post_idx = np.arange(idx + 1, idx + WINDOW + 1)
        ok_pre = (pre_idx >= 0) & (pre_idx < v.second.size)
        ok_post = (post_idx < v.second.size)
        tr["psi"][np.where(ok_pre)[0]] = psi_vi[side]["psi"][pre_idx[ok_pre]]
        tr["psi"][WINDOW + 1 + np.where(ok_post)[0]] = psi_vi[side]["psi"][post_idx[ok_post]]
        tr["vi_out"][np.where(ok_pre)[0]] = v.player_vi[pre_idx[ok_pre], p_out]
        tr["vi_in"][np.where(ok_post)[0]] = v.player_vi[post_idx[ok_post], p_in]

    sub_df = pd.DataFrame(rows_sub)
    sub_df.to_csv(outdir / f"task2_substitutions{tag}.csv", index=False)

    # ------------------------------------------------ 5-min segments
    # The second half holds 2 066 *valid* seconds (spec §7, anchor 1).  Blocks
    # are 300 consecutive valid seconds, so a half yields 6 complete blocks and
    # the remainder is dropped -- exactly the spec's "≈6 段" anchor.
    valid2_idx = np.where((v.period == 2) & v.valid)[0]
    n_seg = valid2_idx.size // SEG
    sec_valid2 = v.second[valid2_idx]

    def seg_of(t0: float) -> int:
        return int(np.searchsorted(sec_valid2, t0)) // SEG

    ev_path = (ROOT / "data/raw" / args.match_id
               / f"{args.match_id}_dynamic_events.csv")
    if ev_path.exists():
        ev = pd.read_csv(ev_path, low_memory=False)
        ev = ev[ev["team_score"].notna()]
    else:
        print(f"  WARNING {ev_path.name} missing: W4 (score) falls back to 0")
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

    rows_seg = []
    for side in TEAMS:
        other = "away" if side == "home" else "home"
        opp_poss = float((poss_all == other).mean()) if poss_all.size else np.nan
        for k in range(n_seg):
            idxs = valid2_idx[k * SEG:(k + 1) * SEG]
            psi_vals = psi_vi[side]["psi"][idxs]
            y = float(np.nanmean(psi_vals)) if np.isfinite(psi_vals).any() else np.nan
            a_prev = any(e["side"] == side and seg_of(e["t0"]) == k - 1 for e in subs)
            a_now = any(e["side"] == side and seg_of(e["t0"]) == k for e in subs)
            n_used = sum(1 for e in subs if e["side"] == side and seg_of(e["t0"]) < k)
            g = goals_at(p2_frame0 + int(sec_valid2[k * SEG]) * 10)
            rows_seg.append(dict(
                match_id=args.match_id, team=names[side], side=side, k=k,
                A=int(a_prev), Y_nats=y,
                W1=k, W2=n_used, W3=int(a_now),
                W4=int(g[side] - g[other]),
                W5=int(side == "home"), W6=opp_poss,
                n_psi=int(np.isfinite(psi_vals).sum()),
                t_start=int(sec_valid2[k * SEG]),
            ))
    seg = pd.DataFrame(rows_seg)
    seg.to_csv(outdir / f"task2_segments{tag}.csv", index=False)
    print(f"second-half valid seconds = {valid2_idx.size}, complete segments = {n_seg}")
    print(seg.round(4).to_string(index=False))

    # ------------------------------------------------ TMLE
    d = seg.dropna(subset=["Y_nats", "W6"]).reset_index(drop=True)
    y = d["Y_nats"].to_numpy()
    y_min, y_max = float(y.min()), float(y.max())
    y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
    d["Y_norm"] = y_norm
    d.to_csv(outdir / f"task2_segments{tag}.csv", index=False)

    A = d["A"].to_numpy(float)
    Wcols = ["W1", "W2", "W3", "W4", "W5", "W6"]
    W = d[Wcols].to_numpy(float)
    full = tmle(A, W, y_norm, seed=args.seed)
    miss = tmle(A, W[:, [0, 1, 3, 4, 5]], y_norm, seed=args.seed)   # drop W3
    glm_full = glm_ate(A, W, y_norm)
    glm_miss = glm_ate(A, W[:, [0, 1, 3, 4, 5]], y_norm)

    scale = y_max - y_min

    # ------------------------------------------------- diagnostics (not locked)
    # permutation test on the treatment label + leave-one-segment-out stability;
    # with N this small the CI alone is uninformative, so both are reported.
    rng = np.random.default_rng(args.seed)
    n_perm = int(getattr(args, "n_perm", 200))
    perm = []
    for _ in range(n_perm):
        ap = rng.permutation(A)
        try:
            perm.append(tmle(ap, W, y_norm, seed=args.seed)["ate"])
        except Exception:                                  # noqa: BLE001
            continue
    perm = np.asarray(perm)
    p_perm = float((np.abs(perm) >= abs(full["ate"])).mean()) if perm.size else np.nan

    loo = []
    for i in range(len(A)):
        keep = np.ones(len(A), dtype=bool)
        keep[i] = False
        try:
            loo.append(tmle(A[keep], W[keep], y_norm[keep], seed=args.seed)["ate"])
        except Exception:                                  # noqa: BLE001
            continue
    loo = np.asarray(loo)

    summary = {
        "match_id": args.match_id,
        "N_segments": int(len(d)),
        "n_treated": int(A.sum()),
        "Y_range_nats": [y_min, y_max],
        "Y_scale": scale,
        "ate_norm": full["ate"],
        "ate_nats": full["ate"] * scale,
        "ci95_norm": [full["ci_low"], full["ci_high"]],
        "ci95_nats": [full["ci_low"] * scale, full["ci_high"] * scale],
        "se_norm": full["se"],
        "epsilon": full["eps"],
        "super_learner": full["learner"],
        "cv_mse": full["cv_mse"],
        "g_range": [full["g_min"], full["g_max"]],
        "ate_glm_norm": glm_full,
        "ate_glm_nats": glm_full * scale,
        "bias_pct_tmle_vs_glm": ((full["ate"] - glm_full) / abs(glm_full) * 100.0
                                 if glm_full != 0 else np.nan),
        "permutation": {
            "n_perm": int(perm.size),
            "p_two_sided": p_perm,
            "null_ate_mean": float(perm.mean()) if perm.size else np.nan,
            "null_ate_sd": float(perm.std()) if perm.size else np.nan,
        },
        "leave_one_out": {
            "ate_min": float(loo.min()) if loo.size else np.nan,
            "ate_max": float(loo.max()) if loo.size else np.nan,
            "ate_sd": float(loo.std()) if loo.size else np.nan,
        },
        "misspecified": {
            "ate_norm": miss["ate"],
            "ate_nats": miss["ate"] * scale,
            "ci95_norm": [miss["ci_low"], miss["ci_high"]],
            "super_learner": miss["learner"],
            "ate_glm_norm": glm_miss,
            "ate_glm_nats": glm_miss * scale,
        },
        "n_events": int(len(sub_df)),
    }
    (outdir / f"task2_tmle_summary{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nTMLE summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"home": "#1f77b4", "away": "#d62728"}

    # (a) Psi_VI across the second half with substitution markers
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.4]})
    ax = axes[0]
    for side in TEAMS:
        m = v.period == 2
        ax.plot(v.second[m] / 60.0, macro_vi[side][m], color=colours[side],
                lw=1.1, label=names[side])
    for e in subs:
        ax.axvline(e["t0"] / 60.0, color=colours[e["side"]], lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("team VI")
    ax.set_title("Team VI in the second half (dashed lines = substitutions)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    for side in TEAMS:
        m = (v.period == 2) & np.isfinite(psi_vi[side]["psi"])
        ax.plot(v.second[m] / 60.0, psi_vi[side]["psi"][m], color=colours[side],
                lw=1.0, label=f"{names[side]} Psi_VI")
    for e in subs:
        ax.axvline(e["t0"] / 60.0, color=colours[e["side"]], lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("second-half time (min)")
    ax.set_ylabel("Psi_VI (nats)")
    ax.set_title("Causal emergence Psi_VI (macro = team VI, micro = 10 players' VI)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"figure_task2_psi_vi_secondhalf{tag}.png", dpi=150)
    plt.close(fig)

    # (b) individual VI of the two players and team Psi_VI, aligned to t0
    x_rel = np.arange(-WINDOW, WINDOW + 1) / 60.0

    def _nanagg(mat):
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(mat, axis=0)
            cnt = np.sum(np.isfinite(mat), axis=0)
            s = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(cnt, 1))
        return m, np.nan_to_num(s, nan=0.0), cnt

    for side in TEAMS:
        events = [r for r in pre_post[side] if np.isfinite(r["psi"]).any()]
        if not events:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

        ax = axes[0]
        for rec in events:
            ax.plot(x_rel[:WINDOW], rec["vi_out"], color="#777777", lw=0.7, alpha=0.45)
            ax.plot(x_rel[WINDOW + 1:], rec["vi_in"], color="#777777", lw=0.7, alpha=0.45)
        mo, so, _ = _nanagg(np.vstack([r["vi_out"] for r in events]))
        mi, si, _ = _nanagg(np.vstack([r["vi_in"] for r in events]))
        ax.plot(x_rel[:WINDOW], mo, color="#d62728", lw=2.0, label="replaced player (pre)")
        ax.fill_between(x_rel[:WINDOW], mo - so, mo + so, color="#d62728", alpha=0.18, lw=0)
        ax.plot(x_rel[WINDOW + 1:], mi, color="#2ca02c", lw=2.0, label="substitute (post)")
        ax.fill_between(x_rel[WINDOW + 1:], mi - si, mi + si, color="#2ca02c",
                        alpha=0.18, lw=0)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("time relative to substitution (min)")
        ax.set_ylabel("individual VI")
        ax.set_title("VI of the replaced / introduced player", fontsize=9)
        ax.legend(fontsize=7)

        ax = axes[1]
        mat = np.vstack([r["psi"] for r in events])
        for row in mat:
            ax.plot(x_rel, row, color="#999999", lw=0.6, alpha=0.5)
        m, sem, _ = _nanagg(mat)
        ax.plot(x_rel, m, color=colours[side], lw=2.0,
                label=f"mean over {len(events)} events")
        ax.fill_between(x_rel, m - sem, m + sem, color=colours[side], alpha=0.2, lw=0)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("time relative to substitution (min)")
        ax.set_ylabel("Psi_VI (nats)")
        ax.set_title("team Psi_VI around the substitution", fontsize=9)
        ax.legend(fontsize=7)

        fig.suptitle(f"{names[side]} — +/-5 min around each substitution", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / f"figure_task2_psi_prepost_{side}{tag}.png", dpi=150)
        plt.close(fig)

    # (b2) spec §5.6 `figure_task2_psi_prepost.png`: every event, both teams,
    #      aligned to t0, so the file name and content match the spec table.
    all_events = [r for s in TEAMS for r in pre_post[s] if np.isfinite(r["psi"]).any()]
    if all_events:
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        mat = np.vstack([r["psi"] for r in all_events])
        for row in mat:
            ax.plot(x_rel, row, color="#bbbbbb", lw=0.6, alpha=0.5)
        m, sem, _ = _nanagg(mat)
        ax.plot(x_rel, m, color="#1f77b4", lw=2.2,
                label=f"mean over {len(all_events)} events (both teams)")
        ax.fill_between(x_rel, m - sem, m + sem, color="#1f77b4", alpha=0.2, lw=0)
        ax.axvline(0, color="k", lw=0.9, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("time relative to substitution t0 (min)")
        ax.set_ylabel("Psi_VI (nats)")
        ax.set_title("Task 2: Psi_VI in the +/-5 min around every substitution",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"figure_task2_psi_prepost{tag}.png", dpi=150)
        plt.close(fig)

    # (c) outcome by treatment
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for a, col, lab in ((0, "#999999", "A = 0 (no substitution)"),
                        (1, "#1f77b4", "A = 1 (substitution)")):
        vals = d.loc[d["A"] == a, "Y_nats"].to_numpy()
        if vals.size:
            ax.scatter(np.full(vals.size, a) + np.random.default_rng(0).normal(
                0, 0.04, vals.size), vals, s=45, color=col, alpha=0.85)
            ax.hlines(vals.mean(), a - 0.2, a + 0.2, color=col, lw=2.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A = 0", "A = 1"])
    ax.set_ylabel("segment mean Psi_VI (nats)")
    ax.set_title(f"Outcome by treatment  (ATE={summary['ate_nats']:+.3f} nats)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / f"figure_task2_outcome_by_treatment{tag}.png", dpi=150)
    plt.close(fig)

    print("\nwrote", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
