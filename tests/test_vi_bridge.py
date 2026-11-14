"""vi_bridge 合成数据测试（不依赖 tracking 数据）。

覆盖：转移掩码、冻结变换 + 代理协议、NaN 段处理、耦合检出、确定性。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mte.significance import surrogate_mte  # noqa: E402
from vi_bridge.vi_te import (  # noqa: E402
    _make_closure, disc_fit, pair_mask, run_pair, std_fit)


def _coupled_pair(T=2000, seed=7, tau=0.35):
    """合成对：B 由自身历史 + A 驱动（AR(1)），A 独立 AR(1)。"""
    rng = np.random.default_rng(seed)
    a = np.zeros(T)
    b = np.zeros(T)
    for t in range(1, T):
        a[t] = 0.7 * a[t - 1] + rng.normal()
        b[t] = 0.5 * b[t - 1] + tau * a[t - 1] + rng.normal()
    return a, b


# -------------------------------------------------------------------- masking
def test_pair_mask_basic():
    A = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    B = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    m = pair_mask(A, B)          # t=0..3
    # t=0: B1 NaN -> 0; t=1: B1 NaN -> 0; t=2: A2 NaN -> 0; t=3: ok -> 1
    assert m.tolist() == [False, False, False, True]


def test_pair_mask_never_crosses_gap():
    rng = np.random.default_rng(0)
    A = rng.normal(size=100)
    B = rng.normal(size=100)
    B[40] = np.nan               # 缺口：B[39]->B[40]、B[40]->B[41] 都应被掩
    m = pair_mask(A, B)
    assert not m[39] and not m[40]
    assert m.sum() == 97


# ------------------------------------------------------- frozen transform + TE
def test_closure_frozen_and_deterministic():
    A, B = _coupled_pair()
    mask = pair_mask(A, B)
    fa = std_fit(A)
    fb = std_fit(B)
    fn = _make_closure("gaussian", A, B, mask, fa, fb, None, None, None)
    raw1 = fn(A, B, [])
    raw2 = fn(A, B, [])
    assert raw1 == raw2 and raw1 > 0
    s1 = surrogate_mte(fn, A, B, [], 25, 30, seed=42)
    s2 = surrogate_mte(fn, A, B, [], 25, 30, seed=42)
    np.testing.assert_array_equal(s1, s2)          # 确定性（固定 seed）


def test_run_pair_coupled_vs_null():
    A, B = _coupled_pair(tau=0.35)
    mask = pair_mask(A, B)
    r_c = run_pair(A, B, mask, std_fit(A), std_fit(B), None, None,
                   "gaussian", None, 25, 60, seed=42)
    A0, B0 = _coupled_pair(tau=0.0)                # 无耦合 → TE≈0
    mask0 = pair_mask(A0, B0)
    r_n = run_pair(A0, B0, mask0, std_fit(A0), std_fit(B0), None, None,
                   "gaussian", None, 25, 60, seed=42)
    assert r_c["raw"] > r_n["raw"]                 # 方向检出（L2a 能力）
    assert r_c["p"] <= 1.0 / 61                    # 耦合存在 → 拒绝零
    assert abs(r_c["eff"] - (r_c["raw"] - r_c["surr_mean"])) < 1e-12  # 恒等式
    assert r_c["surr_ratio"] is not None


# ------------------------------------------------------------------ NaN 处理
def test_nan_gap_surrogate_robust():
    """源通道带 NaN 段：代理评估不崩、真实 n_trans == mask.sum()。"""
    A, B = _coupled_pair(T=600)
    A[300:320] = np.nan
    mask = pair_mask(A, B)
    fa = std_fit(A)
    fb = std_fit(B)
    r = run_pair(A, B, mask, fa, fb, None, None, "gaussian", None, 25, 40,
                 seed=42)
    assert r["n_trans"] == int(mask.sum()) == 579
    assert np.isfinite(r["raw"]) and np.isfinite(r["eff"])


def test_disc_fit_nan_safe():
    ch = np.concatenate([np.linspace(0, 1, 50),
                         np.full(30, np.nan), np.linspace(2, 3, 50)])
    fit = disc_fit(ch, 3, seed=42)
    assert fit is not None
    km, mu, std = fit
    fin = ch[np.isfinite(ch)]
    labels = km.predict(((fin - mu) / (std + 1e-9))[:, None])
    assert np.unique(labels).size == 3
    assert disc_fit(np.full(20, np.nan), 3, seed=42) is None   # 全 NaN


def test_run_pair_discrete_end_to_end():
    A, B = _coupled_pair(T=1000, tau=0.4)
    mask = pair_mask(A, B)
    r = run_pair(A, B, mask, std_fit(A), std_fit(B), disc_fit(A, 3, 42),
                 disc_fit(B, 3, 42), "discrete", None, 25, 40, seed=42)
    assert np.isfinite(r["raw"]) and r["p"] <= 1.0 / 41
