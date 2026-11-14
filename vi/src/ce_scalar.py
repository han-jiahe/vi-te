"""Causal emergence with a **scalar** macro feature.

The specification (实现方案 §3) locks the Gaussian estimator to the closed form
for two scalar variables:

    I(A; B) = -1/2 ln(1 - rho^2(A, B))

with ``rho`` the Pearson correlation over the estimation window and ``rho^2``
clipped at ``1 - 1e-9``.  For 1x1 variables this is algebraically identical to
the general Gaussian formula ``1/2 ln(det S_A det S_B / det S_AB)`` used in
``mi.py`` -- the identity is asserted in ``scripts/selftest_vi.py``.

    Psi(V) = I(V_t ; V_{t+1}) - sum_k I(X_k(t) ; V_{t+1})
"""

from __future__ import annotations

import numpy as np

RHO2_MAX = 1.0 - 1e-9


def _cum(a: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1,) + a.shape[1:], dtype=float), np.cumsum(a, axis=0)])


def _mi_from_sums(w, sx, sy, sxx, syy, sxy):
    """MI from windowed sums; returns (n_windows, n_cols)."""
    num = w * sxy - sx * sy
    var_x = w * sxx - sx * sx
    var_y = w * syy - sy * sy
    den = np.sqrt(np.clip(var_x, 0.0, None) * np.clip(var_y, 0.0, None))
    rho = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    rho2 = np.clip(rho * rho, 0.0, RHO2_MAX)
    return -0.5 * np.log1p(-rho2)


def windowed_mi(x: np.ndarray, y: np.ndarray, win: int) -> np.ndarray:
    """Sliding-window MI between each column of ``x`` (n,k) and ``y`` (n,).

    One value per window end; window ``i`` covers samples ``[i, i+win)``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    w = float(win)
    starts = np.arange(0, n - win + 1)
    ends = starts + win
    cx, cy = _cum(x), _cum(y)
    cxx, cyy = _cum(x * x), _cum(y * y)
    cxy = _cum(x * y)
    sx = cx[ends] - cx[starts]
    sy = cy[ends] - cy[starts]
    sxx = cxx[ends] - cxx[starts]
    syy = cyy[ends] - cyy[starts]
    sxy = cxy[ends] - cxy[starts]
    return _mi_from_sums(w, sx, sy, sxx, syy, sxy)


def mi_scalar(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    da = a - a.mean()
    db = b - b.mean()
    den = np.sqrt((da * da).sum() * (db * db).sum())
    rho = float((da * db).sum() / den) if den > 0 else 0.0
    return float(-0.5 * np.log1p(-min(rho * rho, RHO2_MAX)))


def _runs(mask: np.ndarray):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    return np.split(idx, breaks + 1)


def _fill_gaps(a: np.ndarray, valid: np.ndarray,
               max_gap: int | None) -> np.ndarray:
    """Linearly interpolate interior gaps of unusable samples.

    A "gap" is a run of samples where ``valid`` is False (or the value is
    NaN).  Only interior gaps (both neighbours usable) of length ``<= max_gap``
    are filled; ``max_gap=None`` fills interior gaps of any length.  This is
    the "interpolate the dead-ball segments" fix from 报告 §4 item 2 -- it
    changes the definition of the resulting Psi, so callers must opt in.
    """
    out = np.asarray(a, dtype=float).copy()
    good = np.asarray(valid, dtype=bool) & np.isfinite(out)
    if not good.any():
        return out
    bad_idx = np.where(~good)[0]
    if bad_idx.size == 0:
        return out
    runs = np.split(bad_idx, np.flatnonzero(np.diff(bad_idx) != 1) + 1)
    for r in runs:
        if r.size == 0 or r[0] == 0 or r[-1] == out.size - 1:
            continue                      # leading/trailing: no left/right anchor
        if max_gap is not None and r.size > max_gap:
            continue
        lo, hi = r[0] - 1, r[-1] + 1
        if good[lo] and good[hi]:
            out[r] = np.interp(r.astype(float), [float(lo), float(hi)],
                               [out[lo], out[hi]])
    return out


def psi_series(macro: np.ndarray, sources: np.ndarray, valid: np.ndarray,
               win: int | None = None, max_gap: int | None = 0) -> dict:
    """Causal emergence for a scalar macro feature.

    Parameters
    ----------
    macro : (n,) macro feature V at each second
    sources : (n, k) micro features X at each second
    valid : (n,) bool, seconds that may enter the estimate
    win : window length in samples; ``None`` -> one estimate per contiguous run
    max_gap : interior gaps of unusable seconds up to this many samples are
        linearly interpolated before the windows are built (``None`` = any
        length).  Default 0 = the strict contiguous-window behaviour, which
        leaves the Psi definition unchanged.

    Returns
    -------
    dict with ``psi``, ``i_macro``, ``i_micro_sum`` (all length n, NaN where
    undefined) and the same three numbers averaged over defined entries.
    """
    macro = np.asarray(macro, dtype=float)
    sources = np.atleast_2d(np.asarray(sources, dtype=float).T).T
    n = macro.size

    if max_gap is None or max_gap > 0:
        macro = _fill_gaps(macro, valid, max_gap)
        sources = np.column_stack(
            [_fill_gaps(sources[:, k], valid, max_gap)
             for k in range(sources.shape[1])])
        ok = np.isfinite(macro) & np.isfinite(sources).all(axis=1)
    else:
        ok = np.asarray(valid, dtype=bool)

    pair_ok = np.zeros(n, dtype=bool)
    pair_ok[:-1] = ok[:-1] & ok[1:]
    pair_ok[:-1] &= np.isfinite(macro[:-1]) & np.isfinite(macro[1:])
    pair_ok[:-1] &= np.isfinite(sources[:-1]).all(axis=1)

    psi = np.full(n, np.nan)
    i_macro_out = np.full(n, np.nan)
    i_sum_out = np.full(n, np.nan)

    for run in _runs(pair_ok):
        y = macro[run + 1]
        v = macro[run][:, None]
        x = sources[run]
        if win is None or run.size < win:
            i_m = np.array([mi_scalar(v[:, 0], y)])
            i_s = np.array([[mi_scalar(x[:, k], y) for k in range(x.shape[1])]]).sum(axis=1)
            target = run[-1:]
        else:
            i_m = windowed_mi(v, y, win)[:, 0]
            acc = np.zeros_like(i_m)
            for k in range(x.shape[1]):
                acc += windowed_mi(x[:, k:k + 1], y, win)[:, 0]
            i_s = acc
            target = run[win - 1:]
        if target.size != i_m.size:
            continue
        psi[target] = i_m - i_s
        i_macro_out[target] = i_m
        i_sum_out[target] = i_s

    def _mean(a):
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    return {
        "psi": psi,
        "i_macro": i_macro_out,
        "i_micro_sum": i_sum_out,
        "psi_mean": _mean(psi),
        "i_macro_mean": _mean(i_macro_out),
        "i_micro_sum_mean": _mean(i_sum_out),
        "n_windows": int(np.isfinite(psi).sum()),
    }
