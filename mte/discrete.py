"""
Discrete conditional plug-in MTE (PROTOCOL.md sec. 7) - small systems only.

Each channel is K-means quantised into symbols; the conditional TE is then
the plug-in histogram estimate

    MTE = sum_{y', y, x, z} p(y', y, x, z) log_b[ p(y' | y, x, z) / p(y' | y, z) ]

computed as the sample mean of cell log-ratios (identical to the cell sum,
same convention as markov-te's te/binning.py).  The joint histogram grows
like K^(players + 1), so this estimator is restricted to 3-5 player
subsystems; run_player_mte skips pairs whose cell count exceeds
n / config.min_samples_per_cell and reports them in the skipped grid.

The K-means fits are FROZEN across the surrogate protocol: fit once on the
real channels, only re-assign labels to shuffled surrogates (quantize with
the returned fitted objects).
"""
from typing import List, Sequence

import numpy as np
from sklearn.cluster import KMeans


def quantize_channels(channels: Sequence[np.ndarray], n_clusters: int,
                      seed: int = 42):
    """K-means quantise each channel into n_clusters symbols.

    Returns (symbols, fits): symbols[i] is an int array of the same length
    as channels[i]; fits[i] = (km, mu, std) lets callers re-assign labels to
    SHUFFLED versions of channel i with the frozen fit (never refit on
    surrogate data)."""
    symbols, fits = [], []
    for ch in channels:
        ch = np.asarray(ch, dtype=float)
        if ch.ndim == 1:
            ch = ch[:, None]
        mu, std = ch.mean(axis=0), ch.std(axis=0)
        chs = (ch - mu) / (std + 1e-9)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        symbols.append(km.fit_predict(chs))
        fits.append((km, mu, std))
    return symbols, fits


def quantize_with(fit, ch: np.ndarray) -> np.ndarray:
    """Re-assign labels with a frozen fit: fit = (km, mu, std)."""
    km, mu, std = fit
    ch = np.asarray(ch, dtype=float)
    if ch.ndim == 1:
        ch = ch[:, None]
    return km.predict((ch - mu) / (std + 1e-9))


def conditional_discrete_mte(Y: np.ndarray, X: np.ndarray, Z: np.ndarray,
                             base: float = 2.0) -> float:
    """Plug-in MTE on integer symbol arrays.

    Y : (n,) target symbols y' ; X : (n,) source-history symbols x ;
    Z : (n, m) conditioning symbols (m >= 1; column 0 is the target history
    y under the build_triples convention).  Both p(y'|y,x,z) and p(y'|y,z)
    condition on ALL columns of Z.
    """
    Y = np.asarray(Y).ravel()
    X = np.asarray(X).ravel()
    Z = np.atleast_2d(np.asarray(Z))
    if Z.shape[1] == 0:
        raise ValueError("discrete conditional MTE needs m >= 1 conditioning "
                         "symbols (plain pairwise TE: pass target history as Z)")
    n = len(Y)
    assert n == len(X) == len(Z) > 0

    def code_of(*cols):
        _, inv = np.unique(np.stack(cols, axis=1), axis=0, return_inverse=True)
        return inv

    c_yxz = code_of(Y, X, *[Z[:, j] for j in range(Z.shape[1])])
    c_xz = code_of(X, *[Z[:, j] for j in range(Z.shape[1])])
    c_yz = code_of(Y, *[Z[:, j] for j in range(Z.shape[1])])
    c_z = code_of(*[Z[:, j] for j in range(Z.shape[1])])

    p_yxz = np.bincount(c_yxz, minlength=c_yxz.max() + 1) / n
    p_xz = np.bincount(c_xz, minlength=c_xz.max() + 1) / n
    p_yz = np.bincount(c_yz, minlength=c_yz.max() + 1) / n
    p_z = np.bincount(c_z, minlength=c_z.max() + 1) / n

    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.log((p_yxz[c_yxz] / p_xz[c_xz]) / (p_yz[c_yz] / p_z[c_z])) / np.log(base)
    logr = np.where(np.isfinite(logr), logr, 0.0)
    return float(np.mean(logr))
