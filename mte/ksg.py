"""
Conditional kNN (KSG) MTE - Frenzel-Pompe conditional form (PROTOCOL.md sec. 5).

    I(X;Y|Z) = psi(k) + < psi(n_Z + 1) - psi(n_XZ + 1) - psi(n_YZ + 1) >

where for each sample the radius is the distance to its k-th neighbour in
the FULL joint space (X, Y, Z), Chebyshev (max) norm, and every ball count
n_Z / n_XZ / n_YZ uses the SAME radius in the respective marginal space,
strict "<" convention (radius scaled by 1 - 1e-12, self subtracted) - the
same strict-count convention as markov-te's te/ksg.py.  All channels must be
standardised per dimension beforehand (Chebyshev radii mix dimensions;
mte.utils.standardize).
"""
from typing import Union

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def _atleast_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def _ball_counts(sub: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per-sample ball counts in `sub` space using radii `r` (full-space
    radii), strict '<', self excluded.  Built on a KDTree of the subspace.

    ``query_ball_point`` accepts a per-point radius array and loops in C;
    the former Python-level per-point call loop is removed (identical
    counts: same ``<=`` semantics, same pre-scaled strict radius, self
    pair subtracted)."""
    tree = cKDTree(sub)
    lists = tree.query_ball_point(sub, r, p=np.inf)
    return np.asarray([len(lst) for lst in lists], dtype=np.int64) - 1


def conditional_ksg_mte(X, Y, Z, k: int = 3, base: float = 2.0,
                        eps_scale: float = 1.0 - 1e-12) -> float:
    """I(X;Y|Z) via the Frenzel-Pompe conditional KSG estimator.

    X, Y, Z: (n, *) arrays from mte.utils.build_triples (Z may have 0
    columns -> plain pairwise KSG TE).  Requires n > k.
    """
    X = _atleast_2d(X)
    Y = _atleast_2d(Y)
    Z = _atleast_2d(Z) if np.asarray(Z).size else np.zeros((len(X), 0))
    n = len(X)
    assert n == len(Y) == len(Z) > k, f"need n > k, got n={n}, k={k}"

    xyz = np.hstack([X, Y, Z])
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1, p=np.inf)   # +1: self at distance 0
    r = dists[:, k] * eps_scale                      # strict "<" radius

    n_z = _ball_counts(Z, r) if Z.shape[1] else np.full(n, n - 1)
    n_xz = _ball_counts(np.hstack([X, Z]), r)
    n_yz = _ball_counts(np.hstack([Y, Z]), r)

    with np.errstate(invalid="ignore"):
        val = digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1)
                                   - digamma(n_yz + 1))     # nats
    return float(val) / np.log(base)
