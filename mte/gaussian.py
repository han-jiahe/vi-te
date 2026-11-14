"""
Conditional-Gaussian MTE (PROTOCOL.md sec. 6).

Under a joint-Gaussian assumption the conditional mutual information has the
closed form

    I(X;Y|Z) = 1/2 log[ |S_XZ| |S_YZ| / ( |S_Z| |S_XYZ| ) ]

where S_(.) are sample covariance matrices of the stacked variables.
This is the multivariate Granger measure: X = z^A_{t-1}, Y = z^B_t,
Z = [z^B_{t-1}, z^rest_{t-1}].  Ridge (trace-scaled) regularises near-singular
covariances, the same convention as markov-te's te/gaussian.py.

Cross-check: Geweke's chi^2 test - under the null of no X -> Y given Z,
n * 2 * MTE_nats ~ chi^2(df = dim(X) * dim(Y)) (gaussian_mte_pvalue).
"""
from typing import Union

import numpy as np
from numpy.linalg import slogdet
from scipy.stats import chi2


def _atleast_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def _ridge_cov(M: np.ndarray, ridge: float) -> np.ndarray:
    """Sample covariance with trace-scaled ridge (unit-free shrinkage).

    np.cov squeezes a single-variable covariance to a 0-d scalar (e.g. Z with
    exactly one column), so re-wrap to 2-d before tracing."""
    C = np.atleast_2d(np.cov(M, rowvar=False, ddof=1))
    tr = np.trace(C) / C.shape[0]
    return C + np.eye(C.shape[0]) * tr * ridge


def conditional_gaussian_mte(X, Y, Z, base: float = 2.0,
                             ridge: float = 1e-3) -> float:
    """I(X;Y|Z) in `base` units (default bits), Gaussian closed form.

    X, Y, Z: (n, *) arrays from mte.utils.build_triples (Z may have 0
    columns, in which case this reduces to the pairwise Gaussian TE).
    """
    X = _atleast_2d(X)
    Y = _atleast_2d(Y)
    Z = _atleast_2d(Z) if np.asarray(Z).size else np.zeros((len(X), 0))
    n = len(X)
    assert n == len(Y) == len(Z) > 0

    XZ = np.hstack([X, Z])
    YZ = np.hstack([Y, Z])
    XYZ = np.hstack([X, Y, Z])

    S_Z = _ridge_cov(Z, ridge) if Z.shape[1] else np.eye(1)
    S_XZ = _ridge_cov(XZ, ridge)
    S_YZ = _ridge_cov(YZ, ridge)
    S_XYZ = _ridge_cov(XYZ, ridge)

    val = 0.5 * (slogdet(S_XZ)[1] + slogdet(S_YZ)[1]
                 - slogdet(S_Z)[1] - slogdet(S_XYZ)[1])   # nats
    return val / np.log(base)


def gaussian_mte_pvalue(te: float, n_samples: int, dx: int, dy: int,
                        base: float = 2.0) -> float:
    """Geweke chi^2 p-value: under H0, n * 2 * MTE_nats ~ chi^2(dx * dy).

    `te` is the estimate in `base` units; the statistic is converted to
    nats internally (Geweke's measure is defined on nats).
    """
    stat = n_samples * 2.0 * te * np.log(base)
    return float(chi2.sf(stat, df=int(dx * dy)))
