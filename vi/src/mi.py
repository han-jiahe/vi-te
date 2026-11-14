"""Gaussian (linear) mutual-information estimators, as used by Rosas et al. (2020).

For jointly Gaussian variables the mutual information has a closed form

    I(A; B) = 1/2 * log( det(Sigma_A) * det(Sigma_B) / det(Sigma_AB) )

which is the estimator the paper refers to as the "Gaussian mutual information
estimator ... implemented in the publicly available toolbox provided by Rosas
et al.".  Log-determinants are obtained from eigenvalues so that numerically
singular (rank deficient) sample covariances degrade gracefully instead of
producing -inf.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _logdet_batch(cov: np.ndarray) -> np.ndarray:
    """log|det(C)| for a stack of symmetric matrices, robust to tiny negatives."""
    eig = np.linalg.eigvalsh(cov)
    return np.log(np.clip(eig, _EPS, None)).sum(axis=-1)


def mi_from_cov(cov_a: np.ndarray, cov_b: np.ndarray, cov_ab: np.ndarray) -> np.ndarray:
    """Batched mutual information from block covariances.

    All arguments carry a leading batch axis.
    """
    return 0.5 * (_logdet_batch(cov_a) + _logdet_batch(cov_b) - _logdet_batch(cov_ab))


def gaussian_mi(a: np.ndarray, b: np.ndarray) -> float:
    """Plain plug-in Gaussian MI between two sample matrices (n_samples, n_features)."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the same number of samples")
    ja = a - a.mean(axis=0)
    jb = b - b.mean(axis=0)
    cab = np.cov(np.hstack([ja, jb]), rowvar=False)
    cab = np.atleast_2d(cab)
    da = a.shape[1]
    return float(mi_from_cov(cab[None, :da, :da], cab[None, da:, da:], cab[None])[0])


def rolling_mi(z: np.ndarray, d_first: int, win: int | None) -> np.ndarray:
    """Mutual information between ``z[:, :d_first]`` and ``z[:, d_first:]``.

    With ``win`` set, the estimate is repeated over a sliding window (step 1
    sample) and one value per window end is returned; otherwise a single
    estimate over all samples is returned.
    """
    n, d = z.shape
    if win is None or win > n:
        return np.array([gaussian_mi(z[:, :d_first], z[:, d_first:])])

    starts = np.arange(0, n - win + 1)
    ends = starts + win

    # cumulative sums of raw moments -> O(1) covariance per window
    cum_outer = np.zeros((n + 1, d, d))
    cum_outer[1:] = np.cumsum(z[:, :, None] * z[:, None, :], axis=0)
    cum_sum = np.vstack([np.zeros((1, d)), np.cumsum(z, axis=0)])

    cross = cum_outer[ends] - cum_outer[starts]                     # sum z z^T
    s = cum_sum[ends] - cum_sum[starts]                             # sum z
    mu = s / win
    cov = cross - s[:, :, None] * mu[:, None, :]                     # win * covariance

    out = mi_from_cov(
        cov[:, :d_first, :d_first],
        cov[:, d_first:, d_first:],
        cov,
    )
    return out
