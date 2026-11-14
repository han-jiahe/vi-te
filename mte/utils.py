"""Channel preprocessing helpers for the MTE estimators.

Conventions (PROTOCOL.md sec. 4/9):
  * channels are (T, d) arrays aligned at the same frame t;
  * build_triples forms the order-1 MTE sample (X, Y, Z) from raw channels;
  * PCA is FROZEN across the surrogate protocol: fit once on the real
    channel, transform everything else (same convention as the K-means
    freezing in D:\\MarkovSoccer\\markov-te).
"""
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA


def lag(x: np.ndarray, l: int = 1) -> np.ndarray:
    """x[t - l] aligned with x[t]: returns x[:-l]."""
    return np.asarray(x)[:-l]


def downsample(x: np.ndarray, stride: int) -> np.ndarray:
    """Take every stride-th row (10 Hz / stride = new rate)."""
    return np.asarray(x)[::stride]


def build_triples(A: np.ndarray, B: np.ndarray,
                  R_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order-1 MTE sample from raw per-player channels (all length T).

    X = z^A_{t-1} (source history),  Y = z^B_t (target future),
    Z = [z^B_{t-1}, z^R1_{t-1}, ..., z^Rm_{t-1}] (target history + the rest).

    Every returned array has n = T - 1 rows; row s is the transition
    t = s -> t = s + 1.  A and B may be the same-length (T, d) arrays
    after PCA/standardisation.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    X = A[:-1]
    Y = B[1:]
    z_parts = [B[:-1]]
    for R in R_list:
        z_parts.append(np.asarray(R, dtype=float)[:-1])
    Z = np.column_stack(z_parts) if z_parts else np.zeros((len(X), 0))
    return X, Y, Z


def standardize(x: np.ndarray, mu=None, std=None):
    """Z-score per column; optionally re-use frozen (mu, std)."""
    x = np.asarray(x, dtype=float)
    if mu is None:
        mu, std = x.mean(axis=0), x.std(axis=0)
    return (x - mu) / (std + 1e-9), mu, std


def pca_fit(x: np.ndarray, n_components: int):
    """Fit PCA on a (T, d) channel; returns the fitted PCA object."""
    pca = PCA(n_components=n_components)
    pca.fit(np.asarray(x, dtype=float))
    return pca


def pca_transform(pca, x: np.ndarray) -> np.ndarray:
    """Apply a FROZEN PCA fit (the surrogate protocol never refits)."""
    return pca.transform(np.asarray(x, dtype=float))
