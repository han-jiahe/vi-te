"""
Significance testing for the MTE estimates (PROTOCOL.md sec. 10).

Block-shuffle surrogates: the SOURCE channel A is cut into blocks and the
blocks are randomly re-ordered, destroying the cross-channel dependence
while keeping the within-block autocorrelation of A; B and the conditioning
channels are untouched.  All channel transforms (PCA, standardisation,
K-means fits) are FROZEN on the real data - surrogates only re-transform,
never refit (refitting per surrogate would confound the null with
preprocessing variation).

    TE_eff = raw - mean(surrogates)          (bias-corrected estimate)
    p      = (#{surr >= raw} + 1) / (S + 1)  (one-sided upper tail)
"""
from typing import Callable, List, Tuple

import numpy as np


def block_shuffle(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Cut x into blocks of block_len rows and permute the block order."""
    x = np.asarray(x)
    n = len(x)
    n_blocks = int(np.ceil(n / block_len))
    idx = np.arange(n)
    blocks = [idx[b * block_len:(b + 1) * block_len] for b in range(n_blocks)]
    rng.shuffle(blocks)
    perm = np.concatenate(blocks)
    return x[perm]


def surrogate_mte(fn: Callable[[np.ndarray, np.ndarray, List[np.ndarray]], float],
                  A: np.ndarray, B: np.ndarray, R_list: List[np.ndarray],
                  block_len: int, n_surrogates: int = 200,
                  seed: int = 42) -> np.ndarray:
    """S surrogate MTE values under the block-shuffle null.

    fn(A, B, R_list) -> float is the estimator closure; it applies the lag
    internally (build_triples), so shuffling the RAW channel A by blocks and
    re-lagging is the correct TE null (in-block autocorrelation of A is
    preserved, the cross-channel link to B/R is destroyed).
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_surrogates, dtype=float)
    for s in range(n_surrogates):
        A_sh = block_shuffle(A, block_len, rng)
        out[s] = fn(A_sh, B, R_list)
    return out


def effective_te(te_raw: float, te_surrogates: np.ndarray) -> Tuple[float, float]:
    """Returns (te_effective, p_value): surrogate-corrected TE and the
    one-sided upper-tail p-value."""
    surr = np.asarray(te_surrogates)
    eff = float(te_raw - surr.mean())
    p = float((np.sum(surr >= te_raw) + 1) / (len(surr) + 1))
    return eff, p
