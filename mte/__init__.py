"""
mte - player-level transfer entropy (TE) estimators (PROTOCOL.md).

Pairwise transfer entropy between two per-player GVRNN latent chains:

    TE(A -> B) = I( z^A_{t-1} ; z^B_t | z^B_{t-1} )

i.e. the conditional mutual information I(X; Y | Z) of the triple
X = z^A_{t-1} (source history), Y = z^B_t (target future),
Z = z^B_{t-1} (target history) - the Schreiber TE of order (k, l) = (1, 1).
No third-party channels are conditioned on.

The estimators are generic conditional-MI estimators (they accept an
arbitrary Z); the TE specialisation Z = [z^B_{t-1}] is provided by the
mte.te.transfer_entropy_* wrappers.

  * conditional_gaussian_mte : Gaussian closed form (PROTOCOL.md sec. 6)
  * conditional_ksg_mte      : Frenzel-Pompe conditional kNN (sec. 5)
  * conditional_discrete_mte : K-means quantised plug-in (sec. 7)

All estimators take already-aligned (X, Y, Z) arrays; use mte.utils.build_triples
to build them from raw per-player channels, and mte.significance.surrogate_mte
for the block-shuffle null test (sec. 9).
"""
from .base import TEResult
from .gaussian import conditional_gaussian_mte, gaussian_mte_pvalue
from .ksg import conditional_ksg_mte
from .discrete import conditional_discrete_mte
from .significance import block_shuffle, surrogate_mte, effective_te
from .utils import build_triples, downsample, lag, pca_fit, pca_transform, standardize
from .te import (transfer_entropy_gaussian, transfer_entropy_ksg,
                 transfer_entropy_discrete)

__all__ = [
    "TEResult",
    "conditional_gaussian_mte", "gaussian_mte_pvalue",
    "conditional_ksg_mte",
    "conditional_discrete_mte",
    "transfer_entropy_gaussian", "transfer_entropy_ksg",
    "transfer_entropy_discrete",
    "block_shuffle", "surrogate_mte", "effective_te",
    "build_triples", "downsample", "lag", "pca_fit", "pca_transform",
    "standardize",
]
