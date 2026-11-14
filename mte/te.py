"""
Pairwise transfer entropy (order-1, k = l = 1) between two latent chains.

    TE(A -> B) = I( z^A_{t-1} ; z^B_t | z^B_{t-1} )

i.e. the conditional mutual information I(X; Y | Z) of the triple
X = z^A_{t-1} (source history), Y = z^B_t (target future), Z = z^B_{t-1}
(target history) - the Schreiber TE of embedding (k, l) = (1, 1), with the
discrete sum replaced by an estimator on continuous data.

Three estimators (the same I(X;Y|Z) machinery as the conditional estimators
in this package, specialised to Z = [z^B_{t-1}]):

  * transfer_entropy_gaussian : joint-Gaussian closed form
        TE = 1/2 log2[ |S_XZ| |S_YZ| / ( |S_Z| |S_XYZ| ) ]
      (for 1-dim standardised chains this collapses to
        TE = -1/2 log2(1 - rho^2_{XY.Z}),
       rho_{XY.Z} the partial correlation of X and Y given Z)
  * transfer_entropy_ksg      : Frenzel-Pompe KSG (nonparametric)
  * transfer_entropy_discrete : K-means quantised plug-in (Schreiber's
      counting formula applied to the quantised symbols)

A and B are per-player latent channels (T,) or (T, d); callers are expected
to have applied PCA / standardisation FIRST - the estimators themselves do
not refit any transform (PROTOCOL.md sec. 4).
"""
import numpy as np

from .discrete import conditional_discrete_mte, quantize_channels
from .gaussian import conditional_gaussian_mte
from .ksg import conditional_ksg_mte
from .utils import build_triples


def transfer_entropy_gaussian(A, B, base: float = 2.0, ridge: float = 1e-3) -> float:
    """TE(A->B) under the joint-Gaussian assumption (bits)."""
    X, Y, Z = build_triples(A, B, [])
    return conditional_gaussian_mte(X, Y, Z, base=base, ridge=ridge)


def transfer_entropy_ksg(A, B, k: int = 3, base: float = 2.0,
                         eps_scale: float = 1.0 - 1e-12) -> float:
    """TE(A->B) via the Frenzel-Pompe conditional KSG estimator (bits)."""
    X, Y, Z = build_triples(A, B, [])
    return conditional_ksg_mte(X, Y, Z, k=k, base=base, eps_scale=eps_scale)


def transfer_entropy_discrete(A, B, n_clusters: int = 3, seed: int = 42,
                              base: float = 2.0) -> float:
    """TE(A->B) via K-means quantisation + the Schreiber counting formula
    (bits).  Fits are applied to A and B only (no third-party channels)."""
    syms, _ = quantize_channels([A, B], n_clusters, seed=seed)
    sA, sB = syms
    X = sA[:-1]
    Y = sB[1:]
    Z = np.column_stack([sB[:-1]])     # target history as the only condition
    return conditional_discrete_mte(Y, X, Z, base=base)
