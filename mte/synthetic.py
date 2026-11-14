"""
Synthetic three-channel VAR(1) systems with (approximately) known MTE ground
truth, for validating the three estimators before touching real data
(PROTOCOL.md sec. 13).

System:

    C_t = a_c C_{t-1} + b_xc X_{t-1} + e_c
    X_t = a_x X_{t-1} + b_cx C_{t-1} + c_cx C_t + e_x
    Y_t = a_y Y_{t-1} + b_xy X_{t-1} + b_cy C_{t-1} + c_cy C_t + e_y

with i.i.d. Gaussian noise e ~ N(0, sigma^2), all channels 1-dimensional.
The c_cx / c_cy terms are CONTEMPORANEOUS drives (same time step) - this is
the honest model of the shared-input confound of the GVRNN latents: the
current observation (ball position) enters every player's encoder in the
same frame.

Scenario table:

    name             b_xy  c_cx  c_cy  a_c   expected behaviour
    null             0     0     0     0.5   no edges at all
    direct           0.5   0     0     0.5   X -> Y only; MTE(X->Y|C) > 0
    common_driver    0     1.0   1.0   0.95  C drives X and Y in the SAME
                                             frame (shared-input confound):
                                             X_{t-1} and Y_{t-1} are two noisy
                                             views of C_{t-1}, so pairwise
                                             TE(X->Y) is spurious > 0 while
                                             MTE(X->Y|C_{t-1}) ~ 0  <- the KEY
                                             conditioning test

Known limitation, documented honestly: a pure LAG-1 relay X -> C -> Y is a
two-step path that order-1 TE cannot see at all (neither pairwise nor
conditional) - detecting it needs order-2 TE, a documented future work.

The ground truth is the Gaussian closed-form conditional MI evaluated on a
LONG chain (200k steps) - the plug-in has converged there; it uses the same
formula as mte.gaussian.conditional_gaussian_mte without ridge.
"""
import numpy as np

from .gaussian import conditional_gaussian_mte
from .utils import build_triples


def simulate_chain(T: int, a_x: float = 0.5, a_y: float = 0.5, a_c: float = 0.5,
                   b_xy: float = 0.0, b_xc: float = 0.0,
                   b_cx: float = 0.0, b_cy: float = 0.0,
                   c_cx: float = 0.0, c_cy: float = 0.0,
                   sigma: float = 0.2, seed: int = 0) -> dict:
    """Simulate the VAR(1) system; returns dict with keys X, Y, C (all (T,)
    float arrays, channel X plays the role of "player A", Y of "player B",
    C of the conditioning player)."""
    rng = np.random.default_rng(seed)
    X = np.zeros(T)
    Y = np.zeros(T)
    C = np.zeros(T)
    for t in range(1, T):
        C[t] = a_c * C[t - 1] + b_xc * X[t - 1] + rng.normal(0, sigma)
        X[t] = a_x * X[t - 1] + b_cx * C[t - 1] + c_cx * C[t] + rng.normal(0, sigma)
        Y[t] = a_y * Y[t - 1] + b_xy * X[t - 1] + b_cy * C[t - 1] + c_cy * C[t] + rng.normal(0, sigma)
    return {"X": X, "Y": Y, "C": C}


def gt_mte(chain: dict, pair: str = "X->Y", with_C: bool = True) -> float:
    """Ground-truth MTE on a simulated chain (long-chain Gaussian closed
    form).  pair: 'X->Y' (or 'Y->X'); with_C: condition on C or not
    (pairwise TE when False)."""
    A = chain["X"] if pair == "X->Y" else chain["Y"]
    B = chain["Y"] if pair == "X->Y" else chain["X"]
    R = [chain["C"]] if with_C else []
    X, Y, Z = build_triples(A, B, R)
    return conditional_gaussian_mte(X, Y, Z, base=2.0, ridge=0.0)


def scenarios(T: int = 4000, GT_T: int = 200_000, seed: int = 0) -> dict:
    """All four validation scenarios: short analysis chains + long GT chains."""
    defs = {
        "null":          dict(b_xy=0.0, c_cx=0.0, c_cy=0.0, a_c=0.5),
        "direct":        dict(b_xy=0.5, c_cx=0.0, c_cy=0.0, a_c=0.5),
        "common_driver": dict(b_xy=0.0, c_cx=1.0, c_cy=1.0, a_c=0.95),
    }
    out = {}
    for name, kw in defs.items():
        out[name] = {
            "chain": simulate_chain(T, seed=seed, **kw),
            "gt": simulate_chain(GT_T, seed=seed + 1, **kw),
        }
    return out
