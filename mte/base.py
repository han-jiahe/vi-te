"""Shared result container for the TE estimators."""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TEResult:
    """One TE estimate for one ordered player pair.

    method : "gaussian" | "ksg" | "discrete"
    pair   : (A, B) player indices, A -> B (ball included as a node)
    te     : raw TE in bits (base 2)
    n_samples : number of (X, Y, Z) triples used
    params : estimator-specific hyperparameters
    """
    method: str
    pair: tuple
    te: float
    n_samples: int
    params: Dict[str, Any] = field(default_factory=dict)
