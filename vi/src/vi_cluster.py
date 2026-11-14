"""Clustering Variation of Information (VI), ported from BlueDream_v3plus.

Source: https://github.com/han-jiahe/BlueDream_v3plus
        ``vi_analysis/clustering.py``      (vi, vi_per_node, _contingency)
        ``vi_analysis/clustering_vi.py``   (compute_vi_rates)

This replaces the Voronoi "influence" VI that was used before.  The two are
different quantities with the same name:

    Voronoi VI (old)  = cell area / pitch area, dimensionless in [0, 1]
    Clustering VI     = Variation of Information between two clusterings,
                        in bits, paper arXiv:2102.11182 eq. (2)

    VI(X;Y) = -sum_ij r_ij [ log2(r_ij / p_i) + log2(r_ij / q_j) ]      (bit)
    VI^{p_ij} = -r_ij [ log2(r_ij/p_i) + log2(r_ij/q_j) ] / |x_i ∩ y_j|  (bit)
    VI_rate(t) = VI(C_{t-lag}, C_t) / (time_t - time_{t-lag})           (bit/s)

Nodes are the 22 on-pitch players **plus the two goals** (24 nodes when both
sides are complete).  A team's VI is the sum over its 11 players and its own
goal node, so ``home + away == total VI`` per timestamp (it is *not* 1).
"""

from __future__ import annotations

import numpy as np

# goal node ids, following the upstream convention
GOAL_HOME = -1
GOAL_AWAY = -2


# --------------------------------------------------------------------- graph
def hierarchical_clustering(ids, xy):
    """Nearest-neighbour single-linkage clustering of the 24 nodes.

    A faithful port of ``hierarchical_clustering_by_nearest_neighbor``: each
    node links to its nearest neighbour, and nodes already in different
    clusters are merged.  The iteration order is the order of ``ids``, which is
    fixed by the caller so the result is deterministic.
    """
    n = len(ids)
    if n == 0:
        return {}
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nearest = d.argmin(axis=1)

    assign: dict[int, int] = {}
    counter = 0
    for k in range(n):
        pid = ids[k]
        nid = ids[nearest[k]]
        pc = assign.get(pid)
        nc = assign.get(nid)
        if pc is None and nc is None:
            assign[pid] = counter
            assign[nid] = counter
            counter += 1
        elif nc is None:
            assign[nid] = pc
        elif pc is None:
            assign[pid] = nc
        elif pc != nc:
            for q, c in list(assign.items()):
                if c == nc:
                    assign[q] = pc
    return assign


# ------------------------------------------------------------------- pid -> cid
def _contingency(assign_a: dict[int, int], assign_b: dict[int, int]):
    """Contingency table between two assignments (paper eq. 2).

    A node present in only one side is padded as a singleton cluster on the
    other side, so substitutions / missed detections still carry their VI.
    """
    union = sorted(set(assign_a) | set(assign_b))
    n = len(union)

    def index_map(assign):
        cids = sorted(set(assign.values()))
        cid_to_idx = {c: i for i, c in enumerate(cids)}
        m = {}
        nxt = len(cid_to_idx)
        for tid in union:
            if tid in assign:
                m[tid] = cid_to_idx[assign[tid]]
            else:
                m[tid] = nxt
                nxt += 1
        return m, nxt

    a_map, k = index_map(assign_a)
    b_map, l_ = index_map(assign_b)

    cnt = np.zeros((k, l_))
    pair = {}
    for tid in union:
        i, j = a_map[tid], b_map[tid]
        cnt[i, j] += 1
        pair[tid] = (i, j)
    rij = cnt / n if n else np.zeros_like(cnt)
    return rij, rij.sum(axis=1), rij.sum(axis=0), pair, cnt


def vi(assign_a: dict[int, int], assign_b: dict[int, int]) -> float:
    """Total Variation of Information between two clusterings, in bits."""
    rij, pi, qj, _, _ = _contingency(assign_a, assign_b)
    total = 0.0
    k, l_ = rij.shape
    for i in range(k):
        for j in range(l_):
            r = rij[i, j]
            if r > 0:
                total -= r * (np.log2(r / pi[i]) + np.log2(r / qj[j]))
    return float(total)


def vi_per_node(assign_a: dict[int, int],
                assign_b: dict[int, int]) -> dict[int, float]:
    """Per-node VI contribution (paper eq. 4); sums to :func:`vi`."""
    rij, pi, qj, pair, cnt = _contingency(assign_a, assign_b)
    out = {}
    for tid, (i, j) in pair.items():
        size = int(cnt[i, j])
        if size > 0:
            r = rij[i, j]
            out[tid] = float(-r * (np.log2(r / pi[i]) + np.log2(r / qj[j])) / size) + 0.0
        else:
            out[tid] = 0.0
    return out


# ---------------------------------------------------------------- self-check
def selfcheck() -> None:
    """The upstream assertions, so a port error cannot pass silently."""
    A = {1: 0, 2: 0}
    assert abs(vi(A, A)) < 1e-12
    assert all(abs(v) < 1e-12 for v in vi_per_node(A, A).values())

    A = {1: 0, 2: 0, 3: 0, 4: 0}
    B = {1: 0, 2: 0, 3: 1, 4: 1}
    assert abs(vi(A, B) - 1.0) < 1e-9
    assert all(abs(v - 0.25) < 1e-9 for v in vi_per_node(A, B).values())

    A = {1: 0, 2: 0, 3: 0}
    B = {1: 0, 2: 0}
    expect = -((2 / 3) * np.log2(2 / 3) + (1 / 3) * np.log2(1 / 3))
    assert abs(vi(A, B) - expect) < 1e-9
    per = vi_per_node(A, B)
    assert set(per) == {1, 2, 3}
    assert abs(sum(per.values()) - vi(A, B)) < 1e-9
    assert all(v >= 0 for v in per.values())

    rng = np.random.default_rng(0)
    for _ in range(50):
        ids = list(range(1, 7))
        A = {t: int(rng.integers(0, 3)) for t in ids}
        B = {t: int(rng.integers(0, 3)) for t in ids}
        assert abs(sum(vi_per_node(A, B).values()) - vi(A, B)) < 1e-9
    print("vi_cluster self-check passed")
