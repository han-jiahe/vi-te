"""Voronoi Influence (VI), ported from the BlueDream_v3plus project.

Source: https://github.com/han-jiahe/BlueDream_v3plus  (`vi_analysis/voronoi_vi.py`)

The upstream implementation hard-codes a 105 m x 68 m pitch; the SkillCorner
open data uses 104 m x 68 m, so every geometric constant here is a parameter
(see 实现方案 §1.2 / §2: "必须参数化成 104×68，否则面积差 ~1%").

Algorithm (unchanged from upstream):
  1. collect the 22 players' coordinates in pitch space (metres);
  2. add 8 mirror points per player across the four touchlines so that every
     interior cell is bounded;
  3. `scipy.spatial.Voronoi` tessellation;
  4. clip each player's cell to the pitch rectangle with shapely;
  5. VI_i = clipped_area / pitch_area.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box

PITCH_LENGTH = 104.0          # metres, touchline
PITCH_WIDTH = 68.0            # metres, goal line
PITCH_AREA = PITCH_LENGTH * PITCH_WIDTH      # 7072 m^2

_EPS = 1e-6                    # jitter for coincident points


def add_boundary_mirrors(points: np.ndarray,
                         length: float = PITCH_LENGTH,
                         width: float = PITCH_WIDTH) -> np.ndarray:
    """Original points plus their 8 mirror images across the four sides."""
    x = points[:, 0]
    y = points[:, 1]
    L, W = length, width
    mirrors = np.column_stack([
        np.concatenate([-x, 2 * L - x, x, x, -x, 2 * L - x, -x, 2 * L - x]),
        np.concatenate([y, y, -y, 2 * W - y, -y, -y, 2 * W - y, 2 * W - y]),
    ])
    return np.vstack([points, mirrors])


def compute_frame_vi(points: np.ndarray,
                     length: float = PITCH_LENGTH,
                     width: float = PITCH_WIDTH) -> np.ndarray:
    """Individual VI for every player in one frame.

    Parameters
    ----------
    points : (n, 2) array of pitch coordinates in metres, corner origin.

    Returns
    -------
    (n,) array of VI = clipped Voronoi cell area / pitch area.

    Notes
    -----
    Players occasionally stand a few metres outside the touchlines (throw-in
    takers, goalkeepers).  Mirroring such a point across the line it is already
    outside of puts the mirror *inside* the pitch, where it steals area from
    the real cells and the 22 cells stop tiling the pitch.  Coordinates are
    therefore clamped to ``[1 mm, L - 1 mm] x [1 mm, W - 1 mm]`` before the
    tessellation: a deviation of at most a few metres on 0.7 % of the
    observations, but the cells then tile the pitch exactly (coverage 1.000).
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0)

    pad = 1e-3
    pts = np.clip(np.asarray(points, dtype=float), [pad, pad],
                  [length - pad, width - pad]).copy()
    # coincident players would make the tessellation degenerate
    for k in range(1, n):
        same = np.all(np.abs(pts[:k] - pts[k]) < _EPS, axis=1)
        while same.any():
            pts[k] += np.array([_EPS, _EPS])
            same = np.all(np.abs(pts[:k] - pts[k]) < _EPS, axis=1)

    all_points = add_boundary_mirrors(pts, length, width)
    try:
        vor = Voronoi(all_points)
    except Exception:                                   # pragma: no cover
        return np.zeros(n)

    pitch_box = box(0.0, 0.0, length, width)
    area = np.zeros(n)
    for i in range(n):
        region_idx = vor.point_region[i]
        verts = vor.regions[region_idx]
        if not verts or -1 in verts:
            continue
        cell = Polygon(vor.vertices[verts])
        try:
            clipped = cell.intersection(pitch_box)
        except Exception:                               # pragma: no cover
            continue
        if not clipped.is_empty:
            area[i] = clipped.area
    return area / (length * width)


def compute_second_vi(positions: dict, teams: dict,
                      length: float = PITCH_LENGTH,
                      width: float = PITCH_WIDTH) -> tuple[dict, dict]:
    """VI for one second.

    Parameters
    ----------
    positions : {player_id: (X, Y)} corner-origin metres, on-pitch players only
    teams : {player_id: team label}

    Returns
    -------
    (player_vi, team_vi) dicts
    """
    ids = list(positions)
    if not ids:
        return {}, {}
    pts = np.array([positions[p] for p in ids], dtype=float)
    vi = compute_frame_vi(pts, length, width)
    player_vi = {pid: float(v) for pid, v in zip(ids, vi)}
    team_vi = defaultdict(float)
    for pid, v in player_vi.items():
        team_vi[teams[pid]] += v
    return player_vi, dict(team_vi)
