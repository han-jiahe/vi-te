"""Build the VI table for one match and check the paper-VI self-check anchors.

The VI definition changed on 2026-11-09: it is now the Variation of Information
between two nearest-neighbour clusterings (arXiv:2102.11182), in bits, not the
Voronoi cell-area share.  The anchors below are the new ones; the old
"sum of both teams == 1.000" identity no longer applies.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.ce_scalar import _fill_gaps, mi_scalar, psi_series, windowed_mi
from src.mi import gaussian_mi
from src.vi_cluster import selfcheck as vi_selfcheck
from src.vi_pipeline import ViMatch

mid = sys.argv[1] if len(sys.argv) > 1 else "1886347"

print("=== ported clustering-VI self-check (upstream assertions) ===")
vi_selfcheck()

print("\n--- gap handling checks (synthetic, no data needed) ---")
rng = np.random.default_rng(1)
x = rng.normal(size=200)
valid = np.ones(200, dtype=bool)
valid[100:105] = False
xg = x.copy()
xg[100:105] = np.nan

filled = _fill_gaps(xg, valid, max_gap=10)
assert np.isfinite(filled).all(), "5-sample gap should be filled at max_gap=10"
assert np.allclose(filled[100:105],
                   np.interp(np.arange(100, 105), [99, 105], [x[99], x[105]]))
assert np.isnan(_fill_gaps(xg, valid, max_gap=3)[100:105]).all()
assert np.isfinite(_fill_gaps(xg, valid, max_gap=None)).all()

xg2 = x.copy()
xg2[:4] = np.nan
v2 = np.ones(200, dtype=bool)
v2[:4] = False
assert np.isnan(_fill_gaps(xg2, v2, max_gap=None)[:4]).all(), \
    "leading NaNs must stay NaN"

macro = np.arange(300, dtype=float)
src = np.column_stack([macro, macro * 0.5 + 1])
vm = np.ones(300, dtype=bool)
vm[150:165] = False
p_strict = psi_series(macro, src, vm, win=30, max_gap=0)
p_legacy = psi_series(macro, src, vm, win=30)
p_filled = psi_series(macro, src, vm, win=30, max_gap=20)
assert np.allclose(np.nan_to_num(p_strict["psi"]), np.nan_to_num(p_legacy["psi"]))
assert p_filled["n_windows"] > p_strict["n_windows"], \
    (p_strict["n_windows"], p_filled["n_windows"])
print(f"gap checks passed; windows strict={p_strict['n_windows']} "
      f"filled={p_filled['n_windows']}")

t0 = time.time()
v = ViMatch(mid, ROOT / "data" / "raw" / mid)
print(f"\nbuild took {time.time()-t0:.1f} s")

if "--hz10" in sys.argv:
    print("\n--- E3: 10 Hz (paper) VI rate, downsampled to the 1 Hz grid ---")
    t0 = time.time()
    v10 = ViMatch(mid, ROOT / "data" / "raw" / mid, hz=10)
    print(f"10 Hz build took {time.time()-t0:.1f} s")
    ok = np.isfinite(v10.total_vi)
    print(f"10 Hz VI rate defined on {int(ok.sum())} seconds "
          f"(1 Hz: {int(np.isfinite(v.total_vi).sum())})")
    print(f"total VI rate [bit/s]: median 1 Hz={np.nanmedian(v.total_vi):.4f} "
          f"10 Hz={np.nanmedian(v10.total_vi):.4f} "
          f"(10/1 ratio={np.nanmedian(v10.total_vi)/np.nanmedian(v.total_vi):.2f})")
    sums10 = np.nansum(v10.team_vi[ok], axis=1)
    print(f"home + away == total VI (10 Hz): max|diff| = "
          f"{np.max(np.abs(sums10 - v10.total_vi[ok])):.2e}")
    common = ok & np.isfinite(v.total_vi) & np.isfinite(v10.total_vi)
    h1, h10 = v.team_vi[common, 0], v10.team_vi[common, 0]
    a1, a10 = v.team_vi[common, 1], v10.team_vi[common, 1]
    print(f"corr(1 Hz home, 10 Hz home) = {np.corrcoef(h1, h10)[0,1]:+.3f}")
    print(f"corr(1 Hz away, 10 Hz away) = {np.corrcoef(a1, a10)[0,1]:+.3f}")
    print(f"max|total VI| (10 Hz, bit/s) = {np.nanmax(np.abs(v10.total_vi)):.4f}")
    # regression guard for the stale-copy bug: the 10 Hz build must not claim
    # wildly more defined seconds than the 1 Hz build (the buggy mapping gave
    # 5330 vs 4403); with the fix the counts are close.  (Seconds where the
    # two series coincide are legitimate: static play, or a single clustering
    # flip at the second boundary.)
    stale = np.abs(h1 - h10) < 1e-12
    print(f"seconds with identical 1 Hz/10 Hz value: {int(stale.sum())} "
          f"({100 * stale.mean():.1f}%) -- static/boundary-flip seconds")
    n1 = int(np.isfinite(v.total_vi).sum())
    n10 = int(np.isfinite(v10.total_vi).sum())
    assert abs(n10 - n1) / max(n1, 1) < 0.1, \
        f"10 Hz defined seconds ({n10}) drifted from 1 Hz ({n1})"

print("\n--- anchor 1: valid seconds ---")
print(f"valid seconds = {int(v.valid.sum())} (target ~4405 for 1886347)")
for per in (1, 2):
    sel = v.period == per
    print(f"  P{per}: grid={int(sel.sum())}, valid={int((sel & v.valid).sum())}")

print("\n--- anchor 2 (new): node contributions sum to the total VI ---")
ok = np.isfinite(v.total_vi)
print(f"VI rate defined on {int(ok.sum())} seconds")
print(f"total VI rate [bit/s]: min={np.nanmin(v.total_vi):.4f} "
      f"median={np.nanmedian(v.total_vi):.4f} max={np.nanmax(v.total_vi):.4f}")
print(f"total VI (bit, undivided): max={np.nanmax(v.total_vi_abs):.4f} "
      f"(ceiling log2(24)={np.log2(24):.4f})")
nz = v.total_vi[ok]
print(f"non-negative total VI: {bool((nz >= -1e-12).all())}")
print(f"player VI min={np.nanmin(v.player_vi):.6f} non-negative: "
      f"{bool((np.nanmin(v.player_vi) >= -1e-12))}")
sums = np.nansum(v.team_vi[ok], axis=1)
print(f"home + away == total VI: max|diff| = {np.max(np.abs(sums - v.total_vi[ok])):.2e}")

print("\n--- anchor 3: the two teams are no longer collinear ---")
h = v.team_vi[ok, 0]
a = v.team_vi[ok, 1]
print(f"corr(home VI, away VI) = {np.corrcoef(h, a)[0,1]:+.3f}   "
      "(old Voronoi VI gave -1.000 exactly)")
print(f"corr(home VI_t, home VI_t+1) = {np.corrcoef(h[:-1], h[1:])[0,1]:+.3f}")
print(f"corr(away VI_t, away VI_t+1) = {np.corrcoef(a[:-1], a[1:])[0,1]:+.3f}")

print("\n--- anchor 4: substitutions / 5-min segments ---")
subs = v.substitutions()
print(f"substitution events = {len(subs)} (target ~7)")
n_p2 = int(((v.period == 2) & v.valid).sum())
print(f"second-half VALID seconds = {n_p2}; complete 5-min segments = {n_p2 // 300}")

print("\n--- identity: scalar closed form == general Gaussian formula ---")
rng = np.random.default_rng(0)
a1 = rng.normal(size=500)
b1 = 0.6 * a1 + rng.normal(size=500)
print(f"  closed form={mi_scalar(a1, b1):.8f}  "
      f"general={gaussian_mi(a1[:, None], b1[:, None]):.8f}")
w = windowed_mi(a1[:, None], b1, 60)[:, 0]
print(f"  windowed: n={w.size} mean={w.mean():.4f} global={mi_scalar(a1, b1):.4f}")
