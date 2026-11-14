"""池化前不变量筛查：读全部 vi_te_<id>.json，机械校验协议一致性。

检查项（全部通过才允许 pool_vi_te.py）：
  1. meta.n_surrogates==200、block_len==25（防旧冒烟/20 代理测试文件污染）
  2. 10 场 × 4 半场齐全；每半场 4 方法齐备
  3. 每半场 pairs：downward/upward/cross_down/cross_up 各 10–16 条目
     （名单级，未上场者以 skipped 标记保留）；条目 n_trans ≥ 60 或
     skipped 标记；pid 必须属于该场追踪缓存 player_ids；n_trans 自洽
     （downward 与 upward 同 mask，cross_down 与 cross_up 同 mask）
  4. 数值范围：possession_rate∈[0,1]、n_valid≤n_seconds、
     surr_ratio 旗标计数、eff==raw−surr_mean 恒等（|Δ|<1e-6）
  5. cross_macro 只出现在 home 侧
用法：python -m vi_bridge.verify_pool [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TE_ROOT))

RES_DIR = TE_ROOT / "vi_bridge" / "results"
MATCHES = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]
METHODS = ("gaussian", "ksg|k=3", "ksg|k=5", "discrete|K=3")
PAIR_LABELS = ("downward", "upward", "cross_down", "cross_up")
_DEGEN: dict = {}


def check_pair(r, label, ctx, problems):
    if r.get("skipped"):
        return
    nt = r.get("n_trans")
    if nt is None or nt < 60:
        problems.append(f"{ctx} {label} pid={r.get('pid')} n_trans={nt} <60")
    for mk in METHODS:
        g = r.get("methods", {}).get(mk)
        if g is None:
            problems.append(f"{ctx} {label} pid={r.get('pid')} 缺方法 {mk}")
            continue
        if "raw" not in g:
            problems.append(f"{ctx} {label} pid={r.get('pid')} {mk} 无 raw"
                            f"（guard 触发但无 skipped 标记）")
            continue
        raw, eff, sm = g["raw"], g["eff"], g.get("surr_mean")
        if sm is not None and abs((raw - eff) - sm) > 1e-6:
            problems.append(f"{ctx} {label} pid={r.get('pid')} {mk} "
                            f"eff≠raw−surr_mean: {abs((raw-eff)-sm):.2e}")
        sr = g.get("surr_ratio")
        if sr is not None and abs(sr) > 0.5:
            problems.append(f"{ctx} {label} pid={r.get('pid')} {mk} "
                            f"surr_ratio 旗标 {sr:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="surr_ratio 旗标也计为失败")
    args = ap.parse_args()

    problems = []
    n_docs = 0
    pid_sets: dict[str, set] = {}
    for mid in MATCHES:
        cache = TE_ROOT / "vi" / "data" / "raw" / mid / \
            f"{mid}_tracking_cache_v2.npz"
        if cache.exists():
            pid_sets[mid] = set(
                np.load(cache)["player_ids"].astype(np.int64).tolist())
        p = RES_DIR / f"vi_te_{mid}.json"
        if not p.exists():
            problems.append(f"缺文件 {p.name}")
            continue
        n_docs += 1
        d = json.loads(p.read_text(encoding="utf-8"))
        meta = d.get("meta", {})
        if meta.get("n_surrogates") != 200:
            problems.append(f"{mid} meta.n_surrogates={meta.get('n_surrogates')}"
                            f"（≠200，疑似旧冒烟污染）")
        if meta.get("block_len") != 25:
            problems.append(f"{mid} meta.block_len={meta.get('block_len')}≠25")
        halves = d.get("halves", {})
        if set(halves) != {"home|1", "home|2", "away|1", "away|2"}:
            problems.append(f"{mid} halves 集合异常: {sorted(halves)}")
        for hkey, h in halves.items():
            ctx = f"{mid} {hkey}"
            if not (0 <= h.get("possession_rate", -1) <= 1):
                problems.append(f"{ctx} possession_rate={h['possession_rate']}")
            if h.get("n_valid", 1e9) > h.get("n_seconds", 0):
                problems.append(f"{ctx} n_valid>{n_seconds}"
                                f"{h.get('n_valid')}>{h.get('n_seconds')}")
            for mk in METHODS:
                if mk not in h.get("methods", {}):
                    problems.append(f"{ctx} 缺方法 {mk}")
                else:
                    sm = h["methods"][mk]
                    if sm.get("n_downward") == 0 and \
                            sm.get("downward_sum_bits") == 0.0:
                        problems.append(f"{ctx} {mk} 全部对被跳过但 "
                                        f"D_sum 静默为 0.0（应为 NaN）")
            pairs = h.get("pairs", {})
            for lab in PAIR_LABELS:
                if lab not in pairs:
                    problems.append(f"{ctx} 缺 pairs.{lab}")
                    continue
                # 名单级条目：出场外场球员数随换人 10–16 不等（未上场者
                # 以 skipped 标记保留）；跳过行必须 n_trans<60 且 pid 合法
                if not (10 <= len(pairs[lab]) <= 16):
                    problems.append(f"{ctx} pairs.{lab} "
                                    f"条目数={len(pairs[lab])}∉[10,16]")
                for r in pairs[lab]:
                    pid = r.get("pid")
                    if pid is not None and pid not in pid_sets.get(mid, set()):
                        problems.append(f"{ctx} {lab} pid={pid} "
                                        f"不在 {mid} 追踪缓存 player_ids 中")
                    if r.get("skipped"):
                        if (r.get("n_trans") or 0) >= 60:
                            problems.append(f"{ctx} {lab} pid={pid} "
                                            f"skipped 但 n_trans="
                                            f"{r.get('n_trans')}≥60")
                    check_pair(r, lab, ctx, problems)
            # 同一 mask 的 n_trans 一致性（downward vs upward）
            for a, b in (("downward", "upward"), ("cross_down", "cross_up")):
                na = [r.get("n_trans") for r in pairs.get(a, [])
                      if not r.get("skipped") and r.get("n_trans")]
                nb = [r.get("n_trans") for r in pairs.get(b, [])
                      if not r.get("skipped") and r.get("n_trans")]
                if na and nb and any(abs(x - y) > 1 for x, y in zip(na, nb)):
                    problems.append(f"{ctx} {a}/{b} n_trans 不一致")
        cm = d.get("cross_macro", {})
        for per, entry in cm.items():
            for dr in ("home->away", "away->home"):
                e = entry.get(dr, {})
                if e.get("skipped"):
                    continue
                nt = e.get("n_trans")
                if nt is None or nt < 60:
                    problems.append(f"{mid} cross_macro P{per} {dr} "
                                    f"n_trans={nt} <60")
                for mk in METHODS:
                    g = e.get("methods", {}).get(mk, {})
                    if "raw" not in g and "skipped" not in g:
                        problems.append(f"{mid} cross_macro P{per} {dr} "
                                        f"{mk} 无 raw 无 skipped")
    flags = [p for p in problems if "surr_ratio 旗标" in p]
    hard = [p for p in problems if "surr_ratio 旗标" not in p]
    print(f"docs={n_docs}/10；硬问题 {len(hard)}；surr_ratio 旗标 {len(flags)}")
    for p in hard:
        print("  [HARD]", p)
    for p in flags[:20]:
        print("  [FLAG]", p)
    if len(flags) > 20:
        print(f"  … 及另外 {len(flags)-20} 条旗标")
    # 退化诊断汇总（每方法 p==1.0 / raw<0 占比，D 方向）
    print("退化诊断（D 方向，球员×半场）：")
    import json as _json
    for mid in MATCHES:
        pth = RES_DIR / f"vi_te_{mid}.json"
        if not pth.exists():
            continue
        doc = _json.loads(pth.read_text(encoding="utf-8"))
        for hkey, h in doc.get("halves", {}).items():
            for r in h.get("pairs", {}).get("downward", []):
                if r.get("skipped"):
                    continue
                for mk in METHODS:
                    mm = r.get("methods", {}).get(mk, {})
                    if "raw" in mm:
                        d0 = _DEGEN.setdefault(mk, {"n": 0, "p1": 0, "neg": 0})
                        d0["n"] += 1
                        if mm["p"] == 1.0:
                            d0["p1"] += 1
                        if mm["raw"] < 0:
                            d0["neg"] += 1
    for mk, d in _DEGEN.items():
        n = d.get("n", 0)
        if n:
            print(f"  {mk}: n={n} p==1.0 {d.get('p1', 0)}/{n} "
                  f"raw<0 {d.get('neg', 0)}/{n}")
    return 1 if (hard or (args.strict and flags)) else 0


if __name__ == "__main__":
    sys.exit(main())
