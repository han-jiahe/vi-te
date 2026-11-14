"""一次性修复：vi_te.py 早期版本 _skip_row 把节点索引当 pid 存入跳过行。

对每个 vi_te_<id>.json：跳过行的 pid 是节点索引（< 球员 id 量级），
真实 pid = tracking_cache_v2.npz 的 player_ids[节点索引]。就地改写并断言。
用法：python -m vi_bridge.fix_skipped_pids
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TE_ROOT))

RES_DIR = TE_ROOT / "vi_bridge" / "results"
VI_RAW = TE_ROOT / "vi" / "data" / "raw"
MATCHES = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]


def main() -> int:
    n_fixed = 0
    for mid in MATCHES:
        p = RES_DIR / f"vi_te_{mid}.json"
        if not p.exists():
            print(f"缺文件 {p.name}，跳过")
            continue
        cache = VI_RAW / mid / f"{mid}_tracking_cache_v2.npz"
        player_ids = np.load(cache)["player_ids"].astype(np.int64)
        doc = json.loads(p.read_text(encoding="utf-8"))
        changed = 0
        for hkey, h in doc.get("halves", {}).items():
            for lab, rows in h.get("pairs", {}).items():
                for r in rows:
                    if not r.get("skipped"):
                        continue
                    old = int(r["pid"])
                    if old >= len(player_ids):
                        print(f"[ERR] {mid} {hkey} {lab} pid={old} 超出"
                              f" player_ids 长度 {len(player_ids)}")
                        return 1
                    r["pid"] = int(player_ids[old])
                    changed += 1
        if changed:
            p.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        n_fixed += changed
        print(f"{mid}: 修正 {changed} 条跳过行 pid")
    print(f"合计 {n_fixed} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
