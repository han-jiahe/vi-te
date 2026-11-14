"""vi_bridge 10 场并行运行器。

用法：
    python -m vi_bridge.run_all --workers 5
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TE_ROOT))

MATCHES = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]
RAW_DIR = TE_ROOT / "vi" / "data" / "raw"
OUT_DIR = TE_ROOT / "vi_bridge" / "results"


def run_one(match_id: str, n_surrogates: int, block_len: int, ks, n_clusters,
            seed: int) -> dict:
    """Worker 进程入口（模块级函数，可 pickle）。"""
    from vi_bridge.vi_te import ViTeMatch
    out = OUT_DIR / f"vi_te_{match_id}.json"
    t0 = time.time()
    try:
        vtm = ViTeMatch(match_id, RAW_DIR, n_surrogates=n_surrogates,
                        block_len=block_len, ks=ks, n_clusters=n_clusters,
                        seed=seed, verbose=True)
        vtm.run_to_json(out)
        return dict(match_id=match_id, status="ok",
                    t_s=round(time.time() - t0, 1))
    except Exception as e:                                  # noqa: BLE001
        return dict(match_id=match_id, status="error", error=str(e),
                    trace=traceback.format_exc(limit=3),
                    t_s=round(time.time() - t0, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--matches", default=None,
                    help="逗号分隔的场次覆盖（默认全部 10 场）")
    ap.add_argument("--n-surrogates", type=int, default=200)
    ap.add_argument("--block-len", type=int, default=25)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    matches = ([m.strip() for m in args.matches.split(",") if m.strip()]
               if args.matches else MATCHES)
    missing = [m for m in matches if not (RAW_DIR / m / f"{m}_match.json").exists()]
    if missing:
        print(f"WARNING missing raw data for: {missing}", flush=True)
    matches = [m for m in matches if m not in missing]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_one, m, args.n_surrogates, args.block_len,
                             args.k, args.n_clusters, args.seed)
                   for m in matches]
        for fut in futures:
            r = fut.result()
            tag = "OK " if r["status"] == "ok" else "ERR"
            print(f"[{tag}] {r['match_id']} {r.get('t_s', '')}s "
                  f"{r.get('error', '')}", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
