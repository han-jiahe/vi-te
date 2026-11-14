"""vi_bridge 单场 CLI。

用法（在 D:/MarkovSoccerTE 下）：
    python -m vi_bridge.run_vi_te --match-id 1886347 \
        --raw-dir D:/MarkovSoccerTE/vi/data/raw \
        --out D:/MarkovSoccerTE/vi_bridge/results/vi_te_1886347.json \
        --n-surrogates 200 --block-len 25 --k 3 5 --n-clusters 3

冒烟（单半场、减代理）：
    python -m vi_bridge.run_vi_te --match-id 1886347 --halves home|1 \
        --n-surrogates 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from vi_bridge.vi_te import ViTeMatch


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--raw-dir", default=str(Path(r"D:/MarkovSoccerTE/vi/data/raw")))
    ap.add_argument("--out", default=None,
                    help="输出 JSON（默认 vi_bridge/results/vi_te_<id>.json）")
    ap.add_argument("--n-surrogates", type=int, default=200)
    ap.add_argument("--block-len", type=int, default=25)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--halves", default=None,
                    help="只算指定半场，逗号分隔如 home|1,away|2（冒烟用）")
    ap.add_argument("--methods", default=None,
                    help="只算指定估计器，逗号分隔如 gaussian,ksg,discrete")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    halves = None
    if args.halves:
        halves = {h.strip() for h in args.halves.split(",") if h.strip()}
    methods = None
    if args.methods:
        methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "vi_bridge" / "results"
        / f"vi_te_{args.match_id}.json")

    t0 = time.time()
    vtm = ViTeMatch(
        args.match_id, args.raw_dir,
        n_surrogates=args.n_surrogates, block_len=args.block_len,
        ks=args.k, n_clusters=args.n_clusters, seed=args.seed,
        halves=halves, methods=methods, verbose=not args.quiet)
    doc = vtm.run_to_json(out)
    print(f"total {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
