"""Run Task 1 and Task 2 for every match that is present locally, then pool.

    python scripts/run_all.py                 # all matches under data/raw
    python scripts/run_all.py --only 1886347 1899585

Only matches with the four raw files present are processed; the rest are
skipped so the script can be re-run while downloads are still in progress.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = sys.executable
NEEDED = ["{}_match.json", "{}_tracking_extrapolated.jsonl",
          "{}_dynamic_events.csv"]


def complete(mid: str, raw: Path) -> bool:
    d = raw / mid
    return all((d / p.format(mid)).exists() for p in NEEDED)


def run(script: str, *args: str) -> int:
    cmd = [PY, str(ROOT / "scripts" / script), *args]
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    ids = args.only or sorted(p.name for p in raw.iterdir()
                              if p.is_dir() and p.name.isdigit())
    ready = [m for m in ids if complete(m, raw)]
    print(f"matches present and complete: {ready}")
    if not ready:
        print("nothing to do")
        return 1

    for mid in ready:
        if run("task1_vi_ce.py", "--match-id", mid) != 0:
            print(f"task1 failed for {mid}")
        if run("task2_tmle.py", "--match-id", mid) != 0:
            print(f"task2 failed for {mid}")

    if len(ready) > 1:
        run("task2_pool.py")
    else:
        print("\nonly one match: skipping the pooled TMLE (spec §6 needs ~10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
