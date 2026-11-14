"""Unattended chain: wait for the downloader, then run the whole analysis.

    python scripts/auto_pipeline.py --wait-pid 1848

Stages
------
1. wait    block until the fetch process exits (or ``output/fetch.log``
           contains "done:"), then pause a moment to let files flush
2. check   list which matches have all the files they need; skip the rest
3. analyse ``scripts/run_all.py``  (Task 1 + Task 2 per match + pooled TMLE)
4. report  ``scripts/build_report_docx.py``  (rebuild the Word report)

Everything is appended to ``output/pipeline.log``.  A sentinel at
``data/raw/.stop_pipeline`` aborts between stages.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "output" / "pipeline.log"
STOP = ROOT / "data" / "raw" / ".stop_pipeline"
NEEDED = ("{}_match.json", "{}_tracking_extrapolated.jsonl",
          "{}_dynamic_events.csv")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        return str(pid) in out
    except Exception:                                          # noqa: BLE001
        return False


def fetch_done() -> bool:
    f = ROOT / "output" / "fetch.log"
    if not f.exists():
        return False
    try:
        return "done:" in f.read_text(encoding="utf-8", errors="ignore")
    except Exception:                                          # noqa: BLE001
        return False


def run(script: str, *args: str) -> int:
    cmd = [PY, str(ROOT / "scripts" / script), *args]
    log("$ " + " ".join(cmd[1:]))
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log("    " + line)
    proc.wait()
    log(f"    -> exit {proc.returncode}")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=14.0)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    log("=" * 70)
    log(f"auto_pipeline start (waiting for pid={args.wait_pid})")
    deadline = time.time() + args.max_hours * 3600

    while time.time() < deadline:
        if STOP.exists():
            log("stop sentinel present, aborting")
            return 1
        if fetch_done():
            log("fetch log reports done")
            break
        if args.wait_pid and not pid_alive(args.wait_pid):
            log("fetch process is gone")
            break
        time.sleep(args.poll)
    else:
        log("timed out waiting for the downloader")
        return 1
    time.sleep(10)

    raw = ROOT / "data" / "raw"
    usable, partial = [], []
    for d in sorted(p for p in raw.iterdir() if p.is_dir() and p.name.isdigit()):
        if all((d / n.format(d.name)).exists() for n in NEEDED):
            usable.append(d.name)
        else:
            partial.append(d.name)
    log(f"usable matches ({len(usable)}): {usable}")
    if partial:
        log(f"incomplete, skipped ({len(partial)}): {partial}")
    if not usable:
        log("nothing to analyse")
        return 1

    if STOP.exists():
        log("stop sentinel present, aborting before analysis")
        return 1
    rc = run("run_all.py")
    if rc != 0:
        log(f"run_all.py returned {rc}; building the report anyway")

    if STOP.exists():
        log("stop sentinel present, aborting before the report")
        return 1
    run("build_report_docx.py")

    log("auto_pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
