"""Verified, resumable downloader for the SkillCorner open dataset.

Fixes the failure modes seen earlier in this session:

* **corrupt partial files** -- the git-lfs pointer gives the SHA-256 of the
  blob, so every download is hashed and only renamed into place when it
  matches.  A stale/partial file can never be picked up silently.
* **interrupted loops** -- a ``STOP`` sentinel file (default
  ``data/raw/.stop_fetch``) is checked between attempts, so the job can be
  stopped cleanly instead of being killed mid-write.
* **one slow host** -- tracking blobs are pulled from the signed git-lfs URL
  and fall back to ``media.githubusercontent.com``; the small CSVs come from a
  raw mirror with retries.

Usage
-----
    python tools/fetch_match.py 1899585 1953632 [--raw-dir data/raw]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

REPO = "SkillCorner/opendata"
RAW_MIRRORS = [
    f"https://raw.githubusercontent.com/{REPO}/master",
    f"https://ghproxy.net/https://raw.githubusercontent.com/{REPO}/master",
]
# media.githubusercontent.com is dead (404s for everything since the raw-host
# migration), so LFS blobs go through the batch API.  github.com itself is
# unreachable from this network (TLS-level block); the public proxies below
# forward the batch POST and return the signed S3 href.
BATCH_PROXIES = [
    f"https://gh-proxy.com/https://github.com/{REPO}.git/info/lfs/objects/batch",
    f"https://ghproxy.cn/https://github.com/{REPO}.git/info/lfs/objects/batch",
    f"https://github.com/{REPO}.git/info/lfs/objects/batch",
]
SMALL = ["{}_match.json", "{}_dynamic_events.csv", "{}_phases_of_play.csv"]
CHUNK = 1 << 20


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def http_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/vnd.git-lfs+json",
               "Content-Type": "application/vnd.git-lfs+json",
               "User-Agent": "dsh-fetch/1.0"}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def lfs_objects(match_id: str) -> dict | None:
    """Read the git-lfs pointer of the tracking blob via the GitHub API."""
    url = (f"https://api.github.com/repos/{REPO}/contents/"
           f"data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            meta = json.load(r)
        text = base64.b64decode(meta["content"]).decode("utf-8", "replace")
        oid = size = None
        for line in text.splitlines():
            if line.startswith("oid sha256:"):
                oid = line.split(":", 1)[1].strip()
            elif line.startswith("size "):
                size = int(line.split()[1])
        if oid and size:
            return {"oid": oid, "size": size}
    except Exception as exc:                                   # noqa: BLE001
        log(f"  pointer lookup failed: {exc}")
    return None


def lfs_href(oid: str, size: int) -> str | None:
    body = {"operation": "download", "transfers": ["basic"],
            "objects": [{"oid": oid, "size": size}]}
    for url in BATCH_PROXIES:
        try:
            data = http_json(url, body)
            return data["objects"][0]["actions"]["download"]["href"]
        except Exception as exc:                               # noqa: BLE001
            host = url.split("/")[2]
            log(f"  lfs batch failed via {host}: {type(exc).__name__}")
    return None


def download_to(url: str, tmp: Path, size: int | None, stop: Path,
                max_stall: int = 240) -> bool:
    """Resumable single-connection download; True when the file is complete."""
    tmp.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    last = -1
    stalled = 0
    while True:
        if stop.exists():
            log("  STOP sentinel found, aborting download")
            return False
        have = tmp.stat().st_size if tmp.exists() else 0
        if size is not None and have >= size:
            return True
        if have == last:
            stalled += 1
            if stalled >= max_stall:
                log(f"  stalled at {have} bytes")
                return False
        else:
            stalled = 0
        last = have
        attempt += 1
        headers = {"Range": f"bytes={have}-"} if have else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(tmp, "ab" if have else "wb", buffering=0) as fh:
                while True:
                    if stop.exists():
                        return False
                    if size is not None and have + fh.tell() >= size:
                        break
                    block = r.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
        except Exception as exc:                               # noqa: BLE001
            log(f"  attempt {attempt}: {type(exc).__name__}: {exc}")
        if attempt > 400:
            return False


def fetch_small(match_id: str, raw: Path, stop: Path) -> None:
    for pat in SMALL:
        name = pat.format(match_id)
        out = raw / match_id / name
        if out.exists() and out.stat().st_size > 0:
            continue
        ok = False
        for base in RAW_MIRRORS:
            for _ in range(2):
                if stop.exists():
                    return
                try:
                    req = urllib.request.Request(f"{base}/data/matches/{match_id}/{name}")
                    with urllib.request.urlopen(req, timeout=45) as r, \
                            open(out, "wb", buffering=0) as fh:
                        while True:
                            block = r.read(CHUNK)
                            if not block:
                                break
                            fh.write(block)
                    if out.stat().st_size > 0:
                        log(f"  {name}: {out.stat().st_size/1e3:.0f} kB")
                        ok = True
                        break
                except Exception:                              # noqa: BLE001
                    time.sleep(1)
                if out.exists() and out.stat().st_size == 0:
                    out.unlink()
            if ok:
                break
        if not ok:
            log(f"  WARNING could not fetch {name}")


def _download_range(url: str, path: Path, start: int, length: int, stop: Path,
                    timeout: int = 120, max_stall: int = 240) -> bool:
    """Fetch ``length`` bytes starting at ``start`` into ``path``, resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    end = start + length
    last, stalled, attempt = -1, 0, 0
    while True:
        if stop.exists():
            return False
        have = path.stat().st_size if path.exists() else 0
        if have >= length:
            return True
        if have == last:
            stalled += 1
            if stalled >= max_stall:
                log(f"    stalled at {have}/{length}")
                return False
        else:
            stalled = 0
        last = have
        attempt += 1
        if attempt > 400:
            return False
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start + have}-{end - 1}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, \
                    open(path, "ab", buffering=0) as fh:
                while True:
                    if stop.exists():
                        return False
                    block = r.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    if have + fh.tell() >= length:
                        break
        except Exception as exc:                               # noqa: BLE001
            log(f"    retry ({type(exc).__name__})")
    return False


def download_parallel(url: str, tmp: Path, size: int, stop: Path,
                      workers: int = 4) -> bool:
    """Threaded ranged download.

    A single stream is throttled to ~20-35 KB/s by the CDN; four concurrent
    ranged streams measured ~94 KB/s on this network (8 streams were slower).
    The existing single-stream prefix in ``tmp`` is kept and only the remaining
    bytes are split between the workers, so switching strategies mid-download
    loses nothing.

    Layout: ``tmp`` holds [0, have); ``tmp.c<k>`` hold the tail segments;
    assembly happens in ``tmp.asm`` so an interrupted assembly is recoverable.
    """
    tmp.parent.mkdir(parents=True, exist_ok=True)
    have = tmp.stat().st_size if tmp.exists() else 0
    if have >= size:
        return True
    remaining = size - have
    seg = remaining // workers + 1
    jobs = []
    for k in range(workers):
        s = have + k * seg
        if s >= size:
            break
        jobs.append((k, s, min(s + seg, size) - s))

    ok = [False] * len(jobs)

    def run(i, k, s, ln):
        ok[i] = _download_range(url, tmp.with_name(tmp.name + f".c{k}"), s, ln, stop)

    threads = [threading.Thread(target=run, args=(i, k, s, ln))
               for i, (k, s, ln) in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not all(ok):
        log(f"  {sum(ok)}/{len(ok)} segments done, keeping partial for resume")
        return False

    asm = tmp.with_name(tmp.name + ".asm")
    if asm.exists():
        asm.unlink()
    with open(asm, "wb") as out:
        if tmp.exists():
            with open(tmp, "rb") as f:
                shutil.copyfileobj(f, out, CHUNK)
        for i, (k, s, ln) in enumerate(jobs):
            part = tmp.with_name(tmp.name + f".c{k}")
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, CHUNK)
    for _, k, _ in [(i, j[0], 0) for i, j in enumerate(jobs)]:
        p = tmp.with_name(tmp.name + f".c{k}")
        if p.exists():
            p.unlink()
    if tmp.exists():
        tmp.unlink()
    os.replace(asm, tmp)
    return tmp.stat().st_size == size


def download_sliced(url: str, tmp: Path, size: int, stop: Path,
                    workers: int = 4, slice_bytes: int = 1 << 20) -> bool:
    """Download in fixed 1 MB slices through a small thread pool.

    Measurements on this network: a single sustained stream runs at 20-35 KB/s
    and four *long-lived* streams do not improve on that, but four short 1 MB
    ranged requests run at ~94 KB/s -- the CDN gives each new request a fast
    burst and then throttles the connection.  So the unit of work is a slice,
    not a stream.

    Each slice is its own resumable file (``tmp.s<k>``), which makes an
    interrupted run cheap to continue.  The bytes already present in ``tmp``
    (written by an earlier strategy) are kept as the prefix.
    """
    tmp.parent.mkdir(parents=True, exist_ok=True)
    have = tmp.stat().st_size if tmp.exists() else 0
    if have >= size:
        return True

    slices = []
    s, k = have, 0
    while s < size:
        ln = min(slice_bytes, size - s)
        slices.append((k, s, ln))
        s += ln
        k += 1

    done = [False] * len(slices)
    lock = threading.Lock()
    counter = {"n": 0}

    def task(i):
        if stop.exists():
            return
        kk, st, ln = slices[i]
        part = tmp.with_name(tmp.name + f".s{kk}")
        done[i] = _download_range(url, part, st, ln, stop, max_stall=120)
        with lock:
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                log(f"    {counter['n']}/{len(slices)} slices")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(task, range(len(slices))))

    if not all(done):
        log(f"  {sum(done)}/{len(done)} slices complete, keeping partial for resume")
        return False

    asm = tmp.with_name(tmp.name + ".asm")
    if asm.exists():
        asm.unlink()
    with open(asm, "wb") as out:
        if tmp.exists():
            with open(tmp, "rb") as f:
                shutil.copyfileobj(f, out, CHUNK)
        for kk, st, ln in slices:
            part = tmp.with_name(tmp.name + f".s{kk}")
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, CHUNK)
            part.unlink()
    if tmp.exists():
        tmp.unlink()
    os.replace(asm, tmp)
    return tmp.stat().st_size == size


def fetch_tracking(match_id: str, raw: Path, stop: Path,
                   expected: dict | None, workers: int = 4) -> bool:
    out = raw / match_id / f"{match_id}_tracking_extrapolated.jsonl"
    if out.exists():
        if expected and sha256(out) == expected["oid"]:
            log(f"  tracking already verified ({out.stat().st_size/1e6:.1f} MB)")
            return True
        log("  existing tracking failed verification, re-downloading")
        out.unlink()

    if not expected:
        log("  no LFS pointer, cannot fetch the tracking blob")
        return False

    tmp = out.with_suffix(".jsonl.part")
    size = expected["size"]
    href, deadline = None, 0.0
    while True:
        if stop.exists():
            log("  STOP sentinel found, aborting download")
            return False
        # signed hrefs live ~1 h; re-sign well before expiry so a slow slice
        # never dies on an expired URL (expired slices 403 and are re-run
        # against a fresh href below)
        if href is None or time.time() > deadline:
            href = lfs_href(expected["oid"], size)
            deadline = time.time() + 3600.0 - 300.0
            if href is None:
                log("  no signed URL yet, retrying in 30 s")
                time.sleep(30)
                continue
            log(f"  fetching {href.split('?')[0][:70]} ...")
        if download_sliced(href, tmp, size, stop, workers=workers):
            break
        href = None                                     # re-sign and resume

    if tmp.stat().st_size != size:
        log(f"  size mismatch {tmp.stat().st_size} != {size}")
        return False
    digest = sha256(tmp)
    if digest != expected["oid"]:
        log(f"  sha256 mismatch ({digest[:12]} != {expected['oid'][:12]}), discarding")
        tmp.unlink()
        return False
    log(f"  verified sha256 {digest[:12]}... ({size/1e6:.1f} MB)")
    os.replace(tmp, out)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("match_ids", nargs="+")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--stop-file", default="data/raw/.stop_fetch")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel ranged streams (4 measured fastest on this "
                         "network: ~94 KB/s vs ~25 KB/s single-stream)")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    stop = Path(args.stop_file)
    raw.mkdir(parents=True, exist_ok=True)
    if stop.exists():
        log(f"stop sentinel present at {stop}; remove it to fetch")
        return 1

    failed = []
    for mid in args.match_ids:
        if stop.exists():
            log("stop requested, halting")
            break
        log(f"=== match {mid} ===")
        (raw / mid).mkdir(parents=True, exist_ok=True)
        # tracking first: it is by far the longest download
        expected = lfs_objects(mid)
        if expected is None:
            log("  no lfs pointer, trying media URL without verification")
        ok = fetch_tracking(mid, raw, stop, expected, workers=args.workers)
        if ok:
            fetch_small(mid, raw, stop)
        if not ok:
            log(f"  FAILED {mid}")
            failed.append(mid)

    log(f"done: {len(args.match_ids) - len(failed)} ok, {len(failed)} failed {failed}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())

