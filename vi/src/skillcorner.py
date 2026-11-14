"""Reader for SkillCorner open tracking data (broadcast tracking, 10 fps).

The raw files are the ones published in https://github.com/SkillCorner/opendata:

    {id}_match.json                   lineups, roles, playing time, pitch size
    {id}_tracking_extrapolated.jsonl  one JSON object per broadcast frame
    {id}_dynamic_events.csv           Game-Intelligence dynamic events
    {id}_phases_of_play.csv           phases of play

Parsing 90 MB of JSON per match is slow, so the parsed arrays are cached as an
``.npz`` next to the raw data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS_RAW = 10.0


@dataclass
class TeamInfo:
    side: str               # "home" / "away"
    team_id: int
    name: str
    short_name: str
    direction_by_period: dict[int, str]


class Match:
    """Parsed tracking data for one match, cached to ``cache_dir``."""

    def __init__(self, match_id: str, raw_dir: Path, cache_dir: Path | None = None):
        self.match_id = str(match_id)
        self.raw_dir = Path(raw_dir)
        cache_dir = Path(cache_dir or self.raw_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"{self.match_id}_tracking_cache_v2.npz"

        meta_path = self.raw_dir / f"{self.match_id}_match.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if cache.exists():
            self._load_cache(cache)
        else:
            self._parse_jsonl()
            self._save_cache(cache)

        self._build_player_table()
        self._build_periods()

    # ------------------------------------------------------------------ parse
    def _parse_jsonl(self) -> None:
        path = self.raw_dir / f"{self.match_id}_tracking_extrapolated.jsonl"
        player_ids: list[int] = []
        pid_index: dict[int, int] = {}
        frames: list[int] = []
        periods: list[int] = []
        ball: list[tuple[float, float]] = []
        poss_pid: list[int] = []
        poss_group: list[str] = []
        rows: list[np.ndarray] = []
        det_rows: list[np.ndarray] = []

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:      # truncated tail of a partial file
                    break
                pd = d.get("player_data") or []
                if not pd:
                    continue
                new = [p for p in pd
                       if p.get("player_id") is not None and p["player_id"] not in pid_index]
                for p in new:
                    pid_index[p["player_id"]] = len(player_ids)
                    player_ids.append(p["player_id"])
                xy = np.full((len(player_ids), 2), np.nan, dtype=np.float32)
                det = np.zeros(len(player_ids), dtype=bool)
                for p in pd:
                    j = pid_index.get(p.get("player_id"))
                    if j is None:
                        continue
                    xy[j, 0] = p["x"] if p["x"] is not None else np.nan
                    xy[j, 1] = p["y"] if p["y"] is not None else np.nan
                    det[j] = bool(p.get("is_detected"))
                frames.append(d["frame"])
                periods.append(d.get("period") or 0)
                b = d.get("ball_data") or {}
                ball.append((b.get("x") if b.get("x") is not None else np.nan,
                             b.get("y") if b.get("y") is not None else np.nan))
                ps = d.get("possession") or {}
                poss_pid.append(ps.get("player_id") or -1)
                grp = (ps.get("group") or "").replace(" team", "")
                poss_group.append(grp)
                rows.append(xy)
                det_rows.append(det)

        n = len(frames)
        self.player_ids = np.asarray(player_ids, dtype=np.int64)
        self.frame = np.asarray(frames, dtype=np.int64)
        self.period = np.asarray(periods, dtype=np.int8)
        self.ball = np.asarray(ball, dtype=np.float32).reshape(n, 2)
        self.poss_player_id = np.asarray(poss_pid, dtype=np.int64)
        self.poss_group = np.asarray(poss_group, dtype=object)
        # players discovered after the first frame widen later rows only
        width = len(player_ids)
        track = np.full((n, width, 2), np.nan, dtype=np.float32)
        det = np.zeros((n, width), dtype=bool)
        for i, row in enumerate(rows):
            track[i, :row.shape[0]] = row
            det[i, :det_rows[i].shape[0]] = det_rows[i]
        self.track_xy = track
        self.track_detected = det

    def _save_cache(self, path: Path) -> None:
        np.savez_compressed(
            path,
            player_ids=self.player_ids,
            frame=self.frame,
            period=self.period,
            ball=self.ball,
            poss_player_id=self.poss_player_id,
            poss_group=self.poss_group.astype(str),
            track_xy=self.track_xy,
            track_detected=self.track_detected,
        )

    def _load_cache(self, path: Path) -> None:
        z = np.load(path, allow_pickle=False)
        self.player_ids = z["player_ids"]
        self.frame = z["frame"]
        self.period = z["period"]
        self.ball = z["ball"]
        self.poss_player_id = z["poss_player_id"]
        self.poss_group = z["poss_group"].astype(object)
        self.track_xy = z["track_xy"]
        self.track_detected = z["track_detected"]

    # ----------------------------------------------------------------- tables
    def _build_player_table(self) -> None:
        m = self.meta
        self.teams = {
            "home": TeamInfo("home", m["home_team"]["id"], m["home_team"]["name"],
                             m["home_team"]["short_name"], {}),
            "away": TeamInfo("away", m["away_team"]["id"], m["away_team"]["name"],
                             m["away_team"]["short_name"], {}),
        }
        sides = m.get("home_team_side") or []
        for i, d in enumerate(sides):
            self.teams["home"].direction_by_period[i + 1] = d
            self.teams["away"].direction_by_period[i + 1] = (
                "left_to_right" if d == "right_to_left" else "right_to_left")

        n_players = len(self.player_ids)
        pidx = {int(pid): i for i, pid in enumerate(self.player_ids)}
        self.p_index = pidx
        self.p_side = np.array([""] * n_players, dtype=object)
        self.p_number = np.full(n_players, -1, dtype=np.int64)
        self.p_is_gk = np.zeros(n_players, dtype=bool)
        self.p_short = np.array([""] * n_players, dtype=object)

        # playing_time[(player_index, period)] = (start_frame, end_frame)
        self.playing = {}
        side_of_team = {t.team_id: t.side for t in self.teams.values()}
        for p in m["players"]:
            # the tracking file identifies players by `trackable_object`
            pi = None
            for k in ("trackable_object", "id", "team_player_id"):
                if p.get(k) is not None and p[k] in pidx:
                    pi = pidx[p[k]]
                    break
            if pi is None:
                continue
            self.p_side[pi] = side_of_team.get(p["team_id"], "")
            self.p_number[pi] = p["number"]
            self.p_short[pi] = p.get("short_name") or ""
            role = (p.get("player_role") or {}).get("name") or ""
            self.p_is_gk[pi] = role == "Goalkeeper"
            for bp in (p.get("playing_time") or {}).get("by_period") or []:
                per = int(str(bp["name"]).split("_")[-1])
                self.playing[(pi, per)] = (bp["start_frame"], bp["end_frame"])

        self.pitch_length = float(m.get("pitch_length") or 105)
        self.pitch_width = float(m.get("pitch_width") or 68)

    def _build_periods(self) -> None:
        self.period_bounds = {}
        for p in self.meta["match_periods"]:
            self.period_bounds[p["period"]] = (p["start_frame"], p["end_frame"])

    # ---------------------------------------------------------------- resample
    def resample_1hz(self) -> dict:
        """Down-sample the 10 fps tracking to 1 Hz on an integer-second grid.

        Fields are grouped per period by ``floor((frame - period_start)/10)`` and
        represented by the frame closest to the exact second boundary.
        """
        out = {"second": [], "period": [], "frame": [], "row": [], "ball": [],
               "poss_group": [], "poss_player_id": []}
        mask = np.isin(self.period, [1, 2])
        for per in (1, 2):
            start, end = self.period_bounds.get(per, (None, None))
            if start is None:
                continue
            sel = np.where(mask & (self.period == per))[0]
            if sel.size == 0:
                continue
            frames = self.frame[sel]
            k = np.floor((frames - start) / FPS_RAW).astype(np.int64)
            # pick, per second index, the frame nearest to start + 10*k
            target = start + FPS_RAW * k
            order = np.lexsort((np.abs(frames - target), k))
            sel_sorted = sel[order]
            k_sorted = k[order]
            first = np.ones(k_sorted.size, dtype=bool)
            first[1:] = k_sorted[1:] != k_sorted[:-1]
            picked = sel_sorted[first]
            kp = k_sorted[first]
            out["second"].append(kp)
            out["period"].append(np.full(kp.size, per, dtype=np.int8))
            out["frame"].append(self.frame[picked])
            out["row"].append(picked)
            out["ball"].append(self.ball[picked])
            out["poss_group"].append(self.poss_group[picked])
            out["poss_player_id"].append(self.poss_player_id[picked])

        res = {k: np.concatenate(v) for k, v in out.items()}
        return res
