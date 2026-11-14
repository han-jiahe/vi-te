"""1 Hz pitch-state table for one SkillCorner match, with per-player Voronoi VI.

Implements 实现方案_VI因果涌现与换人TMLE §1 (data, coordinates, down-sampling,
roster) and §2 (VI), so that both Task 1 and Task 2 read the same table.

Conventions (locked by the spec):
  * pitch 104 m x 68 m, corner origin:  X = x + 52,  Y = y + 34
  * second ``t`` of a period starts at ``start_frame + 10 t``; its position is
    the mean over the ``is_detected == True`` frames inside that 1 s window
  * a second is valid only when *both* teams have their full on-pitch roster
    (11 players incl. the goalkeeper) and all of those positions are available
  * VI is computed on the 22 on-pitch players, pitch area 7072 m^2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .skillcorner import Match
from .vi_cluster import (GOAL_AWAY, GOAL_HOME, hierarchical_clustering, vi,
                         vi_per_node)

# pitch geometry (metres, corner origin); the legacy Voronoi module in vi.py is
# kept only as the upstream "add-on" and is no longer used by this pipeline
PITCH_LENGTH = 104.0
PITCH_WIDTH = 68.0

FPS = 10
TEAMS = ("home", "away")


class ViMatch:
    def __init__(self, match_id: str, raw_dir: Path,
                 length: float = PITCH_LENGTH, width: float = PITCH_WIDTH,
                 verbose: bool = True, hz: int = 1):
        self.match = Match(match_id, raw_dir)
        self.match_id = str(match_id)
        self.length = length
        self.width = width
        self.area = length * width
        self.hz = int(hz)
        self._build(verbose)
        if self.hz == 10:
            # paper-native VI rate: lag-1 at 10 fps, downsampled to the 1 Hz
            # grid (overwrites the 1 Hz series built above)
            self._build_vi_10hz()
        self._build_players()

    # ------------------------------------------------------------------ build
    def _build(self, verbose: bool) -> None:
        m = self.match
        n_players = m.player_ids.size
        self.player_ids = m.player_ids
        frame_to_row = np.full(int(m.frame.max()) + 1, -1, dtype=np.int64)
        frame_to_row[m.frame] = np.arange(m.frame.size)

        seconds, periods, positions, detected, usable = [], [], [], [], []
        poss = []
        for per in (1, 2):
            start, end = m.period_bounds[per]
            n_sec = (end - start + 1) // FPS
            for t in range(n_sec):
                f0 = start + FPS * t
                frames = np.arange(f0, f0 + FPS)
                frames = frames[(frames <= end) & (frames < frame_to_row.size)]
                rows = frame_to_row[frames]
                rows = rows[rows >= 0]
                sec_pos = np.full((n_players, 2), np.nan)
                ok = rows.size > 0
                if rows.size:
                    p = m.track_xy[rows]                      # (k, P, 2)
                    d = m.track_detected[rows]
                    finite = np.isfinite(p[:, :, 0]) & np.isfinite(p[:, :, 1])
                    sel = d & finite
                    cnt = sel.sum(axis=0)
                    # mean over detected frames; if a player was never detected
                    # inside the second, fall back to the extrapolated frames
                    fallback = (~(cnt > 0)) & finite.any(axis=0)
                    sums = np.where(sel[:, :, None], p, 0.0).sum(axis=0)
                    denom = np.maximum(cnt, 1)
                    out = sums / denom[:, None]
                    if fallback.any():
                        fin = finite[:, fallback]
                        fb = (np.where(fin[:, :, None], p[:, fallback], 0.0).sum(axis=0)
                              / np.maximum(fin.sum(axis=0), 1)[:, None])
                        out[fallback] = fb
                    sec_pos = np.where((cnt[:, None] > 0) | fallback[:, None], out, np.nan)
                seconds.append(t)
                periods.append(per)
                positions.append(sec_pos)
                detected.append(ok)
                usable.append((f0, f0 + frames.size - 1))
                if rows.size:
                    g = m.poss_group[rows]
                    g = g[(g == "home") | (g == "away")]
                    if g.size:
                        poss.append("home" if (g == "home").sum() >= (g == "away").sum()
                                    else "away")
                    else:
                        poss.append("")
                else:
                    poss.append("")
        self.second = np.asarray(seconds, dtype=np.int64)
        self.period = np.asarray(periods, dtype=np.int8)
        self.poss_group = np.asarray(poss, dtype=object)
        self.sec_pos_raw = np.asarray(positions)          # (S, P, 2), centre origin
        self.sec_has_frames = np.asarray(detected, dtype=bool)
        self.sec_frames = np.asarray(usable, dtype=np.int64)

        # centre origin -> corner origin (spec §1.2)
        self.pos = self.sec_pos_raw.copy()
        self.pos[:, :, 0] = self.pos[:, :, 0] + self.length / 2.0
        self.pos[:, :, 1] = self.pos[:, :, 1] + self.width / 2.0

        # roster per second (spec §1.4)
        self.on_pitch = np.zeros((self.second.size, n_players), dtype=bool)
        for per in (1, 2):
            idx = np.where(self.period == per)[0]
            if idx.size == 0:
                continue
            for pi in range(n_players):
                span = m.playing.get((pi, per))
                if span is None:
                    continue
                s, e = span
                self.on_pitch[idx, pi] = ((self.sec_frames[idx, 1] >= s)
                                          & (self.sec_frames[idx, 0] <= e))

        # validity: the second has tracking, both rosters hold 11 players, and
        # every on-pitch player's position is available
        counts = np.column_stack([
            self.on_pitch[:, [i for i in range(n_players) if m.p_side[i] == s]].sum(axis=1)
            for s in TEAMS
        ])
        finite = np.isfinite(self.pos[:, :, 0])
        has_pos = np.all(~self.on_pitch | finite, axis=1)
        self.valid = ((counts == 11).all(axis=1) & has_pos
                      & self.sec_has_frames)
        if verbose:
            for per in (1, 2):
                sel = self.period == per
                print(f"      P{per}: {sel.sum()} s on the grid, "
                      f"{int((sel & self.valid).sum())} valid")
            print(f"      total valid: {int(self.valid.sum())} s")

        # ---------------------------------------------------------------- VI
        # Paper VI (arXiv:2102.11182): Variation of Information between the
        # nearest-neighbour clustering at t-lag and at t, divided by dt.
        # Nodes = 22 on-pitch players + the two goals (24 when complete).
        # VI_rate(t) = sum_over_nodes VI^p(t) / dt   [bit/s]
        n = self.second.size
        self.player_vi = np.full((n, n_players), np.nan)   # VI rate per player
        self.goal_vi = np.full((n, 2), np.nan)             # VI rate per goal
        self.team_vi = np.full((n, 2), np.nan)             # players + own goal
        self.total_vi = np.full(n, np.nan)                 # total VI rate
        self.total_vi_abs = np.full(n, np.nan)             # total VI (bit, undivided)

        side_of = {i: m.p_side[i] for i in range(n_players)}
        pid_index = {int(pid): i for i, pid in enumerate(self.player_ids)}

        # a team defends the goal it attacks away from
        goal_team = {}
        for per in (1, 2):
            d = m.teams["home"].direction_by_period.get(per, "left_to_right")
            goal_team[per] = {
                GOAL_HOME: "home" if d == "left_to_right" else "away",
                GOAL_AWAY: "away" if d == "left_to_right" else "home",
            }

        for per in (1, 2):
            period_idx = np.where((self.period == per) & self.valid)[0]
            if period_idx.size < 2:
                continue
            gteam = goal_team[per]
            cluster: dict[int, dict[int, int]] = {}
            team_of_at: dict[int, dict[int, str]] = {}
            for t in period_idx:
                on = np.where(self.on_pitch[t])[0]
                nodes = [int(self.player_ids[i]) for i in on]
                xy = [tuple(self.pos[t, i]) for i in on]
                nodes += [GOAL_HOME, GOAL_AWAY]
                xy += [(0.0, self.width * 0.5), (self.length, self.width * 0.5)]
                order = np.argsort(np.asarray(nodes, dtype=np.int64))
                nodes = [nodes[k] for k in order]
                pts = np.asarray([xy[k] for k in order], dtype=float)
                cluster[t] = hierarchical_clustering(nodes, pts)
                team_map = {}
                for tid in nodes:
                    if tid in gteam:
                        team_map[tid] = gteam[tid]
                    else:
                        team_map[tid] = side_of[pid_index[tid]]
                team_of_at[t] = team_map

            for a, b in zip(period_idx[:-1], period_idx[1:]):
                dt = float(self.second[b] - self.second[a])
                if dt <= 0:
                    continue
                total = vi(cluster[a], cluster[b])
                self.total_vi_abs[b] = total
                self.total_vi[b] = total / dt
                for tid, contrib in vi_per_node(cluster[a], cluster[b]).items():
                    rate = contrib / dt
                    if tid == GOAL_HOME:
                        self.goal_vi[b, 0] = rate
                    elif tid == GOAL_AWAY:
                        self.goal_vi[b, 1] = rate
                    else:
                        j = pid_index.get(tid)
                        if j is not None:
                            self.player_vi[b, j] = rate
                # a node that vanished between a and b still carries VI: take
                # its team from whichever frame has it, so home + away is exact
                team_union = dict(team_of_at[a])
                team_union.update(team_of_at[b])
                for k, side in enumerate(TEAMS):
                    total_k = 0.0
                    for tid, team in team_union.items():
                        if team != side:
                            continue
                        if tid == GOAL_HOME:
                            v = self.goal_vi[b, 0]
                        elif tid == GOAL_AWAY:
                            v = self.goal_vi[b, 1]
                        else:
                            j = pid_index.get(tid)
                            v = self.player_vi[b, j] if j is not None else np.nan
                        if np.isfinite(v):
                            total_k += v
                    self.team_vi[b, k] = total_k

        self.goal_side = goal_team

    def _build_vi_10hz(self) -> None:
        """Rebuild the VI series at the paper's native rate (E3).

        The default pipeline computes VI between clusterings one *second*
        apart (dt = 1 s).  Paper arXiv:2102.11182 computes it between
        consecutive 10 fps frames (dt = 0.1 s) and then downsamples to 1 Hz.
        Here every consecutive valid frame pair contributes VI / 0.1 s, and
        second ``t`` carries the mean of its pair rates (``total_vi``,
        ``player_vi``, ``goal_vi``, ``team_vi``) and the *sum* of its pair
        VIs (``total_vi_abs``), so downstream consumers keep the 1 Hz grid.
        The per-pair identities (sum over nodes == total; home + away ==
        total) are preserved by averaging.
        """
        m = self.match
        n_players = self.player_ids.size
        nf = int(m.frame.size)

        # frame-level roster (playing spans are inclusive frame ranges)
        on_frame = np.zeros((nf, n_players), dtype=bool)
        per = m.period
        for pi in range(n_players):
            for p in (1, 2):
                span = m.playing.get((pi, p))
                if span is None:
                    continue
                sel = (per == p) & (m.frame >= span[0]) & (m.frame <= span[1])
                on_frame[sel, pi] = True

        counts = np.column_stack([
            on_frame[:, [i for i in range(n_players)
                         if m.p_side[i] == s]].sum(axis=1)
            for s in TEAMS
        ])
        finite = np.isfinite(m.track_xy[:, :, 0]) & np.isfinite(m.track_xy[:, :, 1])
        has_pos = np.all(~on_frame | finite, axis=1)
        valid_f = (counts == 11).all(axis=1) & has_pos

        # corner origin for the goal geometry
        x = m.track_xy[:, :, 0] + self.length / 2.0
        y = m.track_xy[:, :, 1] + self.width / 2.0

        side_of = {i: m.p_side[i] for i in range(n_players)}
        pid_index = {int(pid): i for i, pid in enumerate(self.player_ids)}
        goal_team = {}
        for p in (1, 2):
            d = m.teams["home"].direction_by_period.get(p, "left_to_right")
            goal_team[p] = {
                GOAL_HOME: "home" if d == "left_to_right" else "away",
                GOAL_AWAY: "away" if d == "left_to_right" else "home",
            }

        # raw frame -> 1 Hz row (the second that contains the frame).  The
        # jsonl drops frames without player data, so m.frame is NOT dense:
        # index by the raw frame number, not the row position.
        f_to_t = np.full(int(m.frame.max()) + 1, -1, dtype=np.int64)
        for t in range(self.second.size):
            f0, f1 = self.sec_frames[t]
            f_to_t[f0:f1 + 1] = t

        # per-second accumulators
        n = self.second.size
        acc_total = np.zeros(n)
        acc_abs = np.zeros(n)
        acc_cnt = np.zeros(n, dtype=np.int64)
        acc_player = np.zeros((n, n_players))
        acc_goal = np.zeros((n, 2))
        acc_team = np.zeros((n, 2))

        dt = 1.0 / FPS
        prev_f = -1
        prev_cluster: dict[int, dict[int, int]] | None = None
        prev_team: dict[int, str] | None = None
        for f in range(nf):
            if not valid_f[f]:
                prev_f = -1
                prev_cluster = None
                continue
            on = np.where(on_frame[f])[0]
            nodes = [int(self.player_ids[i]) for i in on]
            xy = [(float(x[f, i]), float(y[f, i])) for i in on]
            nodes += [GOAL_HOME, GOAL_AWAY]
            xy += [(0.0, self.width * 0.5), (self.length, self.width * 0.5)]
            order = np.argsort(np.asarray(nodes, dtype=np.int64))
            nodes = [nodes[k] for k in order]
            pts = np.asarray([xy[k] for k in order], dtype=float)
            cluster = hierarchical_clustering(nodes, pts)
            gteam = goal_team[int(per[f])]
            team_map = {}
            for tid in nodes:
                if tid in gteam:
                    team_map[tid] = gteam[tid]
                else:
                    team_map[tid] = side_of[pid_index[tid]]

            # pair only raw-consecutive frames (dt = 0.1 s exactly); a frame
            # dropped from the jsonl breaks the pair, since the rate across
            # the gap is undefined
            if prev_f >= 0 and m.frame[f] == m.frame[prev_f] + 1 \
                    and prev_cluster is not None:
                t = int(f_to_t[m.frame[f]])
                if t >= 0:
                    total = vi(prev_cluster, cluster)
                    acc_total[t] += total / dt
                    acc_abs[t] += total
                    acc_cnt[t] += 1
                    node_rate = {tid: c / dt for tid, c in
                                 vi_per_node(prev_cluster, cluster).items()}
                    # a node that vanished between f-1 and f still carries VI:
                    # take its team from whichever frame has it, so
                    # home + away == total exactly (same rule as the 1 Hz block)
                    team_union = dict(prev_team)
                    team_union.update(team_map)
                    for k, side in enumerate(TEAMS):
                        tot_k = 0.0
                        for tid, team in team_union.items():
                            if team != side:
                                continue
                            v = node_rate.get(tid)
                            if v is None:
                                continue
                            if tid == GOAL_HOME:
                                acc_goal[t, 0] += v
                            elif tid == GOAL_AWAY:
                                acc_goal[t, 1] += v
                            else:
                                j = pid_index.get(tid)
                                if j is not None:
                                    acc_player[t, j] += v
                            tot_k += v
                        acc_team[t, k] += tot_k
            prev_f, prev_cluster, prev_team = f, cluster, team_map

        ok = acc_cnt > 0
        # seconds without a valid frame pair are undefined under the 10 Hz
        # definition: NaN them out rather than keeping the 1 Hz values, so
        # the two definitions never mix inside one series
        self.total_vi[~ok] = np.nan
        self.total_vi_abs[~ok] = np.nan
        self.player_vi[~ok] = np.nan
        self.goal_vi[~ok] = np.nan
        self.team_vi[~ok] = np.nan
        self.total_vi[ok] = acc_total[ok] / acc_cnt[ok]
        self.total_vi_abs[ok] = acc_abs[ok]
        self.player_vi[ok] = acc_player[ok] / acc_cnt[ok, None]
        self.goal_vi[ok] = acc_goal[ok] / acc_cnt[ok, None]
        self.team_vi[ok] = acc_team[ok] / acc_cnt[ok, None]

    def _build_players(self) -> None:
        m = self.match
        self.outfield = {
            side: np.array([i for i in range(self.player_ids.size)
                            if m.p_side[i] == side and not m.p_is_gk[i]])
            for side in TEAMS
        }
        self.gk = {
            side: np.array([i for i in range(self.player_ids.size)
                            if m.p_side[i] == side and m.p_is_gk[i]])
            for side in TEAMS
        }

    # ---------------------------------------------------------------- helpers
    def team_index(self, side: str) -> int:
        return TEAMS.index(side)

    def outfield_order(self, side: str, t: int) -> np.ndarray:
        """The team's on-pitch outfield players at second ``t``, sorted by id."""
        cand = self.outfield[side]
        present = cand[self.on_pitch[t, cand]]
        return present[np.argsort(self.player_ids[present])]

    def player_name(self, pi: int) -> str:
        return str(self.match.p_short[pi]) or str(self.player_ids[pi])

    def substitutions(self) -> list[dict]:
        """Rebuild substitution events from playing times (spec §5.1)."""
        m = self.match
        events = []
        for side in TEAMS:
            team_pi = [i for i in range(self.player_ids.size) if m.p_side[i] == side]
            # players who only appear in the second half
            came_on = []
            for pi in team_pi:
                span1 = m.playing.get((pi, 1))
                span2 = m.playing.get((pi, 2))
                if span1 is None and span2 is not None:
                    came_on.append((pi, span2[0]))
            went_off = [(pi, m.playing[(pi, 2)][1]) for pi in team_pi
                        if (pi, 2) in m.playing]
            period2_start = m.period_bounds[2][0]
            for pin, start in sorted(came_on, key=lambda x: x[1]):
                # nearest player whose second-half end is within 1 frame
                cand = [(abs(end - start), pi) for pi, end in went_off
                        if abs(end - start) <= 1 or abs(end + 1 - start) <= 1]
                if not cand:
                    cand = [(abs(end - start), pi) for pi, end in went_off]
                cand.sort()
                if not cand:
                    continue
                pout = cand[0][1]
                events.append(dict(
                    team=side, t0=int(round((start - period2_start) / FPS)),
                    frame=start, p_in=int(pin), p_out=int(pout),
                    p_in_name=self.player_name(pin),
                    p_out_name=self.player_name(pout),
                ))
                went_off = [(pi, e) for pi, e in went_off if pi != pout]
        events.sort(key=lambda e: (e["team"], e["t0"]))
        return events
