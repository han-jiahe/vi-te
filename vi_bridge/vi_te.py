"""TE ↔ VI 因果涌现桥接引擎（vi_bridge 核心）。

把 MarkovSoccerTE 的两两 TE 估计器栈（``mte.*``，order-1、仅条件于目标自身
历史，bits/log2）接到 vi 项目（``D:/MarkovSoccerTE/vi``）的 VI 因果涌现框架
上，实现 Rosas 2020 (PLoS Comput Biol 16(12):e1008289,
doi:10.1371/journal.pcbi.1008289) 实用判据的 TE 版本：

  V   = 球队 VI 率（team_vi，bit/s，1 Hz；10 名外场球员 + 本队球门贡献）
  X_k = 第 k 名外场球员的个体 VI 率（player_vi，固定球员身份）
  D_k = TE(V→X_k)        下行因果（Rosas 定义 D = Σ_i I(V_t; X^i_{t+1} | X^i_t)）
  U_k = TE(X_k→V)        上行因果（对偶项）
  Δ   = ΣD − ΣU          净下行（Rosas 判据：Δ>0 ⇒ downward causation）
  Γ_TE = I(V_t;V_{t+1}) − ΣU_k   TE 条件形式的因果涌现指标（Ψ_TE）
  Ψ_uncond = I(V_t;V_{t+1}) − Σ_k I(X_k(t);V_{t+1})  vi 项目锁定的无条件版（nats）

三层检验教训内建（见 _backup/results/三层检验报告.md）：
  * 全部 TE 带 block-shuffle 代理（block-len 25、默认 200 次、seed 42），
    eff = raw − mean(surr)（精确恒等式），代理 mean/sd/ratio 持久化；
  * 变换（标准化 / K-means 符号化）在真实链上冻结拟合，代理只打乱源通道；
  * 跨队负对照：TE(V_opp→X_own,k) 与 TE(X_own,k→V_opp) 与队内对拥有完全相同的
    VI 滑窗重叠结构、但无队内因果 ⇒ 队内 − 跨队 = 重叠伪影校正（D*/U*）；
  * 转移掩码只在同一连续有效秒段内成立（VI 率跨段无定义），逐对独立；
  * n_trans ≥ 60 守卫；离散 K=3；KSG k=3/5；全部 1 维标量通道。

单位：TE 为 bits（log2，mte 约定）；Ψ / 互信息为 nats（vi 约定，×ln2 换算）。

VI 滑窗重叠声明（诚实口径）：1 Hz VI 率在秒 t 由聚类对 (t−1, t) 定义，因此
V_t 与 X_k(t+1) 共享秒 t 的聚类——lag-1 TE 的 D/U 会被该重叠对称抬高；跨队
负对照量化的正是这一伪影（两队的 VI 率都是同一全局聚类的函数）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

TE_ROOT = Path(__file__).resolve().parents[1]      # D:/MarkovSoccerTE
VI_ROOT = TE_ROOT / "vi"
for p in (str(TE_ROOT), str(VI_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from mte.discrete import conditional_discrete_mte       # noqa: E402
from mte.gaussian import conditional_gaussian_mte       # noqa: E402
from mte.ksg import conditional_ksg_mte                 # noqa: E402
from mte.significance import effective_te, surrogate_mte  # noqa: E402


def _vi_src():
    """延迟导入 vi 项目源码（构造 VI 链 / 计算 Ψ 时才需要）。

    在本仓库内运行时 vi/ 已由上方 sys.path 提供；作为依赖被
    te-emergence 仓库安装时不会加载（te-emergence 只用纯
    numpy/sklearn 的变换与估计函数，不构造 VI 链）。
    """
    from src.ce_scalar import mi_scalar                  # noqa: E402
    from src.vi_pipeline import TEAMS, ViMatch           # noqa: E402
    return mi_scalar, TEAMS, ViMatch

LN2 = float(np.log(2.0))

PAIR_LABELS = ("downward", "upward", "cross_down", "cross_up")
# 每对独立代理种子：打破"全体对共享同一打乱序列"的蒙特卡洛相关
LAB_SEED_IDX = {lab: i for i, lab in enumerate(PAIR_LABELS)}
CM_SEED_IDX = {"home->away": 0, "away->home": 1}
MIN_N_TRANS = 60           # 少于 60 个有效转移的对不估计（守卫）
MIN_N_EVAL = 10            # 代理评估的最小有效行数（高斯/KSG）
MIN_N_DISC = 30            # 离散法的最小有效行数


def mkey(method: str, k=None) -> str:
    """估计器配置键（与试点命名一致）。"""
    if method == "ksg":
        return f"ksg|k={k}"
    if method == "discrete":
        return "discrete|K=3"
    return method


# ------------------------------------------------------------------ transforms
def std_fit(ch: np.ndarray):
    """在有限值上拟合标准化 (mu, std)；<2 个有限值返回 None。"""
    fin = ch[np.isfinite(ch)]
    if fin.size < 2:
        return None
    return float(fin.mean()), float(fin.std())


def disc_fit(ch: np.ndarray, n_clusters: int, seed: int):
    """在有限值上拟合 K-means 符号化，返回 (km, mu, std)；失败返回 None。

    mte.discrete.quantize_channels 的 NaN 安全适配（真实链含离场 NaN 段）：
    拟合只用有限值，标准化参数冻结 (mu, std)，NaN 位置在代理评估时被掩码丢弃。
    """
    fin = ch[np.isfinite(ch)]
    if fin.size < max(10, 3 * n_clusters):
        return None
    mu, std = float(fin.mean()), float(fin.std())
    chs = ((fin - mu) / (std + 1e-9))[:, None]
    try:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(chs)
    except Exception:
        return None
    return km, mu, std


# --------------------------------------------------------------------- masking
def pair_mask(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(T-1,) 布尔掩码：转移 t→t+1 要求 A[t]、B[t]、B[t+1] 全部有限。

    1 Hz VI 率在秒 t 由聚类对 (t−1, t) 定义，VI 率无定义处即为 NaN；
    掩码只依赖三值有限性，不额外引用 vi 的 valid 掩码（valid 是位置特征
    口径，与 VI 有限性在段边界处并不逐点等价）。有限性仍保证转移两端不
    跨无效间隙（间隙处 VI 无定义 → 必有 NaN）。
    """
    T = A.shape[0]
    if T < 2:
        return np.zeros(0, dtype=bool)
    t = np.arange(T - 1)
    return np.isfinite(A[t]) & np.isfinite(B[t]) & np.isfinite(B[t + 1])


# ------------------------------------------------------------- estimator layer
def _make_closure(method: str, A_raw: np.ndarray, B_raw: np.ndarray,
                  mask: np.ndarray, fit_a, fit_b, fit_a_d, fit_b_d, k: int):
    """构造估计器闭包 fn(A_sh, B, R_list)（mte/significance 协议）。

    A_sh 是被块打乱的源通道（真实 A 也走同一闭包）；B 由代理协议保持真实，
    故 Y/Z 用冻结变换预计算一次。掩码位置固定；打乱后落在掩码位置上的 NaN
    行（源通道离场段）被丢弃（代理有效 n 略小于真实 n，方向保守，已声明）。
    """
    t_arr = np.where(mask)[0]
    if method in ("gaussian", "ksg"):
        mu_a, std_a = fit_a
        mu_b, std_b = fit_b
        b_std = (B_raw - mu_b) / (std_b + 1e-9)
        Y = b_std[t_arr + 1]
        Z = b_std[t_arr]

        if method == "gaussian":
            def fn(A_sh, B, R_list):
                a = (A_sh - mu_a) / (std_a + 1e-9)
                X = a[t_arr]
                ok = np.isfinite(X)
                if ok.sum() < MIN_N_EVAL:
                    return np.nan          # 守卫：不注入 0.0（会虚抬 eff）
                return conditional_gaussian_mte(
                    X[ok][:, None], Y[ok][:, None], Z[ok][:, None])
        else:
            def fn(A_sh, B, R_list):
                a = (A_sh - mu_a) / (std_a + 1e-9)
                X = a[t_arr]
                ok = np.isfinite(X)
                if ok.sum() <= k:
                    return np.nan
                return conditional_ksg_mte(
                    X[ok][:, None], Y[ok][:, None], Z[ok][:, None], k=k)
    else:  # discrete
        km, mu_a, std_a = fit_a_d
        kmb, mu_b, std_b = fit_b_d
        b_s = (B_raw - mu_b) / (std_b + 1e-9)
        fin_b = np.isfinite(b_s)
        sym_b = np.full(b_s.shape, -1, dtype=np.int64)
        sym_b[fin_b] = kmb.predict(b_s[fin_b][:, None])
        Y = sym_b[t_arr + 1]
        Z = sym_b[t_arr]

        def fn(A_sh, B, R_list):
            a = (A_sh - mu_a) / (std_a + 1e-9)
            fin = np.isfinite(a)
            sym = np.full(a.shape, -1, dtype=np.int64)
            sym[fin] = km.predict(a[fin][:, None])
            X = sym[t_arr]
            ok = X >= 0
            if ok.sum() < MIN_N_DISC:
                return np.nan
            return conditional_discrete_mte(Y[ok], X[ok], Z[ok][:, None])

    return fn


def run_pair(A_raw: np.ndarray, B_raw: np.ndarray, mask: np.ndarray,
             fit_a, fit_b, fit_a_d, fit_b_d, method: str, k,
             block_len: int, n_surrogates: int, seed: int) -> dict:
    """单对 TE：raw + 代理 + eff/p + 代理统计（三层检验 L1 表的数据源）。

    代理中守卫触发的 NaN 不计入（均值与 p 均按有限代理数归一），并记录
    n_surr_used——避免把 0.0 注入代理分布虚抬 eff。
    """
    fn = _make_closure(method, A_raw, B_raw, mask, fit_a, fit_b, fit_a_d,
                       fit_b_d, k)
    raw = fn(A_raw, B_raw, [])
    surr = np.asarray(surrogate_mte(fn, A_raw, B_raw, [], block_len,
                                    n_surrogates, seed), dtype=float)
    fin = np.isfinite(surr)
    n_fin = int(fin.sum())
    if n_fin:
        surr_v = surr[fin]
        eff = float(raw - surr_v.mean())
        p = float((surr_v >= raw).sum() + 1) / (n_fin + 1)
        surr_mean = float(surr_v.mean())
        surr_sd = float(surr_v.std())
    else:
        eff = p = surr_mean = surr_sd = np.nan
    out = dict(
        n_trans=int(mask.sum()),
        raw=float(raw),
        eff=eff,
        p=p,
        surr_mean=surr_mean,
        surr_sd=surr_sd,
        surr_ratio=(float(surr_mean / abs(raw)) if abs(raw) >= 1e-9
                    else None),
    )
    if n_fin < n_surrogates:
        out["n_surr_used"] = n_fin
    return out


# ------------------------------------------------------------------ match layer
class ViTeMatch:
    """单场比赛的 TE↔VI 桥接计算。

    参数与 run_vi_te CLI 一致；``halves`` 限制只算指定半场（冒烟用），
    ``methods`` 限制估计器集合（冒烟用）。
    """

    def __init__(self, match_id, raw_dir, n_surrogates: int = 200,
                 block_len: int = 25, ks=(3, 5), n_clusters: int = 3,
                 seed: int = 42, halves=None, methods=None, verbose: bool = True):
        self.match_id = str(match_id)
        self.n_surrogates = int(n_surrogates)
        self.block_len = int(block_len)
        self.ks = tuple(int(k) for k in ks)
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)
        self.verbose = verbose
        self.halves = halves          # None 或 {"home|1", ...}
        self.methods = methods        # None 或 {"gaussian", "ksg", "discrete"}

        t0 = time.time()
        raw_path = Path(raw_dir)
        child = raw_path / self.match_id
        if (child / f"{self.match_id}_match.json").exists():
            raw_path = child      # raw_dir 是父目录（vi 项目布局）
        _, _, ViMatch = _vi_src()
        self.v = ViMatch(self.match_id, raw_path, hz=1, verbose=False)
        self._t_build = time.time() - t0

        self._method_keys = []
        for m in ("gaussian", "ksg", "discrete"):
            if self.methods is not None and m not in self.methods:
                continue
            if m == "ksg":
                self._method_keys.extend(mkey("ksg", k) for k in self.ks)
            elif m == "discrete":
                self._method_keys.append(mkey("discrete"))
            else:
                self._method_keys.append(mkey(m))
        self._cross_macro: dict[str, dict] = {}
        self._roles = self._build_roles()

    # ------------------------------------------------------------- utilities
    def _build_roles(self) -> dict:
        """{pid: (role_name, position_group)} 来自 match.json 元数据。"""
        m = self.v.match
        pidx = m.p_index
        out: dict[int, tuple] = {}
        for p in m.meta.get("players", []):
            for k in ("trackable_object", "id", "team_player_id"):
                if p.get(k) is not None and p[k] in pidx:
                    pi = pidx[p[k]]
                    role = (p.get("player_role") or {})
                    out[pi] = (str(role.get("name") or ""),
                               str(role.get("position_group") or ""))
                    break
        return out

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _half_indices(self, per: int) -> np.ndarray:
        return np.where(self.v.period == per)[0]

    def _chains_half(self, per: int) -> dict:
        _, TEAMS, _ = _vi_src()
        idx = self._half_indices(per)
        chains = {
            "home_macro": self.v.team_vi[idx, TEAMS.index("home")],
            "away_macro": self.v.team_vi[idx, TEAMS.index("away")],
        }
        for side in TEAMS:
            for pi in self.v.outfield[side]:
                chains[f"player:{side}:{pi}"] = self.v.player_vi[idx, pi]
        return idx, chains

    @staticmethod
    def _skip_row(pid: int, name: str, n_trans: int, reason: str) -> dict:
        return dict(pid=int(pid), name=name, n_trans=int(n_trans),
                    methods={}, skipped=reason)

    # ------------------------------------------------------------------ halves
    def _psi_uncond(self, macro: np.ndarray,
                    players: dict[int, np.ndarray]) -> dict:
        """无条件 Ψ（vi 项目 Task 2 口径、固定球员身份）：
        I(V_t;V_{t+1}) − Σ_k I(X_k(t);V_{t+1})，mi_scalar（高斯相关 MI，nats），
        在半场内所有有效连续转移上全局计算（非滚动窗）。"""
        mi_scalar, _, _ = _vi_src()
        t = np.arange(macro.size - 1)
        V, Vp = macro[t], macro[t + 1]
        mm = np.isfinite(V) & np.isfinite(Vp)
        i_macro = float(mi_scalar(V[mm], Vp[mm])) if mm.sum() >= 20 else np.nan
        s = 0.0
        per_player = {}
        for pi, ch in players.items():
            X = ch[t]
            mk = np.isfinite(X) & np.isfinite(Vp)
            if mk.sum() >= 20:
                mi = float(mi_scalar(X[mk], Vp[mk]))
                s += mi
                per_player[int(self.v.player_ids[pi])] = mi
        return dict(i_macro_nats=i_macro,
                    i_micro_sum_uncond_nats=float(s),
                    psi_uncond_nats=(float(i_macro - s) if np.isfinite(i_macro)
                                     else np.nan),
                    n_trans_macro=int(mm.sum()),
                    per_player_mi_nats=per_player)

    def _psi_slot(self, per: int, side: str) -> dict:
        """vi 项目 Task 2 的精确构造（槽位口径）：micro[t] = outfield_order
        的 10 个在场外场球员 VI（按 id 排序），valid 要求 10 人全在场；
        在半场内所有连续有效转移上全局计算（非滚动窗）——与 vi 项目
        psi_series(win=60) 的滚动窗估计是同一量的不同估计，符号应一致。"""
        mi_scalar, TEAMS, _ = _vi_src()
        idx = self._half_indices(per)
        T = idx.size
        macro = self.v.team_vi[idx, TEAMS.index(side)]
        micro = np.full((T, 10), np.nan)
        ok = np.zeros(T, dtype=bool)
        for t in range(T):
            ids = self.v.outfield_order(side, idx[t])
            if ids.size == 10:
                micro[t] = self.v.player_vi[idx[t], ids]
                ok[t] = True
        ok &= np.isfinite(macro) & np.isfinite(micro).all(axis=1)
        pm = ok[:-1] & ok[1:]
        if pm.sum() < 20:
            return dict(psi_slot_nats=np.nan, i_macro_slot_nats=np.nan,
                        i_micro_sum_slot_nats=np.nan, n_trans_slot=0)
        i_macro = float(mi_scalar(macro[:-1][pm], macro[1:][pm]))
        s = sum(float(mi_scalar(micro[:-1][pm, j], macro[1:][pm]))
                for j in range(10))
        return dict(psi_slot_nats=float(i_macro - s),
                    i_macro_slot_nats=float(i_macro),
                    i_micro_sum_slot_nats=float(s),
                    n_trans_slot=int(pm.sum()))

    def _half(self, per: int, side: str) -> dict:
        idx, chains = self._chains_half(per)
        opp = "away" if side == "home" else "home"
        own_macro, opp_macro = f"{side}_macro", f"{opp}_macro"

        # 冻结变换拟合（真实链、有限值；逐半场逐链）
        fits_std = {name: std_fit(ch) for name, ch in chains.items()}
        fits_disc = {name: disc_fit(ch, self.n_clusters, self.seed)
                     for name, ch in chains.items()}

        # 对集合：队内 D/U + 跨队负对照（同结构、无队内因果）。
        # 口径：遍历该队全部名单外场球员（含未上场替补）；D_sum 实际为
        # "该半场 n_trans≥60 的出场外场球员之和"，人数随换人 10–13 不等
        # （未上场者被守卫跳过）。
        pairs = []
        for pi in self.v.outfield[side]:
            pname = f"player:{side}:{pi}"
            pairs.append(("downward", own_macro, pname, pi))
            pairs.append(("upward", pname, own_macro, pi))
            pairs.append(("cross_down", opp_macro, pname, pi))
            pairs.append(("cross_up", pname, opp_macro, pi))

        results = {lab: [] for lab in PAIR_LABELS}
        t0 = time.time()
        for lab, a_name, b_name, pi in pairs:
            A, B = chains[a_name], chains[b_name]
            mask = pair_mask(A, B)
            n = int(mask.sum())
            name = self.v.player_name(pi)
            if n < MIN_N_TRANS:
                results[lab].append(self._skip_row(
                    int(self.v.player_ids[pi]), name, n,
                    f"n_trans<{MIN_N_TRANS}"))
                continue
            row = dict(pid=int(self.v.player_ids[pi]), name=name, n_trans=n,
                       methods={})
            for mk_ in self._method_keys:
                method = mk_.split("|")[0]
                k = int(mk_.split("k=")[1]) if "k=" in mk_ else None
                if fits_std[a_name] is None or fits_std[b_name] is None:
                    row["methods"][mk_] = dict(skipped="no_std_fit")
                    continue
                if method == "discrete" and (fits_disc[a_name] is None
                                             or fits_disc[b_name] is None):
                    row["methods"][mk_] = dict(skipped="no_disc_fit")
                    continue
                row["methods"][mk_] = run_pair(
                    A, B, mask, fits_std[a_name], fits_std[b_name],
                    fits_disc[a_name], fits_disc[b_name], method, k,
                    self.block_len, self.n_surrogates,
                    self.seed + 10_000 * pi
                    + 1_000 * LAB_SEED_IDX[lab])
            results[lab].append(row)
        t_pairs = time.time() - t0

        # 跨队宏观对（每 period 只算一次，存 match 级，挂在 home 半场跑）
        if side == "home":
            cm = {}
            for direction, (a_name, b_name) in (
                    ("home->away", ("home_macro", "away_macro")),
                    ("away->home", ("away_macro", "home_macro"))):
                A, B = chains[a_name], chains[b_name]
                mask = pair_mask(A, B)
                n = int(mask.sum())
                entry = dict(n_trans=n, methods={})
                if n >= MIN_N_TRANS:
                    for mk_ in self._method_keys:
                        method = mk_.split("|")[0]
                        k = int(mk_.split("k=")[1]) if "k=" in mk_ else None
                        if fits_std[a_name] is None or fits_std[b_name] is None:
                            entry["methods"][mk_] = dict(skipped="no_std_fit")
                            continue
                        if method == "discrete" and (
                                fits_disc[a_name] is None
                                or fits_disc[b_name] is None):
                            entry["methods"][mk_] = dict(skipped="no_disc_fit")
                            continue
                        entry["methods"][mk_] = run_pair(
                            A, B, mask, fits_std[a_name], fits_std[b_name],
                            fits_disc[a_name], fits_disc[b_name], method, k,
                            self.block_len, self.n_surrogates,
                            self.seed + 900_000
                            + 100 * CM_SEED_IDX[direction])
                else:
                    entry["skipped"] = f"n_trans<{MIN_N_TRANS}"
                cm[direction] = entry
            self._cross_macro[str(per)] = cm

        # 半场级汇总（逐方法）：D/U 求和、净下行 Δ、伪影校正 Δ*、Γ_TE/Ψ_TE
        methods_summary = {}
        for mk_ in self._method_keys:
            def eff_list(lab):
                return [r["methods"][mk_]["eff"] for r in results[lab]
                        if "raw" in r["methods"].get(mk_, {})]

            D, U = eff_list("downward"), eff_list("upward")
            CD, CU = eff_list("cross_down"), eff_list("cross_up")
            D_sum = float(sum(D))
            U_sum = float(sum(U))
            cd_mean = float(np.mean(CD)) if CD else np.nan
            cu_mean = float(np.mean(CU)) if CU else np.nan
            D_star = [d - cd_mean for d in D] if CD else []
            U_star = [u - cu_mean for u in U] if CU else []
            methods_summary[mk_] = dict(
                downward_sum_bits=D_sum,
                upward_sum_bits=U_sum,
                delta_bits=float(D_sum - U_sum),
                cross_down_mean_bits=cd_mean,
                cross_up_mean_bits=cu_mean,
                downward_star_sum_bits=float(sum(D_star)) if D_star else np.nan,
                upward_star_sum_bits=float(sum(U_star)) if U_star else np.nan,
                delta_star_bits=(float(sum(D_star) - sum(U_star))
                                 if D_star and U_star else np.nan),
                n_downward=int(len(D)), n_upward=int(len(U)),
                n_cross_down=int(len(CD)), n_cross_up=int(len(CU)),
            )

        # 无条件 Ψ（固定球员口径）+ vi 项目槽位口径复算
        own_players = {pi: chains[f"player:{side}:{pi}"]
                       for pi in self.v.outfield[side]}
        psi_uncond = self._psi_uncond(chains[own_macro], own_players)
        psi_slot = self._psi_slot(per, side)

        # Γ_TE = I(V;V') − ΣU（bits 与 nats 双单位；与 vi 的 Ψ 同量纲时用 nats）
        for mk_, sm in methods_summary.items():
            i_bits = (psi_uncond["i_macro_nats"] / LN2
                      if np.isfinite(psi_uncond["i_macro_nats"]) else np.nan)
            sm["gamma_te_bits"] = (i_bits - sm["upward_sum_bits"]
                                   if np.isfinite(i_bits) else np.nan)
            sm["gamma_te_nats"] = (sm["gamma_te_bits"] * LN2
                                   if np.isfinite(sm["gamma_te_bits"])
                                   else np.nan)

        poss = self.v.poss_group[idx]
        valid_idx = self.v.valid[idx]
        poss_valid = poss[valid_idx]
        # 口径与 vi 项目 task1 一致：分母只计归属明确的秒（home/away）
        attributed = np.isin(poss_valid, ["home", "away"])
        poss_rate = (float((poss_valid[attributed] == side).mean())
                     if attributed.any() else np.nan)

        players_meta = []
        for pi in self.v.outfield[side]:
            role, group = self._roles.get(pi, ("", ""))
            chain = chains[f"player:{side}:{pi}"]
            players_meta.append(dict(
                pid=int(self.v.player_ids[pi]), name=self.v.player_name(pi),
                role=role, position_group=group,
                n_finite_seconds=int(np.isfinite(chain).sum()),
                on_this_half=bool(np.isfinite(chain).sum() > 0),
            ))

        self._log(f"  [{side}|P{per}] n_trans≈"
                  f"{results['downward'][0]['n_trans'] if results['downward'] and 'n_trans' in results['downward'][0] else 0}"
                  f" pairs={len(pairs)} {t_pairs:.1f}s")

        return dict(
            period=int(per), side=side,
            team=self.v.match.teams[side].short_name,
            n_seconds=int(idx.size),
            n_valid=int(self.v.valid[idx].sum()),
            possession_rate=poss_rate,
            psi=dict(**psi_uncond, **psi_slot),
            methods=methods_summary,
            pairs={lab: results[lab] for lab in PAIR_LABELS},
            players_meta=players_meta,
        )

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        _, TEAMS, _ = _vi_src()
        halves = {}
        for side in TEAMS:
            for per in (1, 2):
                key = f"{side}|{per}"
                if self.halves is not None and key not in self.halves:
                    continue
                halves[key] = self._half(per, side)

        subs = self.v.substitutions()
        meta = dict(
            match_id=self.match_id,
            teams={s: self.v.match.teams[s].short_name for s in TEAMS},
            hz=1,
            protocol=(
                "pairwise order-1 TE, conditioned solely on the target's own "
                "history z^B_{t-1}; estimators = gaussian closed-form / KSG "
                "Frenzel-Pompe k=3,5 / discrete K-means K=3; transforms "
                "frozen on real chains; block-shuffle surrogates (block-len "
                f"{self.block_len}, n={self.n_surrogates}, seed={self.seed}); "
                "eff = raw - mean(surrogates) (exact identity); transition "
                "mask = both chain ends finite within the same contiguous "
                "valid-second run; cross-team pairs are the overlap-artifact "
                "negative controls"),
            n_seconds=int(self.v.second.size),
            n_valid=int(self.v.valid.sum()),
            n_substitutions=len(subs),
            substitutions=[dict(team=s["team"], t0=s["t0"], frame=s["frame"],
                                p_in_name=s["p_in_name"],
                                p_out_name=s["p_out_name"]) for s in subs],
            estimators=self._method_keys,
            n_surrogates=self.n_surrogates,
            block_len=self.block_len,
            seed=self.seed,
            min_n_trans=MIN_N_TRANS,
            t_build_s=round(self._t_build, 1),
            vi_bridge_version=__import__("vi_bridge").__version__,
        )
        return dict(meta=meta, halves=halves, cross_macro=self._cross_macro)

    def run_to_json(self, out_path) -> dict:
        doc = self.run()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False),
                            encoding="utf-8")
        self._log(f"wrote {out_path}")
        return doc
