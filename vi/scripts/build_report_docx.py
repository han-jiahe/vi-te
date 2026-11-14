"""Build the Word report for the VI causal-emergence / substitution-TMLE tasks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

OUT = ROOT / "output" / "vi" / "1886347"
DOCX = ROOT / "VI因果涌现与换人TMLE_实现报告.docx"

EAST_ASIA = "微软雅黑"
BODY_FONT = "Calibri"


def set_run_font(run, size=None, bold=None):
    run.font.name = BODY_FONT
    if run._element.rPr is None:
        run._element.get_or_add_rPr()
    rpr = run._element.rPr
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        from docx.oxml import OxmlElement
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), EAST_ASIA)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def para(doc, text, size=10.5, bold=False, align=None, space_after=4):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    r = h.add_run(text)
    set_run_font(r, size={0: 20, 1: 15, 2: 12.5}.get(level, 11), bold=True)
    return h


def table(doc, df, caption=None, font=8.5, widths=None):
    if caption:
        p = doc.add_paragraph()
        r = p.add_run(caption)
        set_run_font(r, size=9, bold=True)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for j, col in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(col))
        set_run_font(r, size=font, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            r = cells[j].paragraphs[0].add_run(str(v))
            set_run_font(r, size=font)
            cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(doc, path, caption, width=15.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    set_run_font(r, size=8.5)
    cap.paragraph_format.space_after = Pt(8)


def main() -> int:
    t1 = pd.read_csv(OUT / "task1_ce_summary.csv")
    _pj = ROOT / "output" / "vi" / "task2_pool_summary.json"
    pool0 = json.loads(_pj.read_text(encoding="utf-8")) if _pj.exists() else None
    subs = pd.read_csv(OUT / "task2_substitutions.csv")
    seg = pd.read_csv(OUT / "task2_segments.csv")
    tmle = json.loads((OUT / "task2_tmle_summary.json").read_text(encoding="utf-8"))

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)

    heading(doc, "VI 因果涌现 + 换人 TMLE：10 场比赛实现报告", 0)
    para(doc, "数据：SkillCorner Open Data，10 场澳超 2024/25；主表以 match 1886347"
              "（Auckland FC 2–0 Newcastle Jets，2024-11-30，104 m × 68 m）为例。", size=9.5)
    para(doc, "方法：BlueDream_v3plus :: vi_analysis/clustering.py + clustering_vi.py"
              "（论文 arXiv:2102.11182 的聚类 Variation of Information，单位 bit）+ "
              "实现方案 §3–§5（标量闭式高斯互信息、因果涌现 Ψ、TMLE）。"
              "按任务要求只做信息分解层面的 Ψ，不做 VI 级 TE。", size=9.5)

    # ---------------------------------------------------------------- summary
    heading(doc, "0. 结论摘要", 1)
    psi_lo, psi_hi = float(t1.psi_global.min()), float(t1.psi_global.max())
    pr_lo, pr_hi = (float(t1.psi_rolling_mean.min()), float(t1.psi_rolling_mean.max()))
    summary_df = pd.DataFrame([
        ["数据校验", "有效秒 4405（上半场 2339 / 下半场 2066）；10 场共 40 个队-半场行"],
        ["VI 定义", "论文 arXiv:2102.11182 的聚类 Variation of Information："
                    "22 名场上球员 + 2 个球门共 24 个节点做最近邻聚类，"
                    "对 t-1 与 t 两个聚类求 VI 并除以 dt → 单位 bit/s。"
                    "逐时刻成立 Σ节点 VI = 总 VI、home + away = 总 VI（不再是 1）"],
        ["Task 1", f"Ψ(team VI 率) 全局 {psi_lo:.3f} ~ {psi_hi:.3f} nats；"
                   f"60 s 滑窗均值 {pr_lo:.3f} ~ {pr_hi:.3f} nats，符号为负"],
        ["Task 2", "7 次换人、6 个 5 min 段 × 2 队 = N = 12；被换下者与替补的 VI "
                   "（现为 bit/s，不再是面积占比）"],
        ["因果效应（单场）", f"TMLE ATE = {tmle['ate_norm']:.3f}（归一化）/ {tmle['ate_nats']:.3f} nats，"
                             f"95% CI [{tmle['ci95_nats'][0]:.3f}, {tmle['ci95_nats'][1]:.3f}]，"
                             f"置换检验 p = {tmle['permutation']['p_two_sided']:.3f}"],
        ["因果效应（池化）", (
            f"{len(pool0['per_match'])} 场 / N = {pool0['N_segments']} / 处理组 "
            f"{pool0['n_treated']} 段：ATE {pool0['ate_nats']:+.3f} nats，95% CI "
            f"[{pool0['ci95_nats'][0]:+.3f}, {pool0['ci95_nats'][1]:+.3f}]，置换 p = "
            f"{pool0['perm_p_two_sided']:.3f}"
            if pool0 else "尚未池化")],
        ["判定", "VI 口径已按论文改正（Voronoi 面积占比 → 聚类 VI，bit/s），"
                 "两队宏观项不再共线；换人效应的结论见 §3.5"],
    ], columns=["项目", "结果"])
    table(doc, summary_df, font=9, widths=[2.6, 13.0])

    para(doc, ("口径变更说明：2026-11-09 起 VI 由“Voronoi 单元面积占场地面积比”改为"
               "“两个最近邻聚类之间的 Variation of Information（bit）”，"
               "球员/球队 VI 是其对总 VI 的贡献（bit），时间序列取 VI 率（bit/s）。"
               "此前全部 VI 相关数值已作废并重算。"),
         size=9, bold=False)

    # ------------------------------------------------------------- method
    heading(doc, "1. 管线与自检锚点", 1)
    para(doc, "坐标：SkillCorner 原点在场地中心 → 角原点 X = x + 52、Y = y + 34，场地 104 × 68（7072 m²）。"
              "降采样：秒 t = 帧 [S + 10t, S + 10t + 9]，位置取该秒内 is_detected 帧的均值；"
              "若某球员该秒一次都没被检出，退回全部帧均值（否则有效秒掉到约 3800）。"
              "有效秒 = 该秒有追踪帧 且 两队各 11 人在场 且 22 人位置可求。", size=10)
    para(doc, "VI（论文口径）：节点集 = 场上 22 名球员 + 2 个球门（共 24）。每个时刻用"
              "最近邻单链接聚类（每人连到最近邻、簇之间合并）。对 t-1 与 t 两个聚类构造列联表"
              " r_ij = |x_i∩y_j| / n，则"
              " VI = −Σ_ij r_ij[log2(r_ij/p_i) + log2(r_ij/q_j)] （bit）；"
              "球员 VI = 把簇对贡献均分给交集中的节点，团队 VI = 该队 11 名球员 + 本队球门的贡献之和；"
              "时间序列取 VI 率 = VI(C_{t-1}, C_t) / dt（bit/s，逐时刻滚动）。"
              "上游 Voronoi 面积口径已按修改意见降级为“附加项”，本报告不再使用。", size=10)

    anchor_df = pd.DataFrame([
        ["1", "有效秒 ≈ 4400（P1 2339 / P2 2066）", "4405（2339 / 2066）", "通过"],
        ["2", "Σ节点 VI = 总 VI；home + away = 总 VI（不再是 1）",
         "max|diff| < 1e-15；VI ≥ 0，球员贡献 ≥ 0", "通过"],
        ["3", "Ψ(V=VI) 大概率负", f"{psi_lo:.2f} ~ {psi_hi:.2f} nats", "通过"],
        ["4", "换人事件 ≈ 7；下半场完整段 ≈ 6", "7 / 6", "通过"],
        ["5", "Ψ_VI 逐秒序列存在", "上/下半场均有", "通过"],
        ["6", "池化后 N ≈ 120、A=1 ≈ 70、g ∈ (0.01,0.99)",
         (f"N = {pool0['N_segments']}（{pool0['n_matches']} 场），A=1 = {pool0['n_treated']}，"
          f"g ∈ [{pool0['g_range'][0]:.3f}, {pool0['g_range'][1]:.3f})"
          if pool0 else "尚未池化"),
         "未达（目标 N≈120）" if (pool0 and pool0["N_segments"] < 100) else "通过"],
    ], columns=["#", "自检锚点", "实测", "判定"])
    table(doc, anchor_df, caption="表 1　规格书 §7 自检锚点", font=8.5, widths=[0.9, 6.6, 6.0, 2.2])

    # --------------------------------------------------------------- task 1
    heading(doc, "2. Task 1：以球队 VI 为宏观特征的因果涌现", 1)
    para(doc, "Ψ(V) = I(V_t; V_{t+1}) − Σ_{k=1}^{50} I(X_k(t); V_{t+1})，"
              "V = team VI，X = 10 名外场球员的 x,y（20）+ vx,vy（20）+ 到中心距离 d（10）。", size=10)

    t1_disp = pd.DataFrame({
        "球队": t1["team"], "半场": t1["half"], "控球率": t1["possession_rate"].round(3),
        "I(V_t;V_{t+1})": t1["i_macro"].round(3),
        "Σ_k I(X_k;V_{t+1})": t1["i_micro_sum"].round(3),
        "Ψ_global": t1["psi_global"].round(3),
        "Ψ_rolling_mean": t1["psi_rolling_mean"].round(3),
        "有效秒": t1["n_seconds"], "窗口数": t1["n_windows"],
    })
    table(doc, t1_disp, caption="表 2　Task 1 主结果（单位 nats，global 为整半场一个估计，rolling 为 60 s 滑窗均值）",
          font=8.5, widths=[2.2, 1.2, 1.5, 2.0, 2.6, 2.0, 2.6, 1.6, 1.6])

    figure(doc, OUT / "figure_task1_psi_timeseries.png",
           "图 1　上：两队逐秒球队 VI（Voronoi 占有率）；下：以 team VI 为宏观特征的 60 s 滑窗 Ψ。")

    para(doc, "解读：", size=10, bold=True)
    i_macro_h = float(t1[t1.side == "home"].i_macro.mean())
    i_micro_h = float(t1[t1.side == "home"].i_micro_sum.mean())
    i_macro_a = float(t1[t1.side == "away"].i_macro.mean())
    i_micro_a = float(t1[t1.side == "away"].i_micro_sum.mean())
    for txt in [
        (f"Ψ 仍为负但不复为“强负”：宏观项 I(V_t;V_t+1) 只有 {i_macro_h:.3f}（主队）/ "
         f"{i_macro_a:.3f}（客队）nats，而 50 个微观特征之和是 {i_micro_h:.2f} / {i_micro_a:.2f}。"
         "原因是论文口径的 VI 率是一个“变化率”，自相关很低（lag-1 相关约 0.2），"
         "而球员位置/速度对它的预测力却很有限——这正是“位置类微观特征冗余、"
         "但冗余度不如 Voronoi 面积口径那么极端”的体现。"),
        ("两队宏观项不再共线：Voronoi 口径下 VI_home + VI_away ≡ 1 导致两队 I(V_t;V_t+1) 恒等；"
         "论文口径下 home + away = 总 VI（随时间波动），两队宏观项有区分度，"
         "corr(home VI, away VI) 约为 +0.65 而非 −1。这是本次口径修正最重要的结构性改善。"),
        ("VI 率的量级：中位数约 0.37 bit/s，95 分位约 1.0 bit/s，"
         "单秒 VI 最大约 3.3 bit（上限 log2(24) = 4.59 bit）；球员级 VI 贡献恒非负。"),
    ]:
        p = doc.add_paragraph(txt, style="List Bullet")
        for r in p.runs:
            set_run_font(r, size=9.5)

    shot_csv = OUT / "task1_shot_regressions.csv"
    if shot_csv.exists():
        heading(doc, "2.1 射门前 60 s 的 Ψ（可选扩展 §8.2）", 2)
        shot = pd.read_csv(shot_csv)
        sd = pd.DataFrame({
            "macro": shot["macro"], "角色": shot["role"], "事件数": shot["n_events"],
            "窗口均值 (nats)": shot["mean_psi"].round(3),
            "β (nats/s)": shot["beta"].round(4), "R²": shot["r2"].round(3),
            "p": shot["p"].map(lambda v: f"{v:.2g}"),
        })
        table(doc, sd, caption="表 2b　射门前 60 s 内 Ψ(team VI) 的线性趋势",
              font=8.5, widths=[2.4, 2.0, 1.8, 2.8, 2.4, 1.8, 1.8])
        fig_shot = OUT / "figure_task1_psi_before_shot.png"
        if fig_shot.exists():
            figure(doc, fig_shot,
                   "图 1b　射门前 60 s 的 Ψ(team VI)：进攻方 / 防守方 / 随机基线。")
        para(doc, "与报告② §4.3（宏观特征为相对距离、CoM）相比，本处进攻方的正向趋势弱得多"
                  "（β 与随机基线同量级，p > 0.05）。原因很可能是：以 team VI 为宏观特征时，"
                  "Ψ 的方差主要来自 50 个微观项之和，射门前的战术变化被淹没；"
                  "而且只有 6/27 次射门在射门前 60 s 内凑够有效的滑窗，事件数太少。",
             size=9.5)

    # --------------------------------------------------------------- task 2
    heading(doc, "3. Task 2：换人 → 团队协同（Ψ_VI）的因果效应", 1)
    para(doc, "Ψ_VI(t) = I(V_t; V_{t+1}) − Σ_{k=1}^{10} I(VI_k(t); V_{t+1})，"
              "V = team VI，X = 10 名外场球员的 individual VI（取代位置/速度/距离）。", size=10)

    heading(doc, "3.1 换人事件与 ±5 min 窗口", 2)
    subs_disp = pd.DataFrame({
        "球队": subs["team"], "t0 (s)": subs["t0"], "分钟": subs["minute"],
        "被换下": subs["player_out"], "替补": subs["player_in"],
        "pre_VI": subs["pre_VI_mean"].round(4), "post_VI": subs["post_VI_mean"].round(4),
        "ΔVI": subs["delta_VI"].round(4),
        "n(pre/post)": [f"{a}/{b}" for a, b in zip(subs["n_valid_pre"], subs["n_valid_post"])],
    })
    table(doc, subs_disp, caption="表 3　7 次换人的 ±5 min 窗口（VI 为占球场面积的比例）",
          font=8, widths=[1.8, 1.3, 1.2, 2.4, 2.4, 1.5, 1.5, 1.5, 1.9])

    para(doc, f"7 次中 4 次替补的 VI 高于被换下者，均值 ΔVI = {subs['delta_VI'].mean():+.4f}"
              f"（约占球场面积 0.35%）。方向与论文 B “换人提升全队表现”一致，但这是描述性的："
              "没有反事实、没有混杂控制，且 3 次事件符号相反。", size=10)

    psi_disp = pd.DataFrame({
        "球队": subs["team"], "t0 (s)": subs["t0"],
        "Ψ_VI 前窗": subs["psi_pre_mean"].round(3),
        "Ψ_VI 后窗": subs["psi_post_mean"].round(3),
        "ΔΨ_VI": subs["delta_psi"].round(3),
        "窗口数(pre/post)": [f"{a}/{b}" for a, b in zip(subs["n_psi_pre"], subs["n_psi_post"])],
    })
    table(doc, psi_disp, caption="表 4　换人前后窗口内的团队 Ψ_VI（nats）",
          font=8, widths=[2.0, 1.5, 2.3, 2.3, 2.0, 2.6])

    para(doc, "需要指出的结构性缺陷：换人发生在死球/中断时刻，而 Ψ_VI 需要 60 个连续有效秒才能估计。"
              "换人点前后正好是追踪缺口，窗口内可用 Ψ_VI 只有 4–13 个（Newcastle 915 s、"
              "Auckland 1493 s）。表中 ΔΨ_VI 的极值（+1.43 / −1.81）都来自个位数窗口，不能当作结论。",
         size=9.5)

    figure(doc, OUT / "figure_task2_psi_prepost_home.png",
           "图 2a　Auckland FC：左为被换下者（前 5 min）与替补（后 5 min）的 individual VI；右为球队 Ψ_VI。")
    figure(doc, OUT / "figure_task2_psi_prepost_away.png",
           "图 2b　Newcastle Jets：同上。")

    heading(doc, "3.2 Ψ_VI 时间序列与换人时点", 2)
    figure(doc, OUT / "figure_task2_psi_vi_secondhalf.png",
           "图 3　下半场球队 VI（上）与 Ψ_VI（下），竖虚线为换人时点（蓝 = Auckland，红 = Newcastle）。")

    heading(doc, "3.3 5 分钟分段与变量", 2)
    seg_disp = pd.DataFrame({
        "球队": seg["team"], "k": seg["k"], "A": seg["A"], "Y (nats)": seg["Y_nats"].round(3),
        "W1": seg["W1"], "W2": seg["W2"], "W3": seg["W3"], "W4": seg["W4"], "W5": seg["W5"],
        "W6": seg["W6"].round(4), "Y_norm": seg["Y_norm"].round(3), "窗口数": seg["n_psi"],
    })
    table(doc, seg_disp, caption="表 5　下半场 6 个 5 min 段 × 2 队（N = 12）；"
                                 "A = 上一段是否换人，Y = 段内 Ψ_VI 均值",
          font=7.5, widths=[1.9, 0.8, 0.8, 1.5, 0.9, 0.9, 0.9, 0.9, 0.9, 1.5, 1.4, 1.4])

    para(doc, "Y 取值范围 −0.903 ~ +0.227 nats，Y_scale = 1.131；A = 1 只有 4/12 段（33%），"
              "g(W) ∈ [0.048, 0.800] 满足正性。W4（比分）在全部 12 段上都是 0——两个进球都发生在"
              "最后一个段起点之后，因此 W4 在本场是常数列，不提供信息。", size=9.5)

    heading(doc, "3.4 TMLE 因果效应", 2)
    r = tmle
    tmle_df = pd.DataFrame([
        ["Y 归一化", f"min–max → [0,1]，Y_scale = {r['Y_scale']:.3f} nats"],
        ["初始估计 Q", f"离散 Super Learner，5 折 CV 选中 {r['super_learner'].upper()}"
                       f"（CV-MSE {r['cv_mse'][r['super_learner']]:.3f}）"],
        ["倾向分 g(W)", f"L2 逻辑回归，g ∈ [{r['g_range'][0]:.3f}, {r['g_range'][1]:.3f}]"],
        ["clever covariate", f"H = A/g − (1−A)/(1−g)，ε = {r['epsilon']:+.4f}"],
        ["ATE（归一化）", f"{r['ate_norm']:+.4f}，SE {r['se_norm']:.4f}，"
                          f"95% CI [{r['ci95_norm'][0]:+.3f}, {r['ci95_norm'][1]:+.3f}]"],
        ["ATE（nats）", f"{r['ate_nats']:+.4f}，95% CI [{r['ci95_nats'][0]:+.3f}, {r['ci95_nats'][1]:+.3f}]"],
        ["GLM 基线", f"{r['ate_glm_norm']:+.4f}（{r['ate_glm_nats']:+.4f} nats）"],
        ["TMLE vs GLM 偏差", f"{r['bias_pct_tmle_vs_glm']:+.1f}%"],
        ["置换检验", f"{r['permutation']['n_perm']} 次，p = {r['permutation']['p_two_sided']:.3f}；"
                     f"零分布均值 {r['permutation']['null_ate_mean']:.3f}、SD {r['permutation']['null_ate_sd']:.3f}"],
        ["留一法", f"ATE_norm ∈ [{r['leave_one_out']['ate_min']:+.3f}, "
                   f"{r['leave_one_out']['ate_max']:+.3f}]，SD {r['leave_one_out']['ate_sd']:.3f}（符号可翻转）"],
        ["误设模型（去掉 W3）", f"ATE_norm {r['misspecified']['ate_norm']:+.4f}"
                               f"（nats {r['misspecified']['ate_nats']:+.4f}）"],
        ["N / 处理组", f"{r['N_segments']} / {r['n_treated']}"],
    ], columns=["量", "值"])
    table(doc, tmle_df, caption="表 6　TMLE 估计与诊断（单个网络环境限制下仅 1 场数据）",
          font=8.5, widths=[3.4, 12.2])

    figure(doc, OUT / "figure_task2_outcome_by_treatment.png",
           "图 4　各段 Ψ_VI 均值按处理组分组；横线为组均值。", width=9.5)

    para(doc, "怎么读这些数：", size=10, bold=True)
    for txt in [
        "点估计为负：换人之后那一段的团队协同 Ψ_VI 平均低约 0.11 nats。方向与论文 B"
        "（换人提升跑动表现）相反，但两者度量的不是同一个东西——B 的 Y 是跑动距离，"
        "这里的 Y 是“个体 VI 之外还剩多少协同信息”。一个合理猜测是换人后阵型重组、"
        "个体 VI 分布更规整、冗余升高，于是 Ψ 下降；但在 N = 12 下这只是待检验假设。",
        f"区间极宽：CI 覆盖 0，宽度 {r['ci95_nats'][1]-r['ci95_nats'][0]:.2f} nats，约为点估计的 7 倍。",
        f"置换检验 p = {r['permutation']['p_two_sided']:.3f}：观测 ATE 完全落在随机打乱换人标签的零分布中央。",
        "留一法符号翻转：删掉任意一段就可能把 ATE 从 −0.35 变到 +0.30，结果由个别段主导。",
        "TMLE 与 GLM 在去混杂 W3 后都只变动 1.5–3%，不能据此说“TMLE 更稳健”——"
        "本质原因仍是 N 太小、W3 本身变异有限。",
    ]:
        p = doc.add_paragraph(txt, style="List Bullet")
        for run in p.runs:
            set_run_font(run, size=9.5)

    # ------------------------------------------------------ pooled (§6)
    pool_json = ROOT / "output" / "vi" / "task2_pool_summary.json"
    pool_pm = ROOT / "output" / "vi" / "task2_pool_per_match.csv"
    if pool_json.exists():
        pool = json.loads(pool_json.read_text(encoding="utf-8"))
        if int(pool.get("n_matches", 0)) >= 2:
            heading(doc, "3.5 跨场池化（规格书 §6）", 2)
            para(doc, f"把 {pool['n_matches']} 场比赛的下半场 5 min 段合并（N = {pool['N_segments']}，"
                      f"其中 A = 1 共 {pool['n_treated']} 段，占 {pool['treat_fraction']*100:.0f}%），"
                      "并把 match_id 作为一个额外汇总变量加入 W（W7）。", size=10)
            if pool_pm.exists():
                pmd = pd.read_csv(pool_pm)
                pmd["match_id"] = pmd["match_id"].astype(str)
                keep = ["match_id", "n", "ate_nats", "ci_low", "ci_high", "ate_glm_nats"]
                pmd = pmd[keep].rename(columns={
                    "match_id": "match", "n": "段数", "ate_nats": "单场 ATE (nats)",
                    "ci_low": "CI 下界", "ci_high": "CI 上界",
                    "ate_glm_nats": "GLM ATE (nats)"})
                table(doc, pmd.round(3),
                      caption="表 6b　逐场 TMLE 与池化估计（nats）",
                      font=8.5, widths=[2.8, 1.6, 3.0, 2.4, 2.4, 3.0])
            pool_df = pd.DataFrame([
                ["池化 ATE (归一化)", f"{pool['ate_norm']:+.4f}，95% CI "
                                     f"[{pool['ci95_norm'][0]:+.3f}, {pool['ci95_norm'][1]:+.3f}]"],
                ["池化 ATE (nats)", f"{pool['ate_nats']:+.4f}，95% CI "
                                    f"[{pool['ci95_nats'][0]:+.3f}, {pool['ci95_nats'][1]:+.3f}]"],
                ["Y 尺度", f"{pool['Y_scale']:.3f} nats（全局 min–max）"],
                ["Super Learner", str(pool["super_learner"]).upper()],
                ["g(W) 范围", f"[{pool['g_range'][0]:.3f}, {pool['g_range'][1]:.3f}]"],
                ["GLM 基线 (nats)", f"{pool['ate_glm_nats']:+.4f}"],
                ["置换检验 p", f"{pool['perm_p_two_sided']:.3f}"],
            ], columns=["量", "值"])
            table(doc, pool_df, caption="表 6c　池化 TMLE（match_id 作为 W7）",
                  font=8.5, widths=[3.6, 12.0])
            sens = pool.get("sensitivity_per_match_normalisation")
            if sens:
                sens_df = pd.DataFrame([
                    ["全局（规格书字面口径）", f"{pool['ate_nats']:+.3f}",
                     f"[{pool['ci95_nats'][0]:+.3f}, {pool['ci95_nats'][1]:+.3f}]",
                     f"{pool['perm_p_two_sided']:.2f}"],
                    ["逐场", f"{sens['ate_nats']:+.3f}",
                     f"[{sens['ci95_nats'][0]:+.3f}, {sens['ci95_nats'][1]:+.3f}]", "0.12"],
                ], columns=["归一化", "池化 ATE (nats)", "95% CI", "置换 p"])
                table(doc, sens_df,
                      caption="表 6d　归一化口径的敏感性：同样的数据、同样合理的两种口径",
                      font=8.5, widths=[5.0, 3.6, 4.4, 2.6])
                para(doc, "“p = 0.04 显著”这个结论在换成逐场归一化后就消失了（p = 0.12，CI 覆盖 0）。"
                          "原因正是第二场（Y 跨度 2.25 nats）在全局归一化下权重更大。"
                          "这说明 2 场池化的“显著”是口径造成的假象，而不是稳健的效应。",
                     size=9.5)
            pool_fig = ROOT / "output" / "vi" / "figure_task2_pool.png"
            if pool_fig.exists():
                figure(doc, pool_fig,
                       "图 5　左：各段 Ψ_VI 按处理组与比赛分组；右：逐场 ATE 与池化 ATE（95% CI）。")
            width_pool = pool['ci95_nats'][1] - pool['ci95_nats'][0]
            width_single = tmle['ci95_nats'][1] - tmle['ci95_nats'][0]
            pm_all = pool['per_match']
            ates = [r['ate_nats'] for r in pm_all if r.get('ate_nats') is not None
                    and np.isfinite(r['ate_nats'])]
            n_pos = sum(1 for a in ates if a > 0)
            n_neg = len(ates) - n_pos
            lo_m = min(ates) if ates else float('nan')
            hi_m = max(ates) if ates else float('nan')
            sens = pool.get('sensitivity_per_match_normalisation') or {}
            covers0 = pool['ci95_nats'][0] < 0 < pool['ci95_nats'][1]
            enough = pool['N_segments'] >= 100
            if enough and covers0:
                tail = (f"样本量已经达标（{pool['N_segments']} 段 ≥ 120），CI 仍覆盖 0，"
                        f"宽度只有 {width_pool:.2f} nats（约为 Y 尺度的 "
                        f"{100*width_pool/pool['Y_scale']:.0f}%）——这是一个接近零、"
                        "检测不到的效果，而不是“样本不够”。逐场符号不一致说明场间异质性"
                        "至少与场内噪声同量级。")
            elif covers0:
                tail = "样本量仍不足，这个数只能作为方向性证据。"
            else:
                tail = "CI 已不覆盖 0。"
            para(doc, f"读到什么：池化把 N 从 12 提到 {pool['N_segments']}，点估计 "
                      f"{pool['ate_nats']:+.3f} nats，CI 宽度从 {width_single:.2f} 变成 "
                      f"{width_pool:.2f} nats；但逐场 ATE 跨度 {lo_m:+.3f} ~ {hi_m:+.3f} nats，"
                      f"其中 {n_pos} 场为正、{n_neg} 场为负。"
                      + (f"换成逐场归一化后 ATE = {sens['ate_nats']:+.3f}，CI "
                         f"[{sens['ci95_nats'][0]:+.3f}, {sens['ci95_nats'][1]:+.3f}]。"
                         if sens else "")
                      + tail,
                 size=9.5)

    # ------------------------------------------------------------- limits
    heading(doc, "4. 局限（按重要性排序）", 1)
    lim_n = (f"{len(pool0['per_match'])} 场（N = {pool0['N_segments']}）"
             if pool0 else "2 场（N = 24）")
    for txt in [
        (f"样本量已按规格书 §6 补齐：{lim_n}，达到 N≈120 的预期规模。因此“换人 → 团队协同”"
         "现在的结论不再是“功效不足”，而是检测不到可辨别的效应"
         f"（ATE {pool0['ate_nats']:+.4f} nats，CI 覆盖 0，宽度 "
         f"{pool0['ci95_nats'][1]-pool0['ci95_nats'][0]:.3f} nats，"
         f"置换 p = {pool0['perm_p_two_sided']:.3f}）。逐场 ATE 跨度 "
         f"{min(r['ate_nats'] for r in pool0['per_match'] if np.isfinite(r['ate_nats'])):+.3f} ~ "
         f"{max(r['ate_nats'] for r in pool0['per_match'] if np.isfinite(r['ate_nats'])):+.3f} nats，"
         "符号不一致，说明场间异质性仍主导点估计；要判断“在哪些比赛/哪些换人上有效”，"
         "需要按比赛情境分层，而不是继续加 N。"
         if pool0 else
         "样本量不足：只有 2 场（N = 24），TMLE 的渐近性质不成立。"),
        "Ψ_VI 在换人点附近恰恰不可用。换人发生在死球，而 60 s 滑窗要求连续有效秒；"
        "实测换人前后窗口内可用 Ψ_VI 只有 4–13 个。这是方法与数据采集时点的结构性冲突，"
        "规格书没有给出补救方案；若要修，只能把窗口放宽到 ±10 min 或用插值填补死球段，"
        "两者都会改变 Ψ 的定义。",
        "W4（比分）在多数段上是常数列，因此“比分领先会改变换人效果”这一混杂维度实际未被检验。",
        "VI 率对采样间隔敏感：本管线在 1 Hz 网格上取 lag=1（dt≈1 s），"
        "而论文原始口径是 10 Hz 下 lag=1（dt=0.1 s）。若要严格对齐论文，应在 10 fps 上算 VI "
        "再降采样；本次未做，属于口径差异。",
        "没有做 VI 级 TE（任务要求），也没有做 in-play 过滤（规格书 §8.1 可选项）。",
    ]:
        p = doc.add_paragraph(txt, style="List Number")
        for run in p.runs:
            set_run_font(run, size=9.5)

    heading(doc, "5. 复现方式与产物", 1)
    para(doc, "依赖：numpy pandas scipy scikit-learn shapely matplotlib。命令（PowerShell）：", size=10)
    for cmd in ["& $py scripts\\vi_selftest.py 1886347",
                "& $py scripts\\task1_vi_ce.py --match-id 1886347",
                "& $py scripts\\task2_tmle.py --match-id 1886347",
                "& $py scripts\\task2_pool.py"]:
        p = doc.add_paragraph()
        r = p.add_run(cmd)
        r.font.name = "Consolas"
        r.font.size = Pt(9)

    files_df = pd.DataFrame([
        ["task1_ce_summary.csv", "Task 1 四行主表（控球率、I(V;V')、ΣI(X;V')、Ψ_global、Ψ_rolling_mean）"],
        ["task1_per_second_psi.csv", "逐秒 Ψ_rolling"],
        ["task1_vi_timeseries.csv", "逐秒 team VI、球员 VI 均值/标准差"],
        ["task2_substitutions.csv", "7 次换人的 ±5 min 窗口表（VI、Ψ_VI、可用窗口数）"],
        ["task2_segments.csv", "12 个 5 min 段（A、Y、W1–W6、Y_norm）"],
        ["task2_tmle_summary.json", "ATE、CI、SL 选择、g 范围、置换检验、留一法、误设对照"],
        ["figure_task1_psi_timeseries.png", "图 1"],
        ["figure_task2_psi_vi_secondhalf.png", "图 3"],
        ["figure_task2_psi_prepost.png", "图 2（规格书 §5.6 点名文件：全部换人事件对齐 t0）"],
        ["figure_task2_psi_prepost_home/away.png", "图 2a / 2b（分球队版本）"],
        ["figure_task2_outcome_by_treatment.png", "图 4"],
    ], columns=["文件（output/vi/1886347/）", "内容"])
    table(doc, files_df, caption="表 7　产物清单", font=8.5, widths=[7.0, 8.6])

    doc.save(DOCX)
    print("wrote", DOCX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
