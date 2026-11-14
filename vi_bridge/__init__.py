"""vi_bridge —— TE 与 VI 因果涌现的桥接包。

把两两 TE 估计器栈（``mte.*``，order-1、仅条件于目标自身
历史，bits）接到 vi 项目（本仓库 ``vi/``）的 VI 因果涌现框架上：

  V   = 球队 VI 率（team_vi，1 Hz，10 名外场球员 + 本队球门贡献）
  X_k = 第 k 名外场球员的个体 VI 率（player_vi）
  D_k = TE(V→X_k)       = Rosas 2020 下行因果项 I(V_t; X^k_{t+1} | X^k_t)
  U_k = TE(X_k→V)       = 上行因果对偶项
  Δ   = ΣD − ΣU         （Δ>0 ⇒ 下行因果，Rosas 判据）
  Γ_TE = I(V_t;V_{t+1}) − ΣU_k     TE 条件形式的因果涌现/解耦指标
  Ψ_uncond = I(V_t;V_{t+1}) − Σ_k I(X_k(t);V_{t+1})   vi 项目锁定的无条件版（nats）

Rosas 2020: PLoS Comput Biol 16(12):e1008289, doi:10.1371/journal.pcbi.1008289.

子模块：
  vi_te       —— 核心引擎（ViMatch 链提取 / 掩码 / TE 层 / 代理协议 / Ψ 复算）
  run_vi_te   —— 单场 CLI
  run_all     —— 10 场并行运行器
  pool_vi_te  —— 汇总 + 统计 + 图
  latent_te   —— GVRNN 潜空间宏观链交叉校验（第二阶段）
"""

__version__ = "0.1.0"
