# vi-te —— VI 空间主链提取与成对转移熵（TE）计算

从 SkillCorner 足球轨迹数据中提取**可变性强度（Variability Intensity, VI）**
链，构造球队宏观 VI 链 `V_t` 与球员微观 VI 链 `X_k(t)`（1 Hz），再计算
宏观 ↔ 微观之间的**成对条件转移熵**——这是 Rosas 因果涌现框架
（下行因果 `D_t = Σ_i I(V_t; X^i_{t+1} | X^i_t)`）的 TE 估计实现：

```
D_k = TE(V→X_k) = I(V_t; X_k(t+1) | X_k(t))   （精确 Rosas 下行因果，1 阶）
U_k = TE(X_k→V) = I(X_k(t); V_{t+1} | V_t)    （上行对偶项）
cross_* = TE(V_opp→X_own,k)                   （跨队负对照：重叠伪影基线）
Ψ_uncond = I(V_t;V_{t+1}) − Σ_k I(X_k(t);V_{t+1})   （vi 项目 Task-2 涌现口径）
```

四个估计器口径：gaussian 闭式 / KSG（Frenzel-Pompe，k∈{3,5}）/ 离散
（K-means K=3）；代理检验 200 块置换（block=25，冻结标准化），
`eff = raw − mean(surr)`，`p = (#surr≥raw+1)/(S+1)`。

## 仓库布局

| 路径 | 内容 |
|---|---|
| `vi/src/` | VI 项目源码（`skillcorner.py` 数据加载、`vi_cluster.py`/`vi.py` 链提取、`ce_scalar.py` MI/Ψ 估计、`vi_pipeline.py` 管道） |
| `vi/scripts/` | VI 项目分析脚本（task1 因果涌现 / task2 表现结果 / TMLE / 自检） |
| `vi/tools/` | `fetch_match.py` 数据获取工具 |
| `vi_bridge/` | **TE↔VI 桥接包**：`vi_te.py` 引擎、`run_vi_te.py` 单场、`run_all.py` 十场并行、`verify_pool.py` 质控、`fix_skipped_pids.py` 修复 |
| `mte/` | 成对 TE 估计器栈（与 gvrnn-te 同源） |
| `tests/` | pytest 用例（7 个，全部合成数据，`python -m pytest tests/ -q`） |

## 安装

```bash
git clone https://github.com/han-jiahe/vi-te.git
cd vi-te
pip install -e .
# 或作为依赖（下游 gvrnn-te 仓库的 te_emergence 包正是这样调用）：
# pip install git+https://github.com/han-jiahe/vi-te.git
```

依赖：`numpy scipy scikit-learn pandas shapely matplotlib`（见
[requirements.txt](requirements.txt)）。`vi/src/skillcorner.py` 需要 `shapely`
解析球场几何。

> 说明：`vi/` 树不打包安装——其脚本自带仓库内 `sys.path` 定位
> （`from src.ce_scalar import …`），请在仓库根目录运行；
> `vi_bridge.vi_te` 对 vi 源码采用延迟导入，作为 pip 依赖安装时
> 只提供纯 numpy/sklearn 的 TE 估计与变换函数，构造 VI 链的功能
> 在仓库内运行时可用。

## 快速开始（工作流）

> 数据说明：默认使用 **SkillCorner opendata**（见下方「数据许可」）。

```bash
# 0. 数据：下载 SkillCorner opendata 到本地，例如 D:/SkillCorner_opendata/data，
#    并把其中 20 场（Auckland 相关）的 tracking 放成 vi 项目布局：
#    vi/data/raw/<match_id>/tracking_cache_v2.npz（由 vi/tools/fetch_match.py 辅助）

# 1. VI 链自检（vi 项目口径复现）
python -m vi.scripts.vi_selftest

# 2. 单场 TE 全量（4 方向 × 4 方法口径，200 代理）
python -m vi_bridge.run_vi_te --match-id 1886347 \
    --raw-dir vi/data/raw --out results/vi_te_1886347.json

# 3. 十场并行
python -m vi_bridge.run_all

# 4. 质控（池化前不变量筛查；汇总/统计/图由下游 gvrnn-te 仓库的
#    te_emergence.pool 完成）
python -m vi_bridge.verify_pool
```

产出物（`vi/data/`、`vi/output/`、`vi_bridge/results/`）已加入
`.gitignore`，不上传仓库。

## 测试

```bash
python -m pytest tests/ -q     # 7 用例，全部合成数据，无外部依赖
```

## 数据许可（SkillCorner opendata）

- 数据来源：**[SkillCorner / opendata](https://github.com/SkillCorner/opendata)**（MIT License，© SkillCorner）。
- 本仓库**不包含任何比赛数据**；请自行从 SkillCorner opendata 下载。
- **使用该数据须注明来源**（"Data: SkillCorner Open Data, MIT licensed, © SkillCorner"），
  并在发布时**通知 SkillCorner**（见其仓库说明）。
- 本仓库代码以 MIT 许可发布（见 [LICENSE](LICENSE)），与数据许可相互独立。

## 关联仓库

- [gvrnn-te](https://github.com/han-jiahe/gvrnn-te) —— GVRNN 潜链提取与 TE（同一 `mte` 估计器栈）
- [gvrnn-te](https://github.com/han-jiahe/gvrnn-te) —— GVRNN 潜链提取与 TE（其 `te_emergence` 包为因果涌现计算层，通过 `pip install git+…/vi-te.git` 调用本仓库）

## 引用

- Rosas et al. 2020, PLoS Comput Biol 16(12):e1008289, DOI 10.1371/journal.pcbi.1008289（因果涌现）
- Schreiber 2000, PRL 85:461（转移熵）
- Kraskov et al. 2004, PRE 69:066138 / Frenzel & Pompe 2007, PRL 99:204101（KSG）
