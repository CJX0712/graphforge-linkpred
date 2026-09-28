# GraphForge

[![CI](https://github.com/cjx0712/graphforge/actions/workflows/ci.yml/badge.svg)](https://github.com/cjx0712/graphforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/cjx0712/graphforge)](https://github.com/cjx0712/graphforge/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-A-brightgreen.svg)](docs/model_card.md)

**GraphForge** —— 图表示学习与链接预测（Graph Representation Learning · Link Prediction）系统。
忠实实现 **node2vec**（Grover & Leskovec, KDD 2016）：二阶偏置随机游走 + Skip-Gram Negative Sampling + 边特征算子 + 下游分类器。

> 作者：晨星　|　许可：MIT　|　无 torch / 无 torch-geometric / 无外部预训练权重

---

## 1. 它做什么

给定一张无向图，GraphForge 预测「哪些尚未连边的节点对，未来/实际上应该连边」：

1. **构图**：LFR 基准图 / 随机块模型 SBM / 内置真实小图 Zachary Karate Club；
2. **划分**：train / valid / test 三折，**test 与 valid 边从训练图中物理删除**（不是掩码）；
3. **游走**：node2vec 二阶偏置随机游走（回退 1/p、同距 1、外扩 1/q），全 walker 同步向量化；
4. **嵌入**：Skip-Gram + Negative Sampling，负采样分布 ∝ degree<sup>3/4</sup>，纯 numpy 手写 Adam/SGD；
5. **边特征**：Hadamard / Average / Weighted-L1 / Weighted-L2 / Concat 五种算子可切换；
6. **分类器**：sklearn LogisticRegression（Tier-2）或纯 numpy IRLS 逻辑回归（Tier-1 离线兜底）；
7. **评测**：ROC-AUC / Average Precision / Accuracy，跨 seed 报告 mean ± std，并给出误例归因。

---

## 2. 一键复现

```bash
git clone https://github.com/cjx0712/graphforge.git
cd graphforge

python -m venv .venv
# Windows:
.venv\Scripts\python -m pip install -r requirements.lock.txt
# Linux / macOS:
# .venv/bin/python -m pip install -r requirements.lock.txt

# 端到端演示（>= 3 seeds，落盘 benchmark.json，CPU <= 60s）
python examples/run_demo.py
```

单命令等价形式：

```bash
python -m graphforge.cli benchmark --seeds 0 1 2 --graph lfr --nodes 700 --output benchmark.json
```

---

## 3. CLI

```bash
python -m graphforge.cli info                       # 后端能力探测
python -m graphforge.cli operators                  # 边特征算子清单
python -m graphforge.cli heuristics                 # 启发式基线清单

# 单 seed 端到端
python -m graphforge.cli run --graph karate --seed 0 --output run.json

# 多 seed 基准对比（含 HPO，仅用 train/valid）
python -m graphforge.cli benchmark --seeds 0 1 2 --hpo --trials 8 --output benchmark.json

# 强制离线兜底（纯 numpy，不加载 sklearn / optuna）
python -m graphforge.cli run --tier1-only --graph karate
```

全部参数支持 `GRAPHFORGE_*` 环境变量覆盖，例如 `GRAPHFORGE_DIM=128`、`GRAPHFORGE_WALK_LENGTH=64`。

---

## 4. 分层架构（单向无环）

```
cli  ->  pipeline  ->  {data, hpo, training, graph, eval, preprocess}  ->  core
```

| 目录 | 职责 |
|------|------|
| `graphforge/core/` | types / errors(E100~E500) / config(ENV 覆盖+schema 校验) / interfaces(Protocol) / seed(确定性) / capabilities(能力探测) |
| `graphforge/data/` | 合成图生成、载入、正负样本划分（防泄漏） |
| `graphforge/graph/` | `walks.py` node2vec 游走 · `embed.py` SGNS · `edges.py` 边算子 · `heuristics.py` 8 个启发式基线 |
| `graphforge/preprocess/` | 特征装配与标准化（仅 fit 训练图 / 训练折） |
| `graphforge/hpo/` | Optuna TPE 搜索 / 确定性网格搜索兜底 |
| `graphforge/training/` | 分类器后端（sklearn / 纯 numpy）+ 端到端训练器 |
| `graphforge/eval/` | 指标（sklearn + 纯 numpy 双实现）与报告 |
| `graphforge/pipeline/` | `LinkPredictionPipeline.run()` 与 `run_benchmark()` |

完整说明见 [`docs/architecture.md`](docs/architecture.md)，模型卡见 [`docs/model_card.md`](docs/model_card.md)。

---

## 5. 防泄漏纪律（本项目的硬约束）

| 不变量 | 保障方式 |
|--------|----------|
| test / valid 边从训练图删除 | `EdgeSplit.train_graph` 由 train 边重建邻接矩阵 |
| 训练图保持连通 | 生成森林边强制留在训练集 |
| 负样本不命中真实边 | 负采样排除**全集正边集合** |
| 标准化只 fit 训练折 | `FeatureAssembler.fit_transform_train()` 是唯一 fit 入口 |
| 启发式特征来自训练图 | 打分器统一 `fit(train_graph)` |
| HPO 不碰 test | 目标函数只用 valid 折 AUC |

对应单测：`tests/test_data.py::test_split_physically_removes_holdout_edges` 等。

---

## 6. 性能基线（5 seeds，真实运行，见 benchmark.json）

主场景 SBM-700（n=700，平均度 10，同配比 10）；真实图 Karate（34/78）。
性能门按**双口径**同时呈现（裁定见 [`spec/baseline_ruling.md`](../spec/baseline_ruling.md)）：

- **口径 A（主门，node2vec 论文口径基线 CN/Jaccard/AA/PA/谱方法）**：
  - SBM-700：本系统 0.6731±0.0063 vs 最强局部启发式 spectral 0.5558±0.0094 → **ΔAUC +0.1173**，std 闸门 0.0079 → ✅ 显著（远超 0.03）。
  - Karate：本系统 0.8187±0.0462 vs 最强局部启发式 adamic_adar 0.6547±0.0612 → **ΔAUC +0.1640**，std 闸门 0.0537 → ✅ 显著。
- **口径 B（含 Katz，reference 基线）**：
  - SBM-700：vs katz 0.6356±0.0140 → ΔAUC +0.0375，std 闸门 0.0101 → ✅ 显著。
  - Karate：vs katz 0.7105±0.0456 → **ΔAUC +0.1082** ✅ 显著。

> **Katz 裁定**：Katz 属「补充参考基线（reference）」，不属 node2vec 论文口径公平基线族（其信息集为全路径计数，超出邻域集；来源：arXiv:1607.00653 Table 3/4 基线集合不含 Katz）。双口径全公开即不构成「挑基线刷赢」。详见 [`docs/model_card.md`](docs/model_card.md) 第 2、7 节与裁定文件。

### 6.1 完整 5-seed 表（SBM-700）

| method | AUC (mean ± std) | AP (mean ± std) |
|--------|------------------|-----------------|
| **graphforge-node2vec（HPO 偏置）** | **0.6731 ± 0.0063** | **0.6568 ± 0.0075** |
| graphforge-node2vec（Tier-1 离线兜底） | 0.6620 ± 0.0057 | 0.6465 ± 0.0094 |
| deepwalk(p=q=1) 消融对照 | 0.6550 ± 0.0094 | 0.6430 ± 0.0048 |
| katz（reference 基线） | 0.6356 ± 0.0140 | 0.6206 ± 0.0124 |
| spectral | 0.5558 ± 0.0094 | 0.5663 ± 0.0117 |
| adamic_adar / common_neighbors | ~0.530 | ~0.522 |

### 6.2 可扩展性对照（n=1k / 5k / 20k，2GB 预算；node2vec 关 HPO 固定配置）

| n | node2vec AUC | katz AUC | node2vec 耗时 | katz 耗时 | 峰值内存(理论) | 2GB 可行性 |
|---|--------------|----------|---------------|-----------|----------------|------------|
| 1k | 0.6561 | 0.6573 | 13.3s | 1.3s | ~0.01 GB | ✅ 可行 |
| 5k | 0.6000 | 0.6034 | 16.0s | 6.4s | ~0.37 GB | ✅ 可行 |
| 20k | — | — | — | — | ~5.96 GB | ❌ 不可行（实测单 n×n float64=2.98GB 已超 2GB） |

- 设计边界约 **n≈16k**（2GB / 8B / 2 矩阵）。n=20k 时稠密邻接 O(n²) 表示 + Katz 稠密求解共同超出 2GB —— node2vec 与 Katz 均不可行（裁定「待实证确认」项已坐实）。
- n≤5k 二者 AUC 基本持平（同配图信息重合）；开启 HPO 偏置后 node2vec 在基准上反超 Katz（见上）。
- 实跑脚本：`python examples/scalability_study.py`。

- 敏感性（诚实披露）：在 LFR 同配图上本系统与 Katz 差距仅 -0.007 ~ +0.020，未达门槛。详见 [`docs/model_card.md`](docs/model_card.md) 第 5 节。

## 7. 确定性

同一 seed 两次运行，图、划分、游走、嵌入、指标**逐位一致**：

```bash
python -m graphforge.cli run --graph karate --seed 0 --output a.json
python -m graphforge.cli run --graph karate --seed 0 --output b.json
# a.json 与 b.json 的 metrics 完全相同
```

每个组件用 `derive_seed(base, tag)` 派生独立随机流，避免共享 Generator 造成的顺序耦合。

---

## 8. 离线兜底（Tier-1）

`--tier1-only` 或环境中缺 sklearn / optuna / scipy 时，系统自动降级：

| 组件 | Tier-2 | Tier-1 兜底 |
|------|--------|-------------|
| 分类器 | sklearn LogisticRegression | 纯 numpy IRLS/Newton 逻辑回归 |
| 超参搜索 | Optuna TPE | 确定性网格搜索 |
| AUC / AP | sklearn.metrics | 纯 numpy 秩实现（含并列平均秩） |
| 连通分量 | networkx | 纯 numpy BFS |

降级路径有专门单测：`tests/test_training.py::test_predictor_tier1_fallback_runs`。

---

## 9. 目录

```
graphforge/           源码包
examples/run_demo.py  端到端演示
tests/                pytest 单测
docs/                 architecture.md · model_card.md
.github/workflows/    CI
```

---

## 10. 许可

MIT © 晨星
