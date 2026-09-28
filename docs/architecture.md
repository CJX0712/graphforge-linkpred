# GraphForge 架构说明

作者：晨星　|　版本：v0.1.0

---

## 1. 设计目标

| 目标 | 落地方式 |
|------|----------|
| 经典方法忠实实现 | node2vec（Grover & Leskovec, KDD 2016）原文算法，不引入自研「新 SOTA」 |
| 干净环境一键复现 | 固定依赖清单 + 零手工步骤，`python examples/run_demo.py` 即可 |
| 依赖可降级 | 所有 Tier-2 后端 try/except + `available_*()` 探测，缺失走纯 numpy Tier-1 |
| 无网络依赖 | 不使用 torch / torch-geometric / gensim，无预训练权重下载 |
| 结果可复核 | 同 seed 逐位可复现，benchmark.json 每个数字来自真实运行并读回校验 |

---

## 2. 分层与调用方向（单向无环）

```
                       cli.py
                          |
                     pipeline/  (link_prediction.py, benchmark.py)
                          |
   +----------+-----------+-----------+----------+---------+
   |          |           |           |          |         |
 data/     hpo/       training/    graph/     eval/    preprocess/
   |          |           |           |          |         |
   +----------+-----------+-----------+----------+---------+
                          |
                       core/
```

`core` 不依赖任何上层；`eval` 被 `hpo` 复用（指标计算）但 `eval` 不反向依赖 `hpo`。

### 2.1 `core/` —— 契约与地基

| 模块 | 内容 |
|------|------|
| `types.py` | `GraphData` / `EdgeSplit` / `EmbeddingResult` / `SeedResult` / `BenchmarkResult` / `FailureCase` |
| `errors.py` | 统一错误码：E0xx 基类、E1xx 配置、E2xx 数据、E3xx 图算法、E4xx 训练、E5xx 评测 |
| `config.py` | `GraphForgeConfig` dataclass；`GRAPHFORGE_*` 环境变量覆盖；`validate_config()` 做 schema 校验 |
| `interfaces.py` | `GraphSource` / `WalkSampler` / `Embedder` / `EdgeOperator` / `Classifier` / `Scorer` / `Tuner` / `Pipeline` 八个 Protocol |
| `seed.py` | `derive_seed(base, tag)` 为每个组件派生独立随机流，避免共享 Generator 的顺序耦合 |
| `capabilities.py` | `available_sklearn()` 等探测；`forced_tier1()` 上下文用于离线兜底演练 |

### 2.2 `data/` —— 图与划分

- `synthetic.py`：`stochastic_block_model`（经典 SBM）、`lfr_graph`（LFR 基准，networkx 抽样失败时降级到度修正 SBM）、`karate_club`（内置真实小图）。
- `split.py`：划分保证四条不变量
  1. `train_graph` 由 train 边**重建**邻接矩阵 → test/valid 边被物理删除；
  2. 生成森林边强制留在训练集 → 训练图连通；
  3. 负样本从「非全集正边」中采样 → 不会把真实边误标为负类；
  4. 各折正负样本 1:1 并打乱。

### 2.3 `graph/` —— 领域算法

#### `walks.py` node2vec 二阶偏置游走

从 `prev --t--> cur` 走向候选 `x` 的未归一化权重为

```
alpha(prev, x) * w(cur, x)
alpha = 1/p   x == prev              （回退，d = 0）
        1     (prev, x) 有边         （同距，d = 1）
        1/q   其它                   （外扩，d = 2）
```

实现要点：**所有 walker 同步走一步**。把邻接表 padding 成 `(n, max_deg)` 的 `nbr_ids / nbr_weights / nbr_mask`，
每步只需 `nbr_ids[cur]` 一次 gather + 逐元素 α + 累积分布比较，复杂度与 walker 数线性、与步数无关地摊薄。
第一步按边权均匀（无 prev），孤立节点原地不动。

#### `embed.py` Skip-Gram + Negative Sampling

- 正样本：游走窗口内的 `(center, context)`，窗口长度逐条游走按 `U[1, window]` 动态采样；
- 负样本：`degree^{3/4}` 分布（word2vec 的 3/4 次幂平滑），累积分布 + `searchsorted` 采样；
- 优化：Adam（默认）/ SGD（线性衰减），minibatch 向量化；
- 梯度回传：`sgns_forward_backward()` 用 batched matmul 而非 `np.einsum`（后者在 `(B,K,d)` 模式上无 BLAS、慢 10 倍以上）；
- 梯度累加：`_scatter_add()` 走 `argsort + np.add.reduceat`，比 `np.add.at` 快一个量级。

损失：`-log σ(s_pos) - Σ_k log σ(-s_neg)`，其中 `s = v_center · u_context`。

#### `edges.py` 边特征算子

`hadamard / average / weighted_l1 / weighted_l2 / concat`，前四种对端点交换对称。

#### `heuristics.py` 启发式基线

`common_neighbors / adamic_adar / resource_allocation / jaccard / preferential_attachment /
katz / spectral / random`。Katz 用 `np.linalg.solve` 解 `(I - βA)K = βA`，β 默认 `0.5/ρ(A)`（ρ 由幂迭代估计）。
`spectral` 用秩-k 截断 SVD 重构 `A_hat` 后取 `A_hat[u, v]`（Eckart–Young 最优低秩逼近）。

### 2.4 `preprocess/` —— 特征装配

`FeatureAssembler` 把「嵌入算子特征 + 度特征 + 可选启发式特征」拼成一个边特征矩阵。
`fit_transform_train()` 是**唯一**允许 fit 标准化器的入口，valid/test 只做 `transform()`。

### 2.5 `hpo/` —— 超参搜索

Optuna TPE（Tier-2）/ 确定性网格（Tier-1）。搜索空间：`p, q ∈ [0.25, 4]`（对数均匀）、`dim ∈ {32, 64, 128}`。
目标函数只用 train 折训练、valid 折打分，**test 折全程不参与**。

### 2.6 `training/` —— 分类器与训练器

`Node2VecLinkPredictor` 串起 游走 → 嵌入 → 边特征 → 分类器。
分类器后端：`SklearnLogisticRegression`（lbfgs）或 `NumpyLogisticRegression`（IRLS/Newton，含偏置项与 L2）。

### 2.7 `eval/` —— 指标与报告

`roc_auc_score` / `average_precision_score` 均有 sklearn 与纯 numpy 双实现。
**递归坑规避**：sklearn 指标一律以 `_sk_` 前缀别名导入。
`analyze_failures()` 输出误报 / 漏检各 top-k 及归因文本。

---

## 3. 确定性设计

```
seed ──derive_seed──> graph / split / walk / embed / hpo-trial-i / classifier
```

每个组件持有自己的 `np.random.Generator`，因此：
- 增删某个组件的随机调用不会污染其它组件；
- 同一 seed 两次运行，图、划分、游走、嵌入、指标**逐位一致**。

---

## 4. 离线兜底矩阵

| 组件 | Tier-2（有 sklearn/scipy/optuna/networkx） | Tier-1（纯 numpy） |
|------|-------------------------------------------|--------------------|
| 逻辑回归 | sklearn `LogisticRegression(lbfgs)` | 自实现 IRLS/Newton |
| 超参搜索 | Optuna TPE | 确定性网格 `GRID_CANDIDATES` |
| ROC-AUC / AP | `sklearn.metrics` | 秩实现（并列取平均秩） |
| 最大连通分量 | `networkx.connected_components` | 纯 numpy BFS |
| LFR 生成 | `networkx.LFR_benchmark_graph` | 度修正 SBM |
| 截断 SVD | 全 SVD（n ≤ 1500） | 随机化 SVD |

`--tier1-only` 或 `capabilities.forced_tier1()` 可强制走 Tier-1，该路径有专门单测。

---

## 5. 性能与预算

- demo 端到端目标 ≤ 60s CPU、内存峰值 ≤ 2GB；
- 主要成本：SGNS 的 `(B, K, d)` 批量打分与梯度。通过 batched matmul + `reduceat` scatter-add + float32 把单 batch 成本压到 ~2ms（B=2048, K=5, d=64）；
- 图规模控制在 700~900 节点，配合 `num_walks` / `walk_length` / `window` 控制正负样本对数在 10^6 量级。

---

## 6. 已知限制

1. 图以稠密 `(n, n)` 邻接矩阵表示，节点数超过 ~5000 时内存与 SVD 成本显著上升；
2. HPO 默认只在第一个 seed 的 train/valid 上做一次，超参跨 seed 复用；
3. 纯 numpy 逻辑回归为全批量 IRLS，特征维度过大时 Hessian 求解成本上升；
4. 合成基准图上，node2vec 类方法与 Katz 等强全局启发式的差距本身很小（详见 model_card 的敏感性说明）。
