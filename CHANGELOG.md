# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/) 格式，版本号遵循语义化版本。

## [0.1.0] - 2026-09-27

作者：晨星

### Added
- `core`：types / errors(E100~E500) / config(`GRAPHFORGE_*` 环境变量覆盖 + schema 校验) /
  interfaces(Protocol 契约) / seed(确定性派生) / capabilities(Tier-1 / Tier-2 能力探测)
- `data`：LFR 基准图（networkx，失败降级度修正 SBM）、随机块模型 SBM、内置真实小图
  Zachary Karate Club；train/valid/test 划分保证 holdout 边物理删除与训练图连通
- `graph`：node2vec 二阶偏置随机游走（向量化，全 walker 同步走一步）、
  Skip-Gram + Negative Sampling（degree^{3/4} 负采样，纯 numpy Adam/SGD）、
  五种边特征算子（Hadamard / Average / Weighted-L1 / Weighted-L2 / Concat）、
  八个启发式基线（CN / AA / RA / Jaccard / PA / Katz / 谱方法 / Random）
- `preprocess`：边特征装配与标准化，标准化器只在训练折 fit
- `hpo`：Optuna TPE 搜索（仅用 train/valid）+ 确定性网格搜索兜底
- `training`：sklearn LogisticRegression（Tier-2）+ 纯 numpy IRLS 逻辑回归（Tier-1）
- `eval`：ROC-AUC / AP / Accuracy（sklearn 与纯 numpy 双实现）+ 误例归因
- `pipeline`：端到端流水线与跨 seed 基准对比（mean ± std + 显著性判定）
- `cli`：info / operators / heuristics / run / benchmark 五个子命令
- 文档：architecture.md、model_card.md、README（五徽章）
- CI：ruff lint + pytest + demo 冒烟 + Tier-1 兜底冒烟

### Notes
- 不使用 torch / torch-geometric / gensim / lancedb，无任何需联网下载的预训练权重
- 所有可选依赖均 try/except + `available_*()` 探测，缺失时自动降级
