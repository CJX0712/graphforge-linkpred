"""node2vec 链接预测训练器：游走 -> 嵌入 -> 边特征 -> 分类器。

作者: 晨星
"""

from __future__ import annotations

import time

import numpy as np

from ..core.capabilities import backend_tier
from ..core.config import GraphForgeConfig
from ..core.errors import TrainingError
from ..core.seed import derive_seed, make_rng, normalize_seed
from ..core.types import EdgeSplit, EmbeddingResult
from ..graph.embed import SkipGramNS
from ..graph.walks import Node2VecWalker
from ..preprocess.features import FeatureAssembler
from .classifier import make_classifier


class Node2VecLinkPredictor:
    """端到端 node2vec 链接预测模型。"""

    def __init__(
        self,
        config: GraphForgeConfig,
        seed: int | None = None,
        force_tier1: bool = False,
        **overrides: object,
    ) -> None:
        self.config = config
        self.seed = normalize_seed(config.seed if seed is None else seed)
        self.force_tier1 = bool(force_tier1)
        self.overrides: dict[str, object] = dict(overrides)

        self.walker = Node2VecWalker()
        self.assembler: FeatureAssembler | None = None
        self.embedding: EmbeddingResult | None = None
        self.classifier = None
        self.backend_name = ""
        self._train_features: np.ndarray | None = None
        self.walk_count = 0
        self.pair_count = 0
        self.seconds = 0.0
        self.params_used: dict[str, object] = {}

    # ------------------------------------------------------------------ 取值
    def _param(self, key: str, default: object) -> object:
        if key in self.overrides and self.overrides[key] is not None:
            return self.overrides[key]
        return getattr(self.config, key, default)

    # ------------------------------------------------------------------- fit
    def fit(self, split: EdgeSplit) -> "Node2VecLinkPredictor":
        started = time.perf_counter()
        cfg = self.config
        p = float(self._param("p", cfg.p))
        q = float(self._param("q", cfg.q))
        walk_length = int(self._param("walk_length", cfg.walk_length))
        num_walks = int(self._param("num_walks", cfg.num_walks))
        dim = int(self._param("dim", cfg.dim))
        window = int(self._param("window", cfg.window))
        negative = int(self._param("negative_samples", cfg.negative_samples))
        epochs = int(self._param("epochs", cfg.epochs))
        operator = str(self._param("edge_operator", cfg.edge_operator))
        use_heur = bool(self._param("use_heuristic_features", cfg.use_heuristic_features))
        normalize = bool(self._param("normalize_embeddings", cfg.normalize_embeddings))
        classifier_c = float(self._param("classifier_c", cfg.classifier_c))
        learning_rate = float(self._param("learning_rate", cfg.learning_rate))
        batch_size = int(self._param("batch_size", cfg.batch_size))
        optimizer = str(self._param("optimizer", cfg.optimizer))
        beta1 = float(self._param("beta1", cfg.beta1))
        beta2 = float(self._param("beta2", cfg.beta2))
        eps = float(self._param("eps", cfg.eps))

        self.walker.fit(split.train_graph)
        walk_rng = make_rng(derive_seed(self.seed, "walk"))
        walks = self.walker.sample(walk_rng, num_walks, walk_length, p, q)
        self.walk_count = int(walks.shape[0] * walks.shape[1])

        embedder = SkipGramNS(
            dim=dim,
            window=window,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=optimizer,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            seed=derive_seed(self.seed, "embed"),
        )
        embed_rng = make_rng(derive_seed(self.seed, "embed"))
        self.embedding = embedder.fit(split.train_graph, walks, rng=embed_rng)
        self.pair_count = int(self.embedding.params.get("n_pairs", 0))
        if normalize:
            norms = np.linalg.norm(self.embedding.matrix, axis=1, keepdims=True)
            self.embedding.matrix = self.embedding.matrix / np.maximum(norms, 1e-12)

        self.assembler = FeatureAssembler(
            operator=operator,
            use_heuristic_features=use_heur,
            standardize=True,
        ).fit(split.train_graph)

        x_train = self.assembler.fit_transform_train(self.embedding.matrix, split.train_edges)
        self._train_features = x_train
        if split.train_labels.size and np.unique(split.train_labels).size < 2:
            raise TrainingError("训练折标签只含单一类别", code="E400")

        self.classifier = make_classifier(
            prefer=cfg.classifier,
            c=classifier_c,
            max_iter=cfg.max_iter,
            force_tier1=self.force_tier1,
        ).fit(x_train, split.train_labels)
        self.backend_name = f"{self.classifier.backend()}/{backend_tier(self.force_tier1)}"

        self.params_used = {
            "p": p,
            "q": q,
            "walk_length": walk_length,
            "num_walks": num_walks,
            "dim": dim,
            "window": window,
            "negative_samples": negative,
            "epochs": epochs,
            "edge_operator": operator,
            "normalize_embeddings": normalize,
            "use_heuristic_features": use_heur,
            "classifier_backend": self.classifier.backend(),
            "classifier_c": classifier_c,
        }
        self.seconds = time.perf_counter() - started
        return self

    # --------------------------------------------------------------- predict
    def predict_proba(self, edges: np.ndarray) -> np.ndarray:
        if self.embedding is None or self.assembler is None or self.classifier is None:
            raise TrainingError("模型尚未 fit", code="E400")
        features = self.assembler.transform(self.embedding.matrix, np.asarray(edges, np.int64))
        return np.asarray(self.classifier.predict_proba(features), dtype=np.float64)

    def fit_predict(self, split: EdgeSplit, edges: np.ndarray) -> np.ndarray:
        return self.fit(split).predict_proba(edges)

    # ------------------------------------------------------------ 特征复用
    @property
    def train_features(self) -> np.ndarray:
        """训练折特征矩阵（已按 train 折标准化），供其它后端复用同一嵌入。"""
        if self._train_features is None:
            raise TrainingError("模型尚未 fit，无法复用训练特征", code="E400")
        return self._train_features

    def features_for(self, edges: np.ndarray) -> np.ndarray:
        """给定边集合，返回标准化后的边特征矩阵。"""
        if self.embedding is None or self.assembler is None:
            raise TrainingError("模型尚未 fit", code="E400")
        return self.assembler.transform(self.embedding.matrix, np.asarray(edges, dtype=np.int64))
