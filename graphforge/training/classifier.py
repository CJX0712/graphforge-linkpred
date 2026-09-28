"""下游链接预测分类器。

Tier-2：sklearn.linear_model.LogisticRegression（可用时）
Tier-1：纯 numpy IRLS/Newton 逻辑回归（缺失 sklearn 时自动降级）

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.capabilities import available_sklearn
from ..core.errors import BackendError, TrainingError


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class NumpyLogisticRegression:
    """纯 numpy 逻辑回归（IRLS / Newton 法，含 L2 与偏置项）。"""

    def __init__(self, c: float = 1.0, max_iter: int = 50, tol: float = 1e-8) -> None:
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.n_iter_: int = 0

    def backend(self) -> str:
        return "numpy"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise TrainingError("特征与标签形状不匹配", code="E400")
        if np.unique(y).size < 2:
            raise TrainingError("标签只含单一类别，无法训练分类器", code="E400")

        n, d = x.shape
        ones = np.ones((n, 1), dtype=np.float64)
        xb = np.concatenate([x, ones], axis=1)
        weight = np.zeros(d + 1, dtype=np.float64)
        reg = 1.0 / self.c
        reg_vec = np.full(d + 1, reg, dtype=np.float64)
        reg_vec[-1] = 0.0  # 偏置不惩罚

        for iteration in range(self.max_iter):
            prob = _sigmoid(xb @ weight)
            grad = xb.T @ (prob - y) + reg_vec * weight
            curvature = np.clip(prob * (1.0 - prob), 1e-6, 0.25)
            hessian = (xb * curvature[:, None]).T @ xb
            hessian += np.diag(reg_vec) + np.eye(d + 1) * 1e-9
            try:
                step = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:  # pragma: no cover - 数值退化路径
                step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
            weight -= step
            self.n_iter_ = iteration + 1
            if float(np.linalg.norm(step)) < self.tol:
                break

        if not np.all(np.isfinite(weight)):
            raise TrainingError("逻辑回归参数发散", code="E400")
        self.coef_ = weight
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise TrainingError("分类器尚未 fit", code="E400")
        x = np.asarray(x, dtype=np.float64)
        xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        return xb @ self.coef_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision_function(x))


class SklearnLogisticRegression:
    """sklearn 逻辑回归封装（Tier-2）。"""

    def __init__(self, c: float = 1.0, max_iter: int = 1000) -> None:
        if not available_sklearn():
            raise BackendError("sklearn 不可用", code="E403")
        from sklearn.linear_model import LogisticRegression  # 可选后端

        self._model = LogisticRegression(C=float(c), max_iter=int(max_iter), solver="lbfgs")

    def backend(self) -> str:
        return "sklearn"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SklearnLogisticRegression":
        self._model.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64).ravel())
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.decision_function(np.asarray(x, dtype=np.float64)))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict_proba(np.asarray(x, dtype=np.float64)))[:, 1]


def make_classifier(
    prefer: str = "auto",
    c: float = 1.0,
    max_iter: int = 1000,
    force_tier1: bool = False,
):
    """按偏好构造分类器，返回实例（.backend() 给出实际后端）。"""
    use_sklearn = (not force_tier1) and available_sklearn() and prefer in ("auto", "sklearn")
    if use_sklearn:
        try:
            return SklearnLogisticRegression(c=c, max_iter=max_iter)
        except BackendError:  # pragma: no cover - 构造期兜底
            pass
    if prefer == "sklearn" and not force_tier1:
        raise BackendError("指定 sklearn 后端但不可用", code="E403")
    return NumpyLogisticRegression(c=c, max_iter=max_iter)
