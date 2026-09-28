"""可选后端能力探测（Tier-2 加速 / Tier-1 纯 numpy 兜底）。

所有可选依赖一律 try/except ImportError，缺失时返回 False，
上层据此降级到 Tier-1 纯 numpy 实现，绝不硬失败。

作者: 晨星
"""

from __future__ import annotations

import contextlib
import importlib
from functools import lru_cache

# 全局开关：置 True 时所有 Tier-2 后端一律视为缺失，用于离线兜底路径演练与单测。
_FORCE_TIER1 = False


def set_force_tier1(enabled: bool) -> None:
    """强制降级到 Tier-1（纯 numpy），用于验证离线兜底路径。"""
    global _FORCE_TIER1
    _FORCE_TIER1 = bool(enabled)


def force_tier1_enabled() -> bool:
    return _FORCE_TIER1


@lru_cache(maxsize=None)
def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - 探测即容错
        return False
    return True


def available_numpy() -> bool:
    return _has("numpy")


def available_scipy() -> bool:
    return _has("scipy")


def available_sklearn() -> bool:
    return (not _FORCE_TIER1) and _has("sklearn")


def available_networkx() -> bool:
    return _has("networkx")


def available_optuna() -> bool:
    return _has("optuna")


def available_gensim() -> bool:
    """gensim 无 cp313 wheel，仅作可选加速，缺失不影响主链路。"""
    return _has("gensim")


def available_torch() -> bool:
    """GraphForge 不使用 torch；此处仅用于环境报告。"""
    return _has("torch")


def backend_tier(force_tier1: bool = False) -> str:
    """返回 'tier2'（有 sklearn/scipy 加速）或 'tier1'（纯 numpy 兜底）。"""
    if force_tier1:
        return "tier1"
    if available_sklearn() and available_scipy():
        return "tier2"
    return "tier1"


@contextlib.contextmanager
def forced_tier1():
    """上下文内强制 Tier-1（离线兜底演练 / 单测）。"""
    previous = _FORCE_TIER1
    set_force_tier1(True)
    try:
        yield
    finally:
        set_force_tier1(previous)


def capability_report(force_tier1: bool = False) -> dict[str, object]:
    """返回能力探测快照，供日志 / benchmark.json 记录。

    force_tier1=True 时，Tier-2 加速后端（scipy / sklearn / optuna）一律按缺失报告，
    以如实反映离线兜底路径实际可用的能力。
    """
    masked = force_tier1 or force_tier1_enabled()
    return {
        "numpy": available_numpy(),
        "scipy": (not masked) and available_scipy(),
        "sklearn": (not masked) and _has("sklearn"),
        "networkx": available_networkx(),
        "optuna": (not masked) and _has("optuna"),
        "gensim": available_gensim(),
        "torch": available_torch(),
        "tier": backend_tier(masked),
    }
