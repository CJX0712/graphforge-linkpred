"""全局确定性：seed 派生与 RNG 构造。

同 seed 两次运行必须逐位一致，因此每个组件使用 derive_seed 派生的
独立子流，而不是共享一个被多次消费的 Generator。

作者: 晨星
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

_SEED_MIN = 0
_SEED_MAX = 2**32 - 1


def normalize_seed(seed: int) -> int:
    """把任意整数折叠到 [0, 2^32-1]。"""
    return int(seed) % (_SEED_MAX + 1)


def derive_seed(base_seed: int, tag: str) -> int:
    """从 base_seed 与组件标签确定性地派生子 seed。"""
    payload = f"{normalize_seed(base_seed)}::{tag}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (_SEED_MAX + 1)


def set_global_seed(seed: int) -> int:
    """设置 Python / numpy 全局随机源，返回归一化后的 seed。"""
    s = normalize_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    return s


def make_rng(seed: int) -> np.random.Generator:
    """构造独立的 numpy Generator（PCG64）。"""
    return np.random.default_rng(normalize_seed(seed))
