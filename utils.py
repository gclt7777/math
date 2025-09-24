"""通用工具：日志、随机种子、序列化。"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import yaml


def set_global_seed(seed: int) -> None:
    """设置 Python/NumPy 的随机种子，确保可复现。"""

    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """创建目录并返回 Path。"""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def reindex_columns(df, columns: Iterable[str], fill_value: float = 0.0):
    """按给定列顺序补齐 DataFrame。

    pandas 的 ``reindex`` 能一次性完成缺列填充，比逐列判断更高效。
    这里保持 copy，避免链式赋值的潜在副作用。
    """

    if df.empty and not columns:
        return df.copy()
    return df.reindex(columns=list(columns), fill_value=fill_value)


def setup_logger(log_file: str) -> logging.Logger:
    """初始化日志输出（控制台 + 文件，中文友好）。"""

    logger = logging.getLogger("q4_adapt")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
