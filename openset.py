"""开放集拒识策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class OpenSetResult:
    is_rejected: np.ndarray
    scores: np.ndarray
    method: str
    threshold: float


def compute_scores(proba: np.ndarray, method: str = "max_proba") -> np.ndarray:
    if method == "max_proba":
        return proba.max(axis=1)
    if method == "energy":
        return -np.log(np.exp(proba).sum(axis=1))
    raise ValueError("未知的开放集方法")


def apply_open_set(proba: np.ndarray, method: str, threshold: float) -> OpenSetResult:
    scores = compute_scores(proba, method)
    if method == "max_proba":
        is_rejected = scores < threshold
    else:
        is_rejected = scores < threshold  # energy 分数通常越大越可信，阈值留给配置
    return OpenSetResult(is_rejected=is_rejected, scores=scores, method=method, threshold=threshold)
