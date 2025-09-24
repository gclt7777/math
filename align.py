"""特征对齐模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Q4Config
from .dataio import DataBundle


@dataclass
class AlignmentResult:
    source_aligned: Optional[pd.DataFrame]
    target_aligned: pd.DataFrame
    target_raw: pd.DataFrame
    adapter_params: Dict[str, np.ndarray | float | list]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]


def _compute_gap(Xs: Optional[np.ndarray], Xt: np.ndarray) -> Dict[str, float]:
    if Xs is None or len(Xs) == 0:
        mean_src = 0
        cov_src = 0
        mmd = 0
    else:
        mean_src = Xs.mean(axis=0)
        cov_src = np.cov(Xs, rowvar=False)
        mmd = _mmd_rbf(Xs, Xt)
    mean_tgt = Xt.mean(axis=0)
    cov_tgt = np.cov(Xt, rowvar=False)
    mean_gap = float(np.linalg.norm(mean_src - mean_tgt)) if isinstance(mean_src, np.ndarray) else float(np.linalg.norm(mean_tgt))
    cov_gap = float(np.linalg.norm(cov_src - cov_tgt, ord="fro")) if isinstance(cov_src, np.ndarray) else float(np.linalg.norm(cov_tgt, ord="fro"))
    return {
        "mean_diff": mean_gap,
        "cov_fro": cov_gap,
        "mmd_rbf": float(mmd),
    }


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 0.5) -> float:
    if X.size == 0 or Y.size == 0:
        return 0.0
    def kernel(a, b):
        dist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * dist)

    Kxx = kernel(X, X)
    Kyy = kernel(Y, Y)
    Kxy = kernel(X, Y)
    m = X.shape[0]
    n = Y.shape[0]
    return Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n)


def _zscore_to_source(X_src: Optional[np.ndarray], X_tgt: np.ndarray) -> tuple[np.ndarray, Dict[str, list]]:
    if X_src is None or X_src.size == 0:
        mu_ref = X_tgt.mean(axis=0)
        std_ref = X_tgt.std(axis=0)
    else:
        mu_ref = X_src.mean(axis=0)
        std_ref = X_src.std(axis=0)
    mu_t = X_tgt.mean(axis=0)
    std_t = X_tgt.std(axis=0)
    std_ref = np.where(std_ref < 1e-12, 1.0, std_ref)
    std_t = np.where(std_t < 1e-12, 1.0, std_t)
    Xt_norm = (X_tgt - mu_t) / std_t
    Xt_aligned = Xt_norm * std_ref + mu_ref
    return Xt_aligned, {
        "mu_target": mu_t.tolist(),
        "std_target": std_t.tolist(),
        "mu_reference": mu_ref.tolist(),
        "std_reference": std_ref.tolist(),
    }


def _coral(X_src: Optional[np.ndarray], X_tgt: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, Dict[str, list]]:
    if X_src is None or X_src.size == 0:
        mu_src = X_tgt.mean(axis=0)
        cov_src = np.cov(X_tgt, rowvar=False)
    else:
        mu_src = X_src.mean(axis=0)
        cov_src = np.cov(X_src, rowvar=False)
    mu_t = X_tgt.mean(axis=0)
    cov_t = np.cov(X_tgt, rowvar=False)

    dim = X_tgt.shape[1]
    cov_src = cov_src + np.eye(dim) * eps
    cov_t = cov_t + np.eye(dim) * eps

    eigvals_t, eigvecs_t = np.linalg.eigh(cov_t)
    eigvals_s, eigvecs_s = np.linalg.eigh(cov_src)

    cov_t_inv_sqrt = eigvecs_t @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_t, eps))) @ eigvecs_t.T
    cov_s_sqrt = eigvecs_s @ np.diag(np.sqrt(np.maximum(eigvals_s, eps))) @ eigvecs_s.T

    Xt_centered = X_tgt - mu_t
    Xt_aligned = Xt_centered @ cov_t_inv_sqrt @ cov_s_sqrt + mu_src

    return Xt_aligned, {
        "mu_source": mu_src.tolist(),
        "mu_target": mu_t.tolist(),
        "cov_source": cov_src.tolist(),
        "cov_target": cov_t.tolist(),
    }


def align_features(bundle: DataBundle, cfg: Q4Config) -> AlignmentResult:
    X_src = bundle.source_features.to_numpy(dtype=float) if bundle.source_features is not None else None
    X_tgt = bundle.target_features.to_numpy(dtype=float)

    before = _compute_gap(X_src, X_tgt)
    adapter_params: Dict[str, np.ndarray | float | list] = {"method": cfg.adapt.feature_align}

    if cfg.adapt.feature_align == "none":
        Xt_aligned = X_tgt
    elif cfg.adapt.feature_align == "zscore_to_source":
        Xt_aligned, stats = _zscore_to_source(X_src, X_tgt)
        adapter_params.update(stats)
    elif cfg.adapt.feature_align == "coral":
        Xt_aligned, stats = _coral(X_src, X_tgt)
        adapter_params.update(stats)
    else:
        raise ValueError("未知的特征对齐方式")

    after = _compute_gap(X_src, Xt_aligned)

    target_aligned_df = pd.DataFrame(Xt_aligned, columns=bundle.feature_names, index=bundle.target_features.index)

    source_aligned_df = bundle.source_features

    return AlignmentResult(
        source_aligned=source_aligned_df,
        target_aligned=target_aligned_df,
        target_raw=bundle.target_features,
        adapter_params=adapter_params,
        metrics_before=before,
        metrics_after=after,
    )
