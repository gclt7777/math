"""域适配与指标计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

from .config import Q3Config
from .dataio import DataBundle, FeatureData


@dataclass
class AlignmentResult:
    source: FeatureData
    target: FeatureData
    target_raw: FeatureData
    feature_names: List[str]
    model_columns: List[str]
    source_importance: Optional[pd.DataFrame]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    transform_params: Dict[str, object]
    zscore_stats: Dict[str, List[float]]


def _compute_stats(Xs: np.ndarray, Xt: np.ndarray) -> Dict[str, float]:
    mean_diff = np.linalg.norm(Xs.mean(axis=0) - Xt.mean(axis=0))
    cov_diff = np.linalg.norm(np.cov(Xs, rowvar=False) - np.cov(Xt, rowvar=False), ord="fro")
    mmd_val = _mmd_rbf(Xs, Xt)
    return {
        "mean_diff": float(mean_diff),
        "cov_fro": float(cov_diff),
        "mmd_rbf": float(mmd_val),
    }


def _mmd_rbf(Xs: np.ndarray, Xt: np.ndarray, gamma: float = 1.0) -> float:
    def kernel(a, b):
        dist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * dist)

    Ks = kernel(Xs, Xs)
    Kt = kernel(Xt, Xt)
    Kst = kernel(Xs, Xt)
    m = Xs.shape[0]
    n = Xt.shape[0]
    return Ks.sum() / (m * m) + Kt.sum() / (n * n) - 2 * Kst.sum() / (m * n)


def _coral(Xt: np.ndarray, mu_s: np.ndarray, cov_s: np.ndarray, eps: float) -> Tuple[np.ndarray, Dict[str, object]]:
    eps = float(eps)
    mu_t = Xt.mean(axis=0)
    cov_t = np.cov(Xt, rowvar=False)
    cov_s = np.array(cov_s, copy=True)
    dim = Xt.shape[1]
    cov_t = cov_t + np.eye(dim) * eps
    cov_s = cov_s + np.eye(dim) * eps

    # sqrt matrices via eig
    eigvals_t, eigvecs_t = np.linalg.eigh(cov_t)
    eigvals_s, eigvecs_s = np.linalg.eigh(cov_s)

    cov_t_inv_sqrt = eigvecs_t @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_t, eps))) @ eigvecs_t.T
    cov_s_sqrt = eigvecs_s @ np.diag(np.sqrt(np.maximum(eigvals_s, eps))) @ eigvecs_s.T

    Xt_centered = Xt - mu_t
    Xt_aligned = Xt_centered @ cov_t_inv_sqrt @ cov_s_sqrt + mu_s

    params = {
        "mu_source": mu_s.tolist(),
        "mu_target": mu_t.tolist(),
        "cov_source": cov_s.tolist(),
        "cov_target": cov_t.tolist(),
    }
    return Xt_aligned, params


def _zscore_align(Xt: np.ndarray, mu_t: np.ndarray, std_t: np.ndarray, mu_ref: np.ndarray, std_ref: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    std_t = np.where(std_t < 1e-12, 1.0, std_t)
    std_ref = np.where(std_ref < 1e-12, 1.0, std_ref)
    Xt_norm = (Xt - mu_t) / std_t
    Xt_aligned = Xt_norm * std_ref + mu_ref
    return Xt_aligned, {
        "mu_target": mu_t.tolist(),
        "std_target": std_t.tolist(),
        "mu_reference": mu_ref.tolist(),
        "std_reference": std_ref.tolist(),
    }


def fit_transform(bundle: DataBundle, cfg: Q3Config) -> AlignmentResult:
    src_df = bundle.source.features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    tgt_df = bundle.target.features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    Xs = src_df.to_numpy(dtype=float)
    Xt_orig = tgt_df.to_numpy(dtype=float)
    Xt = Xt_orig.copy()

    metrics_before = _compute_stats(Xs, Xt)
    transform_params: Dict[str, object] = {"method": cfg.adapt.method}
    zscore_stats: Dict[str, List[float]] = {}

    if cfg.adapt.method == "coral":
        ref = Xs if cfg.adapt.fit_on in {"source", "source+target"} else Xt
        if cfg.adapt.fit_on == "source+target":
            ref = np.vstack([Xs, Xt])
        mu_s = ref.mean(axis=0)
        cov_s = np.cov(ref, rowvar=False)
        Xt_aligned, params = _coral(Xt, mu_s, cov_s, cfg.adapt.coral_eps)
        transform_params.update(params)
        if cfg.adapt.apply_to == "target":
            Xt = Xt_aligned
        elif cfg.adapt.apply_to == "both":
            Xs, Xt = Xt_aligned, Xt_aligned
        else:
            zscore_stats = params  # 兼容结构
    elif cfg.adapt.method == "zscore":
        ref = Xs if cfg.adapt.fit_on in {"source", "source+target"} else Xt
        if cfg.adapt.fit_on == "source+target":
            ref = np.vstack([Xs, Xt])
        mu_ref = ref.mean(axis=0)
        std_ref = ref.std(axis=0)
        mu_t = Xt.mean(axis=0)
        std_t = Xt.std(axis=0)
        Xt_aligned, stats = _zscore_align(Xt, mu_t, std_t, mu_ref, std_ref)
        transform_params.update(stats)
        zscore_stats = stats
        if cfg.adapt.apply_to in {"target", "both"}:
            Xt = Xt_aligned
        if cfg.adapt.apply_to == "both":
            Xs = _zscore_align(Xs, Xs.mean(axis=0), Xs.std(axis=0), mu_ref, std_ref)[0]
    elif cfg.adapt.method == "mmd":
        transform_params["gamma"] = cfg.adapt.mmd_gamma
        # 无显式变换，仅记录参数
    else:
        transform_params["note"] = "no adaptation applied"

    metrics_after = _compute_stats(Xs, Xt)

    source_features = pd.DataFrame(Xs, columns=src_df.columns, index=bundle.source.features.index)
    target_features = pd.DataFrame(Xt, columns=tgt_df.columns, index=bundle.target.features.index)
    target_raw_features = pd.DataFrame(Xt_orig, columns=tgt_df.columns, index=bundle.target.features.index)

    return AlignmentResult(
        source=FeatureData(source_features, bundle.source.meta),
        target=FeatureData(target_features, bundle.target.meta),
        target_raw=FeatureData(target_raw_features, bundle.target.meta),
        feature_names=bundle.feature_names,
        model_columns=bundle.model_columns,
        source_importance=bundle.source_importance,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        transform_params=transform_params,
        zscore_stats=zscore_stats,
    )
