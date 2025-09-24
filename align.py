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
    target_original: Optional[FeatureData]
    feature_names: List[str]
    model_columns: List[str]
    source_importance: Optional[pd.DataFrame]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    transform_params: Dict[str, object]
    zscore_stats: Dict[str, List[float]]


def _safe_float(value: object) -> Optional[float]:
    """Convert ``value`` to ``float`` if possible, otherwise return ``None``."""

    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    candidate: object = value
    if isinstance(candidate, str):
        candidate = candidate.strip()
        if not candidate:
            return None

    try:
        return float(candidate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        series = pd.to_numeric(pd.Series([candidate]), errors="coerce")
        result = series.iloc[0]
        if pd.isna(result):
            return None
        return float(result)


def _extract_preprocess_stats(params: Dict[str, object]) -> Dict[str, Tuple[float, float]]:
    """从预处理配置中提取均值与标准差，兼容多种导出格式。"""

    stats: Dict[str, Tuple[float, float]] = {}

    if not isinstance(params, dict):
        return stats

    def update_from_mapping(mean_map: Dict[str, object], std_map: Dict[str, object]) -> None:
        common = set(mean_map).intersection(std_map)
        for key in common:
            mean_val = _safe_float(mean_map.get(key))
            std_val = _safe_float(std_map.get(key))
            if mean_val is None or std_val is None:
                continue
            stats[str(key)] = (mean_val, std_val)

    def update_from_lists(columns: List[object], means: List[object], stds: List[object]) -> None:
        if not (len(columns) == len(means) == len(stds)):
            return
        for col, mean, std in zip(columns, means, stds):
            mean_val = _safe_float(mean)
            std_val = _safe_float(std)
            if col is None or mean_val is None or std_val is None:
                continue
            stats[str(col)] = (mean_val, std_val)

    def recurse(node: Dict[str, object]) -> None:
        if not isinstance(node, dict):
            return

        feature_stats = node.get("feature_stats")
        if isinstance(feature_stats, dict):
            for col, val in feature_stats.items():
                if not isinstance(val, dict):
                    continue
                mean_val = _safe_float(val.get("mean") or val.get("mu") or val.get("avg") or val.get("mean_"))
                std_val = _safe_float(val.get("std") or val.get("sigma") or val.get("std_") or val.get("scale"))
                if mean_val is None or std_val is None:
                    continue
                stats[str(col)] = (mean_val, std_val)

        mean_map = None
        std_map = None
        for key in ("mean", "means", "mean_dict"):
            if isinstance(node.get(key), dict):
                mean_map = node[key]  # type: ignore[assignment]
                break
        for key in ("std", "stds", "std_dict", "scale", "scale_dict"):
            if isinstance(node.get(key), dict):
                std_map = node[key]  # type: ignore[assignment]
                break
        if isinstance(mean_map, dict) and isinstance(std_map, dict):
            update_from_mapping(mean_map, std_map)

        columns = None
        for key in ("feature_names", "columns", "feature_columns", "fields"):
            if isinstance(node.get(key), list):
                columns = node[key]  # type: ignore[assignment]
                break
        means = None
        for key in ("mean_", "means", "mean_values", "mean_list"):
            if isinstance(node.get(key), list):
                means = node[key]  # type: ignore[assignment]
                break
        stds = None
        for key in ("scale_", "scales", "std_", "std_values", "std_list"):
            if isinstance(node.get(key), list):
                stds = node[key]  # type: ignore[assignment]
                break
        if columns is not None and means is not None and stds is not None:
            update_from_lists(columns, means, stds)

        for child_key in ("scaler", "standard_scaler", "zscore", "preprocess", "normalizer"):
            child = node.get(child_key)
            if isinstance(child, dict):
                recurse(child)

        steps = node.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    recurse(step.get("params") or step)

    recurse(params)
    return stats


def _apply_preprocess(
    df: pd.DataFrame, stats: Dict[str, Tuple[float, float]]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    if not stats:
        return df.copy(), {}

    df_out = df.copy()
    applied: Dict[str, Dict[str, float]] = {}
    for col, (mean, std) in stats.items():
        if col not in df_out.columns:
            continue
        mean_val = _safe_float(mean)
        std_val = _safe_float(std)
        if mean_val is None or std_val is None:
            continue
        if abs(std_val) < 1e-12:
            std_val = 1.0
        df_out[col] = (df_out[col] - mean_val) / std_val
        applied[col] = {"mean": mean_val, "std": std_val}
    return df_out, applied


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
    src_numeric = bundle.source.features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    tgt_numeric = bundle.target.features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    target_original_df = tgt_numeric.copy()

    preprocess_stats = _extract_preprocess_stats(bundle.preprocess_params)
    src_processed = src_numeric
    tgt_processed = tgt_numeric
    preprocess_summary: Dict[str, object] = {"type": "none"}

    if preprocess_stats:
        src_processed, src_applied = _apply_preprocess(src_numeric, preprocess_stats)
        tgt_processed, tgt_applied = _apply_preprocess(tgt_numeric, preprocess_stats)
        preprocess_summary = {
            "type": "zscore",
            "features": sorted({*src_applied.keys(), *tgt_applied.keys()}),
            "source_stats": src_applied,
            "target_stats": tgt_applied,
        }
    else:
        src_applied = {}
        tgt_applied = {}

    Xs = src_processed.to_numpy(dtype=float)
    Xt_before = tgt_processed.to_numpy(dtype=float)
    Xt = Xt_before.copy()

    metrics_before = _compute_stats(Xs, Xt)
    transform_params: Dict[str, object] = {"method": cfg.adapt.method, "preprocess": preprocess_summary}
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

    source_features = pd.DataFrame(Xs, columns=src_processed.columns, index=bundle.source.features.index)
    target_features = pd.DataFrame(Xt, columns=tgt_processed.columns, index=bundle.target.features.index)
    target_raw_features = pd.DataFrame(Xt_before, columns=tgt_processed.columns, index=bundle.target.features.index)
    target_original_features = pd.DataFrame(
        target_original_df.to_numpy(dtype=float),
        columns=target_original_df.columns,
        index=bundle.target.features.index,
    )

    return AlignmentResult(
        source=FeatureData(source_features, bundle.source.meta),
        target=FeatureData(target_features, bundle.target.meta),
        target_raw=FeatureData(target_raw_features, bundle.target.meta),
        target_original=FeatureData(target_original_features, bundle.target.meta),
        feature_names=bundle.feature_names,
        model_columns=bundle.model_columns,
        source_importance=bundle.source_importance,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        transform_params=transform_params,
        zscore_stats=zscore_stats,
    )
