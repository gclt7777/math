"""域差异量化与机制证据整理。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

from .align import AlignmentResult
from .config import Q3Config
from .infer import PredictionBundle


@dataclass
class ReportBundle:
    domain_gap_df: pd.DataFrame
    mechanism_df: pd.DataFrame
    high_conf_df: pd.DataFrame


def compute_reports(result: AlignmentResult, preds: PredictionBundle, cfg: Q3Config) -> ReportBundle:
    domain_gap_df = _build_domain_gap_table(result)
    mechanism_df = _build_mechanism_table(result, preds, cfg)
    high_conf_df = _build_high_conf_contrib(result, preds, cfg)
    return ReportBundle(
        domain_gap_df=domain_gap_df,
        mechanism_df=mechanism_df,
        high_conf_df=high_conf_df,
    )


def _build_domain_gap_table(result: AlignmentResult) -> pd.DataFrame:
    rows = []
    for metric in sorted(result.metrics_before.keys()):
        before = result.metrics_before.get(metric, float("nan"))
        after = result.metrics_after.get(metric, float("nan"))
        rows.append(
            {
                "metric": metric,
                "before": before,
                "after": after,
                "delta": after - before,
            }
        )
    return pd.DataFrame(rows)


def _group_feature_value(df: pd.DataFrame, columns: List[str]) -> float:
    if not columns:
        return float("nan")
    sub = df[columns]
    if sub.empty:
        return float("nan")
    # 使用均值概括
    return float(sub.mean(axis=1).mean())


def _build_mechanism_table(result: AlignmentResult, preds: PredictionBundle, cfg: Q3Config) -> pd.DataFrame:
    prefixes = cfg.interpret.mech_features_prefix
    if not prefixes:
        prefixes = []
    feature_df = result.target.features.copy()
    feature_df[cfg.data.file_basename_col] = result.target.meta[cfg.data.file_basename_col].values
    feature_df["pred_label"] = preds.segment_df["pred_top1"].values

    rows: List[Dict[str, object]] = []
    for basename, group in feature_df.groupby(cfg.data.file_basename_col):
        row = {
            cfg.data.file_basename_col: basename,
        }
        pred_row = preds.file_df[preds.file_df[cfg.data.file_basename_col] == basename]
        if not pred_row.empty:
            row["pred_label"] = pred_row.iloc[0]["pred_label"]
        else:
            row["pred_label"] = "-"
        for prefix in prefixes:
            cols = [col for col in result.feature_names if col.startswith(prefix) and col in group.columns]
            row[f"{prefix}_energy"] = _group_feature_value(group, cols)
        row["note"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_importance(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    candidates = [col for col in df.columns if col.lower().startswith("importance") or col.lower() in {"importance", "weight", "value"}]
    if not candidates:
        # 尝试特定列
        if {"feature", "importance"}.issubset(df.columns):
            candidates = ["importance"]
        else:
            return pd.DataFrame()
    imp_col = candidates[0]
    feature_col = "feature" if "feature" in df.columns else df.columns[0]
    df_use = df[[feature_col, imp_col]].copy()
    df_use.columns = ["feature", "importance"]
    df_use = df_use.groupby("feature", as_index=False).mean()
    total = df_use["importance"].abs().sum()
    if total > 0:
        df_use["importance_norm"] = df_use["importance"].abs() / total
    else:
        df_use["importance_norm"] = 0.0
    df_use["rank"] = df_use["importance_norm"].rank(ascending=False, method="dense").astype(int)
    df_use = df_use.sort_values("rank")
    return df_use


def _build_high_conf_contrib(result: AlignmentResult, preds: PredictionBundle, cfg: Q3Config) -> pd.DataFrame:
    if not cfg.interpret.use_source_importance or result.source_importance is None or result.source_importance.empty:
        return pd.DataFrame()

    importance_df = _normalize_importance(result.source_importance)
    if importance_df.empty:
        return pd.DataFrame()

    threshold = cfg.inference.unknown_threshold
    segment_info = preds.segment_df.copy()
    high_conf_mask = segment_info["max_prob"] >= threshold
    if not high_conf_mask.any():
        return pd.DataFrame()

    feature_df = result.target.features.loc[segment_info.index]
    feature_df = feature_df.assign(**{
        cfg.data.file_basename_col: segment_info[cfg.data.file_basename_col].values,
        "pred_label": segment_info["pred_top1"].values,
        "max_prob": segment_info["max_prob"].values,
    })

    high_conf_df = feature_df[high_conf_mask].copy()
    if high_conf_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    topk = min(10, len(importance_df))

    importance_map = importance_df.set_index("feature")

    for basename, group in high_conf_df.groupby(cfg.data.file_basename_col):
        mean_vec = group[result.feature_names].mean(axis=0)
        pred_label = group["pred_label"].mode().iat[0] if not group["pred_label"].empty else "-"
        contrib = []
        for feature, value in mean_vec.items():
            if feature not in importance_map.index:
                continue
            imp_norm = float(importance_map.loc[feature, "importance_norm"])
            global_rank = int(importance_map.loc[feature, "rank"])
            imp_raw = float(importance_map.loc[feature, "importance"])
            contrib.append((feature, value * imp_raw, global_rank, imp_raw, imp_norm))
        if not contrib:
            continue
        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        for rank_idx, (feature, approx, global_rank, imp_raw, imp_norm) in enumerate(contrib[:topk], start=1):
            rows.append(
                {
                    cfg.data.file_basename_col: basename,
                    "pred_label": pred_label,
                    "rank": rank_idx,
                    "feature": feature,
                    "approx_contrib": approx,
                    "global_importance_rank": int(global_rank),
                    "global_importance": imp_raw,
                    "global_importance_norm": imp_norm,
                }
            )

    return pd.DataFrame(rows)
