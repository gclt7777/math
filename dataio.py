"""数据载入与列对齐。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import Q4Config
from .utils import reindex_columns


@dataclass
class DataBundle:
    source_features: Optional[pd.DataFrame]
    source_labels: Optional[pd.Series]
    source_meta: Optional[pd.DataFrame]
    target_features: pd.DataFrame
    target_meta: pd.DataFrame
    feature_names: List[str]
    model_columns: List[str]
    raw_target_features: pd.DataFrame


def _load_columns(columns_path: str) -> List[str]:
    with open(columns_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "feature_columns" in payload:
        return list(payload["feature_columns"])
    if "columns" in payload:
        return list(payload["columns"])
    raise KeyError("columns.json 中缺少 feature_columns / columns 字段")


def _prepare_dataframe(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    drop_cols = [col for col in drop_cols if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _select_feature_columns(
    df_src: Optional[pd.DataFrame],
    df_tgt: pd.DataFrame,
    model_columns: List[str],
    prefer_z: bool,
) -> List[str]:
    """决定用于模型输入的特征列。"""

    candidate_cols = [col for col in model_columns if col in df_tgt.columns]
    if df_src is not None:
        candidate_cols = [col for col in candidate_cols if col in df_src.columns]

    if prefer_z:
        z_cols = [col for col in candidate_cols if col.startswith("z_")]
        if z_cols:
            return z_cols

    if candidate_cols:
        return candidate_cols

    # 兜底：目标域存在但模型列缺失时，直接使用交集
    return [col for col in df_tgt.columns if col.startswith("z_") or col in model_columns]


def _align_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return reindex_columns(df, columns)


def load_data(cfg: Q4Config) -> DataBundle:
    """加载源/目标域特征并完成列对齐。"""

    model_columns = _load_columns(cfg.io.q2_columns)

    src_features: Optional[pd.DataFrame] = None
    src_labels: Optional[pd.Series] = None
    src_meta: Optional[pd.DataFrame] = None

    if cfg.io.source_feature_csv:
        df_src = pd.read_csv(cfg.io.source_feature_csv)
        df_src = _prepare_dataframe(df_src, cfg.data.drop_cols)
        label_col = cfg.data.label_col
        if label_col in df_src.columns:
            src_labels = df_src[label_col].copy()
        if label_col in df_src.columns:
            df_src = df_src.drop(columns=[label_col])
        src_meta_cols = [col for col in ["uid", "file_id", "basename", "sensor"] if col in df_src.columns]
        src_meta = df_src[src_meta_cols].copy() if src_meta_cols else None
        src_features = df_src.drop(columns=src_meta_cols, errors="ignore")
    else:
        df_src = None

    df_tgt_raw = pd.read_csv(cfg.io.target_feature_csv)
    df_tgt_raw = _prepare_dataframe(df_tgt_raw, cfg.data.drop_cols)

    target_meta_cols = [col for col in ["file_id", "basename", "uid", "signal_key"] if col in df_tgt_raw.columns]
    if target_meta_cols:
        target_meta = df_tgt_raw[target_meta_cols].copy()
    else:
        target_meta = pd.DataFrame(index=df_tgt_raw.index)
    if "file_id" not in target_meta.columns:
        target_meta["file_id"] = [f"T{i:03d}" for i in range(len(target_meta))]
    target_features = df_tgt_raw.drop(columns=target_meta_cols, errors="ignore")

    feature_cols = _select_feature_columns(src_features, target_features, model_columns, cfg.data.prefer_z_features)

    if src_features is not None:
        src_features = _align_columns(src_features, feature_cols)
    target_aligned = _align_columns(target_features, feature_cols)

    return DataBundle(
        source_features=src_features,
        source_labels=src_labels,
        source_meta=src_meta,
        target_features=target_aligned,
        target_meta=target_meta,
        feature_names=feature_cols,
        model_columns=model_columns,
        raw_target_features=target_features,
    )
