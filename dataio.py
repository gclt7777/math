"""数据载入与列对齐工具。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd  # pyright: ignore[reportMissingImports]

from .config import Q3Config


_LOGGER = logging.getLogger("q3_transfer")


@dataclass
class FeatureData:
    """保存特征矩阵与元信息。"""

    features: pd.DataFrame
    meta: pd.DataFrame


@dataclass
class DataBundle:
    """供后续模块使用的数据集。"""

    source: FeatureData
    target: FeatureData
    feature_names: List[str]
    model_columns: List[str]
    preprocess_params: Dict[str, object]
    source_importance: Optional[pd.DataFrame]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_feature_columns(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    model_columns: List[str],
    prefer_z: bool,
) -> List[str]:
    """决定用于域适配的特征列，优先选择 Z 特征，其次一对一关键特征。"""

    base_cols = [col for col in model_columns if col in df_src.columns or col in df_tgt.columns]
    dual = {"DE_": "target_", "FE_": "target_"}

    def _dual_candidates(prefix: str) -> List[str]:
        pairs = {
            "DE_rms": "rms",
            "DE_peak": "peak",
            "DE_std": "std",
            "DE_kurtosis": "kurtosis",
            "DE_skewness": "skewness",
            "DE_crest_factor": "crest_factor",
            "DE_freq_mean": "freq_mean",
            "DE_freq_std": "freq_std",
            "DE_spectral_centroid": "spectral_centroid",
            "DE_BPFO_amplitude": "BPFO_amplitude",
            "DE_BPFO_harmonics_energy": "BPFO_harmonics_energy",
            "DE_BPFI_amplitude": "BPFI_amplitude",
            "DE_BPFI_harmonics_energy": "BPFI_harmonics_energy",
            "DE_BSF_amplitude": "BSF_amplitude",
            "DE_BSF_harmonics_energy": "BSF_harmonics_energy",
            "DE_low_band_ratio": "low_band_ratio",
            "DE_high_band_ratio": "high_band_ratio",
        }
        selected: List[str] = []
        for src_col, tgt_col in pairs.items():
            if src_col in df_src.columns and tgt_col in df_tgt.columns:
                selected.append(src_col)
        return selected

    if prefer_z:
        z_cols = [col for col in base_cols if col.startswith("z_") and col in df_src.columns and col in df_tgt.columns]
        if z_cols:
            return z_cols

    aligned_cols = _dual_candidates("DE_")
    if aligned_cols:
        return aligned_cols

    intersect = [col for col in base_cols if col in df_src.columns and col in df_tgt.columns]
    if intersect:
        return intersect

    return base_cols


def _prepare_dataframe(df: pd.DataFrame, cfg: Q3Config) -> pd.DataFrame:
    """过滤传感器并删除指定列。"""

    if "sensor" in df.columns and cfg.data.sensor_filter:
        df = df[df["sensor"].isin(cfg.data.sensor_filter)].copy()
    drop_cols = [col for col in cfg.data.drop_cols if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def load_all(cfg: Q3Config) -> DataBundle:
    """载入源/目标域特征并对齐列。"""

    df_src = pd.read_csv(cfg.io.src_feature_csv)
    df_tgt = pd.read_csv(cfg.io.tgt_feature_csv)

    for df in (df_src, df_tgt):
        if "uid" not in df.columns:
            df.insert(0, "uid", [f"auto_{i}" for i in range(len(df))])
        if "sensor" not in df.columns and "sensor_src" in df.columns:
            df["sensor"] = df["sensor_src"]
        if cfg.data.file_basename_col and cfg.data.file_basename_col not in df.columns:
            if "file_path" in df.columns:
                df[cfg.data.file_basename_col] = df["file_path"].apply(
                    lambda x: Path(str(x)).stem if isinstance(x, str) else x
                )

    df_src = _prepare_dataframe(df_src, cfg)
    df_tgt = _prepare_dataframe(df_tgt, cfg)

    columns_config = _load_json(cfg.io.q2_columns)
    if "feature_columns" in columns_config:
        model_columns = list(columns_config["feature_columns"])
    elif "columns" in columns_config:
        model_columns = list(columns_config["columns"])
    else:
        raise KeyError("columns.json 中找不到 feature_columns/columns 字段")

    feature_cols = _select_feature_columns(df_src, df_tgt, model_columns, cfg.data.prefer_z_features)

    meta_cols = [cfg.data.uid_col, cfg.data.file_basename_col, "sensor"]
    if cfg.data.label_col in df_src.columns:
        meta_cols.append(cfg.data.label_col)
    meta_cols = [col for col in meta_cols if col in df_src.columns or col in df_tgt.columns]

    src_meta = df_src[meta_cols].copy()
    tgt_meta = df_tgt[[col for col in meta_cols if col in df_tgt.columns]].copy()

    missing_src = [col for col in model_columns if col not in df_src.columns]
    missing_tgt = [col for col in model_columns if col not in df_tgt.columns]
    if missing_src:
        _LOGGER.warning("源域缺失模型特征列，将以 NaN 占位: %s", missing_src)
    if missing_tgt:
        _LOGGER.warning("目标域缺失模型特征列，将以 NaN 占位: %s", missing_tgt)

    src_features = df_src.reindex(columns=model_columns, copy=True)
    tgt_features = df_tgt.reindex(columns=model_columns, copy=True)

    preprocess_params = _load_json(cfg.io.preprocess_params)

    importance_df = None
    try:
        importance_df = pd.read_csv(cfg.io.q2_importance)
    except FileNotFoundError:
        importance_df = None

    return DataBundle(
        source=FeatureData(src_features, src_meta),
        target=FeatureData(tgt_features, tgt_meta),
        feature_names=feature_cols,
        model_columns=model_columns,
        preprocess_params=preprocess_params,
        source_importance=importance_df,
    )
