"""加载 Q2 训练产物并准备评估数据。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from .config import VizConfig


@dataclass
class DataBundle:
    """封装模型与测试数据。"""

    model: ClassifierMixin
    feature_names: List[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    proba: Optional[np.ndarray]
    decision: Optional[np.ndarray]
    class_order: List[str]
    class_display: Dict[str, str]
    model_classes: List[str]
    cv_results: Optional[pd.DataFrame]
    model_importance: Optional[pd.Series]
    window_df: Optional[pd.DataFrame] = None
    file_df: Optional[pd.DataFrame] = None
    file_y_true: Optional[pd.Series] = None
    file_y_pred: Optional[np.ndarray] = None
    file_proba: Optional[np.ndarray] = None


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_feature_order(columns_json: dict) -> List[str]:
    if isinstance(columns_json, dict):
        if "feature_columns" in columns_json:
            return list(columns_json["feature_columns"])
        if "columns" in columns_json:
            return list(columns_json["columns"])
    raise KeyError("columns.json 缺少 feature_columns 字段")


def _resolve_model_importance(model: ClassifierMixin, feature_names: List[str]) -> Optional[pd.Series]:
    if hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        coef = np.asarray(coef)
        if coef.ndim == 1:
            values = np.abs(coef)
        else:
            values = np.mean(np.abs(coef), axis=0)
        return pd.Series(values, index=feature_names, name="模型重要度").sort_values(ascending=False)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(getattr(model, "feature_importances_"))
        return pd.Series(values, index=feature_names, name="模型重要度").sort_values(ascending=False)
    return None


def load_artifacts(cfg: VizConfig) -> DataBundle:
    """加载模型与测试集，返回 DataBundle。"""

    model = joblib.load(cfg.io.model_path)
    model_classes = list(getattr(model, "classes_", cfg.classes.order))

    columns_json = _load_json(cfg.io.columns_json)
    feature_names = _extract_feature_order(columns_json)

    manifest = _load_json(cfg.io.split_manifest)
    test_uids = set(manifest.get("test", []))
    if not test_uids:
        raise ValueError("split_manifest.json 未提供 test 集 uid")

    features_df = pd.read_csv(cfg.io.features_csv)
    if "uid" not in features_df.columns:
        features_df.insert(0, "uid", [f"auto_{i}" for i in range(len(features_df))])
    features_df = features_df[features_df["uid"].isin(test_uids)].copy()
    if features_df.empty:
        raise ValueError("features_source.csv 中未找到测试集样本")

    missing = [col for col in feature_names if col not in features_df.columns]
    if missing:
        raise KeyError(f"特征列缺失: {missing[:10]}")

    window_df: Optional[pd.DataFrame] = None
    if cfg.io.window_predictions and Path(cfg.io.window_predictions).exists():
        window_df = pd.read_csv(cfg.io.window_predictions)
        if "uid" not in window_df.columns:
            raise KeyError("window_predictions.csv 缺少 uid 列")
        window_df = window_df[window_df["uid"].isin(test_uids)].copy()
        if window_df.empty:
            raise ValueError("window_predictions.csv 中未找到测试集窗口")
        # 按 uid 对齐特征顺序
        features_df = features_df.set_index("uid").loc[window_df["uid"]].reset_index()
    else:
        window_df = None

    X_test = features_df[feature_names].copy()

    if window_df is not None:
        prob_cols = list(cfg.classes.order)
        for col in prob_cols:
            if col not in window_df.columns:
                window_df[col] = 0.0
        window_df = window_df.reset_index(drop=True)
        y_test = window_df["true_label"].astype(str)
        y_pred = window_df["pred_label"].astype(str).to_numpy()
        proba = window_df[prob_cols].to_numpy(dtype=float)
        decision = proba
    else:
        label_col = "label" if "label" in features_df.columns else "fault_type"
        if label_col not in features_df.columns:
            raise KeyError("特征文件缺少标签列 label/fault_type")
        y_test = features_df[label_col].astype(str)
        if hasattr(model, "predict"):
            y_pred = model.predict(X_test)
        else:
            raise AttributeError("模型不支持 predict 方法")
        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        decision = None
        if proba is None and hasattr(model, "decision_function"):
            decision = model.decision_function(X_test)
        elif proba is not None:
            proba_df = pd.DataFrame(proba, columns=model_classes)
            for col in cfg.classes.order:
                if col not in proba_df.columns:
                    proba_df[col] = 0.0
            proba = proba_df[cfg.classes.order].to_numpy(dtype=float)
            decision = proba

    file_df: Optional[pd.DataFrame] = None
    file_y_true: Optional[pd.Series] = None
    file_y_pred: Optional[np.ndarray] = None
    file_proba: Optional[np.ndarray] = None
    if cfg.io.file_predictions and Path(cfg.io.file_predictions).exists():
        file_df = pd.read_csv(cfg.io.file_predictions)
        if {"true_label", "pred_label"}.issubset(file_df.columns):
            prob_cols = list(cfg.classes.order)
            for col in prob_cols:
                if col not in file_df.columns:
                    file_df[col] = 0.0
            file_df = file_df.reset_index(drop=True)
            file_y_true = file_df["true_label"].astype(str)
            file_y_pred = file_df["pred_label"].astype(str).to_numpy()
            file_proba = file_df[prob_cols].to_numpy(dtype=float)

    cv_results = None
    if cfg.io.cv_results and Path(cfg.io.cv_results).exists():
        cv_results = pd.read_csv(cfg.io.cv_results)

    model_importance = _resolve_model_importance(model, feature_names)

    return DataBundle(
        model=model,
        feature_names=feature_names,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        proba=proba,
        decision=decision,
        class_order=list(cfg.classes.order),
        class_display=dict(cfg.classes.display),
        model_classes=model_classes,
        cv_results=cv_results,
        model_importance=model_importance,
        window_df=window_df,
        file_df=file_df,
        file_y_true=file_y_true,
        file_y_pred=file_y_pred,
        file_proba=file_proba,
    )
