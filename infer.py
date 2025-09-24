"""目标域推断与聚合。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

from .align import AlignmentResult
from .config import Q3Config

LOGGER = logging.getLogger("q3_transfer")


@dataclass
class PredictionBundle:
    segment_df: pd.DataFrame
    file_df: pd.DataFrame
    uncertain_df: pd.DataFrame
    proba: np.ndarray
    classes: List[str]


def _prepare_model_input(features: pd.DataFrame, model_columns: List[str]) -> pd.DataFrame:
    df = features.copy()
    missing = [col for col in model_columns if col not in df.columns]
    if missing:
        LOGGER.warning("缺失模型需要的列，将以 0 填充：%s", missing)
        for col in missing:
            df[col] = 0.0
    df = df[model_columns]
    return df


def _ensure_probability(model, X: pd.DataFrame) -> tuple[np.ndarray, List[str]]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        LOGGER.warning("模型不支持 predict_proba，将使用 decision_function Softmax 近似。")
        decision = model.decision_function(X)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        decision = decision - decision.max(axis=1, keepdims=True)
        proba = np.exp(decision)
        proba /= proba.sum(axis=1, keepdims=True)
    classes = list(getattr(model, "classes_", np.arange(proba.shape[1])))
    return proba, classes


def predict_segments(alignment: AlignmentResult, cfg: Q3Config) -> PredictionBundle:
    model = joblib.load(cfg.io.q2_best_model)

    X_input = _prepare_model_input(alignment.target.features, alignment.model_columns)
    proba, classes = _ensure_probability(model, X_input)

    topk = max(1, cfg.inference.topk)
    top_indices = np.argsort(proba, axis=1)[:, ::-1][:, :topk]
    top_labels = np.array([[classes[idx] for idx in row] for row in top_indices])
    top_probs = np.take_along_axis(proba, top_indices, axis=1)

    max_prob = top_probs[:, 0]
    top1 = top_labels[:, 0]

    meta = alignment.target.meta.reset_index(drop=True).copy()
    meta["pred_top1"] = top1
    meta["prob_top1"] = max_prob
    if topk > 1:
        meta["pred_top2"] = top_labels[:, 1]
        meta["prob_top2"] = top_probs[:, 1]
    else:
        meta["pred_top2"] = "-"
        meta["prob_top2"] = 0.0
    meta["max_prob"] = max_prob
    meta["is_unknown"] = meta["max_prob"] < cfg.inference.unknown_threshold

    segment_df = meta.copy()

    basename_col = cfg.data.file_basename_col
    if basename_col not in segment_df.columns:
        LOGGER.warning("目标域数据缺少 %s 列，文件级聚合将退化为按 uid。", basename_col)
        segment_df[basename_col] = segment_df.get(cfg.data.uid_col, segment_df.index.astype(str))

    group_key = basename_col
    grouped = segment_df.groupby(group_key)

    agg_rows: List[Dict[str, object]] = []
    for name, group in grouped:
        probs = np.vstack(group["max_prob"].to_numpy())  # placeholder
        # 使用概率均值
        file_probs = proba[group.index].mean(axis=0) if cfg.inference.agg_mode == "prob_mean" else None
        if cfg.inference.agg_mode == "vote":
            votes = pd.Series(group["pred_top1"]).value_counts()
            top_label = votes.idxmax()
            second_label = votes.iloc[1].name if votes.size > 1 else "-"
            top_score = votes.max() / votes.sum()
            second_score = votes.iloc[1] / votes.sum() if votes.size > 1 else 0.0
        else:
            idx_sorted = np.argsort(file_probs)[::-1]
            top_label = classes[idx_sorted[0]]
            top_score = file_probs[idx_sorted[0]]
            if len(idx_sorted) > 1:
                second_label = classes[idx_sorted[1]]
                second_score = file_probs[idx_sorted[1]]
            else:
                second_label = "-"
                second_score = 0.0
        is_unknown = top_score < cfg.inference.unknown_threshold
        agg_rows.append(
            {
                basename_col: name,
                "pred_label": top_label,
                "prob_label": top_score,
                "second_label": second_label,
                "prob_second": second_score,
                "n_segments": int(len(group)),
                "n_unknown_segments": int(group["is_unknown"].sum()),
                "is_unknown": is_unknown,
            }
        )

    file_df = pd.DataFrame(agg_rows)
    uncertain_df = file_df[file_df["is_unknown"]].copy()

    return PredictionBundle(
        segment_df=segment_df,
        file_df=file_df,
        uncertain_df=uncertain_df,
        proba=proba,
        classes=classes,
    )

