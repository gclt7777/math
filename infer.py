"""目标域推断与聚合。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List

import joblib  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

from .align import AlignmentResult
from .config import Q3Config

LOGGER = logging.getLogger("q3_transfer")

KNOWN_FAULT_LABELS = {"N", "OR", "IR", "B"}

_ALIAS_TO_CANONICAL: Dict[str, str] = {
    "n": "N",
    "normal": "N",
    "normalstate": "N",
    "healthy": "N",
    "good": "N",
    "nofault": "N",
    "normalfault": "N",
    "normaloperation": "N",
    "正常": "N",
    "无故障": "N",
    "or": "OR",
    "outer": "OR",
    "outerrace": "OR",
    "outerring": "OR",
    "outrace": "OR",
    "外圈": "OR",
    "外圈故障": "OR",
    "ir": "IR",
    "inner": "IR",
    "innerrace": "IR",
    "innerring": "IR",
    "inrace": "IR",
    "内圈": "IR",
    "内圈故障": "IR",
    "b": "B",
    "ball": "B",
    "ballfault": "B",
    "ballelement": "B",
    "roller": "B",
    "滚动体": "B",
    "滚动体故障": "B",
}


@dataclass
class PredictionBundle:
    segment_df: pd.DataFrame
    file_df: pd.DataFrame
    submission_df: pd.DataFrame
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

def _normalize_label_token(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    for ch in (" ", "-", "_"):
        text = text.replace(ch, "")
    return text


def _canonical_fault_label(value: object) -> str:
    token = _normalize_label_token(value)
    if not token:
        return "-"
    canonical = _ALIAS_TO_CANONICAL.get(token)
    if canonical:
        return canonical
    upper_text = str(value).strip().upper()
    if upper_text in KNOWN_FAULT_LABELS:
        return upper_text
    return upper_text


def _register_label_mapping(
    decoder: Dict[object, str],
    mapping: Dict[object, object],
) -> None:
    for label, raw_value in mapping.items():
        canonical = _canonical_fault_label(label)
        if canonical not in KNOWN_FAULT_LABELS:
            continue
        for key in {raw_value, str(raw_value)}:
            decoder.setdefault(key, canonical)
        decoder.setdefault(canonical, canonical)


def _register_label_list(decoder: Dict[object, str], values: List[object]) -> None:
    canonical_values = [_canonical_fault_label(val) for val in values]
    if not canonical_values:
        return
    if any(val not in KNOWN_FAULT_LABELS for val in canonical_values):
        return
    for idx, canonical in enumerate(canonical_values):
        for key in {idx, str(idx)}:
            decoder.setdefault(key, canonical)
        decoder.setdefault(canonical, canonical)


def _build_class_decoder(columns_path: str) -> Dict[object, str]:
    decoder: Dict[object, str] = {}
    try:
        with open(columns_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        LOGGER.warning("未找到列配置文件: %s", columns_path)
        return decoder
    except json.JSONDecodeError as exc:
        LOGGER.warning("解析列配置 JSON 失败: %s", exc)
        return decoder

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for key in ("label_mapping", "label_map", "mapping"):
                if key in node and isinstance(node[key], dict):
                    _register_label_mapping(decoder, node[key])
            for key in ("classes", "class_labels", "label_classes", "labels"):
                if key in node and isinstance(node[key], list):
                    _register_label_list(decoder, node[key])
            for value in node.values():
                if isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            if node and isinstance(node[0], (str, int)):
                _register_label_list(decoder, node)
            for item in node:
                if isinstance(item, (dict, list)):
                    _walk(item)

    _walk(config)

    for canonical in KNOWN_FAULT_LABELS:
        decoder.setdefault(canonical, canonical)

    return decoder


def _decode_label(raw_label: object, decoder: Dict[object, str]) -> str:
    if raw_label in decoder:
        return decoder[raw_label]
    key = str(raw_label)
    if key in decoder:
        return decoder[key]
    return _canonical_fault_label(raw_label)

def predict_segments(alignment: AlignmentResult, cfg: Q3Config) -> PredictionBundle:
    model = joblib.load(cfg.io.q2_best_model)

    X_input = _prepare_model_input(alignment.target.features, alignment.model_columns)
    proba, raw_classes = _ensure_probability(model, X_input)

    decoder = _build_class_decoder(cfg.io.q2_columns)
    classes = [_decode_label(cls, decoder) for cls in raw_classes]
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
    meta["pred_fault_type"] = meta["pred_top1"]
    segment_df = meta.copy()

    basename_col = cfg.data.file_basename_col
    if basename_col not in segment_df.columns:
        LOGGER.warning("目标域数据缺少 %s 列，文件级聚合将退化为按 uid。", basename_col)
        segment_df[basename_col] = segment_df.get(cfg.data.uid_col, segment_df.index.astype(str))

    group_key = basename_col
    grouped = segment_df.groupby(group_key)

    agg_rows: List[Dict[str, object]] = []
    submission_rows: List[Dict[str, object]] = []
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

        record = {
            basename_col: name,
            "pred_label": top_label,
            "prob_label": top_score,
            "second_label": second_label,
            "prob_second": second_score,
            "n_segments": int(len(group)),
            "n_unknown_segments": int(group["is_unknown"].sum()),
            "is_unknown": is_unknown,
        }
        agg_rows.append(record)
        submission_rows.append({basename_col: name, "pred_fault_type": top_label})

    file_df = pd.DataFrame(agg_rows)
    uncertain_df = file_df[file_df["is_unknown"]].copy()
    if submission_rows:
        submission_df = pd.DataFrame(submission_rows)
        submission_df = submission_df.sort_values(basename_col).reset_index(drop=True)
    else:
        submission_df = pd.DataFrame(columns=[basename_col, "pred_fault_type"])
    return PredictionBundle(
        segment_df=segment_df,
        file_df=file_df,
        uncertain_df=uncertain_df,
        submission_df=submission_df,
        proba=proba,
        classes=classes,
    )

