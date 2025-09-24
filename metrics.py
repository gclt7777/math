"""指标与报表数据构建。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .align import AlignmentResult


@dataclass
class ReliabilityResult:
    bins: pd.DataFrame
    summary: Dict[str, float]


def build_domain_gap_table(result: AlignmentResult) -> pd.DataFrame:
    rows = []
    for key in sorted(result.metrics_before.keys()):
        rows.append(
            {
                "metric": key,
                "before": result.metrics_before[key],
                "after": result.metrics_after.get(key, np.nan),
                "delta": result.metrics_after.get(key, np.nan) - result.metrics_before[key],
            }
        )
    return pd.DataFrame(rows)


def compute_reliability(proba: np.ndarray, labels: np.ndarray, classes: List[str], n_bins: int = 10) -> ReliabilityResult:
    if len(proba) == 0:
        empty = pd.DataFrame(columns=["bin_lower", "bin_upper", "count", "confidence", "accuracy"])
        return ReliabilityResult(empty, {"brier": np.nan, "ece": np.nan, "n_bins": 0})
    label_to_index = {cls: i for i, cls in enumerate(classes)}
    y_idx = np.array([label_to_index.get(lbl, -1) for lbl in labels])
    valid_mask = y_idx >= 0
    if not valid_mask.any():
        empty = pd.DataFrame(columns=["bin_lower", "bin_upper", "count", "confidence", "accuracy"])
        return ReliabilityResult(empty, {"brier": np.nan, "ece": np.nan, "n_bins": 0})
    proba = proba[valid_mask]
    y_idx = y_idx[valid_mask]

    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_idx).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1
    records = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        records.append(
            {
                "bin_lower": bins[b],
                "bin_upper": bins[b + 1],
                "count": int(mask.sum()),
                "confidence": float(bin_conf),
                "accuracy": float(bin_acc),
            }
        )
        ece += abs(bin_conf - bin_acc) * mask.mean()

    brier = np.mean(np.sum((proba - np.eye(len(classes))[y_idx]) ** 2, axis=1))
    df = pd.DataFrame(records)
    summary = {"brier": float(brier), "ece": float(ece), "n_bins": int(len(df))}
    return ReliabilityResult(df, summary)


def compute_curves(
    proba: np.ndarray,
    labels: np.ndarray,
    classes: List[str],
) -> Dict[str, pd.DataFrame]:
    label_to_index = {cls: i for i, cls in enumerate(classes)}
    y_idx = np.array([label_to_index.get(lbl, -1) for lbl in labels])
    valid_mask = y_idx >= 0
    if not valid_mask.any():
        return {}
    proba = proba[valid_mask]
    y_idx = y_idx[valid_mask]

    curves: Dict[str, pd.DataFrame] = {}
    for idx, cls in enumerate(classes):
        y_true = (y_idx == idx).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, proba[:, idx])
        fpr, tpr, _ = roc_curve(y_true, proba[:, idx])
        curves[f"pr_{cls}"] = pd.DataFrame({"recall": recall, "precision": precision})
        curves[f"roc_{cls}"] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    return curves


def summarize_class_balance(manifest: pd.DataFrame, classes: List[str]) -> Dict[str, int]:
    counts = {cls: 0 for cls in classes}
    if not manifest.empty:
        for cls in classes:
            counts[cls] = int((manifest["pseudo_label"] == cls).sum())
    return counts
