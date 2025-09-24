"""置信度驱动的伪标注自训练。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import Q4Config


@dataclass
class SelfTrainingResult:
    model: object
    manifest: pd.DataFrame
    class_history: List[Dict[str, int]]


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        decision = decision - decision.max(axis=1, keepdims=True)
        proba = np.exp(decision)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba
    raise AttributeError("模型既不支持 predict_proba 也不支持 decision_function")


def run_self_training(
    base_model,
    source_features: Optional[pd.DataFrame],
    source_labels: Optional[pd.Series],
    target_features: pd.DataFrame,
    target_meta: pd.DataFrame,
    cfg: Q4Config,
) -> SelfTrainingResult:
    st_cfg = cfg.adapt.self_training
    if not st_cfg.enabled or source_features is None or source_labels is None:
        return SelfTrainingResult(model=base_model, manifest=pd.DataFrame(), class_history=[])

    classes = list(getattr(base_model, "classes_", sorted(source_labels.unique())))
    class_history: List[Dict[str, int]] = []
    manifest_rows: List[Dict[str, object]] = []
    accepted_mask = np.zeros(len(target_features), dtype=bool)
    current_model = base_model

    for round_idx in range(1, st_cfg.max_iter + 1):
        proba = _predict_proba(current_model, target_features)
        max_proba = proba.max(axis=1)
        pred_indices = proba.argmax(axis=1)
        pred_labels = [classes[idx] for idx in pred_indices]

        eligible = (~accepted_mask) & (max_proba >= st_cfg.conf_threshold)
        if not eligible.any():
            break

        df_candidates = pd.DataFrame({
            "index": np.arange(len(target_features)),
            "label": pred_labels,
            "confidence": max_proba,
        })
        df_candidates = df_candidates[eligible]
        df_candidates = df_candidates.sort_values("confidence", ascending=False)

        selected_indices: List[int] = []
        class_counter: Dict[str, int] = {}
        for _, row in df_candidates.iterrows():
            label = row["label"]
            if class_counter.get(label, 0) >= st_cfg.max_per_class:
                continue
            selected_indices.append(int(row["index"]))
            class_counter[label] = class_counter.get(label, 0) + 1
        if len(selected_indices) < st_cfg.min_new_samples:
            break

        accepted_mask[selected_indices] = True
        if class_history:
            cumulative = class_history[-1].copy()
            for lbl, cnt in class_counter.items():
                cumulative[lbl] = cumulative.get(lbl, 0) + cnt
        else:
            cumulative = class_counter.copy()
        class_history.append(cumulative)

        for idx in selected_indices:
            meta_row = target_meta.iloc[idx] if not target_meta.empty else {}
            manifest_rows.append(
                {
                    "round": round_idx,
                    "index": idx,
                    "file_id": meta_row.get("file_id", meta_row.get("basename", idx)),
                    "pseudo_label": pred_labels[idx],
                    "confidence": float(max_proba[idx]),
                    "accepted": True,
                }
            )

        pseudo_features = target_features.iloc[selected_indices]
        pseudo_labels = pd.Series([pred_labels[i] for i in selected_indices], index=pseudo_features.index)

        train_X = pd.concat([source_features, pseudo_features], axis=0)
        train_y = pd.concat([source_labels, pseudo_labels], axis=0)

        sample_weights = None
        if st_cfg.reweight_source != st_cfg.reweight_target:
            weights_source = np.full(len(source_features), st_cfg.reweight_source)
            weights_target = np.full(len(pseudo_features), st_cfg.reweight_target)
            sample_weights = np.concatenate([weights_source, weights_target])

        new_model = clone(base_model)
        new_model.fit(train_X, train_y, sample_weight=sample_weights)
        current_model = new_model

    manifest_df = pd.DataFrame(manifest_rows)
    return SelfTrainingResult(model=current_model, manifest=manifest_df, class_history=class_history)
