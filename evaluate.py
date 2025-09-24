"""评估指标与解释性分析模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


@dataclass
class EvaluationResult:
    """封装测试集评估产出。"""

    y_true: pd.Series
    y_pred: pd.Series
    proba: pd.DataFrame
    report: pd.DataFrame
    confusion: np.ndarray
    feature_importance: pd.DataFrame
    classes: List[str]
    file_paths: pd.Series
    window_predictions: Optional[pd.DataFrame] = None
    file_predictions: Optional[pd.DataFrame] = None
    file_report: Optional[pd.DataFrame] = None
    file_confusion: Optional[np.ndarray] = None


def _extract_inner_estimator(model) -> object:
    """尝试获取管线或校准器内部的真实分类器。"""

    if isinstance(model, CalibratedClassifierCV):
        if hasattr(model, "estimator") and model.estimator is not None:
            return model.estimator
        if hasattr(model, "base_estimator_"):
            return model.base_estimator_
        if hasattr(model, "base_estimator"):
            return model.base_estimator
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated:
            return calibrated[0].estimator
        return model
    if isinstance(model, Pipeline):
        return model.named_steps.get("clf", model)
    return model


def _direct_importance(estimator, feature_names: List[str]) -> pd.DataFrame | None:
    """尝试读取模型自带的重要度指标。"""

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": values})
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim == 1:
            values = np.abs(coef)
        else:
            values = np.mean(np.abs(coef), axis=0)
        return pd.DataFrame({"feature": feature_names, "importance": values})
    return None


def evaluate_model(model, splits, random_seed: int) -> EvaluationResult:
    """在测试集上计算各类指标并返回结果。"""

    X_test = splits.X_test
    y_test = splits.y_test

    y_pred = model.predict(X_test)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # 将 decision_function 归一化为概率近似
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    else:
        raise ValueError("模型不支持概率输出，无法绘制 ROC/PR。")

    classes = list(model.classes_) if hasattr(model, "classes_") else list(np.unique(y_test))
    proba_df = pd.DataFrame(proba, columns=classes)

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    conf = confusion_matrix(y_test, y_pred, labels=classes)

    estimator = _extract_inner_estimator(model)
    importance_df = _direct_importance(estimator, splits.feature_cols)
    if importance_df is None:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=random_seed,
            n_jobs=-1,
        )
        importance_df = pd.DataFrame(
            {
                "feature": splits.feature_cols,
                "importance": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance", ascending=False)
    else:
        importance_df = importance_df.sort_values("importance", ascending=False)

    uid_series = getattr(splits, "uid_test", pd.Series(range(len(y_test))))
    file_series = getattr(splits, "file_test", pd.Series([None] * len(y_test)))

    y_pred_series = pd.Series(y_pred, name="prediction")
    window_prefix = pd.DataFrame(
        {
            "uid": uid_series.reset_index(drop=True),
            "file_path": file_series.reset_index(drop=True),
            "true_label": y_test.reset_index(drop=True),
            "pred_label": y_pred_series.reset_index(drop=True),
        }
    )
    window_predictions = pd.concat([window_prefix, proba_df.reset_index(drop=True)], axis=1)

    file_predictions: Optional[pd.DataFrame] = None
    file_report_df: Optional[pd.DataFrame] = None
    file_conf: Optional[np.ndarray] = None

    if window_predictions["file_path"].notna().any():
        class_cols = list(proba_df.columns)
        grouped = window_predictions.groupby("file_path", dropna=False)
        file_proba = grouped[class_cols].mean()
        counts = grouped.size()
        true_labels = grouped["true_label"].first()
        proba_values = file_proba.to_numpy()
        pred_idx = np.argmax(proba_values, axis=1)
        pred_labels = [class_cols[i] for i in pred_idx]
        confidences = proba_values[np.arange(len(pred_idx)), pred_idx]

        file_predictions = pd.DataFrame(
            {
                "file_path": file_proba.index,
                "num_windows": counts.values,
                "true_label": true_labels.values,
                "pred_label": pred_labels,
                "pred_confidence": confidences,
            }
        ).reset_index(drop=True)
        file_predictions = pd.concat(
            [file_predictions, file_proba.reset_index(drop=True)],
            axis=1,
        )

        file_report_dict = classification_report(
            true_labels,
            pred_labels,
            output_dict=True,
            zero_division=0,
        )
        file_report_df = pd.DataFrame(file_report_dict).transpose()
        file_conf = confusion_matrix(true_labels, pred_labels, labels=classes)

    return EvaluationResult(
        y_true=y_test,
        y_pred=y_pred_series,
        proba=proba_df,
        report=report_df,
        confusion=conf,
        feature_importance=importance_df,
        classes=classes,
        file_paths=file_series.reset_index(drop=True),
        window_predictions=window_predictions,
        file_predictions=file_predictions,
        file_report=file_report_df,
        file_confusion=file_conf,
    )
