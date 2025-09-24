"""概率校准工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationModel:
    method: str
    temperature: float = 1.0
    isotonic_models: Dict[str, IsotonicRegression] | None = None
    classes: List[str] | None = None

    def apply(self, proba: np.ndarray) -> np.ndarray:
        if self.method == "temperature":
            return apply_temperature(proba, self.temperature)
        if self.method == "isotonic" and self.isotonic_models and self.classes:
            return apply_isotonic(proba, self.isotonic_models, self.classes)
        return proba


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-12
    logits = np.log(np.clip(proba, eps, 1.0))
    scaled = logits / max(temperature, eps)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    exp_sum = exp.sum(axis=1, keepdims=True)
    return exp / exp_sum


def apply_isotonic(proba: np.ndarray, models: Dict[str, IsotonicRegression], classes: List[str]) -> np.ndarray:
    calibrated = np.zeros_like(proba)
    for idx, cls in enumerate(classes):
        model = models.get(cls)
        if model is None:
            calibrated[:, idx] = proba[:, idx]
        else:
            calibrated[:, idx] = model.predict(proba[:, idx])
    calibrated = np.clip(calibrated, 1e-6, 1.0)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return calibrated


def fit_temperature(proba: np.ndarray, labels: np.ndarray, classes: List[str]) -> float:
    eps = 1e-12
    label_to_index = {cls: i for i, cls in enumerate(classes)}
    y_idx = np.array([label_to_index[label] for label in labels])
    best_temp = 1.0
    best_loss = np.inf
    logits = np.log(np.clip(proba, eps, 1.0))
    for temp in np.linspace(0.5, 5.0, 20):
        scaled = apply_temperature(proba, temp)
        loss = -np.mean(np.log(np.clip(scaled[np.arange(len(y_idx)), y_idx], eps, 1.0)))
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp


def fit_isotonic(proba: np.ndarray, labels: np.ndarray, classes: List[str]) -> Dict[str, IsotonicRegression]:
    models: Dict[str, IsotonicRegression] = {}
    label_to_index = {cls: i for i, cls in enumerate(classes)}
    y_idx = np.array([label_to_index[label] for label in labels])
    for idx, cls in enumerate(classes):
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(proba[:, idx], (y_idx == idx).astype(float))
        models[cls] = model
    return models


def calibrate_probabilities(
    proba: np.ndarray,
    labels: np.ndarray,
    classes: List[str],
    method: str,
) -> CalibrationModel:
    if len(proba) == 0 or method == "none":
        return CalibrationModel(method="none", classes=classes)
    if method == "temperature":
        temp = fit_temperature(proba, labels, classes)
        return CalibrationModel(method="temperature", temperature=temp, classes=classes)
    if method == "isotonic":
        models = fit_isotonic(proba, labels, classes)
        return CalibrationModel(method="isotonic", isotonic_models=models, classes=classes)
    return CalibrationModel(method="none", classes=classes)
