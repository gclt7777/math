"""Q4 推理 CLI。"""

from __future__ import annotations

import argparse
import joblib
import pandas as pd
from pathlib import Path

from .config import load_config
from .dataio import _load_columns  # reuse internal helper
from .calibrate import CalibrationModel
from .openset import apply_open_set


def _prepare_input(df: pd.DataFrame, required_cols):
    aligned = df.copy()
    for col in required_cols:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[required_cols]


def _apply_adapter(features: pd.DataFrame, adapter: dict) -> pd.DataFrame:
    method = adapter.get("method", "none")
    X = features.to_numpy(dtype=float)
    if method == "zscore_to_source":
        mu_ref = adapter.get("mu_reference")
        std_ref = adapter.get("std_reference")
        mu_target = adapter.get("mu_target")
        std_target = adapter.get("std_target")
        if mu_ref and std_ref and mu_target and std_target:
            mu_ref = np.array(mu_ref)
            std_ref = np.array(std_ref)
            mu_target = np.array(mu_target)
            std_target = np.where(np.array(std_target) < 1e-12, 1.0, np.array(std_target))
            X = (X - mu_target) / std_target
            X = X * std_ref + mu_ref
    elif method == "coral":
        mu_source = adapter.get("mu_source")
        cov_source = adapter.get("cov_source")
        mu_target = adapter.get("mu_target")
        cov_target = adapter.get("cov_target")
        if mu_source and cov_source and mu_target and cov_target:
            eps = 1e-6
            mu_source = np.array(mu_source)
            cov_source = np.array(cov_source)
            mu_target = np.array(mu_target)
            cov_target = np.array(cov_target)
            dim = cov_target.shape[0]
            cov_source = cov_source + np.eye(dim) * eps
            cov_target = cov_target + np.eye(dim) * eps
            eigvals_t, eigvecs_t = np.linalg.eigh(cov_target)
            eigvals_s, eigvecs_s = np.linalg.eigh(cov_source)
            cov_t_inv_sqrt = eigvecs_t @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_t, eps))) @ eigvecs_t.T
            cov_s_sqrt = eigvecs_s @ np.diag(np.sqrt(np.maximum(eigvals_s, eps))) @ eigvecs_s.T
            X = (X - mu_target) @ cov_t_inv_sqrt @ cov_s_sqrt + mu_source
    return pd.DataFrame(X, columns=features.columns, index=features.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4 推理 CLI")
    parser.add_argument("--config", default="config/q4.yaml")
    parser.add_argument("--input", required=True, help="待预测特征 CSV")
    parser.add_argument("--output", required=True, help="输出预测 CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    package = joblib.load(Path(cfg.io.out_dir) / "q4_final_model.joblib")

    model = package["model"]
    calibration: CalibrationModel = package.get("calibration")
    adapter = package.get("adapter", {})
    thresholds = package.get("thresholds", {})
    feature_columns = package.get("feature_columns")
    classes = package.get("classes")

    df = pd.read_csv(args.input)
    meta_cols = [col for col in ["file_id", "basename", "uid"] if col in df.columns]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame({"file_id": range(len(df))})
    features = df.drop(columns=meta_cols, errors="ignore")

    features = _prepare_input(features, feature_columns)
    features = _apply_adapter(features, adapter)

    proba = model.predict_proba(features)
    if calibration and calibration.method != "none":
        proba = calibration.apply(proba)

    openset_method = thresholds.get("openset_method", "max_proba")
    reject_threshold = thresholds.get("reject_threshold", 0.5)
    openset_res = apply_open_set(proba, openset_method, reject_threshold)

    preds = proba.argmax(axis=1)
    max_proba = proba.max(axis=1)
    labels = [classes[idx] for idx in preds]

    meta["predicted_label"] = labels
    meta["confidence"] = max_proba
    for i, cls in enumerate(classes):
        meta[f"proba_{cls}"] = proba[:, i]
    meta["is_rejected"] = openset_res.is_rejected
    meta.loc[meta["is_rejected"], "predicted_label"] = "UNK"

    meta.to_csv(args.output, index=False)
    print(f"推理完成，结果保存至 {args.output}")


if __name__ == "__main__":
    main()
