"""Q4 自适应 CLI。"""

from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .align import AlignmentResult, align_features
from .calibrate import CalibrationModel, calibrate_probabilities
from .config import Q4Config, load_config
from .dataio import DataBundle, load_data
from .export import export_all, save_model_artifacts
from .metrics import build_domain_gap_table, compute_curves, compute_reliability
from .openset import apply_open_set
from .selftrain import SelfTrainingResult, run_self_training
from .utils import ensure_dir, reindex_columns, save_json, set_global_seed, setup_logger
from .viz import prepare_figures

LABEL_CN = {
    "B": "滚动体故障",
    "IR": "内圈故障",
    "OR": "外圈故障",
    "N": "正常状态",
    "UNK": "未知/拒识",
}


def _load_model_and_classes(path: str) -> Tuple[object, List[str]]:
    model = joblib.load(path)
    classes = list(getattr(model, "classes_", []))
    if not classes:
        raise ValueError("Q2 模型缺少 classes_ 属性，无法继续")
    return model, classes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q4 目标域自适应")
    parser.add_argument("--config", default="config/q4.yaml", help="配置文件路径")
    parser.add_argument("--no-selftrain", action="store_true", help="禁用伪标注自训练")
    parser.add_argument("--align", choices=["none", "zscore_to_source", "coral"], help="覆盖特征对齐方式")
    parser.add_argument("--dry-run", action="store_true", help="仅加载与预测，不导出模型")
    return parser.parse_args()


def _prepare_model_input(df: pd.DataFrame, model_columns: List[str]) -> pd.DataFrame:
    return reindex_columns(df, model_columns)


def _resolve_reliability_labels(
    predictions_df: pd.DataFrame,
    bundle: DataBundle,
    cfg: Q4Config,
    selftrain_result: SelfTrainingResult,
) -> np.ndarray:
    if bundle.target_meta is not None and cfg.data.label_col in bundle.target_meta.columns:
        return bundle.target_meta[cfg.data.label_col].fillna("UNK").to_numpy()
    if not selftrain_result.manifest.empty:
        manifest = selftrain_result.manifest.set_index("index")
        aligned = manifest.reindex(predictions_df.index)["pseudo_label"].fillna("UNK")
        return aligned.to_numpy()
    return np.full(len(predictions_df), "UNK")


def _calibration_dataset(
    selftrain_result: SelfTrainingResult,
    proba_adapted: np.ndarray,
    adapted_model,
    source_input: Optional[pd.DataFrame],
    bundle: DataBundle,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not selftrain_result.manifest.empty:
        idx = selftrain_result.manifest["index"].to_numpy()
        labels = selftrain_result.manifest["pseudo_label"].to_numpy()
        return proba_adapted[idx], labels
    if source_input is not None and bundle.source_labels is not None:
        source_proba = adapted_model.predict_proba(source_input)
        return source_proba, bundle.source_labels.to_numpy()
    return None, None


def _run_calibration(
    cfg: Q4Config,
    selftrain_result: SelfTrainingResult,
    proba_adapted: np.ndarray,
    adapted_model,
    source_input: Optional[pd.DataFrame],
    bundle: DataBundle,
    classes: List[str],
) -> tuple[CalibrationModel, np.ndarray]:
    if not cfg.adapt.calibration.enabled:
        return CalibrationModel(method="none", classes=classes), proba_adapted

    calib_proba, calib_labels = _calibration_dataset(
        selftrain_result,
        proba_adapted,
        adapted_model,
        source_input,
        bundle,
    )
    if calib_proba is None or len(calib_proba) == 0:
        return CalibrationModel(method="none", classes=classes), proba_adapted

    model = calibrate_probabilities(
        calib_proba,
        calib_labels,
        classes,
        cfg.adapt.calibration.method,
    )
    return model, model.apply(proba_adapted)


def _build_curves_if_possible(
    proba: np.ndarray,
    labels: np.ndarray,
    classes: List[str],
) -> Dict[str, pd.DataFrame]:
    if labels is None:
        return {}
    if not np.isin(labels, classes).any():
        return {}
    return compute_curves(proba, labels, classes)


def _summaries(
    predictions_df: pd.DataFrame,
    openset_mask: np.ndarray,
    manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = pd.DataFrame(
        [
            {"metric": "拒识率", "value": float(openset_mask.mean())},
            {"metric": "平均置信度", "value": float(predictions_df["confidence"].mean())},
            {"metric": "伪标注样本数", "value": int(len(manifest_df))},
        ]
    )
    unknown_df = predictions_df[predictions_df["is_rejected"]].copy()
    active_query_df = _active_query(predictions_df)
    return summary_df, unknown_df, active_query_df


def _create_predictions_dataframe(
    meta: pd.DataFrame,
    proba: np.ndarray,
    classes: List[str],
    rejects: np.ndarray,
) -> pd.DataFrame:
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    pred_labels = [classes[idx] for idx in max_idx]
    chinese_labels = [LABEL_CN.get(lbl, lbl) for lbl in pred_labels]

    df = meta.copy()
    df = df.reset_index(drop=True)
    df["predicted_label"] = pred_labels
    df["predicted_label_cn"] = chinese_labels
    df["confidence"] = max_proba
    for i, cls in enumerate(classes):
        df[f"proba_{cls}"] = proba[:, i]
    df["is_rejected"] = rejects
    df.loc[df["is_rejected"], "predicted_label"] = "UNK"
    df.loc[df["is_rejected"], "predicted_label_cn"] = LABEL_CN["UNK"]
    return df


def _active_query(pred_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    df = pred_df.copy()
    df = df.sort_values(["is_rejected", "confidence"]).head(top_k)
    df = df[[c for c in df.columns if c in {"file_id", "basename", "confidence", "is_rejected", "predicted_label_cn"}]]
    df = df.rename(columns={"predicted_label_cn": "current_prediction"})
    df["note"] = np.where(df["is_rejected"], "疑似未知类", "低置信度")
    return df


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.align:
        cfg.adapt.feature_align = args.align
    if args.no_selftrain:
        cfg.adapt.self_training.enabled = False

    set_global_seed(cfg.data.random_seed)

    out_dir = ensure_dir(cfg.io.out_dir)
    ensure_dir(out_dir / "figures")
    ensure_dir(out_dir / "tables")

    logger = setup_logger(cfg.io.log_file)
    logger.info("加载配置: %s", Path(args.config).resolve())

    bundle: DataBundle = load_data(cfg)
    logger.info("目标域样本数: %d", len(bundle.target_features))

    alignment: AlignmentResult = align_features(bundle, cfg)
    logger.info("域差异(对齐前): %s", alignment.metrics_before)
    logger.info("域差异(对齐后): %s", alignment.metrics_after)

    base_model, classes = _load_model_and_classes(cfg.io.q2_model_path)

    target_input = _prepare_model_input(alignment.target_aligned, bundle.model_columns)

    source_input = None
    if alignment.source_aligned is not None and bundle.source_labels is not None:
        source_input = _prepare_model_input(alignment.source_aligned, bundle.model_columns)

    selftrain_result: SelfTrainingResult = run_self_training(
        base_model,
        source_input,
        bundle.source_labels,
        target_input,
        bundle.target_meta,
        cfg,
    )
    adapted_model = selftrain_result.model

    proba_adapted = adapted_model.predict_proba(target_input)
    logger.info("伪标注轮次: %d", len(selftrain_result.class_history))

    calibration_model, proba_calibrated = _run_calibration(
        cfg,
        selftrain_result,
        proba_adapted,
        adapted_model,
        source_input,
        bundle,
        classes,
    )

    openset_cfg = cfg.adapt.open_set
    openset_result = apply_open_set(proba_calibrated, openset_cfg.method, openset_cfg.reject_threshold)

    predictions_df = _create_predictions_dataframe(bundle.target_meta, proba_calibrated, classes, openset_result.is_rejected)

    manifest_df = selftrain_result.manifest
    domain_gap_df = build_domain_gap_table(alignment)

    reliability_labels = _resolve_reliability_labels(predictions_df, bundle, cfg, selftrain_result)

    reliability = compute_reliability(proba_calibrated, reliability_labels, classes)
    curves = _build_curves_if_possible(proba_calibrated, reliability_labels, classes)

    source_counts = bundle.source_labels.value_counts().to_dict() if bundle.source_labels is not None else {cls: 0 for cls in classes}
    summary_df, unknown_df, active_query_df = _summaries(predictions_df, openset_result.is_rejected, manifest_df)

    figures = prepare_figures(
        alignment,
        predictions_df,
        reliability,
        openset_result.scores,
        openset_result.is_rejected,
        openset_cfg.reject_threshold,
        source_counts,
        selftrain_result.class_history,
        curves,
        dpi=cfg.report.figure_dpi,
        openset_method=openset_cfg.method,
        base_model=base_model,
        adapted_model=adapted_model,
        source_features=alignment.source_aligned,
        source_labels=bundle.source_labels,
        target_features=alignment.target_aligned,
    )

    curves_tables: Dict[str, pd.DataFrame] = {name: df for name, df in curves.items()}

    export_all(
        cfg,
        predictions_df,
        manifest_df,
        unknown_df,
        reliability.bins,
        reliability.summary,
        domain_gap_df,
        active_query_df,
        summary_df,
        figures,
        curves_tables,
    )

    thresholds = {
        "reject_threshold": openset_cfg.reject_threshold,
        "conf_threshold": cfg.adapt.self_training.conf_threshold,
        "temperature": getattr(calibration_model, "temperature", 1.0),
        "openset_method": openset_cfg.method,
    }

    if not args.dry_run:
        save_model_artifacts(
            cfg,
            adapted_model,
            calibration_model,
            alignment.adapter_params,
            thresholds,
            bundle.model_columns,
            classes,
        )
        save_json(Path(cfg.io.out_dir) / "q4_unknown_list.json", unknown_df.to_dict(orient="records"))

    logger.info("输出完成，主要文件位于 %s", cfg.io.out_dir)


if __name__ == "__main__":
    main()
