"""导出模型、报表与图像。"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from .config import Q4Config, snapshot_config
from .utils import ensure_dir, save_json, save_yaml
from .viz import FigureRecord


def save_figures(figures: List[FigureRecord], out_dir: Path) -> List[str]:
    paths: List[str] = []
    fig_dir = out_dir
    for record in figures:
        path = fig_dir / record.filename
        record.figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(record.figure)
        paths.append(str(path))
    return paths


def save_tables(payload: Dict[str, pd.DataFrame], out_dir: Path) -> List[str]:
    paths: List[str] = []
    for name, df in payload.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        paths.append(str(path))
    return paths


def save_text_tables(payload: Dict[str, pd.DataFrame], out_dir: Path) -> List[str]:
    paths: List[str] = []
    for name, df in payload.items():
        path = out_dir / f"{name}.html"
        df.to_html(path, index=False, justify="center")
        paths.append(str(path))
    return paths


def export_all(
    cfg: Q4Config,
    predictions: pd.DataFrame,
    pseudolabel_manifest: pd.DataFrame,
    unknown_list: pd.DataFrame,
    reliability_bins: pd.DataFrame,
    reliability_summary: Dict[str, float],
    domain_gap_df: pd.DataFrame,
    active_query_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figures: List[FigureRecord],
    curves_tables: Dict[str, pd.DataFrame],
) -> None:
    out_dir = Path(cfg.io.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "tables")
    ensure_dir(out_dir / "figures")

    predictions.to_csv(out_dir / "q4_predictions.csv", index=False)
    pseudolabel_manifest.to_csv(out_dir / "q4_pseudolabel_manifest.csv", index=False)
    unknown_list.to_csv(out_dir / "q4_unknown_list.csv", index=False)
    reliability_bins.to_csv(out_dir / "q4_calibration_curve.csv", index=False)
    pd.DataFrame([reliability_summary]).to_csv(out_dir / "q4_reliability_summary.csv", index=False)
    domain_gap_df.to_csv(out_dir / "q4_domain_gap_metrics.csv", index=False)
    active_query_df.to_csv(out_dir / "q4_active_query_list.csv", index=False)
    summary_df.to_csv(out_dir / "q4_summary.csv", index=False)

    for name, df in curves_tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    save_figures(figures, out_dir / "figures")

    snapshot = snapshot_config(cfg)
    save_yaml(out_dir / "q4_config_snapshot.yaml", snapshot)


def save_model_artifacts(
    cfg: Q4Config,
    model: object,
    calibration_model,
    adapter_params: Dict[str, object],
    thresholds: Dict[str, float],
    feature_columns: List[str],
    classes: List[str],
) -> None:
    out_dir = Path(cfg.io.out_dir)

    package = {
        "model": model,
        "calibration": calibration_model,
        "adapter": adapter_params,
        "thresholds": thresholds,
        "feature_columns": feature_columns,
        "classes": classes,
    }
    joblib.dump(package, out_dir / "q4_final_model.joblib")
    if cfg.export.save_adapter:
        joblib.dump(adapter_params, out_dir / "q4_adapter.joblib")
    if cfg.export.save_thresholds:
        save_json(out_dir / "q4_thresholds.json", thresholds)
