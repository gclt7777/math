"""导出 PNG/CSV/HTML 与 summary.json。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from .config import VizConfig, snapshot_config
from .metrics import MetricsBundle
from .plots import PlotRecord
from .tables import TableBundle


def save_plots(plots: Iterable[PlotRecord], cfg: VizConfig) -> List[str]:
    paths: List[str] = []
    out_dir = Path(cfg.io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in plots:
        path = out_dir / record.filename
        record.figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(record.figure)
        paths.append(str(path))
    return paths


def save_tables(tables: TableBundle, cfg: VizConfig) -> List[str]:
    out_dir = Path(cfg.io.out_dir)
    paths: List[str] = []

    def _write(df: pd.DataFrame, html: str, base: str) -> None:
        csv_path = out_dir / f"{base}.csv"
        df.to_csv(csv_path, index=False)
        paths.append(str(csv_path))
        html_path = out_dir / f"{base}.html"
        html_path.write_text(html, encoding="utf-8")
        paths.append(str(html_path))

    _write(tables.report_df, tables.report_html, "q2_classification_report")
    _write(tables.cm_counts_df, tables.cm_counts_html, "q2_cm_counts")
    _write(tables.cm_norm_df, tables.cm_norm_html, "q2_cm_normalized")

    if tables.roc_table is not None and tables.roc_html is not None:
        _write(tables.roc_table, tables.roc_html, "q2_roc_auc_table")
    if tables.pr_table is not None and tables.pr_html is not None:
        _write(tables.pr_table, tables.pr_html, "q2_pr_ap_table")
    if tables.cv_summary is not None and tables.cv_html is not None:
        _write(tables.cv_summary, tables.cv_html, "q2_cv_summary")
    if tables.feature_importance is not None and tables.feature_html is not None:
        _write(tables.feature_importance, tables.feature_html, "feature_importance")
    if tables.reliability_df is not None and tables.reliability_html is not None:
        _write(tables.reliability_df, tables.reliability_html, "q2_reliability_bins")
    if tables.file_report_df is not None and tables.file_report_html is not None:
        _write(tables.file_report_df, tables.file_report_html, "q2_file_classification_report")
    if tables.file_cm_counts_df is not None and tables.file_cm_counts_html is not None:
        _write(tables.file_cm_counts_df, tables.file_cm_counts_html, "q2_file_cm_counts")
    if tables.file_cm_norm_df is not None and tables.file_cm_norm_html is not None:
        _write(tables.file_cm_norm_df, tables.file_cm_norm_html, "q2_file_cm_normalized")
    if tables.file_roc_table is not None and tables.file_roc_html is not None:
        _write(tables.file_roc_table, tables.file_roc_html, "q2_file_roc_auc_table")
    if tables.file_pr_table is not None and tables.file_pr_html is not None:
        _write(tables.file_pr_table, tables.file_pr_html, "q2_file_pr_ap_table")
    if tables.file_reliability_df is not None and tables.file_reliability_html is not None:
        _write(tables.file_reliability_df, tables.file_reliability_html, "q2_file_reliability_bins")
    return paths


def save_reliability_metrics(metrics: MetricsBundle, cfg: VizConfig) -> List[str]:
    paths: List[str] = []
    out_dir = Path(cfg.io.out_dir)
    if metrics.reliability is not None:
        brier_path = out_dir / "q2_brier_score.txt"
        brier_path.write_text(f"Brier 分数: {metrics.reliability.brier_score:.4f}\n", encoding="utf-8")
        paths.append(str(brier_path))
    if metrics.file_reliability is not None:
        file_brier_path = out_dir / "q2_file_brier_score.txt"
        file_brier_path.write_text(
            f"文件级 Brier 分数: {metrics.file_reliability.brier_score:.4f}\n",
            encoding="utf-8",
        )
        paths.append(str(file_brier_path))
    return paths


def write_summary_json(plots: List[str], table_paths: List[str], metrics: MetricsBundle, cfg: VizConfig) -> str:
    out_dir = Path(cfg.io.out_dir)
    summary_path = out_dir / "summary.json"
    payload = {
        "config": snapshot_config(cfg),
        "plots": plots,
        "tables": table_paths,
    }
    if metrics.reliability is not None:
        payload["brier_score"] = metrics.reliability.brier_score
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(summary_path)


def save_all(plots: Iterable[PlotRecord], tables: TableBundle, metrics: MetricsBundle, cfg: VizConfig) -> List[str]:
    plot_paths = save_plots(plots, cfg)
    table_paths = save_tables(tables, cfg)
    extra_paths = save_reliability_metrics(metrics, cfg)
    summary_path = write_summary_json(plot_paths, table_paths + extra_paths, metrics, cfg)
    return plot_paths + table_paths + extra_paths + [summary_path]
