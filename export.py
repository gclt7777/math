"""导出 CSV/HTML/PNG 以及汇总。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

from .config import Q3Config, snapshot_config
from .align import AlignmentResult
from .infer import PredictionBundle
from .metrics import ReportBundle
from .viz import FigureRecord


def _write_html_table(df: pd.DataFrame, path: Path, title: str) -> None:
    html = ["<html><head><meta charset='utf-8'><title>{}</title></head><body>".format(title)]
    html.append(f"<h2>{title}</h2>")
    html.append(df.to_html(index=False, border=0, justify="center"))
    html.append("</body></html>")
    path.write_text("\n".join(html), encoding="utf-8")


def save_all(
    preds: PredictionBundle,
    reports: ReportBundle,
    figures: Iterable[FigureRecord],
    alignment: AlignmentResult,
    cfg: Q3Config,
) -> List[str]:
    out_dir = Path(cfg.io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    exported: List[str] = []

    # 预测结果表
    segment_path = out_dir / "predictions_segment.csv"
    preds.segment_df.to_csv(segment_path, index=False)
    exported.append(str(segment_path))

    file_path = out_dir / "predictions_file.csv"
    preds.file_df.to_csv(file_path, index=False)
    exported.append(str(file_path))

    submission_path = out_dir / "predictions_target_fault_type.csv"
    preds.submission_df.to_csv(submission_path, index=False)
    exported.append(str(submission_path))

    uncertain_path = out_dir / "uncertain_cases.csv"
    preds.uncertain_df.to_csv(uncertain_path, index=False)
    exported.append(str(uncertain_path))

    # 域差异与机理表
    domain_gap_path = out_dir / "domain_gap_metrics.csv"
    reports.domain_gap_df.to_csv(domain_gap_path, index=False)
    exported.append(str(domain_gap_path))

    mech_path = out_dir / "mechanism_evidence.csv"
    reports.mechanism_df.to_csv(mech_path, index=False)
    exported.append(str(mech_path))

    contrib_path = out_dir / "high_conf_sample_topk_features.csv"
    reports.high_conf_df.to_csv(contrib_path, index=False)
    exported.append(str(contrib_path))

    # HTML 表格
    _write_html_table(preds.file_df.head(20), tables_dir / "predictions_file_preview.html", "文件级预测示例")
    exported.append(str(tables_dir / "predictions_file_preview.html"))
    _write_html_table(reports.domain_gap_df, tables_dir / "domain_gap_metrics.html", "域差异指标")
    exported.append(str(tables_dir / "domain_gap_metrics.html"))
    if not reports.mechanism_df.empty:
        _write_html_table(reports.mechanism_df, tables_dir / "mechanism_evidence.html", "机理一致性摘要")
        exported.append(str(tables_dir / "mechanism_evidence.html"))
    if not reports.high_conf_df.empty:
        _write_html_table(reports.high_conf_df, tables_dir / "high_conf_topk.html", "高置信特征贡献")
        exported.append(str(tables_dir / "high_conf_topk.html"))

    # 适配参数
    transform_path = out_dir / "adapt_transform.json"
    transform_path.write_text(json.dumps(alignment.transform_params, ensure_ascii=False, indent=2), encoding="utf-8")
    exported.append(str(transform_path))

    # 图像
    for record in figures:
        full_path = out_dir / record.filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        record.figure.savefig(full_path, dpi=cfg.viz.figure_dpi, bbox_inches="tight")
        plt.close(record.figure)
        exported.append(str(full_path))

    # 汇总与报告
    summary = {
        "config": snapshot_config(cfg),
        "outputs": exported,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    exported.append(str(summary_path))

    _write_html_report(preds, reports, figures, cfg)
    exported.append(str(out_dir / "viz_index.html"))

    return exported


def _write_html_report(
    preds: PredictionBundle,
    reports: ReportBundle,
    figures: Iterable[FigureRecord],
    cfg: Q3Config,
) -> None:
    out_dir = Path(cfg.io.out_dir)
    lines = ["<html><head><meta charset='utf-8'><title>Q3 无监督迁移报告</title></head><body>"]
    lines.append("<h1>Q3 无监督迁移诊断报告</h1>")
    lines.append("<h2>文件级预测概览</h2>")
    lines.append(f"<p>生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}</p>")
    lines.append("<table border='0'>")
    lines.append(preds.file_df.head(20).to_html(index=False, border=0, justify="center"))
    if not preds.submission_df.empty:
        lines.append("<h2>目标文件预测结果</h2>")
        lines.append(preds.submission_df.to_html(index=False, border=0, justify="center"))
    lines.append("</table>")

    lines.append("<h2>图表</h2><ul>")
    for record in figures:
        rel_path = record.filename
        lines.append(f"<li><a href='{rel_path}' target='_blank'>{record.description}</a></li>")
    lines.append("</ul>")

    lines.append("<h2>关键表格链接</h2><ul>")
    for name in [
        "predictions_file.csv",
        "uncertain_cases.csv",
        "domain_gap_metrics.csv",
        "mechanism_evidence.csv",
        "high_conf_sample_topk_features.csv",
    ]:
        if (out_dir / name).exists():
            lines.append(f"<li><a href='{name}' target='_blank'>{name}</a></li>")
    lines.append("</ul>")

    lines.append("</body></html>")
    (out_dir / "viz_index.html").write_text("\n".join(lines), encoding="utf-8")
