"""表格格式化与导出数据准备。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from .config import TableFormat, VizConfig
from .metrics import MetricsBundle


@dataclass
class TableBundle:
    report_df: pd.DataFrame
    report_html: str
    cm_counts_df: pd.DataFrame
    cm_counts_html: str
    cm_norm_df: pd.DataFrame
    cm_norm_html: str
    roc_table: Optional[pd.DataFrame]
    roc_html: Optional[str]
    pr_table: Optional[pd.DataFrame]
    pr_html: Optional[str]
    cv_summary: Optional[pd.DataFrame]
    cv_html: Optional[str]
    feature_importance: Optional[pd.DataFrame]
    feature_html: Optional[str]
    reliability_df: Optional[pd.DataFrame]
    reliability_html: Optional[str]
    console_report: str
    console_cv: Optional[str]
    file_report_df: Optional[pd.DataFrame]
    file_report_html: Optional[str]
    file_cm_counts_df: Optional[pd.DataFrame]
    file_cm_counts_html: Optional[str]
    file_cm_norm_df: Optional[pd.DataFrame]
    file_cm_norm_html: Optional[str]
    file_roc_table: Optional[pd.DataFrame]
    file_roc_html: Optional[str]
    file_pr_table: Optional[pd.DataFrame]
    file_pr_html: Optional[str]
    file_reliability_df: Optional[pd.DataFrame]
    file_reliability_html: Optional[str]
    console_file_report: Optional[str]


def build_tables(metrics: MetricsBundle, cfg: VizConfig) -> TableBundle:
    fmt = cfg.table_format

    report_df = metrics.class_report.copy()
    report_df["精确率"] = report_df["精确率"].round(fmt.precision)
    report_df["召回率"] = report_df["召回率"].round(fmt.precision)
    report_df["F1分数"] = report_df["F1分数"].round(fmt.precision)
    report_html = _render_html_table(
        report_df,
        ["精确率", "召回率", "F1分数"],
        fmt,
        title="分类报告",
    )

    cm_counts_df = metrics.confusion_counts.copy()
    cm_counts_html = _render_html_table(
        cm_counts_df,
        [],
        fmt,
        index=False,
        title="混淆矩阵（数量）",
    )

    cm_norm_df = metrics.confusion_normalized.copy()
    cols_norm = [c for c in cm_norm_df.columns if c != "真实\\预测"]
    cm_norm_df[cols_norm] = cm_norm_df[cols_norm].astype(float).round(fmt.precision)
    cm_norm_html = _render_html_table(
        cm_norm_df,
        cols_norm,
        fmt,
        index=False,
        title="混淆矩阵（归一化）",
    )

    roc_html = None
    pr_html = None
    roc_df = metrics.roc_table.copy() if metrics.roc_table is not None else None
    if roc_df is not None:
        roc_df["AUC"] = roc_df["AUC"].astype(float)
        roc_df["AUC"] = roc_df["AUC"].round(fmt.precision)
        roc_html = _render_html_table(
            roc_df,
            ["AUC"],
            fmt,
            index=False,
            title="ROC 面积 (AUC)",
        )

    pr_df = metrics.pr_table.copy() if metrics.pr_table is not None else None
    if pr_df is not None:
        pr_df["AP"] = pr_df["AP"].astype(float)
        pr_df["AP"] = pr_df["AP"].round(fmt.precision)
        pr_html = _render_html_table(
            pr_df,
            ["AP"],
            fmt,
            index=False,
            title="PR 面积 (AP)",
        )

    cv_df = metrics.cv_summary.copy() if metrics.cv_summary is not None else None
    cv_html = None
    if cv_df is not None:
        for col in cv_df.columns:
            if col == "模型":
                continue
            cv_df[col] = pd.to_numeric(cv_df[col], errors="coerce")
            if pd.api.types.is_numeric_dtype(cv_df[col]):
                cv_df[col] = cv_df[col].round(fmt.precision)
        cv_html = _render_html_table(
            cv_df,
            [c for c in cv_df.columns if c != "模型"],
            fmt,
            index=False,
            title="交叉验证汇总",
        )

    feature_df, feature_html = _build_feature_table(metrics, cfg)
    reliability_df = None
    reliability_html = None
    if metrics.reliability is not None:
        reliability_df = metrics.reliability.bins.copy()
        rel_numeric = [c for c in ["平均预测概率", "实际命中率"] if c in reliability_df.columns]
        reliability_html = _render_html_table(
            reliability_df,
            rel_numeric,
            fmt,
            index=False,
            title="校准可信度分箱",
        )

    console_df = report_df.copy()
    for col in ["精确率", "召回率", "F1分数"]:
        console_df[col] = console_df[col].map(lambda v: f"{v:.{fmt.precision}f}")
    console_report = console_df.to_string(index=False)
    console_cv = None
    if cv_df is not None:
        top = cv_df.head(5).copy()
        for col in top.columns:
            if col == "模型":
                continue
            top[col] = pd.to_numeric(top[col], errors="coerce")
            if pd.api.types.is_numeric_dtype(top[col]):
                top[col] = top[col].map(lambda v: f"{v:.{fmt.precision}f}" if pd.notna(v) else "-")
        console_cv = top.to_string(index=False)

    file_report_df = None
    file_report_html = None
    file_cm_counts_df = None
    file_cm_counts_html = None
    file_cm_norm_df = None
    file_cm_norm_html = None
    file_roc_df = None
    file_roc_html = None
    file_pr_df = None
    file_pr_html = None
    file_reliability_df = None
    file_reliability_html = None
    console_file_report = None

    if metrics.file_class_report is not None:
        file_report_df = metrics.file_class_report.copy()
        for col in ["精确率", "召回率", "F1分数"]:
            if col in file_report_df.columns:
                file_report_df[col] = pd.to_numeric(file_report_df[col], errors="coerce").round(fmt.precision)
        file_report_html = _render_html_table(
            file_report_df,
            [c for c in ["精确率", "召回率", "F1分数"] if c in file_report_df.columns],
            fmt,
            title="文件级分类报告",
        )
        console_file = file_report_df.copy()
        for col in ["精确率", "召回率", "F1分数"]:
            if col in console_file.columns:
                console_file[col] = console_file[col].map(lambda v: f"{v:.{fmt.precision}f}" if pd.notna(v) else "-")
        console_file_report = console_file.to_string(index=False)

    if metrics.file_confusion_counts is not None:
        file_cm_counts_df = metrics.file_confusion_counts.copy()
        file_cm_counts_html = _render_html_table(
            file_cm_counts_df,
            [],
            fmt,
            index=False,
            title="文件级混淆矩阵（数量）",
        )

    if metrics.file_confusion_normalized is not None:
        file_cm_norm_df = metrics.file_confusion_normalized.copy()
        cols = [c for c in file_cm_norm_df.columns if c != "真实\\预测"]
        file_cm_norm_df[cols] = file_cm_norm_df[cols].astype(float).round(fmt.precision)
        file_cm_norm_html = _render_html_table(
            file_cm_norm_df,
            cols,
            fmt,
            index=False,
            title="文件级混淆矩阵（归一化）",
        )

    if metrics.file_roc_table is not None:
        file_roc_df = metrics.file_roc_table.copy()
        if "AUC" in file_roc_df.columns:
            file_roc_df["AUC"] = pd.to_numeric(file_roc_df["AUC"], errors="coerce").round(fmt.precision)
        file_roc_html = _render_html_table(
            file_roc_df,
            ["AUC"],
            fmt,
            index=False,
            title="文件级 ROC 面积 (AUC)",
        )

    if metrics.file_pr_table is not None:
        file_pr_df = metrics.file_pr_table.copy()
        if "AP" in file_pr_df.columns:
            file_pr_df["AP"] = pd.to_numeric(file_pr_df["AP"], errors="coerce").round(fmt.precision)
        file_pr_html = _render_html_table(
            file_pr_df,
            ["AP"],
            fmt,
            index=False,
            title="文件级 PR 面积 (AP)",
        )

    if metrics.file_reliability is not None and metrics.file_reliability.bins is not None:
        file_reliability_df = metrics.file_reliability.bins.copy()
        rel_cols = [c for c in ["平均预测概率", "实际命中率"] if c in file_reliability_df.columns]
        file_reliability_html = _render_html_table(
            file_reliability_df,
            rel_cols,
            fmt,
            index=False,
            title="文件级校准可信度分箱",
        )

    return TableBundle(
        report_df=report_df,
        report_html=report_html,
        cm_counts_df=cm_counts_df,
        cm_counts_html=cm_counts_html,
        cm_norm_df=cm_norm_df,
        cm_norm_html=cm_norm_html,
        roc_table=roc_df,
        roc_html=roc_html,
        pr_table=pr_df,
        pr_html=pr_html,
        cv_summary=cv_df,
        cv_html=cv_html,
        feature_importance=feature_df,
        feature_html=feature_html,
        reliability_df=reliability_df,
        reliability_html=reliability_html,
        console_report=console_report,
        console_cv=console_cv,
        file_report_df=file_report_df,
        file_report_html=file_report_html,
        file_cm_counts_df=file_cm_counts_df,
        file_cm_counts_html=file_cm_counts_html,
        file_cm_norm_df=file_cm_norm_df,
        file_cm_norm_html=file_cm_norm_html,
        file_roc_table=file_roc_df,
        file_roc_html=file_roc_html,
        file_pr_table=file_pr_df,
        file_pr_html=file_pr_html,
        file_reliability_df=file_reliability_df,
        file_reliability_html=file_reliability_html,
        console_file_report=console_file_report,
    )


def _wrap_html(content: str, title: str | None = None) -> str:
    heading = title or "Q2 可视化表格"
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"zh-CN\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        f"  <title>{heading}</title>\n"
        "  <style>\n"
        "    body { font-family: 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial', sans-serif;"
        " line-height: 1.6; margin: 24px; background-color: #fbfbfb; color: #222; }\n"
        "    table { margin: 0 auto; border-collapse: collapse; font-size: 15px; }\n"
        "    th, td { padding: 8px 14px; border: 1px solid #d0d0d0; text-align: center; }\n"
        "    th { background: #f2f2f2; font-weight: 600; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"<h2 style=\"text-align:center;\">{heading}</h2>\n"
        f"{content}\n"
        "</body>\n"
        "</html>\n"
    )


def _render_html_table(
    df: pd.DataFrame,
    numeric_cols: List[str],
    fmt: TableFormat,
    index: bool = False,
    title: str | None = None,
) -> str:
    """手工渲染 HTML 表格，支持阈值高亮。"""

    columns = df.columns.tolist()
    html_rows: List[str] = []
    if index:
        header = "<tr><th></th>" + "".join(f"<th>{col}</th>" for col in columns) + "</tr>"
    else:
        header = "<tr>" + "".join(f"<th>{col}</th>" for col in columns) + "</tr>"
    html_rows.append(header)

    for idx, row in df.iterrows():
        cells = []
        if index:
            cells.append(f"<td>{idx}</td>")
        for col in columns:
            val = row[col]
            display = val
            style = ""
            if col in numeric_cols and pd.notna(val) and isinstance(val, (int, float)):
                display = f"{val:.{fmt.precision}f}"
                if fmt.highlight_lower is not None and val < fmt.highlight_lower:
                    style = f' style="background-color: {fmt.highlight_low_color};"'
                if fmt.highlight_upper is not None and val >= fmt.highlight_upper:
                    style = f' style="background-color: {fmt.highlight_high_color};"'
            elif pd.isna(val):
                display = "-"
            cells.append(f"<td{style}>{display}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_body = "\n".join(html_rows)
    table_html = (
        "<table border='0' style=\"border-collapse: collapse; text-align: center;\">\n"
        f"{table_body}\n"
        "</table>"
    )
    return _wrap_html(table_html, title)


def _build_feature_table(metrics: MetricsBundle, cfg: VizConfig) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    model_imp = metrics.model_importance
    perm_imp = metrics.permutation_importance
    if model_imp is None and perm_imp is None:
        return None, None

    rows = []
    if model_imp is not None:
        rows.extend(
            {
                "特征名": idx,
                "重要度": float(val),
                "来源": "模型",
            }
            for idx, val in model_imp.items()
        )
    if perm_imp is not None:
        rows.extend(
            {
                "特征名": idx,
                "重要度": float(val),
                "来源": "置换",
            }
            for idx, val in perm_imp.items()
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("重要度", ascending=False).reset_index(drop=True)
    df["重要度"] = df["重要度"].round(cfg.table_format.precision)
    html = _render_html_table(
        df,
        ["重要度"],
        cfg.table_format,
        index=False,
        title="特征重要度",
    )
    return df, html
