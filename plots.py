"""绘图模块：生成中文图表。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

from .config import VizConfig
from .metrics import MetricsBundle, ReliabilityData


@dataclass
class PlotRecord:
    figure: plt.Figure
    filename: str
    description: str


COLOR_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_confusion_matrices(metrics: MetricsBundle, cfg: VizConfig) -> List[PlotRecord]:
    records: List[PlotRecord] = []

    counts = metrics.confusion_counts.set_index("真实\\预测")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(counts.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(counts.columns)))
    ax.set_xticklabels(counts.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(counts.index)))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（计数）")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = counts.iloc[i, j]
            ax.text(j, i, int(value), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    records.append(PlotRecord(fig, "q2_confusion_matrix.png", "混淆矩阵计数"))

    norm = metrics.confusion_normalized.set_index("真实\\预测")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(norm.astype(float).to_numpy(), cmap="Greens", vmin=0, vmax=1)
    ax2.set_xticks(range(len(norm.columns)))
    ax2.set_xticklabels(norm.columns, rotation=45, ha="right")
    ax2.set_yticks(range(len(norm.index)))
    ax2.set_yticklabels(norm.index)
    ax2.set_xlabel("预测类别")
    ax2.set_ylabel("真实类别")
    ax2.set_title("混淆矩阵（行归一化）")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            value = norm.iloc[i, j]
            ax2.text(j, i, value, ha="center", va="center", color="black")
    fig2.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    records.append(PlotRecord(fig2, "q2_confusion_matrix_norm.png", "混淆矩阵归一化"))

    if metrics.file_confusion_counts is not None:
        file_counts = metrics.file_confusion_counts.set_index("真实\\预测")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        im3 = ax3.imshow(file_counts.to_numpy(), cmap="Purples")
        ax3.set_xticks(range(len(file_counts.columns)))
        ax3.set_xticklabels(file_counts.columns, rotation=45, ha="right")
        ax3.set_yticks(range(len(file_counts.index)))
        ax3.set_yticklabels(file_counts.index)
        ax3.set_xlabel("预测类别")
        ax3.set_ylabel("真实类别")
        ax3.set_title("文件级混淆矩阵（数量）")
        for i in range(file_counts.shape[0]):
            for j in range(file_counts.shape[1]):
                ax3.text(j, i, int(file_counts.iloc[i, j]), ha="center", va="center", color="black")
        fig3.colorbar(im3, ax=ax3)
        fig3.tight_layout()
        records.append(PlotRecord(fig3, "q2_file_confusion_matrix.png", "文件级混淆矩阵计数"))

    if metrics.file_confusion_normalized is not None:
        file_norm = metrics.file_confusion_normalized.set_index("真实\\预测")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        im4 = ax4.imshow(file_norm.astype(float).to_numpy(), cmap="Oranges", vmin=0, vmax=1)
        ax4.set_xticks(range(len(file_norm.columns)))
        ax4.set_xticklabels(file_norm.columns, rotation=45, ha="right")
        ax4.set_yticks(range(len(file_norm.index)))
        ax4.set_yticklabels(file_norm.index)
        ax4.set_xlabel("预测类别")
        ax4.set_ylabel("真实类别")
        ax4.set_title("文件级混淆矩阵（行归一化）")
        for i in range(file_norm.shape[0]):
            for j in range(file_norm.shape[1]):
                val = file_norm.iloc[i, j]
                ax4.text(j, i, val, ha="center", va="center", color="black")
        fig4.colorbar(im4, ax=ax4)
        fig4.tight_layout()
        records.append(PlotRecord(fig4, "q2_file_confusion_matrix_norm.png", "文件级混淆矩阵归一化"))

    return records


def plot_roc_pr(metrics_bundle: MetricsBundle, cfg: VizConfig) -> List[PlotRecord]:
    records: List[PlotRecord] = []
    roc = metrics_bundle.roc_data
    pr = metrics_bundle.pr_data
    if roc is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for idx, (label, data) in enumerate(roc.per_class.items()):
            disp = cfg.classes.display.get(label, label)
            ax.plot(data["fpr"], data["tpr"], color=COLOR_CYCLE[idx % len(COLOR_CYCLE)], label=f"{disp} (AUC={roc.per_class_auc[label]:.3f})")
        ax.plot(roc.micro["fpr"], roc.micro["tpr"], color="#444444", linestyle="--", label=f"micro (AUC={metrics.auc(roc.micro['fpr'], roc.micro['tpr']):.3f})")
        ax.plot([0, 1], [0, 1], color="#999999", linestyle=":")
        ax.set_xlabel("假阳性率")
        ax.set_ylabel("真阳性率")
        ax.set_title("ROC 曲线")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        records.append(PlotRecord(fig, "q2_roc_ovr.png", "ROC 曲线"))
    if pr is not None:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        for idx, (label, data) in enumerate(pr.per_class.items()):
            disp = cfg.classes.display.get(label, label)
            ax2.plot(data["recall"], data["precision"], color=COLOR_CYCLE[idx % len(COLOR_CYCLE)], label=f"{disp} (AP={pr.per_class_ap[label]:.3f})")
        ax2.plot(pr.micro["recall"], pr.micro["precision"], color="#444444", linestyle="--", label="micro")
        ax2.set_xlabel("召回率")
        ax2.set_ylabel("精确率")
        ax2.set_title("PR 曲线")
        ax2.legend(loc="lower left", fontsize=9)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        records.append(PlotRecord(fig2, "q2_pr_ovr.png", "PR 曲线"))

    if metrics_bundle.file_roc_data is not None:
        roc_f = metrics_bundle.file_roc_data
        figf, axf = plt.subplots(figsize=(7, 5))
        for idx, (label, data) in enumerate(roc_f.per_class.items()):
            disp = cfg.classes.display.get(label, label)
            axf.plot(data["fpr"], data["tpr"], color=COLOR_CYCLE[idx % len(COLOR_CYCLE)], label=f"{disp} (AUC={roc_f.per_class_auc[label]:.3f})")
        axf.plot(roc_f.micro["fpr"], roc_f.micro["tpr"], color="#444444", linestyle="--", label=f"micro (AUC={metrics.auc(roc_f.micro['fpr'], roc_f.micro['tpr']):.3f})")
        axf.plot([0, 1], [0, 1], color="#999999", linestyle=":")
        axf.set_xlabel("假阳性率")
        axf.set_ylabel("真阳性率")
        axf.set_title("文件级 ROC 曲线")
        axf.legend(loc="lower right", fontsize=9)
        axf.grid(alpha=0.3)
        figf.tight_layout()
        records.append(PlotRecord(figf, "q2_file_roc_ovr.png", "文件级 ROC"))

    if metrics_bundle.file_pr_data is not None:
        pr_f = metrics_bundle.file_pr_data
        figfp, axfp = plt.subplots(figsize=(7, 5))
        for idx, (label, data) in enumerate(pr_f.per_class.items()):
            disp = cfg.classes.display.get(label, label)
            axfp.plot(data["recall"], data["precision"], color=COLOR_CYCLE[idx % len(COLOR_CYCLE)], label=f"{disp} (AP={pr_f.per_class_ap[label]:.3f})")
        axfp.plot(pr_f.micro["recall"], pr_f.micro["precision"], color="#444444", linestyle="--", label="micro")
        axfp.set_xlabel("召回率")
        axfp.set_ylabel("精确率")
        axfp.set_title("文件级 PR 曲线")
        axfp.legend(loc="lower left", fontsize=9)
        axfp.grid(alpha=0.3)
        figfp.tight_layout()
        records.append(PlotRecord(figfp, "q2_file_pr_ovr.png", "文件级 PR"))
    return records


def plot_feature_importance(metrics_bundle: MetricsBundle) -> Optional[PlotRecord]:
    model_imp = metrics_bundle.model_importance
    perm_imp = metrics_bundle.permutation_importance
    if model_imp is None and perm_imp is None:
        return None

    union = []
    if model_imp is not None:
        union.extend(model_imp.index.tolist())
    if perm_imp is not None:
        union.extend(perm_imp.index.tolist())
    unique_features = list(dict.fromkeys(union))[:30]
    if not unique_features:
        return None

    values_model = [model_imp.get(feat, 0.0) if model_imp is not None else 0.0 for feat in unique_features]
    values_perm = [perm_imp.get(feat, 0.0) if perm_imp is not None else 0.0 for feat in unique_features]

    indices = np.arange(len(unique_features))
    height = 0.35
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(unique_features))))
    if model_imp is not None:
        ax.barh(indices - height / 2, values_model, height=height, color="#1f77b4", label="模型")
    if perm_imp is not None:
        ax.barh(indices + height / 2, values_perm, height=height, color="#ff7f0e", label="置换")
    ax.set_yticks(indices)
    ax.set_yticklabels(unique_features)
    ax.invert_yaxis()
    ax.set_xlabel("重要度")
    ax.set_title("特征重要度对比 (Top-K)")
    ax.legend()
    fig.tight_layout()
    return PlotRecord(fig, "q2_feature_importance.png", "特征重要度")


def plot_model_comparison(metrics_bundle: MetricsBundle) -> Optional[PlotRecord]:
    summary = metrics_bundle.cv_summary
    if summary is None or summary.empty or "F1宏平均(均值)" not in summary.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(summary))))
    ax.barh(summary["模型"], summary["F1宏平均(均值)"], color="#2ca02c")
    ax.invert_yaxis()
    ax.set_xlabel("F1 宏平均(均值)")
    ax.set_title("候选模型对比")
    fig.tight_layout()
    return PlotRecord(fig, "q2_model_comparison.png", "模型对比")


def plot_reliability(data: ReliabilityData | None, filename: str, title: str) -> Optional[PlotRecord]:
    if data is None:
        return None
    df = data.bins
    valid = df[df["样本数"] > 0].copy()
    if valid.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(valid["平均预测概率"], valid["实际命中率"], marker="o", color="#d62728")
    ax.plot([0, 1], [0, 1], linestyle=":", color="#555555")
    ax.set_xlabel("平均预测概率")
    ax.set_ylabel("实际命中率")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return PlotRecord(fig, filename, title)


def generate_all_plots(metrics_bundle: MetricsBundle, cfg: VizConfig) -> List[PlotRecord]:
    records: List[PlotRecord] = []
    if cfg.plots.plot_confusion:
        records.extend(plot_confusion_matrices(metrics_bundle, cfg))
    if cfg.plots.plot_roc_pr:
        records.extend(plot_roc_pr(metrics_bundle, cfg))
    if cfg.plots.plot_feature_importance:
        fi_record = plot_feature_importance(metrics_bundle)
        if fi_record:
            records.append(fi_record)
    if cfg.plots.plot_model_comparison:
        mc_record = plot_model_comparison(metrics_bundle)
        if mc_record:
            records.append(mc_record)
    if cfg.plots.plot_reliability:
        rel_record = plot_reliability(metrics_bundle.reliability, "q2_reliability.png", "可靠性曲线")
        if rel_record:
            records.append(rel_record)
        file_rel_record = plot_reliability(metrics_bundle.file_reliability, "q2_file_reliability.png", "文件级可靠性曲线")
        if file_rel_record:
            records.append(file_rel_record)
    return records
