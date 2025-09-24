"""Q3 可视化，全部中文标注。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:  # pragma: no cover - 可选依赖
    umap = None

from .align import AlignmentResult
from .config import Q3Config
from .infer import PredictionBundle
from .metrics import ReportBundle


@dataclass
class FigureRecord:
    figure: plt.Figure
    filename: str
    description: str


def _embedding(data: np.ndarray, use_umap: bool) -> np.ndarray:
    if use_umap and umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(data)


def render_all(result: AlignmentResult, preds: PredictionBundle, reports: ReportBundle, cfg: Q3Config) -> List[FigureRecord]:
    records: List[FigureRecord] = []

    overview_fig = _plot_target_overview(result, cfg)
    if overview_fig:
        records.append(overview_fig)
    records.extend(_plot_domain_embeddings(result, preds, cfg))
    dashboard_fig = _plot_transfer_dashboard(result, preds, cfg)
    if dashboard_fig:
        records.append(dashboard_fig)
    else:
        records.append(_plot_confidence_hist(preds, cfg))
        records.append(_plot_file_bars(preds, cfg))
    mech_fig = _plot_mechanism(reports, cfg)
    if mech_fig:
        records.append(mech_fig)
    records.append(_plot_domain_gap(reports))
    return [rec for rec in records if rec is not None]


def _plot_target_overview(result: AlignmentResult, cfg: Q3Config) -> Optional[FigureRecord]:
    df = result.target_raw.features.copy()
    meta = result.target.meta
    if df.empty:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("目标域数据特征分析", fontsize=20, fontweight="bold")

    ids_series = meta.get(cfg.data.file_basename_col, pd.Series(range(len(df))))
    ids = ids_series.astype(str).tolist()

    ax_summary = axes[0, 0]
    ax_summary.clear()
    counts = pd.Series(ids).value_counts().sort_index()
    ax_summary.bar(counts.index, counts.values, color="#5DADE2", edgecolor="#154360", alpha=0.85)
    ax_summary.set_title("目标域文件分布", fontweight="bold")
    ax_summary.set_xlabel("文件ID")
    ax_summary.set_ylabel("出现次数")
    for tick in ax_summary.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
    summary_text = (
        f"文件数量: {len(ids)}\n"
        f"特征维度: {df.shape[1]}\n"
    )
    if "sampling_rate" in df.columns and not df["sampling_rate"].empty:
        summary_text += f"采样率(Hz): {np.median(df['sampling_rate'].dropna()):.0f}\n"
    else:
        summary_text += "采样率(Hz): 未知\n"
    if {"signal_length", "sampling_rate"}.issubset(df.columns):
        sr = df["sampling_rate"].replace(0, np.nan)
        duration = (df["signal_length"] / sr).dropna()
        summary_text += f"信号时长(秒): {duration.mean():.2f}" if not duration.empty else "信号时长(秒): 未知"
    else:
        summary_text += "信号时长(秒): 未知"
    ax_summary.text(
        0.02,
        0.95,
        summary_text,
        transform=ax_summary.transAxes,
        fontsize=13,
        va="top",
        bbox=dict(boxstyle="round", facecolor="#F8F9F9", edgecolor="#D5D8DC", alpha=0.92),
    )

    if "signal_length" in df.columns:
        axes[0, 1].hist(df["signal_length"].dropna(), bins=10, color="#58D68D", edgecolor="#1B4F72", alpha=0.8)
        axes[0, 1].set_title("信号长度分布", fontweight="bold")
        axes[0, 1].set_xlabel("采样点数")
        axes[0, 1].set_ylabel("频次")
    else:
        axes[0, 1].set_visible(False)

    for ax, col, title, color in [
        (axes[0, 2], "rms", "RMS 值分布", "#F1948A"),
        (axes[1, 0], "kurtosis", "峭度分布", "#F7DC6F"),
        (axes[1, 1], "crest_factor", "峰值因子分布", "#D7BDE2"),
    ]:
        if col in df.columns:
            ax.hist(df[col].dropna(), bins=12, color=color, edgecolor="#424949", alpha=0.85)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("频次")
        else:
            ax.set_visible(False)

    key_features = [c for c in ["rms", "kurtosis", "crest_factor", "spectral_centroid"] if c in df.columns]
    ax_comp = axes[1, 2]
    if key_features:
        norm_df = df[key_features].copy()
        norm_df = (norm_df - norm_df.min()) / (norm_df.max() - norm_df.min() + 1e-12)
        x = np.arange(len(ids))
        width = 0.75 / len(key_features)
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F5B041"]
        for i, feat in enumerate(key_features):
            ax_comp.bar(x + i * width, norm_df[feat], width=width, color=colors[i % len(colors)], alpha=0.9, label=feat)
        ax_comp.set_title("关键特征对比 (标准化)", fontweight="bold")
        ax_comp.set_xlabel("文件ID")
        ax_comp.set_ylabel("标准化值")
        ax_comp.set_xticks(x + width * (len(key_features) - 1) / 2)
        ax_comp.set_xticklabels(ids, rotation=35)
        ax_comp.legend(frameon=False, loc="upper right")
    else:
        ax_comp.set_visible(False)

    for ax in axes.ravel():
        if ax.has_data():
            ax.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return FigureRecord(fig, "figures/target_feature_overview.png", "目标域特征概览")


def _plot_domain_embeddings(result: AlignmentResult, preds: PredictionBundle, cfg: Q3Config) -> List[FigureRecord]:
    records: List[FigureRecord] = []
    source_labels = result.source.meta.get(cfg.data.label_col, pd.Series(["未知"] * len(result.source.meta))).astype(str)
    target_preds = preds.segment_df["pred_top1"].astype(str)

    default_display = {"N": "正常", "OR": "外圈故障", "IR": "内圈故障", "B": "滚动体故障"}
    display_map = getattr(cfg, "class_display", default_display)
    class_order = getattr(cfg, "class_order", list(preds.classes))

    def build_embeddings(source_feat: pd.DataFrame, target_feat: pd.DataFrame, target_label: pd.Series, domain_suffix: str) -> pd.DataFrame:
        combined = pd.concat([source_feat, target_feat])
        emb = _embedding(combined.values, cfg.viz.use_umap)
        df_embed = pd.DataFrame(emb, columns=["x", "y"], index=combined.index)
        df_embed.loc[source_feat.index, "domain"] = "源域"
        df_embed.loc[target_feat.index, "domain"] = f"目标域({domain_suffix})"
        df_embed.loc[source_feat.index, "label"] = source_labels.reindex(source_feat.index).fillna("未知")
        df_embed.loc[target_feat.index, "label"] = target_label.reindex(target_feat.index).fillna("未知")
        df_embed["label_display"] = df_embed["label"].map(lambda x: display_map.get(x, default_display.get(x, x)))
        return df_embed

    before_df = build_embeddings(result.source.features, result.target_raw.features, target_preds, "适配前")
    after_df = build_embeddings(result.source.features, result.target.features, target_preds, "适配后")

    for name, df_plot, fname in [
        ("域对齐前的特征嵌入", before_df, "embedding_before.png"),
        ("域对齐后的特征嵌入", after_df, "embedding_after.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        markers = {"源域": "o", "目标域(适配前)": "^", "目标域(适配后)": "^"}
        colors = plt.cm.get_cmap("tab10", max(3, len(class_order)))
        for (domain, label), group in df_plot.groupby(["domain", "label_display"]):
            raw_label = group["label"].iloc[0]
            if class_order and raw_label in class_order:
                color_idx = class_order.index(raw_label)
            elif raw_label in default_display:
                color_idx = ["N", "OR", "IR", "B"].index(raw_label)
            else:
                color_idx = 0
            ax.scatter(group["x"], group["y"], s=18, marker=markers.get(domain, "o"),
                       color=colors(color_idx), label=f"{domain}-{label}")
        ax.set_title(name + (" (UMAP)" if cfg.viz.use_umap and umap is not None else " (PCA)"))
        ax.set_xlabel("嵌入维度 1")
        ax.set_ylabel("嵌入维度 2")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        records.append(FigureRecord(fig, f"figures/{fname}", name))
    return records


def _plot_transfer_dashboard(result: AlignmentResult, preds: PredictionBundle, cfg: Q3Config) -> Optional[FigureRecord]:
    if preds.file_df.empty:
        return None

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("轴承故障迁移学习诊断结果", fontsize=18, fontweight="bold")

    # 1. 嵌入图
    ax1 = plt.subplot(2, 3, 1)
    try:
        combined = pd.concat([result.source.features, result.target.features])
        emb = _embedding(combined.values, cfg.viz.use_umap)
        ids = combined.index
        src_len = len(result.source.features)
        ax1.scatter(emb[:src_len, 0], emb[:src_len, 1], c="#3498DB", alpha=0.6, label="源域", s=40)
        ax1.scatter(emb[src_len:, 0], emb[src_len:, 1], c="#E74C3C", marker="s", alpha=0.85, label="目标域", s=70)
    except Exception as exc:  # pragma: no cover - 安全回退
        ax1.text(0.5, 0.5, f"嵌入失败: {exc}", ha="center", va="center")
    ax1.set_title("t-SNE 特征空间分布" if cfg.viz.use_umap else "PCA 特征空间分布")
    ax1.set_xlabel("维度 1")
    ax1.set_ylabel("维度 2")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 2. 预测分布饼图
    ax2 = plt.subplot(2, 3, 2)
    label_cn = {"N": "正常", "OR": "外圈故障", "IR": "内圈故障", "B": "滚动体故障"}
    counts = preds.file_df["pred_label"].value_counts()
    labels = [label_cn.get(lbl, lbl) for lbl in counts.index]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    if not counts.empty:
        ax2.pie(counts.values, labels=labels, autopct="%1.1f%%", colors=colors[: len(counts)], startangle=90)
    ax2.set_title("目标域预测结果分布", fontweight="bold")

    # 3. 置信度分布
    ax3 = plt.subplot(2, 3, 3)
    conf = preds.file_df["prob_label"].to_numpy()
    ax3.hist(conf, bins=8, color="#5DADE2", edgecolor="black", alpha=0.8)
    ax3.axvline(conf.mean(), color="#E74C3C", linestyle="--", label=f"平均置信度: {conf.mean():.3f}")
    ax3.axvline(cfg.inference.unknown_threshold, color="#27AE60", linestyle=":", label="不确定阈值")
    ax3.set_xlabel("置信度")
    ax3.set_ylabel("文件数量")
    ax3.set_title("预测置信度分布", fontweight="bold")
    ax3.legend()

    # 4. 文件级置信度柱状
    ax4 = plt.subplot(2, 3, 4)
    df_sorted = preds.file_df.sort_values(cfg.data.file_basename_col)
    x = np.arange(len(df_sorted))
    color_map: Dict[str, str] = {"B": "#FF6B6B", "IR": "#4ECDC4", "N": "#45B7D1", "OR": "#96CEB4"}
    colors_bar = [color_map.get(lbl, "#9E9E9E") for lbl in df_sorted["pred_label"]]
    bars = ax4.bar(x, df_sorted["prob_label"], color=colors_bar)
    ax4.axhline(cfg.inference.unknown_threshold, color="#27AE60", linestyle=":", linewidth=1)
    ax4.set_xticks(x)
    ax4.set_xticklabels(df_sorted[cfg.data.file_basename_col], rotation=45)
    ax4.set_xlabel("文件ID")
    ax4.set_ylabel("最大概率")
    ax4.set_title("各文件预测置信度", fontweight="bold")

    # 5. 特征重要性
    ax5 = plt.subplot(2, 3, 5)
    importance_df = result.source_importance
    if importance_df is not None and not importance_df.empty:
        cols = [c for c in importance_df.columns if c.lower().startswith("importance") or c.lower() == "importance"]
        if cols:
            col = cols[0]
            tmp = importance_df[["feature", col]].copy()
            tmp.columns = ["feature", "importance"]
            tmp = tmp.groupby("feature", as_index=False).mean()
            tmp = tmp.sort_values("importance", ascending=False).head(10)
            ax5.barh(tmp["feature"], tmp["importance"], color="#F1948A")
            ax5.invert_yaxis()
            ax5.set_xlabel("重要度")
            ax5.set_title("Top 10 特征重要性", fontweight="bold")
        else:
            ax5.set_visible(False)
    else:
        ax5.set_visible(False)

    # 6. 预测结果表
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    lines = ["目标域预测结果", "=" * 40]
    for _, row in df_sorted.iterrows():
        label_name = label_cn.get(row["pred_label"], row["pred_label"])
        lines.append(f"{row[cfg.data.file_basename_col]} : {label_name} ({row['pred_label']}) 置信度 {row['prob_label']:.3f}")
    ax6.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#F2F4F4", alpha=0.9))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return FigureRecord(fig, "figures/transfer_dashboard.png", "迁移诊断仪表盘")


def _plot_confidence_hist(preds: PredictionBundle, cfg: Q3Config) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(preds.file_df["prob_label"], bins=20, color="#1f77b4", alpha=0.8)
    ax.axvline(cfg.inference.unknown_threshold, color="#d62728", linestyle="--", label="不确定阈值")
    ax.set_xlabel("文件级最大概率")
    ax.set_ylabel("文件数量")
    ax.set_title("目标域文件置信度分布")
    ax.legend()
    fig.tight_layout()
    return FigureRecord(fig, "figures/confidence_hist.png", "置信度分布图")


def _plot_file_bars(preds: PredictionBundle, cfg: Q3Config) -> FigureRecord:
    df = preds.file_df.sort_values(cfg.data.file_basename_col)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    ax.bar(x, df["prob_label"], color="#2ca02c")
    ax.axhline(cfg.inference.unknown_threshold, color="#d62728", linestyle="--", linewidth=1)
    ax.set_xlabel("文件")
    ax.set_ylabel("最大概率")
    ax.set_title("文件级预测结果")
    ax.set_xticks(x)
    ax.set_xticklabels(df[cfg.data.file_basename_col], rotation=45, ha="right")
    fig.tight_layout()
    return FigureRecord(fig, "figures/file_predictions.png", "文件级预测")


def _plot_mechanism(reports: ReportBundle, cfg: Q3Config) -> Optional[FigureRecord]:
    df = reports.mechanism_df
    if df.empty:
        return None
    prefixes = [col for col in df.columns if col.endswith("_energy")]
    if not prefixes:
        return None

    fig, ax = plt.subplots(figsize=(13, 6))
    base = df[cfg.data.file_basename_col].astype(str).tolist()
    x = np.arange(len(base))
    width = 0.75 / len(prefixes)
    palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F5B041", "#9B59B6"]

    for idx, prefix in enumerate(prefixes):
        values = df[prefix].fillna(0.0).to_numpy()
        if np.allclose(values, 0):
            values_norm = values
        else:
            v_min, v_max = values.min(), values.max()
            scale = v_max - v_min
            values_norm = (values - v_min) / scale if scale > 1e-12 else np.zeros_like(values)
        label = prefix.replace("_energy", "")
        ax.bar(x + idx * width, values_norm, width=width, color=palette[idx % len(palette)], alpha=0.9,
               label=label)

    ax.set_xlabel("文件")
    ax.set_ylabel("归一化能量")
    ax.set_title("机理特征能量分布", fontweight="bold")
    ax.set_xticks(x + width * (len(prefixes) - 1) / 2)
    ax.set_xticklabels(base, rotation=40)
    ax.legend(frameon=False, ncol=min(len(prefixes), 4))
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    return FigureRecord(fig, "figures/mechanism_features.png", "机理特征图")


def _plot_domain_gap(reports: ReportBundle) -> FigureRecord:
    df = reports.domain_gap_df
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "无域差异指标", ha="center", va="center")
        ax.axis("off")
        return FigureRecord(fig, "figures/domain_gap.png", "域差异指标")
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(df))
    ax.barh(y - 0.15, df["before"], height=0.3, label="适配前")
    ax.barh(y + 0.15, df["after"], height=0.3, label="适配后")
    ax.set_yticks(y)
    ax.set_yticklabels(df["metric"])
    ax.set_xlabel("指标值")
    ax.set_title("域差异指标对比")
    ax.legend()
    fig.tight_layout()
    return FigureRecord(fig, "figures/domain_gap.png", "域差异指标")
