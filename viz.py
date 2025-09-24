"""Q4 中文可视化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .align import AlignmentResult
from .metrics import ReliabilityResult

COLOR_MAP = {
    "B": "#FF6B6B",
    "IR": "#4ECDC4",
    "OR": "#45B7D1",
    "N": "#96CEB4",
    "UNK": "#9E9E9E",
}


@dataclass
class FigureRecord:
    figure: plt.Figure
    filename: str
    description: str


def _setup_chinese_font() -> None:
    preferred = ["SimHei", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


def confidence_histogram(pred_df: pd.DataFrame, threshold: float) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pred_df["confidence"], bins=15, color="#5DADE2", edgecolor="#154360", alpha=0.85)
    ax.axvline(threshold, color="#E74C3C", linestyle="--", label=f"拒识阈值 {threshold:.2f}")
    ax.set_xlabel("置信度")
    ax.set_ylabel("文件数量")
    ax.set_title("目标域置信度分布（含阈值线）")
    ax.legend()
    fig.tight_layout()
    return FigureRecord(fig, "q4_confidence_hist.png", "置信度分布")


def reliability_curve(reliability: ReliabilityResult) -> Optional[FigureRecord]:
    if reliability.bins.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(reliability.bins["confidence"], reliability.bins["accuracy"], marker="o", label="经验准确率")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D", label="理想校准")
    ax.set_xlabel("预测置信度")
    ax.set_ylabel("真实准确率")
    ax.set_title("校准曲线（可靠性图）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    note = (
        f"Brier={reliability.summary['brier']:.4f} | "
        f"ECE={reliability.summary['ece']:.4f} | bins={reliability.summary['n_bins']}"
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=10, ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="#ECF0F1", alpha=0.8))
    fig.tight_layout()
    return FigureRecord(fig, "q4_reliability_curve.png", "校准曲线")


def _embedding_plot(alignment: AlignmentResult, method: str, random_state: int = 42) -> Optional[FigureRecord]:
    if alignment.target_aligned.empty:
        return None
    source = alignment.source_aligned
    target_before = alignment.target_raw
    target_after = alignment.target_aligned

    def reduce(X: np.ndarray) -> np.ndarray:
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=random_state, init="random")
        else:
            reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)

    try:
        if source is not None and len(source) > 1200:
            src_sample = source.sample(n=1200, random_state=random_state)
        else:
            src_sample = source
        tgt_before = target_before
        tgt_after = target_after

        combined_before = pd.concat([src_sample, tgt_before]) if src_sample is not None else tgt_before
        emb_before = reduce(combined_before.values)
        combined_after = pd.concat([src_sample, tgt_after]) if src_sample is not None else tgt_after
        emb_after = reduce(combined_after.values)
    except Exception as exc:  # pragma: no cover
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"降维失败: {exc}", ha="center", va="center")
        ax.axis("off")
        name = "tsne" if method == "tsne" else "pca"
        return FigureRecord(fig, f"q4_{name}_before_after.png", "降维失败")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb, title in zip(
        axes,
        (emb_before, emb_after),
        ("对齐前", "对齐后"),
    ):
        limit_src = len(src_sample) if src_sample is not None else 0
        if src_sample is not None:
            ax.scatter(emb[:limit_src, 0], emb[:limit_src, 1], s=15, c="#3498DB", alpha=0.6, label="源域")
            ax.scatter(emb[limit_src:, 0], emb[limit_src:, 1], s=40, c="#E74C3C", marker="s", alpha=0.85, label="目标域")
        else:
            ax.scatter(emb[:, 0], emb[:, 1], s=40, c="#E74C3C", alpha=0.85, label="目标域")
        ax.set_title(f"{title}")
        ax.set_xlabel("维度1")
        ax.set_ylabel("维度2")
        ax.legend()
        ax.grid(True, alpha=0.3)
    name = "tsne" if method == "tsne" else "pca"
    title = "t-SNE 特征空间：对齐前 vs 对齐后" if method == "tsne" else "PCA 特征空间：对齐前 vs 对齐后"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return FigureRecord(fig, f"q4_{name}_before_after.png", title)


def open_set_distribution(scores: np.ndarray, rejects: np.ndarray, method: str, threshold: float) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[~rejects], bins=15, alpha=0.8, color="#82E0AA", label="接受样本")
    if rejects.any():
        ax.hist(scores[rejects], bins=15, alpha=0.8, color="#F1948A", label="拒识样本")
    ax.axvline(threshold, color="#E74C3C", linestyle="--", label=f"阈值 {threshold:.2f}")
    xlabel = "最大概率" if method == "max_proba" else "能量分数"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("数量")
    ax.set_title("开放集拒识评分分布")
    ax.legend()
    fig.tight_layout()
    return FigureRecord(fig, "q4_open_set_score_dist.png", "开放集分布")


def class_balance_chart(source_counts: Dict[str, int], pseudo_counts_history: List[Dict[str, int]], classes: List[str]) -> Optional[FigureRecord]:
    if not pseudo_counts_history:
        return None
    final_counts = pseudo_counts_history[-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(classes))
    width = 0.35
    source_vals = [source_counts.get(cls, 0) for cls in classes]
    pseudo_vals = [final_counts.get(cls, 0) for cls in classes]
    ax.bar(x - width / 2, source_vals, width, color="#AED6F1", label="源域")
    ax.bar(x + width / 2, pseudo_vals, width, color="#F5B041", label="伪标注")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("样本数量")
    ax.set_title("伪标注后类别数量对比（最终轮）")
    ax.legend()
    fig.tight_layout()
    return FigureRecord(fig, "q4_class_balance_after_st.png", "类别数量对比")


def pr_curve(curves: Dict[str, pd.DataFrame], colors: Dict[str, str]) -> Optional[FigureRecord]:
    pr_items = {k: v for k, v in curves.items() if k.startswith("pr_")}
    if not pr_items:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, df in pr_items.items():
        cls = key.replace("pr_", "")
        ax.plot(df["recall"], df["precision"], label=cls, color=colors.get(cls, None))
    ax.set_xlabel("召回率")
    ax.set_ylabel("精确率")
    ax.set_title("目标域 PR 曲线（代理评估）")
    ax.legend()
    fig.tight_layout()
    return FigureRecord(fig, "q4_pr_ovr_target.png", "PR 曲线")


def roc_curve_plot(curves: Dict[str, pd.DataFrame], colors: Dict[str, str]) -> Optional[FigureRecord]:
    roc_items = {k: v for k, v in curves.items() if k.startswith("roc_")}
    if not roc_items:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, df in roc_items.items():
        cls = key.replace("roc_", "")
        ax.plot(df["fpr"], df["tpr"], label=cls, color=colors.get(cls, None))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D")
    ax.set_xlabel("假阳率")
    ax.set_ylabel("真阳率")
    ax.set_title("目标域 ROC 曲线（代理评估）")
    ax.legend()
    fig.tight_layout()
    return FigureRecord(fig, "q4_roc_ovr_target.png", "ROC 曲线")


def interpretability_panels(
    base_model,
    adapted_model,
    source_features: pd.DataFrame,
    source_labels: pd.Series,
    target_features: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
) -> List[FigureRecord]:
    figs: List[FigureRecord] = []

    feature_names = source_features.columns
    labels_unique = sorted(source_labels.unique())

    shared_cols = [col for col in source_features.columns if col in target_features.columns]
    combined_shared: Optional[pd.DataFrame]
    src_labels_sample = None
    src_len_shared = 0

    if len(shared_cols) >= 2:
        src_sample = source_features[shared_cols]
        if len(src_sample) > 500:
            src_sample = src_sample.sample(500, random_state=42)
        combined_shared = pd.concat([src_sample, target_features[shared_cols]])
        src_labels_sample = source_labels.loc[src_sample.index]
        src_len_shared = len(src_sample)
    else:
        combined_shared = None

    # --------- Pre Interpretability ---------
    fig_pre = plt.figure(figsize=(20, 12))
    fig_pre.suptitle("第一部分：事前可解释性分析", fontsize=18, fontweight="bold")

    importances = np.zeros(len(feature_names))
    if hasattr(base_model, "feature_importances_"):
        importances = np.array(base_model.feature_importances_)
    top_idx = np.argsort(importances)[-6:] if len(feature_names) >= 6 else np.arange(len(feature_names))
    top_features = feature_names[top_idx]
    top_importance = importances[top_idx]

    ax1 = fig_pre.add_subplot(2, 3, 1)
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#A569BD"]
    ax1.barh(range(len(top_features)), top_importance, color=colors[: len(top_features)], alpha=0.85)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels([f.replace("DE_", "") for f in top_features])
    ax1.set_xlabel("重要性分数")
    ax1.set_title("特征重要性排序", fontweight="bold")

    ax2 = fig_pre.add_subplot(2, 3, 2)
    if "DE_kurtosis" in source_features.columns:
        for lbl in labels_unique:
            mask = source_labels == lbl
            ax2.hist(source_features.loc[mask, "DE_kurtosis"], bins=20, alpha=0.6, density=True,
                     label=f"{LABEL_CN.get(lbl, lbl)}", color=COLOR_MAP.get(lbl, "gray"))
        ax2.axvline(3, color="red", linestyle="--")
        ax2.set_title("峭度特征的故障类型分布", fontweight="bold")
        ax2.set_xlabel("峭度值")
        ax2.set_ylabel("密度")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "缺少 DE_kurtosis", transform=ax2.transAxes, ha="center", va="center")

    ax3 = fig_pre.add_subplot(2, 3, 3)
    heat_cols = [col for col in ["DE_rms", "DE_kurtosis", "DE_crest_factor", "DE_BPFO_amplitude", "DE_BPFI_amplitude", "DE_BSF_amplitude"] if col in source_features.columns]
    if len(heat_cols) >= 2:
        corr = source_features[heat_cols].corr()
        im = ax3.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax3.set_xticks(range(len(heat_cols)))
        ax3.set_xticklabels([c.replace("DE_", "") for c in heat_cols], rotation=40)
        ax3.set_yticks(range(len(heat_cols)))
        ax3.set_yticklabels([c.replace("DE_", "") for c in heat_cols])
        for i in range(len(heat_cols)):
            for j in range(len(heat_cols)):
                ax3.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
        fig_pre.colorbar(im, ax=ax3, shrink=0.8)
        ax3.set_title("关键特征相关性矩阵", fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "特征数量不足", transform=ax3.transAxes, ha="center", va="center")

    ax4 = fig_pre.add_subplot(2, 3, 4)
    freq_cols = [col for col in ["DE_BPFO_amplitude", "DE_BPFI_amplitude", "DE_BSF_amplitude"] if col in source_features.columns]
    if freq_cols:
        width = 0.8 / len(freq_cols)
        x = np.arange(len(labels_unique))
        for i, col in enumerate(freq_cols):
            means = [source_features.loc[source_labels == lbl, col].mean() for lbl in labels_unique]
            ax4.bar(x + i * width, means, width, label=col.replace("DE_", "").replace("_amplitude", ""))
        ax4.set_xticks(x + width * (len(freq_cols) - 1) / 2)
        ax4.set_xticklabels([LABEL_CN.get(lbl, lbl) for lbl in labels_unique], rotation=30)
        ax4.set_ylabel("平均能量")
        ax4.set_title("故障特征频率能量对比", fontweight="bold")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "缺少频率特征", transform=ax4.transAxes, ha="center", va="center")

    ax5 = fig_pre.add_subplot(2, 3, 5)
    if "DE_rms" in source_features.columns:
        for lbl in labels_unique:
            ax5.hist(source_features.loc[source_labels == lbl, "DE_rms"], bins=20, density=True, alpha=0.6,
                     label=LABEL_CN.get(lbl, lbl), color=COLOR_MAP.get(lbl, "gray"))
        ax5.set_title("RMS 特征分布", fontweight="bold")
        ax5.set_xlabel("RMS")
        ax5.set_ylabel("密度")
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "缺少 DE_rms", transform=ax5.transAxes, ha="center", va="center")

    ax6 = fig_pre.add_subplot(2, 3, 6)
    if "DE_crest_factor" in source_features.columns:
        means = [source_features.loc[source_labels == lbl, "DE_crest_factor"].mean() for lbl in labels_unique]
        stds = [source_features.loc[source_labels == lbl, "DE_crest_factor"].std() for lbl in labels_unique]
        x = np.arange(len(labels_unique))
        bars = ax6.bar(x, means, yerr=stds, color=[COLOR_MAP.get(lbl, "gray") for lbl in labels_unique], alpha=0.8, capsize=6)
        ax6.set_xticks(x)
        ax6.set_xticklabels([LABEL_CN.get(lbl, lbl) for lbl in labels_unique], rotation=30)
        ax6.set_ylabel("峰值因子")
        ax6.set_title("峰值因子均值对比", fontweight="bold")
        for bar, mean in zip(bars, means):
            ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.2f}", ha="center", va="bottom")
    else:
        ax6.text(0.5, 0.5, "缺少 DE_crest_factor", transform=ax6.transAxes, ha="center", va="center")

    fig_pre.tight_layout(rect=[0, 0.03, 1, 0.95])
    figs.append(FigureRecord(fig_pre, "figures/q4_pre_interpretability.png", "事前可解释性"))

    # --------- Transfer Process Interpretability ---------
    fig_proc = plt.figure(figsize=(20, 12))
    fig_proc.suptitle("第二部分：迁移过程可解释性分析", fontsize=18, fontweight="bold")

    ax_tsne = fig_proc.add_subplot(2, 3, 1)
    if combined_shared is not None and src_labels_sample is not None:
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined_shared)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_scaled) - 1))
        emb = tsne.fit_transform(combined_scaled)
        for lbl in labels_unique:
            mask = src_labels_sample == lbl
            ax_tsne.scatter(emb[:src_len_shared][mask, 0], emb[:src_len_shared][mask, 1], s=20,
                            color=COLOR_MAP.get(lbl, "gray"), alpha=0.6, label=f"源域-{LABEL_CN.get(lbl, lbl)}")
        ax_tsne.scatter(
            emb[src_len_shared:, 0],
            emb[src_len_shared:, 1],
            s=45,
            marker="s",
            color="#2C3E50",
            alpha=0.9,
            label="目标域",
        )
        ax_tsne.set_title("t-SNE 特征空间分布", fontweight="bold")
        ax_tsne.legend()
        ax_tsne.grid(True, alpha=0.3)
    else:
        ax_tsne.text(0.5, 0.5, "特征维度不足", transform=ax_tsne.transAxes, ha="center", va="center")

    ax_kurt = fig_proc.add_subplot(2, 3, 2)
    if "DE_kurtosis" in source_features.columns and "DE_kurtosis" in target_features.columns:
        ax_kurt.hist(source_features["DE_kurtosis"], bins=20, alpha=0.6, density=True, label="源域", color="#5DADE2")
        ax_kurt.hist(target_features["DE_kurtosis"], bins=15, alpha=0.6, density=True, label="目标域", color="#F5B041")
        ax_kurt.set_title("峭度分布对比", fontweight="bold")
        ax_kurt.set_xlabel("峭度值")
        ax_kurt.set_ylabel("密度")
        ax_kurt.legend()
    else:
        ax_kurt.text(0.5, 0.5, "缺少峭度特征", transform=ax_kurt.transAxes, ha="center", va="center")

    ax_align = fig_proc.add_subplot(2, 3, 3)
    example_feat = shared_cols[0] if shared_cols else None
    if example_feat is not None:
        src_vals = source_features[example_feat]
        tgt_before = alignment.target_raw[example_feat]
        tgt_after = alignment.target_aligned[example_feat]
        ax_align.hist(src_vals, bins=20, alpha=0.5, label="源域", color="#5DADE2", density=True)
        ax_align.hist(tgt_before, bins=15, alpha=0.5, label="目标域(适配前)", color="#E74C3C", density=True)
        ax_align.hist(tgt_after, bins=15, alpha=0.5, label="目标域(适配后)", color="#27AE60", density=True)
        ax_align.set_title("域适应效果示意", fontweight="bold")
        ax_align.legend()
    else:
        ax_align.text(0.5, 0.5, "缺少对齐特征", transform=ax_align.transAxes, ha="center", va="center")

    ax_imp = fig_proc.add_subplot(2, 3, 4)
    if hasattr(base_model, "feature_importances_") and hasattr(adapted_model, "feature_importances_"):
        base_imp = np.array(base_model.feature_importances_)
        adapt_imp = np.array(adapted_model.feature_importances_)
        top_idx2 = np.argsort(base_imp)[-6:]
        labels_imp = [feature_names[i].replace("DE_", "") for i in top_idx2]
        x = np.arange(len(labels_imp))
        width = 0.4
        ax_imp.bar(x - width / 2, base_imp[top_idx2], width, label="迁移前", color="#F1948A", alpha=0.85)
        ax_imp.bar(x + width / 2, adapt_imp[top_idx2], width, label="迁移后", color="#7FB3D5", alpha=0.85)
        ax_imp.set_xticks(x)
        ax_imp.set_xticklabels(labels_imp, rotation=30)
        ax_imp.set_ylabel("重要性")
        ax_imp.set_title("特征重要性变化", fontweight="bold")
        ax_imp.legend()
    else:
        ax_imp.text(0.5, 0.5, "模型缺少重要度信息", transform=ax_imp.transAxes, ha="center", va="center")

    ax_band = fig_proc.add_subplot(2, 3, 5)
    band_pairs = [("DE_low_band_ratio", "low_band_ratio", "低频"), ("DE_high_band_ratio", "high_band_ratio", "高频")]
    plotted = False
    for src_col, tgt_col, label in band_pairs:
        if src_col in source_features.columns and tgt_col in target_features.columns:
            ax_band.bar(label + "-源域", source_features[src_col].mean(), color="#5DADE2", alpha=0.8)
            ax_band.bar(label + "-目标域", target_features[tgt_col].mean(), color="#F5B041", alpha=0.8)
            plotted = True
    if plotted:
        ax_band.set_ylabel("能量比例")
        ax_band.set_title("频带能量分布对比", fontweight="bold")
    else:
        ax_band.text(0.5, 0.5, "缺少频带能量特征", transform=ax_band.transAxes, ha="center", va="center")

    ax_pca = fig_proc.add_subplot(2, 3, 6)
    if combined_shared is not None and src_labels_sample is not None:
        pca = PCA(n_components=2)
        emb = pca.fit_transform(combined_shared)
        for lbl in labels_unique:
            mask = src_labels_sample == lbl
            ax_pca.scatter(emb[:src_len_shared][mask, 0], emb[:src_len_shared][mask, 1], s=20,
                           color=COLOR_MAP.get(lbl, "gray"), alpha=0.6, label=f"源域-{LABEL_CN.get(lbl, lbl)}")
        ax_pca.scatter(
            emb[src_len_shared:, 0],
            emb[src_len_shared:, 1],
            s=45,
            marker="s",
            color="#2C3E50",
            alpha=0.9,
            label="目标域",
        )
        ax_pca.set_title("PCA 特征空间可视化", fontweight="bold")
        ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax_pca.legend()
        ax_pca.grid(True, alpha=0.3)
    else:
        ax_pca.text(0.5, 0.5, "特征维度不足", transform=ax_pca.transAxes, ha="center", va="center")

    fig_proc.tight_layout(rect=[0, 0.03, 1, 0.95])
    figs.append(FigureRecord(fig_proc, "figures/q4_transfer_process.png", "迁移过程可解释性"))

    # --------- Post Interpretability ---------
    fig_post = plt.figure(figsize=(20, 12))
    fig_post.suptitle("第三部分：事后可解释性分析", fontsize=18, fontweight="bold")

    ordered_pred = predictions.copy()
    if "file_id" in ordered_pred.columns:
        ordered_pred = ordered_pred.sort_values("file_id")

    ax_conf = fig_post.add_subplot(2, 3, 1)
    ax_conf.bar(range(len(ordered_pred)), ordered_pred["confidence"], color="#45B7D1", alpha=0.85)
    ax_conf.axhline(0.8, color="#E74C3C", linestyle="--", label="高置信度阈值")
    ax_conf.axhline(threshold, color="#F5B041", linestyle=":", label="拒识阈值")
    if "file_id" in ordered_pred.columns:
        ax_conf.set_xticks(range(len(ordered_pred)))
        ax_conf.set_xticklabels(ordered_pred["file_id"], rotation=35)
    ax_conf.set_ylabel("置信度")
    ax_conf.set_title("目标域预测置信度分析", fontweight="bold")
    ax_conf.legend()
    ax_conf.grid(True, alpha=0.3)

    ax_pie = fig_post.add_subplot(2, 3, 2)
    counts = ordered_pred["predicted_label"].value_counts()
    labels = [LABEL_CN.get(lbl, lbl) for lbl in counts.index]
    ax_pie.pie(counts.values, labels=labels, autopct="%1.1f%%",
               colors=[COLOR_MAP.get(lbl, "#7F8C8D") for lbl in counts.index])
    ax_pie.set_title("目标域预测结果分布", fontweight="bold")

    ax_feat = fig_post.add_subplot(2, 3, 3)
    if hasattr(adapted_model, "feature_importances_"):
        imp = np.array(adapted_model.feature_importances_)
        top_idx3 = np.argsort(imp)[-6:]
        ax_feat.barh([feature_names[i].replace("DE_", "") for i in top_idx3], imp[top_idx3], color="#FF6B6B", alpha=0.85)
        ax_feat.set_title("模型特征重要性", fontweight="bold")
        ax_feat.set_xlabel("重要性")
    else:
        ax_feat.text(0.5, 0.5, "模型不支持重要度", transform=ax_feat.transAxes, ha="center", va="center")

    ax_decision = fig_post.add_subplot(2, 3, 4)
    pca = PCA(n_components=2)
    emb = pca.fit_transform(target_features.values)
    color_labels = [COLOR_MAP.get(lbl, "#7F8C8D") for lbl in ordered_pred["predicted_label"]]
    ax_decision.scatter(emb[:, 0], emb[:, 1], c=color_labels, alpha=0.85, s=50)
    ax_decision.set_title("决策空间可视化 (PCA)", fontweight="bold")
    ax_decision.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax_decision.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax_decision.grid(True, alpha=0.3)

    ax_active = fig_post.add_subplot(2, 3, 5)
    low_conf = ordered_pred.nsmallest(6, "confidence")
    ax_active.barh(low_conf.get("file_id", range(len(low_conf))), low_conf["confidence"], color="#FF6B6B", alpha=0.85)
    ax_active.set_title("待人工复核样本 (Top6)", fontweight="bold")
    ax_active.set_xlabel("置信度")
    for y, val in enumerate(low_conf["confidence"]):
        ax_active.text(val + 0.01, y, f"{val:.3f}", va="center")

    ax_stats = fig_post.add_subplot(2, 3, 6)
    bins = [
        (ordered_pred["confidence"] >= 0.8).sum(),
        ((ordered_pred["confidence"] >= 0.7) & (ordered_pred["confidence"] < 0.8)).sum(),
        (ordered_pred["confidence"] < 0.7).sum(),
    ]
    labels_conf = ["高置信度(≥0.8)", "中等置信度(0.7-0.8)", "低置信度(<0.7)"]
    ax_stats.bar(labels_conf, bins, color=["#45B7D1", "#F5B041", "#FF6B6B"], alpha=0.85)
    ax_stats.set_title("预测置信度统计", fontweight="bold")
    ax_stats.set_ylabel("样本数量")
    for x, val in zip(labels_conf, bins):
        ax_stats.text(x, val + 0.2, str(val), ha="center")

    fig_post.tight_layout(rect=[0, 0.03, 1, 0.95])
    figs.append(FigureRecord(fig_post, "figures/q4_post_interpretability.png", "事后可解释性"))

    return figs


def prepare_figures(
    alignment: AlignmentResult,
    predictions: pd.DataFrame,
    reliability: ReliabilityResult,
    openset_scores: np.ndarray,
    openset_mask: np.ndarray,
    threshold: float,
    class_counts_source: Dict[str, int],
    class_history: List[Dict[str, int]],
    curves: Dict[str, pd.DataFrame],
    dpi: int = 160,
    openset_method: str = "max_proba",
    base_model=None,
    adapted_model=None,
    source_features: Optional[pd.DataFrame] = None,
    source_labels: Optional[pd.Series] = None,
    target_features: Optional[pd.DataFrame] = None,
) -> List[FigureRecord]:
    records: List[FigureRecord] = []

    records.append(confidence_histogram(predictions, threshold))
    reliability_fig = reliability_curve(reliability)
    if reliability_fig:
        records.append(reliability_fig)

    tsne_fig = _embedding_plot(alignment, "tsne")
    if tsne_fig:
        records.append(tsne_fig)
    pca_fig = _embedding_plot(alignment, "pca")
    if pca_fig:
        records.append(pca_fig)

    records.append(open_set_distribution(openset_scores, openset_mask, openset_method, threshold))

    all_classes = list(class_counts_source.keys())
    if class_history:
        for history in class_history:
            for cls in history.keys():
                if cls not in all_classes:
                    all_classes.append(cls)
    class_balance_fig = class_balance_chart(class_counts_source, class_history, all_classes)
    if class_balance_fig:
        records.append(class_balance_fig)

    pr_fig = pr_curve(curves, COLOR_MAP)
    if pr_fig:
        records.append(pr_fig)
    roc_fig = roc_curve_plot(curves, COLOR_MAP)
    if roc_fig:
        records.append(roc_fig)

    if base_model is not None and source_features is not None and source_labels is not None and target_features is not None:
        try:
            interp_figs = interpretability_panels(
                base_model,
                adapted_model if adapted_model is not None else base_model,
                source_features,
                source_labels,
                target_features,
                predictions,
                threshold,
            )
            records.extend(interp_figs)
        except Exception:
            pass

    for rec in records:
        rec.figure.set_dpi(dpi)
    return records
