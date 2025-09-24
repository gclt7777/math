# q2_train/viz.py
# -*- coding: utf-8 -*-
"""
可视化工具（纯 sklearn + matplotlib 实现），避免 scikit-plot 与 scipy 兼容性问题。
输出：
- 混淆矩阵：q2_confusion_matrix.png
- 多分类 ROC(OvR)：q2_roc_ovr.png
- 多分类 PR(OvR)：q2_pr_ovr.png
- 特征重要度柱状图：q2_feature_importance.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Any
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib.font_manager as fm
import warnings


def _setup_chinese_font() -> None:
    """尽量启用中文字体，避免图表文字乱码。"""

    preferred = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "WenQuanYi Micro Hei",
        "Source Han Sans CN",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((font for font in preferred if font in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"[viz] 使用中文字体: {chosen}")
    else:
        warnings.warn(
            "未找到可用的中文字体，图表可能出现乱码。",
            RuntimeWarning,
        )


_setup_chinese_font()


def _to_numpy(arr: Any) -> np.ndarray:
    """兼容 pandas 数据结构，强制转换为 numpy 数组。"""

    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy()
    return np.asarray(arr)

def plot_confusion_matrix(
    y_true: Sequence, y_pred: Sequence, class_names: Sequence[str],
    out_path: str, dpi: int = 160, normalize: Optional[str] = "true"
) -> None:
    """
    绘制混淆矩阵。normalize ∈ {'true','pred','all', None}
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names,
                          normalize=(normalize if normalize in {"true","pred","all"} else None))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title("混淆矩阵")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thresh = np.nanmax(cm) / 2 if np.isfinite(np.nanmax(cm)) else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if (normalize and normalize is not None) else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_roc_ovr(
    y_true: Sequence, y_proba: np.ndarray, class_names: Sequence[str],
    out_path: str, dpi: int = 160
) -> None:
    """
    多分类 OvR ROC 曲线。y_proba 形状为 (n_samples, n_classes)。
    """
    y_true = _to_numpy(y_true)
    y_proba = _to_numpy(y_proba)
    y_true_bin = label_binarize(y_true, classes=class_names)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for i, c in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr,tpr):.3f})")
    # micro 平均
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    ax.plot(fpr, tpr, linestyle="--", label=f"micro (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], ":", label="chance")
    ax.set_xlabel("假阳率 (FPR)")
    ax.set_ylabel("真阳率 (TPR)")
    ax.set_title("ROC 曲线（一对其余）")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_pr_ovr(
    y_true: Sequence, y_proba: np.ndarray, class_names: Sequence[str],
    out_path: str, dpi: int = 160
) -> None:
    """
    多分类 OvR PR 曲线。y_proba 形状为 (n_samples, n_classes)。
    """
    y_true = _to_numpy(y_true)
    y_proba = _to_numpy(y_proba)
    y_true_bin = label_binarize(y_true, classes=class_names)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for i, c in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        ax.plot(recall, precision, label=f"{c} (AP={ap:.3f})")
    ax.set_xlabel("召回率")
    ax.set_ylabel("精确率")
    ax.set_title("精确率-召回率（一对其余）")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_feature_importance(
    model, feature_names: Sequence[str], out_path: str, dpi: int = 160, top_k: int = 30
) -> None:
    """
    绘制特征重要度柱状图：
    - 优先使用 tree/boost 的 feature_importances_
    - 其次使用线性模型 coef_ 的 |coef|（多类取均值）
    - 若都没有，则输出占位图，并在控制台提示可使用 permutation importance
    """
    importances = None
    title_note = ""
    # 树/Boost
    if hasattr(model, "feature_importances_"):
        try:
            importances = np.asarray(model.feature_importances_).astype(float)
            title_note = "feature_importances_"
        except Exception:
            importances = None
    # 线性
    if importances is None and hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            importances = np.abs(coef).astype(float)
        else:
            importances = np.mean(np.abs(coef), axis=0).astype(float)
        title_note = "|coef| (avg over classes)"
    if importances is None or importances.size != len(feature_names):
        # 占位图：提示使用 permutation importance
        fig, ax = plt.subplots(figsize=(6.6, 1.8))
        ax.axis("off")
        ax.text(0.02, 0.6, "此模型不提供内建重要度。\n建议使用 permutation importance 计算后再绘图。",
                ha="left", va="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print("[viz] 模型无内建重要度：已输出提示图。")
        return

    # 选 top-k 可读性更好
    idx = np.argsort(importances)[::-1]
    idx = idx[:min(top_k, len(idx))]
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    ax.bar(range(len(idx)), importances[idx])
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("重要度")
    ax.set_title(f"特征重要度 ({title_note})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
