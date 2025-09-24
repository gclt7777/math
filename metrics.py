"""计算各类评估指标、曲线与表格数据。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize

from .config import FeatureImportanceCfg, VizConfig
from .data_loader import DataBundle


@dataclass
class ROCData:
    per_class: Dict[str, Dict[str, np.ndarray]]  # label -> {fpr, tpr}
    micro: Dict[str, np.ndarray]
    macro_auc: float
    per_class_auc: Dict[str, float]


@dataclass
class PRData:
    per_class: Dict[str, Dict[str, np.ndarray]]  # label -> {precision, recall}
    micro: Dict[str, np.ndarray]
    per_class_ap: Dict[str, float]
    macro_ap: float


@dataclass
class ReliabilityData:
    bins: pd.DataFrame
    brier_score: float


@dataclass
class MetricsBundle:
    """汇总所有可视化所需的指标与表格数据。"""

    class_report: pd.DataFrame
    confusion_counts: pd.DataFrame
    confusion_normalized: pd.DataFrame
    roc_data: Optional[ROCData]
    pr_data: Optional[PRData]
    roc_table: Optional[pd.DataFrame]
    pr_table: Optional[pd.DataFrame]
    cv_summary: Optional[pd.DataFrame]
    model_importance: Optional[pd.Series]
    permutation_importance: Optional[pd.Series]
    reliability: Optional[ReliabilityData]
    file_class_report: Optional[pd.DataFrame] = None
    file_confusion_counts: Optional[pd.DataFrame] = None
    file_confusion_normalized: Optional[pd.DataFrame] = None
    file_roc_data: Optional[ROCData] = None
    file_pr_data: Optional[PRData] = None
    file_roc_table: Optional[pd.DataFrame] = None
    file_pr_table: Optional[pd.DataFrame] = None
    file_reliability: Optional[ReliabilityData] = None


def compute_metrics(bundle: DataBundle, cfg: VizConfig) -> MetricsBundle:
    """主入口：根据模型输出生成指标集合。"""

    class_report = _build_classification_report(
        bundle.y_test,
        bundle.y_pred,
        bundle.class_order,
        bundle.class_display,
    )
    cm_counts, cm_norm = _build_confusion_matrices(
        bundle.y_test,
        bundle.y_pred,
        bundle.class_order,
        bundle.class_display,
    )
    roc_data, pr_data, roc_table, pr_table = _build_roc_pr(
        bundle.y_test,
        bundle.proba,
        bundle.class_order,
        bundle.class_display,
    )
    cv_summary = _summarize_cv(bundle)
    model_imp, perm_imp = _compute_feature_importance(bundle, cfg.feature_importance)
    reliability = _compute_reliability(
        bundle.y_test,
        bundle.y_pred,
        bundle.proba,
        bundle.class_order,
    ) if cfg.plots.plot_reliability else None

    file_class_report = None
    file_cm_counts = None
    file_cm_norm = None
    file_roc_data = None
    file_pr_data = None
    file_roc_table = None
    file_pr_table = None
    file_reliability = None
    if bundle.file_y_true is not None and bundle.file_y_pred is not None:
        file_class_report = _build_classification_report(
            bundle.file_y_true,
            bundle.file_y_pred,
            bundle.class_order,
            bundle.class_display,
        )
        file_cm_counts, file_cm_norm = _build_confusion_matrices(
            bundle.file_y_true,
            bundle.file_y_pred,
            bundle.class_order,
            bundle.class_display,
        )
        file_roc_data, file_pr_data, file_roc_table, file_pr_table = _build_roc_pr(
            bundle.file_y_true,
            bundle.file_proba,
            bundle.class_order,
            bundle.class_display,
        )
        if cfg.plots.plot_reliability:
            file_reliability = _compute_reliability(
                bundle.file_y_true,
                bundle.file_y_pred,
                bundle.file_proba,
                bundle.class_order,
            )

    return MetricsBundle(
        class_report=class_report,
        confusion_counts=cm_counts,
        confusion_normalized=cm_norm,
        roc_data=roc_data,
        pr_data=pr_data,
        roc_table=roc_table,
        pr_table=pr_table,
        cv_summary=cv_summary,
        model_importance=model_imp,
        permutation_importance=perm_imp,
        reliability=reliability,
        file_class_report=file_class_report,
        file_confusion_counts=file_cm_counts,
        file_confusion_normalized=file_cm_norm,
        file_roc_data=file_roc_data,
        file_pr_data=file_pr_data,
        file_roc_table=file_roc_table,
        file_pr_table=file_pr_table,
        file_reliability=file_reliability,
    )


def _build_classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    class_order: List[str],
    class_display: Dict[str, str],
) -> pd.DataFrame:
    report_dict = metrics.classification_report(
        y_true,
        y_pred,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in class_order:
        stats = report_dict.get(label, {})
        rows.append(
            {
                "类别": class_display.get(label, label),
                "精确率": float(stats.get("precision", 0.0)),
                "召回率": float(stats.get("recall", 0.0)),
                "F1分数": float(stats.get("f1-score", 0.0)),
                "支持度": int(stats.get("support", 0)),
            }
        )

    # macro / micro / weighted
    # 宏平均
    macro_stats = report_dict.get("macro avg", {})
    rows.append(
        {
            "类别": "宏平均",
            "精确率": float(macro_stats.get("precision", 0.0)),
            "召回率": float(macro_stats.get("recall", 0.0)),
            "F1分数": float(macro_stats.get("f1-score", 0.0)),
            "支持度": int(macro_stats.get("support", 0)),
        }
    )
    # 微平均 - 使用 accuracy / micro 计算
    micro_precision = metrics.precision_score(y_true, y_pred, labels=class_order, average="micro", zero_division=0)
    micro_recall = metrics.recall_score(y_true, y_pred, labels=class_order, average="micro", zero_division=0)
    micro_f1 = metrics.f1_score(y_true, y_pred, labels=class_order, average="micro", zero_division=0)
    rows.append(
        {
            "类别": "微平均",
            "精确率": float(micro_precision),
            "召回率": float(micro_recall),
            "F1分数": float(micro_f1),
            "支持度": int(len(y_true)),
        }
    )
    # 加权平均
    weighted_stats = report_dict.get("weighted avg", {})
    rows.append(
        {
            "类别": "加权平均",
            "精确率": float(weighted_stats.get("precision", 0.0)),
            "召回率": float(weighted_stats.get("recall", 0.0)),
            "F1分数": float(weighted_stats.get("f1-score", 0.0)),
            "支持度": int(weighted_stats.get("support", 0)),
        }
    )

    return pd.DataFrame(rows)


def _build_confusion_matrices(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    class_order: List[str],
    class_display: Dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = class_order
    cnt = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    norm = metrics.confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    display_labels = [class_display.get(lbl, lbl) for lbl in labels]

    counts_df = pd.DataFrame(cnt, columns=display_labels, index=display_labels)
    counts_df.insert(0, "真实\\预测", display_labels)

    norm_df = pd.DataFrame(norm, columns=display_labels, index=display_labels)
    norm_df.insert(0, "真实\\预测", display_labels)

    return counts_df, norm_df


def _build_roc_pr(
    y_true: pd.Series | np.ndarray,
    proba: Optional[np.ndarray],
    class_order: List[str],
    class_display: Dict[str, str],
) -> tuple[Optional[ROCData], Optional[PRData], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if proba is None:
        return None, None, None, None

    y_true = np.asarray(y_true)
    prob_df = pd.DataFrame(proba, columns=class_order)
    y_prob = prob_df.to_numpy()
    labels = class_order
    display = class_display

    y_bin = label_binarize(y_true, classes=labels)
    roc_per_class: Dict[str, Dict[str, np.ndarray]] = {}
    auc_per_class: Dict[str, float] = {}
    pr_per_class: Dict[str, Dict[str, np.ndarray]] = {}
    ap_per_class: Dict[str, float] = {}

    for idx, label in enumerate(labels):
        fpr, tpr, _ = metrics.roc_curve(y_bin[:, idx], y_prob[:, idx])
        roc_per_class[label] = {"fpr": fpr, "tpr": tpr}
        auc_per_class[label] = metrics.auc(fpr, tpr)

        precision, recall, _ = metrics.precision_recall_curve(y_bin[:, idx], y_prob[:, idx])
        pr_per_class[label] = {"precision": precision, "recall": recall}
        ap_per_class[label] = metrics.average_precision_score(y_bin[:, idx], y_prob[:, idx])

    fpr_micro, tpr_micro, _ = metrics.roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_micro = {"fpr": fpr_micro, "tpr": tpr_micro}

    precision_micro, recall_micro, _ = metrics.precision_recall_curve(y_bin.ravel(), y_prob.ravel())
    pr_micro = {"precision": precision_micro, "recall": recall_micro}

    sorted_labels = sorted(labels)
    prob_sorted = prob_df.reindex(columns=sorted_labels).to_numpy()
    macro_auc = metrics.roc_auc_score(y_true, prob_sorted, multi_class="ovo", labels=sorted_labels)
    macro_ap = float(np.mean(list(ap_per_class.values())))

    roc_table_rows = [
        {"类别": display.get(lbl, lbl), "AUC": _format_float(auc_per_class[lbl], 3)}
        for lbl in labels
    ]
    roc_table_rows.append({"类别": "micro", "AUC": _format_float(metrics.auc(fpr_micro, tpr_micro), 3)})
    roc_table_rows.append({"类别": "macro", "AUC": _format_float(macro_auc, 3)})
    roc_table = pd.DataFrame(roc_table_rows)

    pr_table_rows = [
        {"类别": display.get(lbl, lbl), "AP": _format_float(ap_per_class[lbl], 3)}
        for lbl in labels
    ]
    pr_table_rows.append({"类别": "micro", "AP": _format_float(metrics.average_precision_score(y_bin.ravel(), y_prob.ravel()), 3)})
    pr_table_rows.append({"类别": "macro", "AP": _format_float(macro_ap, 3)})
    pr_table = pd.DataFrame(pr_table_rows)

    roc_data = ROCData(per_class=roc_per_class, micro=roc_micro, macro_auc=macro_auc, per_class_auc=auc_per_class)
    pr_data = PRData(per_class=pr_per_class, micro=pr_micro, per_class_ap=ap_per_class, macro_ap=macro_ap)

    return roc_data, pr_data, roc_table, pr_table


def _summarize_cv(bundle: DataBundle) -> Optional[pd.DataFrame]:
    if bundle.cv_results is None or bundle.cv_results.empty:
        return None

    df = bundle.cv_results.copy()
    model_col = None
    for col in ["model", "model_name", "candidate"]:
        if col in df.columns:
            model_col = col
            break
    if model_col is None:
        return None

    metrics_cols = {
        "mean_test_f1_macro": ("F1宏平均",),
        "mean_test_balanced_accuracy": ("平衡准确率",),
        "mean_test_accuracy": ("准确率",),
    }

    agg_dict = {col: ["mean", "std"] for col in metrics_cols.keys() if col in df.columns}
    if not agg_dict:
        return None

    summary = df.groupby(model_col).agg(agg_dict)
    summary.columns = [
        f"{metrics_cols[col][0]}({'均值' if stat == 'mean' else '标准差'})"
        for col, stat in summary.columns
    ]
    summary = summary.reset_index().rename(columns={model_col: "模型"})
    summary = summary.fillna("-")
    if "F1宏平均(均值)" in summary.columns:
        summary = summary.sort_values("F1宏平均(均值)", ascending=False)
    return summary


def _compute_feature_importance(bundle: DataBundle, cfg: FeatureImportanceCfg) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    model_imp = bundle.model_importance
    perm_imp = None
    if cfg.use_permutation:
        try:
            result = permutation_importance(
                bundle.model,
                bundle.X_test,
                bundle.y_test,
                n_repeats=cfg.permutation_repeats,
                random_state=0,
                n_jobs=-1,
            )
            perm_imp = pd.Series(result.importances_mean, index=bundle.feature_names, name="置换重要度").sort_values(ascending=False)
        except Exception:
            perm_imp = None
    if model_imp is not None:
        model_imp = model_imp.head(cfg.topk)
    if perm_imp is not None:
        perm_imp = perm_imp.head(cfg.topk)
    return model_imp, perm_imp


def _compute_reliability(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    proba: Optional[np.ndarray],
    class_order: List[str],
) -> Optional[ReliabilityData]:
    if proba is None:
        return None

    prob = np.max(proba, axis=1)
    true = (np.asarray(y_true) == y_pred)

    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(prob, bins) - 1
    records = []
    for i in range(len(bins) - 1):
        mask = bin_ids == i
        count = mask.sum()
        if count == 0:
            records.append({
                "概率区间": f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
                "样本数": 0,
                "平均预测概率": "-",
                "实际命中率": "-",
            })
            continue
        avg_prob = prob[mask].mean()
        hit_rate = true[mask].mean()
        records.append({
            "概率区间": f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
            "样本数": int(count),
            "平均预测概率": round(avg_prob, 3),
            "实际命中率": round(hit_rate, 3),
        })

    prob_df = pd.DataFrame(proba, columns=class_order)
    prob_df = prob_df.reindex(columns=class_order).to_numpy()
    brier = metrics.brier_score_loss(label_binarize(y_true, classes=class_order).ravel(), prob_df.ravel())
    return ReliabilityData(bins=pd.DataFrame(records), brier_score=float(brier))


def _format_float(value: float, precision: int) -> str:
    if value is None or np.isnan(value):
        return "-"
    return f"{value:.{precision}f}"
