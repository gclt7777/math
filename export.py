# -*- coding: utf-8 -*-
"""导出模块：将第二问训练产物（模型、评估报表、图表等）写入磁盘。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from .evaluate import EvaluationResult
from .models import SearchResult
from .viz import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_ovr,
    plot_roc_ovr,
)
from .utils import Config, save_joblib, save_json


def _get_primary_pipeline(model):
    """
    获取包含 scaler 的主管线（便于导出 scaler）。
    - 若是 CalibratedClassifierCV，则回溯到底层 estimator。
    - 否则返回原模型。
    """
    if isinstance(model, CalibratedClassifierCV):
        # 新版 sklearn: base_estimator_，旧版通过 calibrated_classifiers_ 取
        if hasattr(model, "base_estimator_"):
            return model.base_estimator_
        if getattr(model, "calibrated_classifiers_", None):
            return model.calibrated_classifiers_[0].estimator
    return model


def export_all(
    cfg: Config,
    search: SearchResult,
    model,
    splits,
    eval_result: EvaluationResult,
    logger,
) -> None:
    """
    统一导出模型、指标、曲线等产物。

    参数说明
    ----------
    cfg : Config
        配置对象（含输出目录、图像 DPI 等）。
    search : SearchResult
        网格搜索/模型选择的结果，包含 cv_results（DataFrame）。
    model : Any
        最佳（并已校准）的分类器。
    splits : Any
        数据拆分与特征信息（需含 use_scaler, feature_cols, uid_* 三个字段）。
    eval_result : EvaluationResult
        最终评估结果（需含 y_true, y_pred, proba, classes, report, feature_importance 等）。
    logger : logging.Logger
        日志记录器。
    """

    out_dir = Path(cfg.io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) 保存模型 ----------
    model_path = out_dir / "best_model.joblib"
    save_joblib(model_path, model)
    logger.info("已保存模型到 %s", model_path)

    # ---------- 2) 若使用原始特征，则尝试导出 scaler ----------
    if getattr(splits, "use_scaler", False):
        pipeline = _get_primary_pipeline(model)
        scaler = None
        if isinstance(pipeline, Pipeline) and "scaler" in pipeline.named_steps:
            scaler = pipeline.named_steps["scaler"]
        if scaler is not None:
            scaler_path = out_dir / "scaler.joblib"
            save_joblib(scaler_path, scaler)
            logger.info("已保存 scaler 到 %s", scaler_path)
        else:
            logger.info("未检测到 scaler 组件，跳过导出。")

    # ---------- 3) 导出列名顺序 ----------
    columns_path = out_dir / "columns.json"
    save_json(columns_path, {"feature_columns": list(splits.feature_cols)})
    logger.info("已保存特征列顺序到 %s", columns_path)

    # ---------- 4) 写入 CV 结果 ----------
    cv_path = out_dir / "cv_results.csv"
    # 防御：确保是 DataFrame
    if isinstance(search.cv_results, pd.DataFrame):
        search.cv_results.to_csv(cv_path, index=False)
        logger.info("已导出 CV 结果到 %s", cv_path)
    else:
        logger.warning("cv_results 不是 DataFrame，跳过写入。")

    # ---------- 5) 写入测试报告 ----------
    report_path = out_dir / "test_report.csv"
    report_df = eval_result.report.copy()
    if report_df.index.name != "label":
        report_df = report_df.reset_index().rename(columns={"index": "label"})
    report_df.to_csv(report_path, index=False)
    logger.info("已导出测试报告到 %s", report_path)

    if eval_result.file_report is not None:
        file_report_path = out_dir / "file_report.csv"
        file_report_df = eval_result.file_report.copy()
        if file_report_df.index.name != "label":
            file_report_df = file_report_df.reset_index().rename(columns={"index": "label"})
        file_report_df.to_csv(file_report_path, index=False)
        logger.info("已导出文件级测试报告到 %s", file_report_path)

    # ---------- 6) 写入特征重要度（表格） ----------
    fi_path = out_dir / "feature_importance.csv"
    # 若评估阶段已计算 permutation/模型内建重要度，这里直接落盘
    if isinstance(eval_result.feature_importance, pd.DataFrame):
        eval_result.feature_importance.to_csv(fi_path, index=False)
        logger.info("已导出特征重要度到 %s", fi_path)
    else:
        logger.info("评估对象未提供 tabular 形式的特征重要度，跳过 CSV 导出。")

    # ---------- 7) 保存拆分清单 ----------
    manifest: Dict[str, list] = {
        "train": list(getattr(splits, "uid_train", [])),
        "val": list(getattr(splits, "uid_val", [])),
        "test": list(getattr(splits, "uid_test", [])),
        "file_train": list(getattr(splits, "file_train", [])),
        "file_val": list(getattr(splits, "file_val", [])),
        "file_test": list(getattr(splits, "file_test", [])),
    }
    manifest_path = out_dir / "split_manifest.json"
    save_json(manifest_path, manifest)
    logger.info("已导出数据拆分清单到 %s", manifest_path)

    # ---------- 8) 导出窗口/文件级预测 ----------
    try:
        if eval_result.window_predictions is not None:
            window_path = out_dir / "window_predictions.csv"
            eval_result.window_predictions.to_csv(window_path, index=False)
            logger.info("已导出窗口级预测到 %s", window_path)
        else:
            logger.warning("未获取窗口级预测明细，跳过导出 window_predictions.csv。")

        if eval_result.file_predictions is not None:
            file_path_csv = out_dir / "file_predictions.csv"
            eval_result.file_predictions.to_csv(file_path_csv, index=False)
            logger.info("已导出文件级预测到 %s", file_path_csv)
        else:
            logger.warning("未聚合文件级预测，跳过导出 file_predictions.csv。")
    except Exception as exc:
        logger.exception("导出窗口/文件级预测失败：%s", exc)

    # ---------- 9) 图表输出 ----------
    # 混淆矩阵：新版 viz 需要 y_true / y_pred / classes
    try:
        plot_confusion_matrix(
            y_true=eval_result.y_true,
            y_pred=eval_result.y_pred,
            class_names=eval_result.classes,
            out_path=str(out_dir / "q2_confusion_matrix.png"),
            dpi=cfg.export.figure_dpi,
            normalize="true",  # 与说明书一致：按真实标签归一化
        )
        logger.info("已生成混淆矩阵图。")

        if eval_result.file_predictions is not None and eval_result.file_confusion is not None:
            file_conf_path = out_dir / "q2_confusion_matrix_filelevel.png"
            plot_confusion_matrix(
                y_true=eval_result.file_predictions["true_label"],
                y_pred=eval_result.file_predictions["pred_label"],
                class_names=eval_result.classes,
                out_path=str(file_conf_path),
                dpi=cfg.export.figure_dpi,
                normalize="true",
            )
            logger.info("已生成文件级混淆矩阵图。")
    except Exception as exc:
        logger.exception("绘制混淆矩阵失败：%s", exc)

    # ROC / PR：需要概率输出，若某模型不支持 proba 则跳过
    if getattr(eval_result, "proba", None) is not None:
        try:
            plot_roc_ovr(
                y_true=eval_result.y_true,
                y_proba=eval_result.proba,
                class_names=eval_result.classes,
                out_path=str(out_dir / "q2_roc_ovr.png"),
                dpi=cfg.export.figure_dpi,
            )
            logger.info("已生成 ROC(OvR) 图。")
        except Exception as exc:
            logger.exception("绘制 ROC 失败：%s", exc)

        try:
            plot_pr_ovr(
                y_true=eval_result.y_true,
                y_proba=eval_result.proba,
                class_names=eval_result.classes,
                out_path=str(out_dir / "q2_pr_ovr.png"),
                dpi=cfg.export.figure_dpi,
            )
            logger.info("已生成 PR(OvR) 图。")
        except Exception as exc:
            logger.exception("绘制 PR 失败：%s", exc)
    else:
        logger.warning("当前模型未提供概率输出（proba=None），跳过 ROC/PR 绘图。")

    # 特征重要度柱状图：新版 viz 需要 (model, feature_names)
    try:
        plot_feature_importance(
            model=model,
            feature_names=list(splits.feature_cols),
            out_path=str(out_dir / "q2_feature_importance.png"),
            dpi=cfg.export.figure_dpi,
            top_k=30,
        )
        logger.info("已生成特征重要度图。")
    except Exception as exc:
        logger.exception("绘制特征重要度失败：%s", exc)

    logger.info("导出完成：模型、CV、测试报告、图表与拆分清单已写入 %s。", out_dir)
