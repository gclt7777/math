"""Q1 可视化命令行入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .config import ensure_output_tree, load_viz_config, snapshot_config
from .export import save_plots, write_summary
from .plots import (
    PlotRecord,
    plot_correlation_matrix,
    plot_data_overview,
    plot_feature_distributions,
    plot_feature_distribution_summary,
    plot_feature_space,
    plot_envelope,
    plot_fft,
    plot_frequency_comparison,
    plot_signal_summary,
    plot_spectrogram,
    plot_time_waveform,
)
from .selectors import SegmentSpec, prepare_segments
from .signal_utils import reconstruct_segment


CHINESE_FONTS = ["SimHei", "Microsoft YaHei", "STHeiti", "Arial Unicode MS", "Sarasa Gothic SC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Q1 可视化命令")
    parser.add_argument("--config", default="config/q1_viz.yaml", help="配置文件路径")
    return parser.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("q1_viz")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def setup_chinese_fonts() -> None:
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = CHINESE_FONTS
    matplotlib.rcParams["figure.max_open_warning"] = 0


def main() -> None:
    args = parse_args()
    cfg = load_viz_config(args.config)
    ensure_output_tree(cfg)
    logger = setup_logger(cfg.io.log_file)
    setup_chinese_fonts()
    logger.info("加载配置成功: %s", Path(args.config).resolve())

    try:
        segments: List[SegmentSpec] = prepare_segments(cfg.selection, cfg.io.source_dirs, cfg.io.target_dirs)
    except FileNotFoundError as exc:
        logger.error("样本选择失败: %s", exc)
        return
    if not segments:
        logger.error("未找到任何可用样本，流程终止。")
        return
    logger.info("已选取 %d 个样本段。", len(segments))

    plot_records: List[PlotRecord] = []
    updated_segments: List[SegmentSpec] = []

    for spec in segments:
        logger.info("处理样本: 标签=%s 预期传感器=%s 文件=%s", spec.label, spec.sensor, spec.file_path)
        try:
            bundle = reconstruct_segment(spec, cfg)
        except Exception as exc:
            logger.warning("样本处理失败 (label=%s 文件=%s): %s", spec.label, spec.file_path, exc)
            continue

        updated_segments.append(bundle.spec)
        if bundle.used_default_rpm:
            logger.warning("样本 %s 未提供 RPM，使用默认值 %.1f", bundle.spec.basename, bundle.rpm_used)
        else:
            logger.info("样本 %s RPM=%.2f", bundle.spec.basename, bundle.rpm_used)

        if cfg.plots.time_waveform:
            plot_records.append(plot_time_waveform(bundle))
            plot_records.append(plot_signal_summary(bundle))
            plot_records.append(plot_frequency_comparison(bundle))
        if cfg.plots.fft_0_1000:
            plot_records.append(plot_fft(bundle, cfg.plots))
        if cfg.plots.envelope_ir and bundle.spec.label.upper() == "IR":
            plot_records.append(plot_envelope(bundle))
        if cfg.plots.spectrogram_ir and bundle.spec.label.upper() == "IR":
            spec_record = plot_spectrogram(bundle)
            if spec_record:
                plot_records.append(spec_record)

    if cfg.plots.features_dist or cfg.plots.corr_matrix:
        feature_df = pd.read_csv(cfg.io.features_csv)
        feature_df = feature_df[feature_df["label"].isin(["B", "IR", "OR", "N"])]
        if cfg.plots.features_dist:
            for feat in cfg.feature_viz.time_feats:
                if cfg.feature_viz.use_z_pref and not feat.startswith("z_"):
                    z_name = f"z_{feat}"
                    if z_name not in feature_df.columns and feat not in feature_df.columns:
                        logger.warning("特征列缺失: %s", z_name)
                elif feat not in feature_df.columns:
                    logger.warning("特征列缺失: %s", feat)
            feature_plots = plot_feature_distributions(feature_df, cfg.feature_viz, ["B", "IR", "OR", "N"])
            if feature_plots:
                logger.info("生成特征分布相关图 %d 张。", len(feature_plots))
                plot_records.extend(feature_plots)
                summary_plot = plot_feature_distribution_summary(feature_df, ["B", "IR", "OR", "N"])
                if summary_plot:
                    plot_records.append(summary_plot)
        if cfg.plots.corr_matrix:
            corr_record = plot_correlation_matrix(feature_df, cfg.feature_viz, ["B", "IR", "OR", "N"])
            if corr_record:
                plot_records.append(corr_record)
                logger.info("生成相关性矩阵图。")
            else:
                logger.warning("相关性矩阵绘制失败：有效特征列不足。")

        overview_plot = plot_data_overview(feature_df)
        if overview_plot:
            plot_records.append(overview_plot)

        embed_plot = plot_feature_space(feature_df, cfg.plots.use_umap)
        if embed_plot:
            plot_records.append(embed_plot)

    if not plot_records:
        logger.warning("未生成任何图表，请检查配置。")
        return

    figures = save_plots(plot_records, cfg)
    summary_path = write_summary(cfg, snapshot_config(cfg), updated_segments or segments, figures)
    logger.info("汇总文件已生成: %s", summary_path)
    logger.info("可视化流程完成，共生成 %d 张图。", len(figures))


def entry_point():  # pragma: no cover
    main()


if __name__ == "__main__":
    main()
