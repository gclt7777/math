"""Q2 可视化命令 CLI。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib

from .config import ensure_output_dirs, load_viz_config
from .data_loader import load_artifacts
from .metrics import compute_metrics
from .plots import generate_all_plots
from .reporting import print_tables
from .tables import build_tables
from .export import save_all


CHINESE_FONTS = ["SimHei", "Microsoft YaHei", "Heiti SC", "STHeiti", "PingFang SC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q2 可视化分析命令")
    parser.add_argument("--config", default="config/q2_viz.yaml", help="配置文件路径")
    return parser.parse_args()


def setup_font() -> None:
    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = CHINESE_FONTS


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("q2_viz")
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


def main() -> None:
    args = parse_args()
    cfg = load_viz_config(args.config)
    ensure_output_dirs(cfg)
    setup_font()
    logger = setup_logger(cfg.io.log_file)

    logger.info("加载配置: %s", Path(args.config).resolve())
    logger.info("模型路径: %s", cfg.io.model_path)

    bundle = load_artifacts(cfg)
    logger.info("测试集样本数: %d", len(bundle.y_test))
    logger.info("类别映射: %s", cfg.classes.display)

    metrics_bundle = compute_metrics(bundle, cfg)
    plots = generate_all_plots(metrics_bundle, cfg)
    tables = build_tables(metrics_bundle, cfg)

    print_tables(tables, logger)

    if tables.cv_summary is not None and not tables.cv_summary.empty:
        best_row = tables.cv_summary.iloc[0]
        best_model = best_row.get("模型", "未知模型")
        logger.info("最佳模型: %s", best_model)

    generated_paths = save_all(plots, tables, metrics_bundle, cfg)
    logger.info("已生成文件 %d 个", len(generated_paths))
    for path in generated_paths:
        logger.info("输出: %s", path)
    if metrics_bundle.reliability is not None:
        logger.info("Brier 分数: %.4f", metrics_bundle.reliability.brier_score)
    if metrics_bundle.file_reliability is not None:
        logger.info("文件级 Brier 分数: %.4f", metrics_bundle.file_reliability.brier_score)


def entry_point():  # pragma: no cover
    main()


if __name__ == "__main__":
    main()
