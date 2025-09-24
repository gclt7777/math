"""Q3 无监督迁移 CLI。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

from .align import fit_transform
from .config import Q3Config, ensure_output_dirs, load_config
from .dataio import load_all
from .export import save_all
from .infer import predict_segments
from .metrics import compute_reports
from .viz import render_all

CHINESE_FONTS = ["SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC", "STHeiti"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q3 无监督迁移诊断")
    parser.add_argument("--config", default="config/q3.yaml", help="配置文件路径")
    return parser.parse_args()


def setup_font(font_path: str | None) -> None:
    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False
    if font_path:
        matplotlib.rcParams["font.sans-serif"] = [font_path]
    else:
        matplotlib.rcParams["font.sans-serif"] = CHINESE_FONTS


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("q3_transfer")
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
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    setup_font(cfg.viz.chinese_font)
    logger = setup_logger(cfg.io.log_file)

    logger.info("加载配置: %s", Path(args.config).resolve())

    data_bundle = load_all(cfg)
    logger.info("源域样本数: %d | 目标域样本数: %d", len(data_bundle.source.features), len(data_bundle.target.features))

    alignment = fit_transform(data_bundle, cfg)
    logger.info("域差异 (适配前): %s", alignment.metrics_before)
    logger.info("域差异 (适配后): %s", alignment.metrics_after)

    preds = predict_segments(alignment, cfg)
    logger.info("文件级预测完成，文件数: %d", len(preds.file_df))

    reports = compute_reports(alignment, preds, cfg)
    figures = render_all(alignment, preds, reports, cfg)

    outputs = save_all(preds, reports, figures, alignment, cfg)
    logger.info("已生成 %d 个产物", len(outputs))
    for path in outputs:
        logger.info("输出: %s", path)

    if not preds.submission_df.empty:
        logger.info(
            "目标文件故障类型预测 (共 %d 个):\n%s",
            len(preds.submission_df),
            preds.submission_df.to_string(index=False),
        )
    logger.info("文件级预测示例:\n%s", preds.file_df.head().to_string(index=False))


def entry_point():  # pragma: no cover
    main()


if __name__ == "__main__":
    main()

