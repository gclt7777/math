"""Q2 训练主入口，串联加载、训练、评估与导出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .data import DataBundle, load_data, split_data
from .evaluate import evaluate_model
from .models import fit_final_model, perform_grid_search
from .export import export_all
from .utils import (
    Config,
    config_snapshot,
    ensure_output_dirs,
    load_config,
    set_global_seed,
    setup_logger,
)


def _subset_bundle(bundle: DataBundle, n_samples: int, seed: int) -> DataBundle:
    """对数据包做分层下采样，确保每个类别保留足够样本。"""

    total = len(bundle.X)
    if n_samples >= total:
        return bundle

    min_required = 4  # 避免后续分层拆分时类别过少
    rng = np.random.default_rng(seed)

    # 分层抽样，优先保持原始比例
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    train_idx, _ = next(splitter.split(bundle.X, bundle.y))
    selected_idx = list(bundle.X.index[train_idx])

    # 若某些类别样本过少，追加该类别样本直至满足最小阈值
    selected_set = set(selected_idx)
    counts = bundle.y.loc[selected_idx].value_counts()
    for label, total_count in bundle.y.value_counts().items():
        current = counts.get(label, 0)
        if current >= min_required or total_count <= current:
            continue

        need = min(min_required, total_count) - current
        if need <= 0:
            continue

        available = [idx for idx in bundle.y[bundle.y == label].index if idx not in selected_set]
        if not available:
            continue
        take = min(need, len(available))
        extra_idx = rng.choice(available, size=take, replace=False)
        selected_idx.extend(extra_idx.tolist())
        selected_set.update(extra_idx.tolist())
        counts[label] = current + take

    # 洗牌保持随机性
    selected_array = np.array(selected_idx)
    rng.shuffle(selected_array)
    final_idx = selected_array.tolist()

    return DataBundle(
        X=bundle.X.loc[final_idx].reset_index(drop=True),
        y=bundle.y.loc[final_idx].reset_index(drop=True),
        uids=bundle.uids.loc[final_idx].reset_index(drop=True),
        sensors=bundle.sensors.loc[final_idx].reset_index(drop=True),
        file_paths=bundle.file_paths.loc[final_idx].reset_index(drop=True),
        feature_cols=bundle.feature_cols,
        use_scaler=bundle.use_scaler,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Q2 源域故障诊断基线训练")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="启用 Demo 模式，仅抽样少量样本快速验证流程",
    )
    parser.add_argument(
        "--demo-size",
        type=int,
        default=200,
        help="Demo 模式下保留的样本数量",
    )
    return parser.parse_args()


def main() -> None:
    """主函数：读取配置、执行训练并导出产物。"""

    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_global_seed(cfg.data.random_seed)
    logger = setup_logger(cfg.io.log_file)

    logger.info("启动 Q2 训练流程，配置文件：%s", args.config)
    logger.info("配置快照：%s", json.dumps(config_snapshot(cfg), ensure_ascii=False))

    if Path(cfg.io.preprocess_params).exists():
        logger.info("检测到 Q1 预处理参数：%s", cfg.io.preprocess_params)

    bundle = load_data(cfg, logger)
    logger.info("使用特征数: %d | 采用标准化: %s", len(bundle.feature_cols), bundle.use_scaler)
    if args.demo:
        logger.info("Demo 模式启用，将样本数量限制为 %d", args.demo_size)
        bundle = _subset_bundle(bundle, args.demo_size, cfg.data.random_seed)
        demo_counts = bundle.y.value_counts().to_dict()
        logger.info("Demo 数据集样本量: %d | 标签分布: %s", len(bundle.X), demo_counts)

    splits = split_data(bundle, cfg)
    logger.info(
        "拆分完成：Train=%d, Val=%d, Test=%d",
        len(splits.X_train),
        len(splits.X_val),
        len(splits.X_test),
    )
    if len(splits.X_val) == 0:
        logger.warning("验证集样本不足，已跳过验证集拆分。")

    search_result = perform_grid_search(splits, cfg, logger)
    logger.info("最佳模型 %s 超参：%s", search_result.best_name, search_result.best_params)

    final_model = fit_final_model(search_result, splits, cfg.calibration, cfg.models.cv.folds)
    logger.info("最终模型训练完成，开始在测试集评估。")

    eval_result = evaluate_model(final_model, splits, cfg.data.random_seed)
    logger.info("测试集 macro F1: %.4f", eval_result.report.loc["macro avg", "f1-score"])
    if eval_result.file_report is not None:
        try:
            file_acc = eval_result.file_report.loc["accuracy", "precision"]
            file_f1 = eval_result.file_report.loc["macro avg", "f1-score"]
            logger.info(
                "文件级指标 -> accuracy: %.4f | macro F1: %.4f | 文件数: %d",
                file_acc,
                file_f1,
                len(eval_result.file_predictions) if eval_result.file_predictions is not None else 0,
            )
        except Exception as exc:
            logger.warning("记录文件级指标失败：%s", exc)

    export_all(cfg, search_result, final_model, splits, eval_result, logger)

    logger.info("Q2 流程结束，产物已写入 %s", cfg.io.out_dir)


if __name__ == "__main__":
    main()
