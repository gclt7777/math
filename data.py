"""数据加载与拆分模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .utils import Config


ALLOWED_LABELS = {"N", "OR", "IR", "B"}


@dataclass
class DataBundle:
    """承载特征矩阵与关联元信息。"""

    X: pd.DataFrame
    y: pd.Series
    uids: pd.Series
    sensors: pd.Series
    file_paths: pd.Series
    feature_cols: List[str]
    use_scaler: bool


@dataclass
class SplitBundle:
    """存放拆分后的训练/验证/测试数据。"""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    uid_train: pd.Series
    uid_val: pd.Series
    uid_test: pd.Series
    file_train: pd.Series
    file_val: pd.Series
    file_test: pd.Series
    feature_cols: List[str]
    use_scaler: bool

    def trainval(self) -> Tuple[pd.DataFrame, pd.Series]:
        """返回训练与验证合并后的数据，用于最终模型训练。"""

        X = pd.concat([self.X_train, self.X_val], axis=0)
        y = pd.concat([self.y_train, self.y_val], axis=0)
        return X, y


def load_data(cfg: Config, logger) -> DataBundle:
    """读取源域特征并完成基础清洗。"""

    df = pd.read_csv(cfg.io.feature_csv)

    if "uid" not in df.columns:
        df.insert(0, "uid", [f"auto_{i}" for i in range(len(df))])

    if "sensor" not in df.columns and "sensor_src" in df.columns:
        df["sensor"] = df["sensor_src"]

    file_paths = df.get("file_path", pd.Series([None] * len(df), index=df.index, name="file_path"))

    if "uid" not in df.columns:
        raise ValueError("特征表缺少 uid 列，无法追踪样本。")

    # 记录原始规模
    logger.info("原始样本量: %d", len(df))

    df = df[df[cfg.data.label_col].isin(ALLOWED_LABELS)].copy()
    logger.info("筛除非法标签后样本量: %d", len(df))

    if "sensor" in df.columns and cfg.data.sensor_filter:
        df = df[df["sensor"].isin(cfg.data.sensor_filter)].copy()
        logger.info("传感器过滤后样本量: %d", len(df))

    uids = df["uid"].copy()
    sensors = df.get("sensor", pd.Series([None] * len(df), index=df.index, name="sensor"))
    file_paths = file_paths.loc[df.index].reset_index(drop=True)

    if "sensor" in df.columns:
        df = df.drop(columns="sensor")

    for col in cfg.data.drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    label_col = cfg.data.label_col
    if label_col not in df.columns:
        raise ValueError(f"清洗后缺少标签列: {label_col}")

    feature_cols = [c for c in df.columns if c != label_col]

    z_cols = [c for c in feature_cols if c.startswith("z_")]
    use_scaler = False
    if cfg.data.prefer_z_features and z_cols:
        feature_cols = z_cols
    else:
        use_scaler = True

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # 记录类别分布
    counts = y.value_counts().to_dict()
    logger.info("标签分布: %s", counts)

    return DataBundle(
        X=X,
        y=y,
        uids=uids.reset_index(drop=True),
        sensors=sensors.reset_index(drop=True),
        file_paths=file_paths,
        feature_cols=feature_cols,
        use_scaler=use_scaler,
    )


def split_data(bundle: DataBundle, cfg: Config) -> SplitBundle:
    """执行分层拆分，先划分测试集再划分验证集。"""

    stratify = bundle.y if cfg.data.stratify_by == cfg.data.label_col else None

    groups = bundle.file_paths.to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.data.test_size, random_state=cfg.data.random_seed)
    trainval_idx, test_idx = next(gss.split(np.zeros(len(groups)), bundle.y, groups=groups))

    X_trainval = bundle.X.iloc[trainval_idx]
    y_trainval = bundle.y.iloc[trainval_idx]
    uid_trainval = bundle.uids.iloc[trainval_idx]
    file_trainval = bundle.file_paths.iloc[trainval_idx]

    X_test = bundle.X.iloc[test_idx]
    y_test = bundle.y.iloc[test_idx]
    uid_test = bundle.uids.iloc[test_idx]
    file_test = bundle.file_paths.iloc[test_idx]

    val_size = cfg.data.val_size
    if val_size <= 0 or val_size >= 1:
        raise ValueError("val_size 必须位于 (0,1)。")

    n_classes = y_trainval.nunique()
    X_train = X_trainval
    y_train = y_trainval
    uid_train = uid_trainval
    file_train = file_trainval

    X_val = X_trainval.iloc[0:0].copy()
    y_val = y_trainval.iloc[0:0].copy()
    uid_val = uid_trainval.iloc[0:0].copy()
    file_val = file_trainval.iloc[0:0].copy()

    remaining_ratio = max(1.0 - cfg.data.test_size, 1e-8)
    val_ratio = cfg.data.val_size / remaining_ratio if remaining_ratio > 0 else 0.0

    trainval_groups = file_trainval.to_numpy()
    if 0 < val_ratio < 1 and len(np.unique(trainval_groups)) > 1 and len(y_trainval) >= 2 * n_classes:
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=cfg.data.random_seed)
        train_idx_rel, val_idx_rel = next(gss_val.split(np.zeros(len(trainval_groups)), y_trainval, groups=trainval_groups))
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]

        X_train = bundle.X.iloc[train_idx]
        y_train = bundle.y.iloc[train_idx]
        uid_train = bundle.uids.iloc[train_idx]
        file_train = bundle.file_paths.iloc[train_idx]

        X_val = bundle.X.iloc[val_idx]
        y_val = bundle.y.iloc[val_idx]
        uid_val = bundle.uids.iloc[val_idx]
        file_val = bundle.file_paths.iloc[val_idx]

    split = SplitBundle(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        uid_train=uid_train.reset_index(drop=True),
        uid_val=uid_val.reset_index(drop=True),
        uid_test=uid_test.reset_index(drop=True),
        file_train=file_train.reset_index(drop=True),
        file_val=file_val.reset_index(drop=True),
        file_test=file_test.reset_index(drop=True),
        feature_cols=bundle.feature_cols,
        use_scaler=bundle.use_scaler,
    )

    # 额外校验：同一原始文件的窗口仅能出现在单一数据子集，避免潜在泄漏。
    file_sets = {
        "train": set(split.file_train.dropna()),
        "val": set(split.file_val.dropna()),
        "test": set(split.file_test.dropna()),
    }
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = file_sets[a].intersection(file_sets[b])
        if inter:
            overlaps[f"{a}_{b}"] = sorted(inter)
    if overlaps:
        raise RuntimeError(
            "检测到数据泄漏：以下数据集共享相同 file_path 窗口 -> "
            + "; ".join(f"{pair}: {paths}" for pair, paths in overlaps.items())
        )

    return split
