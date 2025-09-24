"""工具模块，提供配置解析、日志与序列化等辅助功能。"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import yaml


@dataclass
class IOCfg:
    """输入输出配置。"""

    feature_csv: str
    preprocess_params: str
    out_dir: str
    log_file: str
    raw_source_dirs: List[str] = field(default_factory=list)
    raw_target_dirs: List[str] = field(default_factory=list)

    def resolve(self, root: Path) -> None:
        """将相对路径转换为绝对路径，便于全局一致引用。"""

        self.feature_csv = str((root / self.feature_csv).resolve())
        self.preprocess_params = str((root / self.preprocess_params).resolve())
        self.out_dir = str((root / self.out_dir).resolve())
        self.log_file = str((root / self.log_file).resolve())
        self.raw_source_dirs = [str((root / p).resolve()) for p in self.raw_source_dirs]
        self.raw_target_dirs = [str((root / p).resolve()) for p in self.raw_target_dirs]


@dataclass
class DataCfg:
    """数据处理配置。"""

    label_col: str
    drop_cols: List[str]
    sensor_filter: List[str]
    stratify_by: str
    test_size: float
    val_size: float
    random_seed: int
    prefer_z_features: bool = True


@dataclass
class CandidateCfg:
    """单个模型候选配置。"""

    name: str
    type: str
    params_grid: Dict[str, Any]


@dataclass
class CVCfg:
    """交叉验证配置。"""

    folds: int
    scoring: List[str]
    n_jobs: int


@dataclass
class ModelsCfg:
    """模型搜索配置。"""

    candidates: List[CandidateCfg]
    cv: CVCfg


@dataclass
class ImbalanceCfg:
    """类不平衡处理配置。"""

    use_class_weight: bool = True
    sampler: Optional[Dict[str, Any]] = None


@dataclass
class CalibrationCfg:
    """校准配置。"""

    enabled: bool = True
    method: str = "isotonic"


@dataclass
class ExportCfg:
    """导出配置。"""

    topk_models: int = 1
    save_per_fold: bool = False
    figure_dpi: int = 160


@dataclass
class Config:
    """顶层配置对象。"""

    io: IOCfg
    data: DataCfg
    models: ModelsCfg
    imbalance: ImbalanceCfg
    calibration: CalibrationCfg
    export: ExportCfg


def load_config(path: str) -> Config:
    """读取 YAML 文件并生成配置实例。"""

    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = config_path.parent.parent

    def _parse_candidates(items: List[Dict[str, Any]]) -> List[CandidateCfg]:
        result = []
        for item in items:
            result.append(
                CandidateCfg(
                    name=item["name"],
                    type=item["type"],
                    params_grid=item.get("params_grid", {}),
                )
            )
        return result

    io_cfg = IOCfg(**raw["io"])
    io_cfg.resolve(root)

    data_cfg = DataCfg(**raw["data"])

    cv_raw = raw["models"]["cv"]
    cv_cfg = CVCfg(folds=cv_raw["folds"], scoring=cv_raw["scoring"], n_jobs=cv_raw["n_jobs"])

    model_cfg = ModelsCfg(
        candidates=_parse_candidates(raw["models"]["candidates"]),
        cv=cv_cfg,
    )

    imbalance_cfg = ImbalanceCfg(**raw["imbalance"])
    calibration_cfg = CalibrationCfg(**raw["calibration"])
    export_cfg = ExportCfg(**raw["export"])

    return Config(
        io=io_cfg,
        data=data_cfg,
        models=model_cfg,
        imbalance=imbalance_cfg,
        calibration=calibration_cfg,
        export=export_cfg,
    )


def ensure_output_dirs(cfg: Config) -> None:
    """创建输出与日志目录，避免后续写入失败。"""

    out_dir = Path(cfg.io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.io.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """设置随机种子，确保实验可复现。"""

    random.seed(seed)
    np.random.seed(seed)


def setup_logger(log_file: str) -> logging.Logger:
    """初始化日志记录器，输出到控制台与文件。"""

    logger = logging.getLogger("q2_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """以 UTF-8 保存 JSON 文件，保留中文。"""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_joblib(path: str, obj: Any) -> None:
    """使用 joblib 序列化对象。"""

    joblib.dump(obj, path)


def config_snapshot(cfg: Config) -> Dict[str, Any]:
    """将配置转换为可记录的字典结构。"""

    return {
        "io": asdict(cfg.io),
        "data": asdict(cfg.data),
        "models": {
            "candidates": [asdict(item) for item in cfg.models.candidates],
            "cv": asdict(cfg.models.cv),
        },
        "imbalance": asdict(cfg.imbalance),
        "calibration": asdict(cfg.calibration),
        "export": asdict(cfg.export),
    }
