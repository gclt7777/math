"""Q4 配置解析。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class IOConfig:
    source_feature_csv: Optional[str] = None
    target_feature_csv: str = "artifacts/q1/features_target.csv"
    q2_model_path: str = "artifacts/q2/best_model.joblib"
    q2_columns: str = "artifacts/q2/columns.json"
    q2_scaler: Optional[str] = None
    out_dir: str = "artifacts/q4"
    log_file: str = "artifacts/q4/q4_adapt.log"
    raw_source_dirs: List[str] = field(default_factory=list)
    raw_target_dirs: List[str] = field(default_factory=list)

    def resolve(self, root: Path) -> None:
        self.target_feature_csv = str((root / self.target_feature_csv).resolve())
        if self.source_feature_csv:
            self.source_feature_csv = str((root / self.source_feature_csv).resolve())
        self.q2_model_path = str((root / self.q2_model_path).resolve())
        self.q2_columns = str((root / self.q2_columns).resolve())
        if self.q2_scaler:
            self.q2_scaler = str((root / self.q2_scaler).resolve())
        self.out_dir = str((root / self.out_dir).resolve())
        self.log_file = str((root / self.log_file).resolve())
        self.raw_source_dirs = [str((root / p).resolve()) for p in self.raw_source_dirs]
        self.raw_target_dirs = [str((root / p).resolve()) for p in self.raw_target_dirs]


@dataclass
class DataConfig:
    label_col: str = "label"
    drop_cols: List[str] = field(default_factory=list)
    prefer_z_features: bool = True
    random_seed: int = 2025


@dataclass
class SelfTrainingConfig:
    enabled: bool = True
    conf_threshold: float = 0.85
    max_iter: int = 3
    max_per_class: int = 9999
    min_new_samples: int = 20
    reweight_source: float = 1.0
    reweight_target: float = 1.0


@dataclass
class CalibrationConfig:
    enabled: bool = True
    method: str = "temperature"  # temperature | isotonic | none


@dataclass
class OpenSetConfig:
    enabled: bool = True
    reject_threshold: float = 0.5
    method: str = "max_proba"  # max_proba | energy


@dataclass
class AdaptConfig:
    feature_align: str = "zscore_to_source"  # none|zscore_to_source|coral
    self_training: SelfTrainingConfig = field(default_factory=SelfTrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    open_set: OpenSetConfig = field(default_factory=OpenSetConfig)


@dataclass
class ReportConfig:
    figure_dpi: int = 160
    save_tables: bool = True
    save_figures: bool = True


@dataclass
class ExportConfig:
    save_adapter: bool = True
    save_thresholds: bool = True


@dataclass
class Q4Config:
    io: IOConfig
    data: DataConfig
    adapt: AdaptConfig
    report: ReportConfig
    export: ExportConfig


def load_config(path: str) -> Q4Config:
    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = cfg_path.parent.parent

    io_cfg = IOConfig(**raw.get("io", {}))
    io_cfg.resolve(root)

    data_cfg = DataConfig(**raw.get("data", {}))

    raw_adapt = raw.get("adapt", {})
    st_cfg = SelfTrainingConfig(**raw_adapt.get("self_training", {}))
    cal_cfg = CalibrationConfig(**raw_adapt.get("calibration", {}))
    os_cfg = OpenSetConfig(**raw_adapt.get("open_set", {}))
    adapt_cfg = AdaptConfig(
        feature_align=raw_adapt.get("feature_align", "zscore_to_source"),
        self_training=st_cfg,
        calibration=cal_cfg,
        open_set=os_cfg,
    )

    report_cfg = ReportConfig(**raw.get("report", {}))
    export_cfg = ExportConfig(**raw.get("export", {}))

    if adapt_cfg.feature_align not in {"none", "zscore_to_source", "coral"}:
        raise ValueError("adapt.feature_align 必须是 none/zscore_to_source/coral 之一")
    if adapt_cfg.calibration.method not in {"temperature", "isotonic", "none"}:
        raise ValueError("adapt.calibration.method 仅支持 temperature/isotonic/none")
    if adapt_cfg.open_set.method not in {"max_proba", "energy"}:
        raise ValueError("adapt.open_set.method 仅支持 max_proba/energy")

    return Q4Config(
        io=io_cfg,
        data=data_cfg,
        adapt=adapt_cfg,
        report=report_cfg,
        export=export_cfg,
    )


def snapshot_config(cfg: Q4Config) -> Dict[str, Any]:
    return {
        "io": asdict(cfg.io),
        "data": asdict(cfg.data),
        "adapt": {
            "feature_align": cfg.adapt.feature_align,
            "self_training": asdict(cfg.adapt.self_training),
            "calibration": asdict(cfg.adapt.calibration),
            "open_set": asdict(cfg.adapt.open_set),
        },
        "report": asdict(cfg.report),
        "export": asdict(cfg.export),
    }
