"""配置解析与目录初始化。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


VALID_AGG_MODES = {"prob_mean", "prob_logmean", "prob_max", "prob_median", "vote"}


@dataclass
class IOConfig:
    """输入输出路径设置。"""

    src_feature_csv: str
    tgt_feature_csv: str
    preprocess_params: str
    q2_best_model: str
    q2_columns: str
    q2_importance: str
    out_dir: str
    log_file: str

    def resolve(self, root: Path) -> None:
        self.src_feature_csv = str((root / self.src_feature_csv).resolve())
        self.tgt_feature_csv = str((root / self.tgt_feature_csv).resolve())
        self.preprocess_params = str((root / self.preprocess_params).resolve())
        self.q2_best_model = str((root / self.q2_best_model).resolve())
        self.q2_columns = str((root / self.q2_columns).resolve())
        self.q2_importance = str((root / self.q2_importance).resolve())
        self.out_dir = str((root / self.out_dir).resolve())
        self.log_file = str((root / self.log_file).resolve())


@dataclass
class DataConfig:
    label_col: str = "label"
    uid_col: str = "uid"
    file_basename_col: str = "basename"
    sensor_filter: List[str] = field(default_factory=lambda: ["DE", "FE"])
    prefer_z_features: bool = True
    drop_cols: List[str] = field(default_factory=list)


@dataclass
class AdaptConfig:
    method: str = "coral"
    coral_eps: float = 1e-6
    mmd_kernel: str = "rbf"
    mmd_gamma: float = 0.5
    fit_on: str = "source"
    apply_to: str = "target"


@dataclass
class InferenceConfig:
    agg_mode: str = "prob_mean"
    unknown_threshold: float = 0.55
    topk: int = 2


@dataclass
class VizConfig:
    use_umap: bool = True
    figure_dpi: int = 160
    chinese_font: Optional[str] = None


@dataclass
class InterpretConfig:
    use_source_importance: bool = True
    use_shap: bool = False
    shap_sample: int = 2000
    mech_features_prefix: List[str] = field(default_factory=lambda: ["BPFI", "BPFO", "BSF"])


@dataclass
class ExportConfig:
    save_tables: bool = True
    save_figures: bool = True
    html_report: bool = True


@dataclass
class Q3Config:
    io: IOConfig
    data: DataConfig
    adapt: AdaptConfig
    inference: InferenceConfig
    viz: VizConfig
    interpret: InterpretConfig
    export: ExportConfig


def load_config(path: str) -> Q3Config:
    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = cfg_path.parent.parent

    io_cfg = IOConfig(**raw.get("io", {}))
    io_cfg.resolve(root)

    data_cfg = DataConfig(**raw.get("data", {}))
    adapt_cfg = AdaptConfig(**raw.get("adapt", {}))
    inference_cfg = InferenceConfig(**raw.get("inference", {}))
    viz_cfg = VizConfig(**raw.get("viz", {}))
    interpret_cfg = InterpretConfig(**raw.get("interpret", {}))
    export_cfg = ExportConfig(**raw.get("export", {}))

    # 合法性校验
    if adapt_cfg.method not in {"coral", "mmd", "none", "zscore"}:
        raise ValueError("adapt.method 必须是 coral/mmd/zscore/none 之一")
    if inference_cfg.agg_mode not in VALID_AGG_MODES:
        raise ValueError(
            "inference.agg_mode 仅支持 prob_mean/prob_logmean/prob_max/prob_median/vote"
        )

    return Q3Config(
        io=io_cfg,
        data=data_cfg,
        adapt=adapt_cfg,
        inference=inference_cfg,
        viz=viz_cfg,
        interpret=interpret_cfg,
        export=export_cfg,
    )


def ensure_output_dirs(cfg: Q3Config) -> None:
    out_dir = Path(cfg.io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    Path(cfg.io.log_file).parent.mkdir(parents=True, exist_ok=True)


def snapshot_config(cfg: Q3Config) -> Dict[str, object]:
    return {
        "io": asdict(cfg.io),
        "data": asdict(cfg.data),
        "adapt": asdict(cfg.adapt),
        "inference": asdict(cfg.inference),
        "viz": asdict(cfg.viz),
        "interpret": asdict(cfg.interpret),
        "export": asdict(cfg.export),
    }
