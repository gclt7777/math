"""配置解析模块：读取 q1_viz.yaml 并整合 Q1 基础配置。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from q1_pipeline.config import load_config as load_q1_config, Config as Q1Config


@dataclass
class IOConfig:
    """输入输出与日志路径设置。"""

    source_dirs: List[str]
    target_dirs: List[str]
    features_csv: str
    out_dir: str
    log_file: str

    def resolve(self, root: Path) -> None:
        """将相对路径转换为绝对路径。"""

        self.source_dirs = [str((root / p).resolve()) for p in self.source_dirs]
        self.target_dirs = [str((root / p).resolve()) for p in self.target_dirs]
        self.features_csv = str((root / self.features_csv).resolve())
        self.out_dir = str((root / self.out_dir).resolve())
        self.log_file = str((root / self.log_file).resolve())


@dataclass
class SelectionConfig:
    """样本选择策略。"""

    random_seed: int
    prefer_sensor: List[str]
    sample_per_label: int
    sample_files: Dict[str, str]
    ir_for_tfspec: bool = True


@dataclass
class PlotToggles:
    """图表开关。"""

    time_waveform: bool = True
    fft_0_1000: bool = True
    char_freq_marks: bool = True
    spectrogram_ir: bool = True
    envelope_ir: bool = True
    features_dist: bool = True
    corr_matrix: bool = True
    use_umap: bool = True


@dataclass
class FFTConfig:
    """频谱参数。"""

    max_hz: float = 1000.0
    nfft: int = 4096
    window: str = "hann"


@dataclass
class SpecConfig:
    """时频谱参数。"""

    nperseg: int = 2048
    noverlap: int = 1024
    max_hz: float = 4000.0


@dataclass
class EnvelopeOverride:
    """包络配置覆盖项。"""

    window: Optional[str] = None
    zero_pad: Optional[bool] = None
    fft_max_hz: Optional[float] = None


@dataclass
class FeatureVizConfig:
    """特征可视化参数。"""

    use_z_pref: bool
    time_feats: List[str]
    mech_energy_keys: List[str]
    figure_dpi: int


@dataclass
class VizConfig:
    """整体可视化配置对象。"""

    base_config_path: str
    base_config: Q1Config
    io: IOConfig
    selection: SelectionConfig
    plots: PlotToggles
    fft: FFTConfig
    spec: SpecConfig
    envelope: EnvelopeOverride
    feature_viz: FeatureVizConfig


def load_viz_config(path: str) -> VizConfig:
    """加载可视化配置并返回整合后的 VizConfig。"""

    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    root = cfg_path.parent.parent

    base_cfg_rel = data.get("base_q1_config", "config/q1.yaml")
    base_cfg_path = str((root / base_cfg_rel).resolve())
    base_cfg = load_q1_config(base_cfg_path)

    io_raw = data.get("io", {})
    io_cfg = IOConfig(
        source_dirs=io_raw.get("source_dirs", []),
        target_dirs=io_raw.get("target_dirs", []),
        features_csv=io_raw.get("features_csv", "artifacts/q1/features_source.csv"),
        out_dir=io_raw.get("out_dir", "artifacts/q1/viz"),
        log_file=io_raw.get("log_file", "artifacts/q1/viz.log"),
    )
    io_cfg.resolve(root)

    sel_raw = data.get("selection", {})
    selection_cfg = SelectionConfig(
        random_seed=sel_raw.get("random_seed", 2025),
        prefer_sensor=sel_raw.get("prefer_sensor", ["DE", "FE", "BA"]),
        sample_per_label=sel_raw.get("sample_per_label", 1),
        sample_files={k: str((root / Path(v)).resolve()) for k, v in sel_raw.get("sample_files", {}).items()},
        ir_for_tfspec=sel_raw.get("ir_for_tfspec", True),
    )

    plots_cfg = PlotToggles(**data.get("plots", {}))
    fft_cfg = FFTConfig(**data.get("fft", {}))
    spec_cfg = SpecConfig(**data.get("spec", {}))
    envelope_cfg = EnvelopeOverride(**data.get("envelope", {}))
    fv_raw = data.get("feature_viz", {})
    feature_cfg = FeatureVizConfig(
        use_z_pref=fv_raw.get("use_z_pref", True),
        time_feats=fv_raw.get("time_feats", []),
        mech_energy_keys=fv_raw.get("mech_energy_keys", []),
        figure_dpi=fv_raw.get("figure_dpi", 160),
    )

    return VizConfig(
        base_config_path=base_cfg_path,
        base_config=base_cfg,
        io=io_cfg,
        selection=selection_cfg,
        plots=plots_cfg,
        fft=fft_cfg,
        spec=spec_cfg,
        envelope=envelope_cfg,
        feature_viz=feature_cfg,
    )


def ensure_output_tree(cfg: VizConfig) -> None:
    """保证输出目录结构存在。"""

    out_dir = Path(cfg.io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.io.log_file).parent.mkdir(parents=True, exist_ok=True)


def snapshot_config(cfg: VizConfig) -> Dict[str, Any]:
    """生成 summary.json 的配置快照。"""

    return {
        "base_q1_config": cfg.base_config_path,
        "io": {
            "source_dirs": cfg.io.source_dirs,
            "target_dirs": cfg.io.target_dirs,
            "features_csv": cfg.io.features_csv,
            "out_dir": cfg.io.out_dir,
            "log_file": cfg.io.log_file,
        },
        "selection": asdict(cfg.selection),
        "plots": asdict(cfg.plots),
        "fft": asdict(cfg.fft),
        "spec": asdict(cfg.spec),
        "envelope": asdict(cfg.envelope),
        "feature_viz": asdict(cfg.feature_viz),
    }
