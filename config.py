"""配置解析：读取 q2 可视化 YAML 并准备运行参数。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class IOConfig:
    """输入输出与依赖路径。"""

    model_path: str
    columns_json: str
    split_manifest: str
    features_csv: str
    cv_results: Optional[str]
    window_predictions: Optional[str]
    file_predictions: Optional[str]
    test_report: Optional[str]
    file_report: Optional[str]
    out_dir: str
    log_file: str

    def resolve(self, root: Path) -> None:
        self.model_path = str((root / self.model_path).resolve())
        self.columns_json = str((root / self.columns_json).resolve())
        self.split_manifest = str((root / self.split_manifest).resolve())
        self.features_csv = str((root / self.features_csv).resolve())
        if self.cv_results:
            self.cv_results = str((root / self.cv_results).resolve())
        if self.window_predictions:
            self.window_predictions = str((root / self.window_predictions).resolve())
        if self.file_predictions:
            self.file_predictions = str((root / self.file_predictions).resolve())
        if self.test_report:
            self.test_report = str((root / self.test_report).resolve())
        if self.file_report:
            self.file_report = str((root / self.file_report).resolve())
        self.out_dir = str((root / self.out_dir).resolve())
        self.log_file = str((root / self.log_file).resolve())


@dataclass
class ClassConfig:
    """类别相关设置。"""

    order: List[str] = field(default_factory=lambda: ["N", "OR", "IR", "B"])
    display: Dict[str, str] = field(default_factory=lambda: {
        "N": "正常",
        "OR": "外圈故障",
        "IR": "内圈故障",
        "B": "滚动体故障",
    })


@dataclass
class PlotToggles:
    """图表开关。"""

    plot_confusion: bool = True
    plot_roc_pr: bool = True
    plot_feature_importance: bool = True
    plot_model_comparison: bool = True
    plot_reliability: bool = False


@dataclass
class TableFormat:
    """表格格式化设置。"""

    precision: int = 3
    highlight_lower: Optional[float] = 0.8
    highlight_upper: Optional[float] = 0.9
    highlight_low_color: str = "#fdd0d0"
    highlight_high_color: str = "#d0f0d0"


@dataclass
class FeatureImportanceCfg:
    """特征重要度参数。"""

    topk: int = 30
    permutation_repeats: int = 10
    use_permutation: bool = True


@dataclass
class VizConfig:
    """整体配置对象。"""

    io: IOConfig
    classes: ClassConfig
    plots: PlotToggles
    table_format: TableFormat
    feature_importance: FeatureImportanceCfg


def load_viz_config(path: str) -> VizConfig:
    """读取配置文件并构造 VizConfig。"""

    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = cfg_path.parent.parent

    io_raw = raw.get("io", {})
    io_cfg = IOConfig(
        model_path=io_raw.get("model_path", "artifacts/q2/best_model.joblib"),
        columns_json=io_raw.get("columns_json", "artifacts/q2/columns.json"),
        split_manifest=io_raw.get("split_manifest", "artifacts/q2/split_manifest.json"),
        features_csv=io_raw.get("features_csv", "artifacts/q1/数据集/source_domain_features_simplified.csv"),
        cv_results=io_raw.get("cv_results"),
        window_predictions=io_raw.get("window_predictions", "artifacts/q2/window_predictions.csv"),
        file_predictions=io_raw.get("file_predictions", "artifacts/q2/file_predictions.csv"),
        test_report=io_raw.get("test_report", "artifacts/q2/test_report.csv"),
        file_report=io_raw.get("file_report", "artifacts/q2/file_report.csv"),
        out_dir=io_raw.get("out_dir", "artifacts/q2/viz"),
        log_file=io_raw.get("log_file", "artifacts/q2/viz.log"),
    )
    io_cfg.resolve(root)

    class_cfg = ClassConfig(**raw.get("classes", {}))
    plot_cfg = PlotToggles(**raw.get("plots", {}))
    table_cfg = TableFormat(**raw.get("table_format", {}))
    fi_cfg = FeatureImportanceCfg(**raw.get("feature_importance", {}))

    return VizConfig(
        io=io_cfg,
        classes=class_cfg,
        plots=plot_cfg,
        table_format=table_cfg,
        feature_importance=fi_cfg,
    )


def ensure_output_dirs(cfg: VizConfig) -> None:
    """确保输出与日志目录存在。"""

    out_dir = Path(cfg.io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.io.log_file).parent.mkdir(parents=True, exist_ok=True)


def snapshot_config(cfg: VizConfig) -> Dict[str, object]:
    """生成 summary.json 需要的配置快照。"""

    return {
        "io": {
            "model_path": cfg.io.model_path,
            "columns_json": cfg.io.columns_json,
            "split_manifest": cfg.io.split_manifest,
            "features_csv": cfg.io.features_csv,
            "cv_results": cfg.io.cv_results,
            "window_predictions": cfg.io.window_predictions,
            "file_predictions": cfg.io.file_predictions,
            "test_report": cfg.io.test_report,
            "file_report": cfg.io.file_report,
            "out_dir": cfg.io.out_dir,
            "log_file": cfg.io.log_file,
        },
        "classes": asdict(cfg.classes),
        "plots": asdict(cfg.plots),
        "table_format": asdict(cfg.table_format),
        "feature_importance": asdict(cfg.feature_importance),
    }
