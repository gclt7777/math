"""导出与汇总工具。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any

from .config import VizConfig
from .plots import PlotRecord
from .selectors import SegmentSpec


def save_plot(record: PlotRecord, cfg: VizConfig) -> str:
    """保存单张图像，并视需要生成 SVG。"""

    base = Path(cfg.io.out_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / record.filename
    record.figure.savefig(path, dpi=cfg.feature_viz.figure_dpi, bbox_inches="tight")
    # 避免内存泄漏
    from matplotlib import pyplot as plt

    plt.close(record.figure)
    return str(path)


def save_plots(records: Iterable[PlotRecord], cfg: VizConfig) -> List[Dict[str, Any]]:
    """批量保存绘图并返回记录。"""

    saved = []
    for rec in records:
        path = save_plot(rec, cfg)
        saved.append(
            {
                "category": rec.category,
                "filename": rec.filename,
                "path": path,
                "description": rec.description,
            }
        )
    return saved


def write_summary(
    cfg: VizConfig,
    config_snapshot: Dict[str, Any],
    segments: List[SegmentSpec],
    figures: List[Dict[str, Any]],
) -> str:
    """将汇总信息写入 summary.json。"""

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": config_snapshot,
        "segments": [seg.to_dict() for seg in segments],
        "figures": figures,
    }
    out_path = Path(cfg.io.out_dir) / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return str(out_path)
