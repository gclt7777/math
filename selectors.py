"""段选择与样本配置。"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import SelectionConfig
from .utils import MatEntry, scan_mat_dirs


@dataclass
class SegmentSpec:
    """描述待可视化的段。"""

    label: str
    file_path: str
    basename: str
    sensor: Optional[str]
    prefer_sensors: List[str]
    fs_hint: Optional[int]
    seg_idx: int = 0
    rpm_used: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        """用于 summary.json 的序列化。"""

        data = asdict(self)
        return data


def prepare_segments(
    cfg: SelectionConfig,
    source_dirs: List[str],
    target_dirs: List[str],
) -> List[SegmentSpec]:
    """根据配置从目录中选取样本。"""

    rng = random.Random(cfg.random_seed)
    entries = scan_mat_dirs(source_dirs + target_dirs)

    grouped: Dict[str, List[MatEntry]] = {}
    for entry in entries:
        label = entry.label or "UNKNOWN"
        grouped.setdefault(label, []).append(entry)

    specs: List[SegmentSpec] = []
    desired_labels = ["B", "IR", "OR", "N"]

    for label in desired_labels:
        overrides = cfg.sample_files.get(label)
        chosen_entries: List[MatEntry] = []
        if overrides:
            path = Path(overrides)
            if not path.exists():
                raise FileNotFoundError(f"sample_files[{label}] 路径不存在: {path}")
            chosen_entries.append(
                MatEntry(
                    path=str(path.resolve()),
                    basename=path.stem,
                    label=label,
                    sensor_hints=[s for s in cfg.prefer_sensor],
                    fs_hint=None,
                )
            )
        else:
            candidates = grouped.get(label, [])
            if not candidates:
                continue
            sample_size = min(cfg.sample_per_label, len(candidates))
            chosen_entries = rng.sample(candidates, sample_size)

        for entry in chosen_entries:
            sensor = next((s for s in cfg.prefer_sensor if s in entry.sensor_hints), None)
            specs.append(
                SegmentSpec(
                    label=label,
                    file_path=entry.path,
                    basename=entry.basename,
                    sensor=sensor,
                    prefer_sensors=list(cfg.prefer_sensor),
                    fs_hint=entry.fs_hint,
                )
            )

    return specs
