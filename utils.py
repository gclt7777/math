"""通用工具：文件扫描、标签推断等。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


LABEL_PATTERNS = {
    "B": re.compile(r"(^|[_/\\-])b([_/\\-]|$)", re.IGNORECASE),
    "IR": re.compile(r"(^|[_/\\-])ir([_/\\-]|$)", re.IGNORECASE),
    "OR": re.compile(r"(^|[_/\\-])or([_/\\-]|$)", re.IGNORECASE),
    "N": re.compile(r"(^|[_/\\-])n([_/\\-]|$)", re.IGNORECASE),
}

FS_HINTS = {
    12000: re.compile(r"12k", re.IGNORECASE),
    48000: re.compile(r"48k", re.IGNORECASE),
    32000: re.compile(r"32k", re.IGNORECASE),
}


@dataclass
class MatEntry:
    """描述扫描到的 .mat 文件。"""

    path: str
    basename: str
    label: Optional[str]
    sensor_hints: List[str]
    fs_hint: Optional[int]


def infer_label(path: Path) -> Optional[str]:
    """根据路径推断标签。"""

    parts = list(path.parts)
    parts.append(path.stem)
    for label, pattern in LABEL_PATTERNS.items():
        if any(pattern.search(part) for part in parts):
            return label
    return None


def infer_sensors(path: Path) -> List[str]:
    """根据路径推测可能的传感器。"""

    tokens = []
    for part in path.parts:
        part_lower = part.lower()
        if "de" in part_lower:
            tokens.append("DE")
        if "fe" in part_lower:
            tokens.append("FE")
        if "ba" in part_lower:
            tokens.append("BA")
    if "de" in path.stem.lower():
        tokens.append("DE")
    if "fe" in path.stem.lower():
        tokens.append("FE")
    if "ba" in path.stem.lower():
        tokens.append("BA")
    ordered = []
    for sensor in ["DE", "FE", "BA"]:
        if sensor in tokens and sensor not in ordered:
            ordered.append(sensor)
    return ordered


def infer_fs(path: Path) -> Optional[int]:
    """从路径推断采样率。"""

    parts = list(path.parts)
    parts.append(path.stem)
    for fs, pattern in FS_HINTS.items():
        if any(pattern.search(part) for part in parts):
            return fs
    return None


def scan_mat_dirs(dirs: List[str]) -> List[MatEntry]:
    """扫描目录返回 .mat 文件列表。"""

    entries: List[MatEntry] = []
    for directory in dirs:
        root = Path(directory)
        if not root.exists():
            continue
        for file in sorted(root.rglob("*.mat")):
            entries.append(
                MatEntry(
                    path=str(file.resolve()),
                    basename=file.stem,
                    label=infer_label(file),
                    sensor_hints=infer_sensors(file),
                    fs_hint=infer_fs(file),
                )
            )
    return entries

