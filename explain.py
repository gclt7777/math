"""解释相关工具（当前提供占位实现）。"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def compute_shap_values(*args, **kwargs) -> Optional[pd.DataFrame]:
    """如启用 SHAP，可在此处扩展实现。默认返回 None。"""

    try:
        import shap  # noqa: F401
    except ImportError:  # pragma: no cover - shap 可选
        return None
    # 为简化，此处暂不实现具体 SHAP 逻辑
    return None

