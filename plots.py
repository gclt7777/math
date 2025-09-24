"""绘图函数集合，全部采用中文标题与注释。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

try:
    import umap
except ImportError:  # pragma: no cover - 可选依赖
    umap = None

from .config import FeatureVizConfig, PlotToggles
from .signal_utils import SignalBundle


LABEL_DISPLAY = {
    "BPFI": "内圈故障频率",
    "BPFO": "外圈故障频率",
    "BSF": "滚动体故障频率",
    "FTF": "保持架故障频率",
}

LABEL_COLORS = {
    "B": "#d73027",
    "IR": "#4575b4",
    "OR": "#66bd63",
    "N": "#fdae61",
}


@dataclass
class PlotRecord:
    """统一的绘图结果描述。"""

    figure: plt.Figure
    category: str
    filename: str
    description: str


def _add_freq_markers(ax, bundle: SignalBundle, harmonics: Dict[str, List[float]], max_hz: float) -> List[Line2D]:
    """绘制故障特征频率的虚线，并返回图例句柄。"""

    legend_handles: List[Line2D] = []
    label_color_map = {
        "BPFI": "#d62728",
        "BPFO": "#2ca02c",
        "BSF": "#9467bd",
        "FTF": "#8c564b",
    }

    for key, freqs in harmonics.items():
        color = label_color_map.get(key, "#333333")
        plotted = False
        for freq in freqs:
            if freq <= 0 or freq > max_hz:
                continue
            ax.axvline(freq, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
            plotted = True
        if plotted:
            display_name = LABEL_DISPLAY.get(key, key)
            legend_label = f"{key} / {display_name}"
            legend_handles.append(
                Line2D([0, 1], [0, 0], color=color, linestyle="--", linewidth=1.2, label=legend_label)
            )

    return legend_handles


def plot_time_waveform(bundle: SignalBundle) -> PlotRecord:
    """原始 / 预处理波形对比。"""

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(bundle.time_axis, bundle.raw_segment, color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("幅值")
    axes[0].set_title("原始振动信号")

    axes[1].plot(bundle.time_axis, bundle.processed_segment, color="#ff7f0e", linewidth=1.0)
    axes[1].set_ylabel("幅值")
    axes[1].set_xlabel("时间 (秒)")
    axes[1].set_title("预处理后振动信号")

    title = f"{bundle.spec.label}-{bundle.spec.sensor}-{bundle.spec.basename} - 原始/预处理时域对比"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    filename = f"time_{bundle.spec.label}_{bundle.spec.sensor}_{bundle.spec.basename}.png"
    return PlotRecord(fig, "signals", filename, "时域对比")


def plot_signal_summary(bundle: SignalBundle) -> PlotRecord:
    """生成包含时域、频域与包络概要的多子图。"""

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # 原始信号
    axes[0, 0].plot(bundle.time_axis, bundle.raw_segment, color="#1f77b4", linewidth=0.9)
    axes[0, 0].set_title("原始振动信号")
    axes[0, 0].set_xlabel("时间 (秒)")
    axes[0, 0].set_ylabel("幅值")
    axes[0, 0].grid(alpha=0.2)

    # 预处理信号
    axes[0, 1].plot(bundle.time_axis, bundle.processed_segment, color="#ff7f0e", linewidth=0.9)
    axes[0, 1].set_title("预处理后振动信号")
    axes[0, 1].set_xlabel("时间 (秒)")
    axes[0, 1].set_ylabel("幅值")
    axes[0, 1].grid(alpha=0.2)

    # 频谱 0-1000 Hz
    fft_vals = np.abs(np.fft.rfft(bundle.processed_segment))
    freqs = np.fft.rfftfreq(bundle.processed_segment.size, d=1.0 / bundle.fs_target)
    mask = freqs <= 1000
    axes[1, 0].plot(freqs[mask], fft_vals[mask], color="#2ca02c", linewidth=1.0)
    axes[1, 0].set_title("频谱 (0-1000 Hz)")
    axes[1, 0].set_xlabel("频率 (Hz)")
    axes[1, 0].set_ylabel("幅值")
    axes[1, 0].grid(alpha=0.2)
    char_markers = {
        name: [freq]
        for name, freq in bundle.char_freqs.items()
        if name in {"BPFI", "BPFO", "BSF", "FTF"} and 0 < freq <= 1000
    }
    handles_fft: List[Line2D] = []
    if char_markers:
        handles_fft = _add_freq_markers(axes[1, 0], bundle, char_markers, max_hz=1000.0)

    # 包络谱 0-600 Hz
    env_freqs = bundle.envelope_freqs
    env_mag = bundle.envelope_mag
    env_mask = env_freqs <= 600
    axes[1, 1].plot(env_freqs[env_mask], env_mag[env_mask], color="#d62728", linewidth=1.0)
    axes[1, 1].set_title("包络谱 (0-600 Hz)")
    axes[1, 1].set_xlabel("频率 (Hz)")
    axes[1, 1].set_ylabel("包络幅值")
    axes[1, 1].grid(alpha=0.2)
    handles_env: List[Line2D] = []
    env_markers = {
        name: [freq]
        for name, freq in bundle.char_freqs.items()
        if name in {"BPFI", "BPFO", "BSF", "FTF"} and 0 < freq <= 600
    }
    legend_env = None
    if env_markers:
        handles_env = _add_freq_markers(axes[1, 1], bundle, env_markers, max_hz=600.0)
        legend_env = fig.legend(
            handles_env,
            [handle.get_label() for handle in handles_env],
            loc="center right",
            bbox_to_anchor=(1.03, 0.5),
            frameon=False,
            title="故障特征",
        )

    fig.suptitle(
        f"{bundle.spec.label}-{bundle.spec.sensor}-{bundle.spec.basename} 信号综述",
        fontsize=13,
    )
    tight_rect = [0, 0, 0.84, 1] if legend_env else [0, 0, 1, 0.96]
    fig.tight_layout(rect=tight_rect)
    return PlotRecord(fig, "signals", f"summary_{bundle.spec.label}_{bundle.spec.sensor}_{bundle.spec.basename}.png", "信号综述")


def plot_frequency_comparison(bundle: SignalBundle) -> PlotRecord:
    """更深入的频域比较图，包含功率谱与包络谱。"""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    proc = bundle.processed_segment
    fs = bundle.fs_target
    n = proc.size
    fft_vals = np.abs(np.fft.rfft(proc))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = freqs <= 1500
    axes[0].plot(freqs[mask], fft_vals[mask], color="#1f77b4")
    axes[0].set_title("功率谱密度 (0-1.5kHz)")
    axes[0].set_xlabel("频率 (Hz)")
    axes[0].set_ylabel("幅值")
    axes[0].grid(alpha=0.2)
    fft_markers = {
        name: [freq]
        for name, freq in bundle.char_freqs.items()
        if name in {"BPFI", "BPFO", "BSF", "FTF"} and 0 < freq <= 1500
    }
    if fft_markers:
        _add_freq_markers(axes[0], bundle, fft_markers, max_hz=1500.0)

    env_freqs = bundle.envelope_freqs
    env_mag = bundle.envelope_mag
    env_mask = env_freqs <= 400
    axes[1].plot(env_freqs[env_mask], env_mag[env_mask], color="#ff7f0e")
    axes[1].set_title("包络谱 (0-400Hz)")
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("包络幅值")
    axes[1].grid(alpha=0.2)
    env_markers = {
        name: [freq]
        for name, freq in bundle.char_freqs.items()
        if name in {"BPFI", "BPFO", "BSF", "FTF"} and 0 < freq <= 400
    }
    legend_env = None
    if env_markers:
        handles_env = _add_freq_markers(axes[1], bundle, env_markers, max_hz=400.0)
        legend_env = fig.legend(
            handles_env,
            [handle.get_label() for handle in handles_env],
            loc="center right",
            bbox_to_anchor=(1.03, 0.5),
            frameon=False,
            title="故障特征",
        )

    fig.suptitle(f"{bundle.spec.label}-{bundle.spec.sensor}-{bundle.spec.basename} 频域综述", fontsize=13)
    tight_rect = [0, 0, 0.82, 0.95] if legend_env else [0, 0, 1, 0.95]
    fig.tight_layout(rect=tight_rect)
    return PlotRecord(fig, "spectra", f"freq_summary_{bundle.spec.label}_{bundle.spec.sensor}_{bundle.spec.basename}.png", "频域综述")


def plot_fft(bundle: SignalBundle, toggles: PlotToggles) -> PlotRecord:
    """幅度谱对比。"""

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(bundle.fft_freqs, bundle.fft_raw, label="原始信号", color="#1f77b4")
    ax.plot(bundle.fft_freqs, bundle.fft_processed, label="预处理信号", color="#ff7f0e")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度")
    max_hz = float(bundle.fft_freqs[-1]) if bundle.fft_freqs.size else 0.0
    ax.set_title(f"{bundle.spec.label}-{bundle.spec.sensor}-{bundle.spec.basename} - 频谱 (0-1000 Hz)")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, max_hz)

    freq_handles: List[Line2D] = []
    if toggles.char_freq_marks:
        freq_handles = _add_freq_markers(ax, bundle, bundle.harmonics_fft, max_hz=max_hz)

    if bundle.used_default_rpm:
        text = f"使用默认转速 rpm_default={bundle.rpm_used:.1f}"
        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=9, color="#555555",
                ha="left", va="top")

    handles, labels = ax.get_legend_handles_labels()
    signal_legend = ax.legend(handles, labels, loc="upper right")
    if freq_handles:
        freq_legend = ax.legend(
            freq_handles,
            [handle.get_label() for handle in freq_handles],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.5,
            frameon=False,
            title="故障特征",
        )
        ax.add_artist(signal_legend)

    if freq_handles:
        fig.tight_layout(rect=[0, 0, 0.8, 1])
    else:
        fig.tight_layout()
    filename = f"fft_{bundle.spec.label}_{bundle.spec.sensor}_{bundle.spec.basename}_0_1000.png"
    return PlotRecord(fig, "spectra", filename, "频谱对比")


def plot_envelope(bundle: SignalBundle) -> PlotRecord:
    """包络谱分析，突出故障频率及谐波。"""

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(bundle.envelope_freqs, bundle.envelope_mag, color="#d62728")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("包络幅值")
    ax.set_title(f"{bundle.spec.label}-{bundle.spec.sensor}-{bundle.spec.basename} - 包络谱")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, float(bundle.envelope_freqs[-1]) if bundle.envelope_freqs.size else 0.0)

    _add_freq_markers(ax, bundle, bundle.harmonics_env, max_hz=float(bundle.envelope_freqs[-1]))

    fig.tight_layout()
    filename = f"envelope_{bundle.spec.label}_{bundle.spec.sensor}_{bundle.spec.basename}.png"
    return PlotRecord(fig, "spectra", filename, "包络谱分析")


def plot_spectrogram(bundle: SignalBundle) -> Optional[PlotRecord]:
    """绘制时频谱图。"""

    if bundle.spec_freqs is None or bundle.spec_times is None or bundle.spec_mag_db is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.8))
    pcm = ax.pcolormesh(
        bundle.spec_times,
        bundle.spec_freqs,
        bundle.spec_mag_db,
        shading="auto",
        cmap="magma",
    )
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("频率 (Hz)")
    ax.set_title(f"{bundle.spec.label}-{bundle.spec.sensor}-{bundle.spec.basename} - 时频谱")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("幅度 (dB)")
    fig.tight_layout()

    filename = f"spectrogram_{bundle.spec.label}_{bundle.spec.sensor}_{bundle.spec.basename}.png"
    return PlotRecord(fig, "spectra", filename, "时频谱")


def plot_feature_distributions(df: pd.DataFrame, cfg: FeatureVizConfig, label_order: List[str]) -> List[PlotRecord]:
    """绘制关键特征按标签分布对比图。"""

    records: List[PlotRecord] = []
    features = _resolve_feature_columns(df, cfg)

    for feat in features:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label in label_order:
            subset = df[df["label"].eq(label)][feat]
            if subset.empty:
                continue
            ax.hist(
                subset,
                bins=40,
                alpha=0.4,
                density=True,
                label=label,
                color=LABEL_COLORS.get(label, None),
            )
        ax.set_title(f"{feat} 分布对比（按标签）")
        ax.set_xlabel(feat)
        ax.set_ylabel("频数")
        ax.legend(title="标签")
        fig.tight_layout()
        filename = f"features_{feat}_by_label.png"
        records.append(PlotRecord(fig, "features", filename, "特征分布"))

    energy_records = _plot_energy_bars(df, cfg, label_order)
    records.extend(energy_records)
    return records


def plot_feature_distribution_summary(df: pd.DataFrame, label_order: List[str]) -> Optional[PlotRecord]:
    """类似 notebook 的多特征对比摘要。"""

    if df.empty:
        return None

    keys = ["td_rms", "td_kurt", "td_crest", "td_peak"]
    available = [k for k in keys if k in df.columns]
    if not available:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    flat_axes = axes.flatten()
    used = 0
    for feature in available[:4]:
        ax = flat_axes[used]
        for label in label_order:
            subset = df[df["label"].eq(label)][feature]
            if subset.empty:
                continue
            ax.hist(subset, bins=40, alpha=0.55, density=False, label=label)
        ax.set_title(f"{feature} 分布")
        ax.set_xlabel(feature)
        ax.set_ylabel("概率密度")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        finite_vals = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite_vals.empty:
            upper = float(finite_vals.quantile(0.99))
            lower = float(finite_vals.quantile(0.01))
            if upper > lower:
                left = max(0.0, lower * 0.8)
                right = upper * 1.2
                ax.set_xlim(left, right)
        used += 1

    for idx in range(used, len(flat_axes)):
        fig.delaxes(flat_axes[idx])

    fig.suptitle("关键特征分布对比", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return PlotRecord(fig, "features", "summary_feature_distribution.png", "特征分布综述")


def plot_data_overview(df: pd.DataFrame) -> Optional[PlotRecord]:
    if df.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    if "label" in df.columns:
        counts = df["label"].value_counts()
        axes[0].bar(counts.index, counts.values, color="#4E79A7")
        axes[0].set_title("标签分布")
        axes[0].set_xlabel("标签")
        axes[0].set_ylabel("数量")
        for x, y in zip(counts.index, counts.values):
            axes[0].text(x, y + max(counts.values)*0.02, str(y), ha="center", fontsize=9)
        axes[0].grid(alpha=0.2)

    if "sensor" in df.columns:
        sensor_counts = df["sensor"].value_counts()
        axes[1].bar(sensor_counts.index, sensor_counts.values, color="#F28E2B")
        axes[1].set_title("传感器分布")
        axes[1].set_xlabel("传感器")
        axes[1].set_ylabel("数量")
        axes[1].grid(alpha=0.2)

    feature_cols = [col for col in df.columns if col.startswith("td_")]
    if feature_cols:
        box_data = df[feature_cols].fillna(0)
        axes[2].boxplot([box_data[col] for col in feature_cols], labels=feature_cols)
        axes[2].set_title("时域特征箱线图")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].grid(alpha=0.2)

    if feature_cols:
        corr = df[feature_cols].corr()
        im = axes[3].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[3].set_xticks(range(len(feature_cols)))
        axes[3].set_xticklabels(feature_cols, rotation=45)
        axes[3].set_yticks(range(len(feature_cols)))
        axes[3].set_yticklabels(feature_cols)
        axes[3].set_title("时域特征相关性")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle("数据概览与统计", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return PlotRecord(fig, "features", "data_overview.png", "数据概览")


def plot_feature_space(df: pd.DataFrame, use_umap: bool) -> Optional[PlotRecord]:
    if df.empty or "label" not in df.columns:
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.startswith("td_") or col.startswith("fd_")]
    if len(feature_cols) < 2:
        return None

    data = df[feature_cols].fillna(0.0)
    labels = df["label"].astype(str)

    if use_umap and umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=42)
        proj = reducer.fit_transform(data)
        method = "UMAP"
    else:
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(data)
        method = "PCA"

    fig, ax = plt.subplots(figsize=(7.5, 6))
    color_map = {label: plt.cm.tab10(i % 10) for i, label in enumerate(labels.unique())}
    for label in labels.unique():
        mask = labels == label
        ax.scatter(proj[mask, 0], proj[mask, 1], s=20, alpha=0.7, label=label, color=color_map[label])
    ax.set_title(f"{method} 嵌入下的源域特征分布")
    ax.set_xlabel("维度 1")
    ax.set_ylabel("维度 2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return PlotRecord(fig, "features", "feature_space_embedding.png", f"特征空间 ({method})")


def _resolve_feature_columns(df: pd.DataFrame, cfg: FeatureVizConfig) -> List[str]:
    """根据配置挑选时域特征列。"""

    selected: List[str] = []
    for feat in cfg.time_feats:
        if cfg.use_z_pref:
            z_name = f"z_{feat}" if not feat.startswith("z_") else feat
            if z_name in df.columns:
                selected.append(z_name)
                continue
        if feat in df.columns:
            selected.append(feat)
    return selected


def _plot_energy_bars(df: pd.DataFrame, cfg: FeatureVizConfig, label_order: List[str]) -> List[PlotRecord]:
    """绘制能量/峰值对比柱状图。"""

    records: List[PlotRecord] = []
    for key in cfg.mech_energy_keys:
        candidates = []
        if cfg.use_z_pref:
            candidates.append(f"z_{key}_R_total")
            candidates.append(f"z_{key}_peak")
        candidates.extend([f"{key}_R_total", f"{key}_peak"])
        column = next((c for c in candidates if c in df.columns), None)
        if column is None:
            continue
        values = []
        labels = []
        colors = []
        for label in label_order:
            subset = df[df["label"].eq(label)][column]
            if subset.empty:
                continue
            values.append(float(subset.mean()))
            labels.append(label)
            colors.append(LABEL_COLORS.get(label, None))
        if not values:
            continue

        value_arr = np.array(values, dtype=float)
        abs_values = np.abs(value_arr)
        sorted_abs = np.sort(abs_values)
        secondary_axis = False
        zoom_range = None
        if len(sorted_abs) >= 2 and sorted_abs[-1] > 5 * sorted_abs[-2] > 0:
            secondary_axis = True
            dominant_idx = int(np.argmax(abs_values))
            zoom_vals = np.delete(value_arr, dominant_idx)
            zoom_max = float(zoom_vals.max()) if zoom_vals.size else 0.0
            zoom_min = float(zoom_vals.min()) if zoom_vals.size else 0.0
            pad = max(abs(zoom_max), abs(zoom_min)) * 0.2 if zoom_vals.size else 1.0
            zoom_range = (zoom_min - pad, zoom_max + pad)

        if secondary_axis:
            fig = plt.figure(figsize=(8, 5.2))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax = fig.add_subplot(gs[0])
            ax_zoom = fig.add_subplot(gs[1], sharex=ax)
        else:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax_zoom = None

        ax.bar(labels, values, color=colors, width=0.6)
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylabel(column)
        ax.set_title(f"{column} 平均值对比（按标签）")

        max_val = float(value_arr.max())
        min_val = float(value_arr.min())
        pad_upper = max(0.2 * abs(max_val), 0.05)
        pad_lower = max(0.2 * abs(min_val), 0.05)
        ax.set_ylim(min_val - pad_lower, max_val + pad_upper)

        if ax_zoom is not None and zoom_range is not None:
            ax_zoom.bar(labels, values, color=colors, width=0.6)
            ax_zoom.axhline(0, color="#555555", linewidth=0.6, linestyle="--", alpha=0.6)
            ax_zoom.set_ylim(zoom_range)
            ax_zoom.set_ylabel("放大", fontsize=9)
            ax_zoom.tick_params(axis="x", labelrotation=0)
        fig.tight_layout()
        filename = f"features_{column}_avg_by_label.png"
        records.append(PlotRecord(fig, "features", filename, "能量/峰值对比"))
    return records


def plot_correlation_matrix(df: pd.DataFrame, cfg: FeatureVizConfig, label_order: List[str]) -> Optional[PlotRecord]:
    """绘制 Pearson 相关性热力图。"""

    columns = _resolve_feature_columns(df, cfg)
    for key in cfg.mech_energy_keys:
        candidates = []
        if cfg.use_z_pref:
            candidates.extend([f"z_{key}_R_total", f"z_{key}_peak"])
        candidates.extend([f"{key}_R_total", f"{key}_peak"])
        for col in candidates:
            if col in df.columns and col not in columns:
                columns.append(col)
    columns = [c for c in columns if c in df.columns]
    columns = list(dict.fromkeys(columns))
    if len(columns) < 2:
        return None

    corr = df[columns].corr(method="pearson").fillna(0.0)

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(columns)), max(6, 0.6 * len(columns))))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title("特征相关性矩阵")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson 相关系数")
    fig.tight_layout()

    return PlotRecord(fig, "features", "corr_matrix.png", "特征相关性矩阵")
