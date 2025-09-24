"""信号加载与频谱计算工具。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import resample_poly, spectrogram

from q1_pipeline.io_utils import load_mat_safe
from q1_pipeline.preprocess import preprocess_segment
from q1_pipeline.resample import segment_signal
from q1_pipeline.envelope import envelope_spectrum
from q1_pipeline.bearing import bearing_char_freqs

from .config import VizConfig
from .selectors import SegmentSpec


@dataclass
class SignalBundle:
    """封装用于绘图的各类信号与频谱数据。"""

    spec: SegmentSpec
    fs_src: int
    fs_target: int
    rpm_used: float
    used_default_rpm: bool
    time_axis: np.ndarray
    raw_segment: np.ndarray
    processed_segment: np.ndarray
    fft_freqs: np.ndarray
    fft_raw: np.ndarray
    fft_processed: np.ndarray
    envelope_freqs: np.ndarray
    envelope_mag: np.ndarray
    spec_freqs: Optional[np.ndarray]
    spec_times: Optional[np.ndarray]
    spec_mag_db: Optional[np.ndarray]
    char_freqs: Dict[str, float]
    harmonics_fft: Dict[str, List[float]]
    harmonics_env: Dict[str, List[float]]


def _resample(signal: np.ndarray, fs_src: int, fs_target: int) -> np.ndarray:
    """使用分数重采样至目标采样率。"""

    if fs_src == fs_target:
        return signal.astype(float)
    from math import gcd

    g = gcd(fs_src, fs_target)
    up = fs_target // g
    down = fs_src // g
    return resample_poly(signal.astype(float), up, down)


def _window_fft(signal: np.ndarray, fs: int, nfft: int, window: str) -> tuple[np.ndarray, np.ndarray]:
    """计算带窗单边幅度谱。"""

    n = signal.size
    if window.lower() == "hann":
        win = np.hanning(n)
    else:
        win = np.ones(n)
    x = signal * win
    mag = np.abs(np.fft.rfft(x, n=nfft)) / max(np.sum(win) / len(win), 1e-9)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, mag


def _apply_fft_limit(freqs: np.ndarray, values: np.ndarray, max_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """裁剪频率范围。"""

    mask = freqs <= max_hz
    return freqs[mask], values[mask]


def _build_envelope_config(viz_cfg: VizConfig) -> Dict[str, Optional[float]]:
    """根据覆盖项构造 envelope 设置。"""

    override = {}
    if viz_cfg.envelope.window is not None:
        override["window"] = viz_cfg.envelope.window
    if viz_cfg.envelope.zero_pad is not None:
        override["zero_pad"] = viz_cfg.envelope.zero_pad
    if viz_cfg.envelope.fft_max_hz is not None:
        override["fft_max_hz"] = viz_cfg.envelope.fft_max_hz
    return override


def _harmonics(freq: float, max_order: int, limit: float) -> List[float]:
    """生成不超过指定频率的谐波列表。"""

    harmonics = []
    for k in range(1, max_order + 1):
        value = freq * k
        if value <= 0 or value > limit:
            break
        harmonics.append(value)
    return harmonics


def reconstruct_segment(spec: SegmentSpec, cfg: VizConfig) -> SignalBundle:
    """加载 .mat 文件、重采样并生成绘图所需数据。"""

    payload = load_mat_safe(spec.file_path)

    sensor = spec.sensor
    for candidate in spec.prefer_sensors:
        if payload.signal_map.get(candidate) is not None:
            sensor = candidate
            break
    if sensor is None:
        raise ValueError(f"文件 {spec.file_path} 缺少任何可用传感器通道")
    spec.sensor = sensor

    signal = payload.signal_map[sensor]
    if signal is None:
        raise ValueError(f"文件 {spec.file_path} 传感器 {sensor} 信号为空")

    fs_src = spec.fs_hint or (int(payload.fs_candidates[0]) if payload.fs_candidates else cfg.base_config.sampling.fs_target)
    fs_target = cfg.base_config.sampling.fs_target

    resampled = _resample(signal, fs_src, fs_target)

    seg_len = int(cfg.base_config.sampling.segment_seconds * fs_target)
    overlap = cfg.base_config.sampling.overlap
    segments = segment_signal(resampled, fs_target, cfg.base_config.sampling.segment_seconds, overlap, drop_tail=True)
    if not segments:
        raise ValueError(f"文件 {spec.file_path} 无法生成分段")
    seg_idx = min(spec.seg_idx, len(segments) - 1)
    raw_segment = segments[seg_idx].data.astype(float)

    preprocess_cfg = deepcopy(cfg.base_config.preprocess)
    processed_segment = preprocess_segment(raw_segment.copy(), fs_target, preprocess_cfg)

    time_axis = np.arange(raw_segment.size) / fs_target

    fft_freqs_full, fft_raw_full = _window_fft(raw_segment, fs_target, cfg.fft.nfft, cfg.fft.window)
    _, fft_proc_full = _window_fft(processed_segment, fs_target, cfg.fft.nfft, cfg.fft.window)
    fft_freqs, fft_raw = _apply_fft_limit(fft_freqs_full, fft_raw_full, cfg.fft.max_hz)
    _, fft_processed = _apply_fft_limit(fft_freqs_full, fft_proc_full, cfg.fft.max_hz)

    envelope_cfg = deepcopy(cfg.base_config.envelope)
    overrides = _build_envelope_config(cfg)
    for key, value in overrides.items():
        setattr(envelope_cfg, key, value)
    env_freqs_full, env_mag_full = envelope_spectrum(processed_segment, fs_target, envelope_cfg)
    env_limit = envelope_cfg.fft_max_hz or float(env_freqs_full[-1])
    env_mask = env_freqs_full <= env_limit
    env_freqs = env_freqs_full[env_mask]
    env_mag = env_mag_full[env_mask]

    rpm_array = payload.rpm
    if rpm_array is not None and np.size(rpm_array) > 0:
        rpm_used = float(np.mean(rpm_array))
        used_default = False
    else:
        rpm_used = cfg.base_config.bearing.rpm_default
        used_default = True
    spec.rpm_used = rpm_used

    bearing_params = {
        "n": cfg.base_config.bearing.n,
        "d": cfg.base_config.bearing.d,
        "D": cfg.base_config.bearing.D,
        "theta_deg": cfg.base_config.bearing.theta_deg,
    }
    char_freqs = bearing_char_freqs(rpm_used, bearing_params)

    spec_f, spec_t, spec_Sxx = spectrogram(
        processed_segment,
        fs=fs_target,
        window=cfg.fft.window,
        nperseg=cfg.spec.nperseg,
        noverlap=cfg.spec.noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    spec_mask = spec_f <= cfg.spec.max_hz
    spec_freqs = spec_f[spec_mask]
    spec_mag = np.abs(spec_Sxx[spec_mask, :])
    spec_mag_db = 20 * np.log10(spec_mag + 1e-12)

    harmonics_fft: Dict[str, List[float]] = {}
    harmonics_env: Dict[str, List[float]] = {}
    for key, freq in char_freqs.items():
        if key not in {"BPFI", "BPFO", "BSF", "FTF"}:
            continue
        harmonics_fft[key] = _harmonics(freq, 5, cfg.fft.max_hz)
        harmonics_env[key] = _harmonics(freq, 8, env_limit)

    return SignalBundle(
        spec=spec,
        fs_src=fs_src,
        fs_target=fs_target,
        rpm_used=rpm_used,
        used_default_rpm=used_default,
        time_axis=time_axis,
        raw_segment=raw_segment,
        processed_segment=processed_segment,
        fft_freqs=fft_freqs,
        fft_raw=fft_raw,
        fft_processed=fft_processed,
        envelope_freqs=env_freqs,
        envelope_mag=env_mag,
        spec_freqs=spec_freqs,
        spec_times=spec_t,
        spec_mag_db=spec_mag_db,
        char_freqs=char_freqs,
        harmonics_fft=harmonics_fft,
        harmonics_env=harmonics_env,
    )
