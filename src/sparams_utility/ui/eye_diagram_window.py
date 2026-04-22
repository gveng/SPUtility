from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from sparams_utility.circuit_solver import ChannelSimResult


DEFAULT_EYE_SPAN_UI = 2
EYE_SPAN_CHOICES = (1, 2, 3)
DEFAULT_RENDER_MODE = "Heatmap"
RENDER_MODE_CHOICES = ("Heatmap", "Heatmap + Lines", "Lines")
DEFAULT_QUALITY_PRESET = "Balanced"
QUALITY_PRESET_CHOICES = ("Fast", "Balanced", "HighRes")
DEFAULT_NOISE_RMS_MV: float = 0.0
DEFAULT_JITTER_RMS_PS: float = 0.0
DEFAULT_MAX_EYE_TRACES: int = 20000
DEFAULT_HEATMAP_X_BINS: int = 360
DEFAULT_HEATMAP_Y_BINS: int = 280


def _make_blue_yellow_lut(n: int = 256) -> np.ndarray:
    """Build an ADS-style heatmap LUT: white → blue → cyan → green → yellow."""
    # Colour stops (t, R, G, B) — tuned to match Keysight ADS eye palette.
    # Background stays white; low-density traces are blue; crossings are yellow.
    stops = [
        (0.000, 255, 255, 255),   # white  – zero / background
        (0.030, 255, 255, 255),   # white  – keep background clean
        (0.080,  20,  40, 200),   # medium blue  (sparse traces)
        (0.200,   0, 100, 255),   # bright blue
        (0.360,   0, 210, 240),   # cyan
        (0.500,   0, 220, 180),   # cyan-green
        (0.640,  30, 220,  60),   # green
        (0.790, 180, 240,   0),   # yellow-green
        (0.920, 255, 240,   0),   # yellow
        (1.000, 255, 255,  80),   # bright yellow (peak)
    ]
    lut = np.zeros((n, 4), dtype=np.uint8)
    for i in range(n):
        t = i / max(n - 1, 1)
        # Find bracketing stops and interpolate
        for j in range(len(stops) - 1):
            t0, r0, g0, b0 = stops[j]
            t1, r1, g1, b1 = stops[j + 1]
            if t <= t1:
                s = (t - t0) / max(t1 - t0, 1e-9)
                lut[i] = [
                    int(r0 + (r1 - r0) * s),
                    int(g0 + (g1 - g0) * s),
                    int(b0 + (b1 - b0) * s),
                    255,
                ]
                break
    return lut


def _normalize_eye_span_ui(span_ui: int) -> int:
    return span_ui if span_ui in EYE_SPAN_CHOICES else DEFAULT_EYE_SPAN_UI


def _normalize_render_mode(render_mode: str) -> str:
    return render_mode if render_mode in RENDER_MODE_CHOICES else DEFAULT_RENDER_MODE


def _normalize_quality_preset(quality_preset: str) -> str:
    return quality_preset if quality_preset in QUALITY_PRESET_CHOICES else DEFAULT_QUALITY_PRESET


def _eye_render_config(quality_preset: str, n_workers: int) -> dict[str, float | int]:
    quality = _normalize_quality_preset(quality_preset)
    if quality == "Fast":
        return {
            "max_traces": 6000,
            "x_bins": 220,
            "y_bins": 180,
            "smooth_sigma": 0.9,
            "max_line_traces": 800,
        }
    if quality == "HighRes":
        if n_workers >= 16:
            return {
                "max_traces": 100000,
                "x_bins": 800,
                "y_bins": 600,
                "smooth_sigma": 1.4,
                "max_line_traces": 7000,
            }
        if n_workers >= 12:
            return {
                "max_traces": 80000,
                "x_bins": 720,
                "y_bins": 560,
                "smooth_sigma": 1.3,
                "max_line_traces": 5600,
            }
        return {
            "max_traces": 60000,
            "x_bins": 640,
            "y_bins": 480,
            "smooth_sigma": 1.2,
            "max_line_traces": 4200,
        }
    return {
        "max_traces": DEFAULT_MAX_EYE_TRACES,
        "x_bins": DEFAULT_HEATMAP_X_BINS,
        "y_bins": DEFAULT_HEATMAP_Y_BINS,
        "smooth_sigma": 1.1,
        "max_line_traces": 2000,
    }


def _expected_eye_levels_for_encoding(encoding: str) -> int:
    return 4 if encoding == "PAM4" else 2


def _diagnostic_style_for_score(score: float) -> tuple[str, str]:
    if not np.isfinite(score):
        return "LOW", "#dc2626"
    if score >= 6.0:
        return "HIGH", "#15803d"
    if score >= 3.0:
        return "MEDIUM", "#b45309"
    return "LOW", "#dc2626"


def _decision_marker_positions(span_ui: int) -> list[float]:
    half_span_ui = span_ui / 2.0
    min_marker = int(np.ceil(-half_span_ui))
    max_marker = int(np.floor(half_span_ui))
    return [float(marker) for marker in range(min_marker, max_marker + 1)]


def _compute_voltage_limits(segments: np.ndarray) -> tuple[float, float]:
    low = float(np.min(segments))
    high = float(np.max(segments))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(segments))
        high = float(np.max(segments))
    if high <= low:
        delta = max(1e-3, abs(high) * 0.1 + 1e-3)
        return low - delta, high + delta
    pad = 0.03 * (high - low)
    return low - pad, high + pad


def _build_eye_density(
    segments: np.ndarray,
    time_axis_ui: np.ndarray,
    x_bins: int = DEFAULT_HEATMAP_X_BINS,
    y_bins: int = DEFAULT_HEATMAP_Y_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.tile(time_axis_ui, segments.shape[0])
    y_values = segments.ravel()

    x_edges = np.linspace(float(time_axis_ui[0]), float(time_axis_ui[-1]), x_bins + 1)
    y_min, y_max = _compute_voltage_limits(segments)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)

    density, _, _ = np.histogram2d(x_values, y_values, bins=[x_edges, y_edges])
    return density, x_edges, y_edges


def _chunk_slices(n_items: int, n_chunks: int) -> list[slice]:
    if n_items <= 0:
        return []
    n_chunks = max(1, min(n_chunks, n_items))
    base = n_items // n_chunks
    extra = n_items % n_chunks
    out: list[slice] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        stop = start + size
        out.append(slice(start, stop))
        start = stop
    return out


def _collect_segments_no_jitter_parallel(
    waveform: np.ndarray,
    positions: np.ndarray,
    overlay_samples: int,
    max_traces: int,
    n_workers: int,
) -> np.ndarray:
    if positions.size == 0:
        return np.empty((0, overlay_samples), dtype=float)
    positions = positions[:max_traces]
    if positions.size == 0:
        return np.empty((0, overlay_samples), dtype=float)

    idx_offsets = np.arange(overlay_samples, dtype=np.int64)

    def _worker(chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.empty((0, overlay_samples), dtype=float)
        idx = chunk[:, None] + idx_offsets[None, :]
        return waveform[idx]

    if n_workers <= 1 or positions.size < 2048:
        return _worker(positions.astype(np.int64, copy=False)).astype(float, copy=False)

    slices = _chunk_slices(int(positions.size), n_workers)
    chunks = [positions[s].astype(np.int64, copy=False) for s in slices]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        mats = list(ex.map(_worker, chunks))
    mats = [m for m in mats if m.size]
    if not mats:
        return np.empty((0, overlay_samples), dtype=float)
    return np.vstack(mats).astype(float, copy=False)


def _build_eye_density_parallel(
    segments: np.ndarray,
    time_axis_ui: np.ndarray,
    x_bins: int,
    y_bins: int,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(float(time_axis_ui[0]), float(time_axis_ui[-1]), x_bins + 1)
    y_min, y_max = _compute_voltage_limits(segments)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)

    n_rows = int(segments.shape[0])
    if n_workers <= 1 or n_rows < 1500:
        density, _, _ = np.histogram2d(
            np.tile(time_axis_ui, n_rows),
            segments.ravel(),
            bins=[x_edges, y_edges],
        )
        return density, x_edges, y_edges

    def _worker(chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.zeros((x_bins, y_bins), dtype=float)
        x_vals = np.tile(time_axis_ui, chunk.shape[0])
        y_vals = chunk.ravel()
        d, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])
        return d

    slices = _chunk_slices(n_rows, n_workers)
    chunks = [segments[s, :] for s in slices]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_worker, chunks))
    density = np.sum(parts, axis=0)
    return density, x_edges, y_edges


def _align_to_ui_boundary(start_sample: int, samples_per_ui: int) -> int:
    if samples_per_ui <= 1:
        return start_sample
    phase = start_sample % samples_per_ui
    if phase == 0:
        return start_sample
    return start_sample + (samples_per_ui - phase)


def _build_eye_time_axis(samples_per_ui: int, span_ui: int) -> np.ndarray:
    overlay_samples = max(2, int(span_ui * samples_per_ui))
    return np.arange(overlay_samples, dtype=float) / float(samples_per_ui) - (span_ui / 2.0)


def _score_eye_phase(
    waveform: np.ndarray,
    start_sample: int,
    end_sample: int,
    samples_per_ui: int,
    overlay_samples: int,
    expected_levels: int = 2,
) -> float:
    center_index = overlay_samples // 2
    radius = min(2, max(0, samples_per_ui // 8))
    positions = np.arange(start_sample, end_sample - overlay_samples + 1, samples_per_ui, dtype=int)
    required_segments = max(8, expected_levels * 4)
    if positions.size < required_segments:
        return float("-inf")

    centers = np.empty(positions.size, dtype=float)
    for index, position in enumerate(positions):
        center_start = max(position + center_index - radius, position)
        center_stop = min(position + center_index + radius + 1, position + overlay_samples)
        centers[index] = float(np.mean(waveform[center_start:center_stop]))

    centers.sort()
    gaps = np.diff(centers)
    if gaps.size == 0:
        return float("-inf")

    max_levels = min(max(2, int(expected_levels)), centers.size // 2)
    best_score = float("-inf")
    for level_count in range(max_levels, 1, -1):
        split_count = min(level_count - 1, gaps.size)
        split_indices = np.argpartition(gaps, -split_count)[-split_count:]
        split_indices = np.sort(split_indices)
        clusters = np.split(centers, split_indices + 1)
        if len(clusters) < level_count:
            continue

        cluster_sizes = np.array([cluster.size for cluster in clusters], dtype=float)
        if np.any(cluster_sizes < 2):
            continue

        balance_ratio = float(np.min(cluster_sizes) / float(centers.size))
        min_balance = 0.20 if level_count <= 2 else 0.08
        if balance_ratio < min_balance:
            continue

        medians = np.array([float(np.median(cluster)) for cluster in clusters], dtype=float)
        rail_gaps = np.diff(medians)
        if rail_gaps.size == 0 or np.any(rail_gaps <= 0):
            continue

        gap_strength = float(np.mean(rail_gaps))
        cluster_mads = np.array(
            [float(np.median(np.abs(cluster - np.median(cluster)))) for cluster in clusters],
            dtype=float,
        )
        noise = float(np.mean(cluster_mads))
        score = balance_ratio * gap_strength / max(noise, 1e-12)
        if score > best_score:
            best_score = score

    return best_score


def _find_best_eye_phase_and_score(
    waveform: np.ndarray,
    start_sample: int,
    end_sample: int,
    samples_per_ui: int,
    overlay_samples: int,
    expected_levels: int = 2,
) -> tuple[int, float]:
    aligned_start = _align_to_ui_boundary(start_sample, samples_per_ui)
    best_start = aligned_start
    best_score = float("-inf")

    # Fallback score: maximize center-sample spread when strict clustering is inconclusive.
    fallback_best_start = aligned_start
    fallback_best_score = float("-inf")
    center_index = overlay_samples // 2
    radius = min(2, max(0, samples_per_ui // 8))

    for phase in range(samples_per_ui):
        phase_start = aligned_start + phase
        score = _score_eye_phase(
            waveform,
            phase_start,
            end_sample,
            samples_per_ui,
            overlay_samples,
            expected_levels=expected_levels,
        )
        if score > best_score:
            best_score = score
            best_start = phase_start

        positions = np.arange(phase_start, end_sample - overlay_samples + 1, samples_per_ui, dtype=int)
        if positions.size >= 8:
            centers = np.empty(positions.size, dtype=float)
            for index, position in enumerate(positions):
                center_start = max(position + center_index - radius, position)
                center_stop = min(position + center_index + radius + 1, position + overlay_samples)
                centers[index] = float(np.mean(waveform[center_start:center_stop]))
            spread = float(np.percentile(centers, 90.0) - np.percentile(centers, 10.0))
            if spread > fallback_best_score:
                fallback_best_score = spread
                fallback_best_start = phase_start

    if not np.isfinite(best_score):
        return fallback_best_start, fallback_best_score

    return best_start, best_score


def _find_best_eye_phase(
    waveform: np.ndarray,
    start_sample: int,
    end_sample: int,
    samples_per_ui: int,
    overlay_samples: int,
    expected_levels: int = 2,
) -> int:
    best_start, _ = _find_best_eye_phase_and_score(
        waveform,
        start_sample,
        end_sample,
        samples_per_ui,
        overlay_samples,
        expected_levels=expected_levels,
    )
    return best_start


def _compute_eye_summary(
    segment_matrix: np.ndarray,
    samples_per_ui: int,
) -> dict[str, float]:
    """Compute NRZ eye summary using oscilloscope-style definitions.

    Reported values:
    * One Level / Zero Level: medians of upper/lower clusters at decision center.
    * Eye Amplitude: One Level - Zero Level.
    * Eye Height: vertical opening at decision center using percentile envelopes.
    * Eye Width: contiguous open interval around center, in UI.
    * Eye Crossing: crossing level at +/-0.5 UI, normalized to amplitude (%).
    """
    n_seg, n_samp = segment_matrix.shape

    def _nan_summary() -> dict[str, float]:
        nan = float("nan")
        # Keep legacy aliases (level1/level0/height/width) for compatibility.
        return {
            "one_level": nan,
            "zero_level": nan,
            "eye_amplitude": nan,
            "eye_height": nan,
            "eye_width_ui": nan,
            "eye_crossing_pct": nan,
            "level1": nan,
            "level0": nan,
            "height": nan,
            "width": nan,
        }

    if n_seg < 4 or n_samp < 2:
        return _nan_summary()

    # ── Vertical measurement at the UI centre ──────────────────────────────
    c_radius = max(1, samples_per_ui // 8)
    ci = n_samp // 2
    c_start = max(0, ci - c_radius)
    c_stop = min(n_samp, ci + c_radius + 1)
    center_vals = segment_matrix[:, c_start:c_stop].mean(axis=1)
    center_vals_sorted = np.sort(center_vals)

    # Split into two clusters at the largest gap
    gaps = np.diff(center_vals_sorted)
    split_idx = int(np.argmax(gaps)) + 1
    cluster0 = center_vals_sorted[:split_idx]
    cluster1 = center_vals_sorted[split_idx:]
    if cluster0.size < 2 or cluster1.size < 2:
        return _nan_summary()

    one_level = float(np.median(cluster1))
    zero_level = float(np.median(cluster0))
    eye_amplitude = one_level - zero_level

    # ── Horizontal measurement (eye width) ─────────────────────────────────
    # For each time sample compute the p5 (low envelope) and p95 (high envelope)
    # Eye is "open" at a column when p5 of upper cluster > p95 of lower cluster
    threshold = (one_level + zero_level) / 2.0
    upper_mask = center_vals >= threshold
    lower_mask = ~upper_mask

    upper_segs = segment_matrix[upper_mask]
    lower_segs = segment_matrix[lower_mask]

    if upper_segs.shape[0] < 2 or lower_segs.shape[0] < 2:
        eye_height = float("nan")
        eye_width_ui = float("nan")
    else:
        # per-column 5th/95th percentile
        p5_upper = np.percentile(upper_segs, 5.0, axis=0)
        p95_lower = np.percentile(lower_segs, 95.0, axis=0)
        open_cols = p5_upper > p95_lower   # True where eye is open

        # Height at decision centre (green arrow in reference figure).
        center_col = n_samp // 2
        eye_height = float(p5_upper[center_col] - p95_lower[center_col])

        # Width of the central eye: contiguous open interval around centre
        # (red arrow in reference figure), expressed in UI.
        if not bool(open_cols[center_col]):
            eye_width_ui = 0.0
        else:
            left = center_col
            while left > 0 and bool(open_cols[left - 1]):
                left -= 1
            right = center_col
            while right < (n_samp - 1) and bool(open_cols[right + 1]):
                right += 1
            open_count = right - left + 1
            eye_width_ui = float(open_count) / float(max(samples_per_ui, 1))

    eye_crossing_pct = float("nan")
    if np.isfinite(eye_amplitude) and eye_amplitude > 0.0 and n_samp >= 3:
        activity = _crossing_activity_profile(segment_matrix)
        center_col = n_samp // 2
        half_ui_samples = max(1, int(round(samples_per_ui / 2.0)))
        search_radius = max(2, samples_per_ui // 3)
        crossing_levels: list[float] = []
        for target in (center_col - half_ui_samples, center_col + half_ui_samples):
            lo = max(0, target - search_radius)
            hi = min(n_samp, target + search_radius + 1)
            if hi <= lo:
                continue
            peak = lo + int(np.argmax(activity[lo:hi]))

            left_col = max(0, peak - 1)
            right_col = min(n_samp - 1, peak + 1)
            slope_mag = np.abs(segment_matrix[:, right_col] - segment_matrix[:, left_col])
            q = float(np.percentile(slope_mag, 75.0))
            transition_mask = slope_mag >= q
            if int(np.count_nonzero(transition_mask)) < 4:
                transition_mask = slope_mag > 0.0
            values = segment_matrix[transition_mask, peak]
            if values.size == 0:
                values = segment_matrix[:, peak]
            crossing_levels.append(float(np.median(values)))

        if crossing_levels:
            crossing_level_v = float(np.mean(crossing_levels))
            eye_crossing_pct = float(np.clip(
                100.0 * (crossing_level_v - zero_level) / eye_amplitude,
                0.0,
                100.0,
            ))

    return {
        "one_level": one_level,
        "zero_level": zero_level,
        "eye_amplitude": eye_amplitude,
        "eye_height": eye_height,
        "eye_width_ui": eye_width_ui,
        "eye_crossing_pct": eye_crossing_pct,
        "level1": one_level,
        "level0": zero_level,
        "height": eye_height,
        "width": eye_width_ui,
    }


def _eye_opening_profile(segment_matrix: np.ndarray) -> np.ndarray:
    """Return per-column opening estimate as the largest vertical gap.

    For each time column, sort all trace voltages and use the maximum adjacent
    gap as the eye opening estimator. This is robust for NRZ and works well to
    locate the decision region center.
    """
    n_seg, n_samp = segment_matrix.shape
    if n_seg < 4 or n_samp < 2:
        return np.zeros(n_samp, dtype=float)

    # Vectorized per-column max gap (much faster than Python loop).
    sorted_cols = np.sort(segment_matrix, axis=0)
    gaps = np.diff(sorted_cols, axis=0)
    return np.max(gaps, axis=0)


def _best_opening_index_near_center(opening: np.ndarray) -> int:
    if opening.size == 0:
        return 0
    center = opening.size // 2
    max_open = float(np.max(opening))
    if not np.isfinite(max_open) or max_open <= 0.0:
        return center

    # Keep near-maximum candidates, then choose the one closest to center.
    threshold = max_open * 0.98
    candidates = np.where(opening >= threshold)[0]
    if candidates.size == 0:
        return int(np.argmax(opening))
    nearest = int(candidates[np.argmin(np.abs(candidates - center))])
    return nearest


def _crossing_activity_profile(segment_matrix: np.ndarray) -> np.ndarray:
    """Return per-column transition activity used to locate crossing points.

    High values indicate columns where many traces change quickly in voltage,
    which corresponds to eye crossings for NRZ/PAM signals.
    """
    n_seg, n_samp = segment_matrix.shape
    if n_seg < 2 or n_samp < 3:
        return np.zeros(n_samp, dtype=float)

    slope = np.abs(np.diff(segment_matrix, axis=1))
    activity = np.zeros(n_samp, dtype=float)
    activity[1:] = np.mean(slope, axis=0)
    return activity


def _estimate_crossing_phase_shift_samples(
    segment_matrix: np.ndarray,
    samples_per_ui: int,
    span_ui: int,
) -> int:
    """Estimate sample shift so crossings land at +/-0.5 UI for 2-UI span."""
    if span_ui != 2 or samples_per_ui <= 1:
        return 0

    activity = _crossing_activity_profile(segment_matrix)
    if activity.size < 3 or not np.isfinite(activity).any():
        return 0

    center_idx = activity.size // 2
    half_ui_samples = max(1, int(round(samples_per_ui / 2.0)))
    target_left = center_idx - half_ui_samples
    target_right = center_idx + half_ui_samples

    search_radius = max(2, samples_per_ui // 3)

    left_lo = max(0, target_left - search_radius)
    left_hi = min(activity.size, target_left + search_radius + 1)
    right_lo = max(0, target_right - search_radius)
    right_hi = min(activity.size, target_right + search_radius + 1)
    if left_hi <= left_lo or right_hi <= right_lo:
        return 0

    left_peak = left_lo + int(np.argmax(activity[left_lo:left_hi]))
    right_peak = right_lo + int(np.argmax(activity[right_lo:right_hi]))

    left_err = left_peak - target_left
    right_err = right_peak - target_right
    shift = int(round((left_err + right_err) / 2.0))

    max_shift = max(1, samples_per_ui // 2)
    return int(np.clip(shift, -max_shift, max_shift))


class EyeDiagramWindow(QMainWindow):
    span_changed = Signal(int)
    render_mode_changed = Signal(str)
    quality_preset_changed = Signal(str)

    def __init__(
        self,
        result: ChannelSimResult,
        title: str = "Eye Diagram",
        parent=None,
        initial_span_ui: int = DEFAULT_EYE_SPAN_UI,
        initial_render_mode: str = DEFAULT_RENDER_MODE,
        initial_quality_preset: str = DEFAULT_QUALITY_PRESET,
        statistical_enabled: bool = False,
        noise_rms_mv: float = DEFAULT_NOISE_RMS_MV,
        jitter_rms_ps: float = DEFAULT_JITTER_RMS_PS,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        app = QApplication.instance()
        if app is not None:
            self.setWindowIcon(app.windowIcon())

        self._result = result
        self._eye_span_ui = _normalize_eye_span_ui(int(initial_span_ui))
        self._render_mode = _normalize_render_mode(str(initial_render_mode))
        self._quality_preset = _normalize_quality_preset(str(initial_quality_preset))
        self._statistical_enabled: bool = bool(statistical_enabled)
        self._noise_rms_mv: float = float(noise_rms_mv)
        self._jitter_rms_ps: float = float(jitter_rms_ps)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        info_layout = QHBoxLayout()
        spec = result.driver_spec
        mode = "Differential" if result.is_differential else "Single-ended"
        info_text = (
            f"{mode} | Bitrate: {spec.bitrate_gbps} Gbps | "
            f"Pattern: {spec.prbs_pattern} | Encoding: {spec.encoding} | "
            f"Rise: {spec.rise_time_s * 1e12:.1f} ps | Fall: {spec.fall_time_s * 1e12:.1f} ps | "
            f"Bits: {spec.num_bits}"
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        self._eye_span_combo = QComboBox()
        for span_ui in EYE_SPAN_CHOICES:
            self._eye_span_combo.addItem(f"{span_ui} UI", span_ui)
        combo_index = self._eye_span_combo.findData(self._eye_span_ui)
        if combo_index >= 0:
            self._eye_span_combo.setCurrentIndex(combo_index)
        self._eye_span_combo.currentIndexChanged.connect(self._on_eye_span_changed)
        info_layout.addWidget(QLabel("Span"))
        info_layout.addWidget(self._eye_span_combo)

        self._render_mode_combo = QComboBox()
        for mode in RENDER_MODE_CHOICES:
            self._render_mode_combo.addItem(mode, mode)
        render_mode_index = self._render_mode_combo.findData(self._render_mode)
        if render_mode_index >= 0:
            self._render_mode_combo.setCurrentIndex(render_mode_index)
        self._render_mode_combo.currentIndexChanged.connect(self._on_render_mode_changed)
        info_layout.addWidget(QLabel("Render"))
        info_layout.addWidget(self._render_mode_combo)

        self._quality_preset_combo = QComboBox()
        for preset in QUALITY_PRESET_CHOICES:
            self._quality_preset_combo.addItem(preset, preset)
        quality_index = self._quality_preset_combo.findData(self._quality_preset)
        if quality_index >= 0:
            self._quality_preset_combo.setCurrentIndex(quality_index)
        self._quality_preset_combo.currentIndexChanged.connect(self._on_quality_preset_changed)
        info_layout.addWidget(QLabel("Quality"))
        info_layout.addWidget(self._quality_preset_combo)
        layout.addLayout(info_layout)

        self._diagnostics_label = QLabel()
        self._diagnostics_label.setWordWrap(True)
        layout.addWidget(self._diagnostics_label)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet(
            "QLabel { "
            "font-family: Consolas, 'Courier New', monospace; "
            "font-weight: 700; "
            "font-size: 12px; "
            "color: #052e16; "
            "background: #ecfdf5; "
            "border: 1px solid #16a34a; "
            "border-radius: 4px; "
            "padding: 4px 8px; "
            "}"
        )
        self._summary_label.setMinimumHeight(26)
        layout.addWidget(self._summary_label)
        self._eye_summary: dict[str, float] = {}

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.22)
        self._plot_widget.setLabel("bottom", "time, UI", color="#cc1a00", size="11pt")
        self._plot_widget.setLabel("left", "Density", color="#cc1a00", size="11pt")
        self._plot_widget.setTitle("Eye Diagram")
        layout.addWidget(self._plot_widget)

        self._draw_eye_diagram()

    @property
    def eye_summary(self) -> dict[str, float]:
        """Return last-computed eye summary.

        Keys:
        - one_level, zero_level, eye_amplitude, eye_height: volts
        - eye_width_ui: UI
        - eye_width_ps, bit_period_ps: picoseconds
        - eye_crossing_pct: percent

        Legacy aliases are kept for compatibility:
        - level1, level0, height, width, width_ps
        """
        return dict(self._eye_summary)

    def export_report_state(self) -> dict[str, object]:
        """Return eye diagram settings and measurements for report export."""
        spec = self._result.driver_spec
        return {
            "window_title": self.windowTitle(),
            "mode": "Differential" if self._result.is_differential else "Single-ended",
            "bitrate_gbps": float(spec.bitrate_gbps),
            "encoding": str(spec.encoding),
            "prbs_pattern": str(spec.prbs_pattern),
            "num_bits": int(spec.num_bits),
            "rise_time_ps": float(spec.rise_time_s) * 1e12,
            "fall_time_ps": float(spec.fall_time_s) * 1e12,
            "voltage_high_v": float(spec.voltage_high_v),
            "voltage_low_v": float(spec.voltage_low_v),
            "eye_span_ui": int(self._eye_span_ui),
            "render_mode": str(self._render_mode),
            "quality_preset": str(self._quality_preset),
            "statistical_enabled": bool(self._statistical_enabled),
            "noise_rms_mv": float(self._noise_rms_mv),
            "jitter_rms_ps": float(self._jitter_rms_ps),
            "measurements": dict(self._eye_summary),
        }

    def grab_eye_plot_pixmap(self):
        """Return a snapshot of the eye plot area."""
        return self._plot_widget.grab()

    def _on_eye_span_changed(self, index: int) -> None:
        span_ui = self._eye_span_combo.itemData(index)
        if span_ui is None:
            return
        self._eye_span_ui = _normalize_eye_span_ui(int(span_ui))
        self.span_changed.emit(self._eye_span_ui)
        self._draw_eye_diagram()

    def _on_render_mode_changed(self, index: int) -> None:
        render_mode = self._render_mode_combo.itemData(index)
        if render_mode is None:
            return
        self._render_mode = _normalize_render_mode(str(render_mode))
        self.render_mode_changed.emit(self._render_mode)
        self._draw_eye_diagram()

    def _on_quality_preset_changed(self, index: int) -> None:
        quality_preset = self._quality_preset_combo.itemData(index)
        if quality_preset is None:
            return
        self._quality_preset = _normalize_quality_preset(str(quality_preset))
        self.quality_preset_changed.emit(self._quality_preset)
        self._draw_eye_diagram()

    def _draw_eye_diagram(self) -> None:
        self._plot_widget.clear()
        result = self._result
        ui_s = result.ui_s
        dt = result.time_s[1] - result.time_s[0] if len(result.time_s) > 1 else 1e-12
        samples_per_ui = max(1, int(round(ui_s / dt)))
        n_workers = max(1, os.cpu_count() or 1)
        render_cfg = _eye_render_config(self._quality_preset, n_workers)
        max_eye_traces = int(render_cfg["max_traces"])

        waveform = result.waveform_v
        total_samples = len(waveform)

        # Discard first and last 5% to avoid transient edges
        margin = int(total_samples * 0.05)
        end_sample = total_samples - margin

        # Overlay a sample-accurate UI window centered on the optimal decision phase.
        time_axis_ui = _build_eye_time_axis(samples_per_ui, self._eye_span_ui)
        overlay_samples = len(time_axis_ui)
        expected_levels = _expected_eye_levels_for_encoding(result.driver_spec.encoding)
        start_sample, phase_score = _find_best_eye_phase_and_score(
            waveform,
            margin,
            end_sample,
            samples_per_ui,
            overlay_samples,
            expected_levels=expected_levels,
        )

        _use_jitter = self._statistical_enabled and self._jitter_rms_ps > 0.0
        _jitter_sigma_samples = (self._jitter_rms_ps * 1e-12 / dt) if _use_jitter else 0.0
        _rng = np.random.default_rng() if (self._statistical_enabled and (self._jitter_rms_ps > 0.0 or self._noise_rms_mv > 0.0)) else None

        def _collect_segments(start_at: int) -> tuple[np.ndarray, int]:
            traces = 0
            max_start = end_sample - overlay_samples
            if max_start < start_at:
                return np.empty((0, overlay_samples), dtype=float), 0

            positions = np.arange(start_at, max_start + 1, samples_per_ui, dtype=np.int64)
            if positions.size == 0:
                return np.empty((0, overlay_samples), dtype=float), 0

            if _use_jitter and _rng is not None and _jitter_sigma_samples > 0.0:
                positions = positions[:max_eye_traces]
                jitter = _rng.normal(0.0, _jitter_sigma_samples, size=positions.size).astype(np.int64)
                actual = positions + jitter
                actual = np.clip(actual, 0, max_start)
                idx_offsets = np.arange(overlay_samples, dtype=np.int64)
                idx = actual[:, None] + idx_offsets[None, :]
                segs = waveform[idx].astype(float, copy=False)
                traces = int(segs.shape[0])
                return segs, traces

            segs = _collect_segments_no_jitter_parallel(
                waveform,
                positions,
                overlay_samples,
                max_eye_traces,
                n_workers,
            )
            traces = int(segs.shape[0])
            return segs, traces

        segment_matrix, traces_drawn = _collect_segments(start_sample)

        if traces_drawn == 0:
            self._diagnostics_label.setText("Diagnostics: insufficient data to render eye diagram.")
            self._summary_label.setText("")
            self._eye_summary = {}
            self._plot_widget.setTitle("Eye Diagram (insufficient data)")
            return

        # Align crossings around -0.5 UI and +0.5 UI for the 2-UI display.
        phase_shift_samples = _estimate_crossing_phase_shift_samples(
            segment_matrix,
            samples_per_ui,
            self._eye_span_ui,
        )
        if phase_shift_samples != 0:
            start_sample = max(margin, min(start_sample + phase_shift_samples, end_sample - overlay_samples))
            segment_matrix, traces_drawn = _collect_segments(start_sample)

        if self._statistical_enabled and self._noise_rms_mv > 0.0 and _rng is not None:
            segment_matrix = segment_matrix + _rng.normal(0.0, self._noise_rms_mv * 1e-3, segment_matrix.shape)
        render_heatmap = self._render_mode in {"Heatmap", "Heatmap + Lines"}
        render_lines = self._render_mode in {"Lines", "Heatmap + Lines"}

        y_min: float
        y_max: float
        if render_heatmap:
            density, x_edges, y_edges = _build_eye_density_parallel(
                segment_matrix,
                time_axis_ui,
                x_bins=int(render_cfg["x_bins"]),
                y_bins=int(render_cfg["y_bins"]),
                n_workers=n_workers,
            )
            # log1p gives a smooth gradient from sparse (blue) to dense (yellow).
            # Power-law ^0.35 was making every non-zero bin map to maximum (binary look).
            density_image = np.log1p(density)
            density_image = gaussian_filter(density_image, sigma=float(render_cfg["smooth_sigma"]))

            # Clip top 0.5% so bright crossing pixels don't wash out the colour scale.
            foreground = density_image[density_image > 0]
            if foreground.size > 0:
                p_high = float(np.percentile(foreground, 99.5))
            else:
                p_high = 1e-6
            p_high = max(p_high, 1e-6)

            image_item = pg.ImageItem(density_image, axisOrder="col-major")
            image_item.setRect(
                QRectF(
                    float(x_edges[0]),
                    float(y_edges[0]),
                    float(x_edges[-1] - x_edges[0]),
                    float(y_edges[-1] - y_edges[0]),
                )
            )
            image_item.setLookupTable(_make_blue_yellow_lut())
            image_item.setLevels((0.0, p_high))
            self._plot_widget.addItem(image_item)
            y_min, y_max = float(y_edges[0]), float(y_edges[-1])
        else:
            y_min, y_max = _compute_voltage_limits(segment_matrix)

        if render_lines:
            line_pen = pg.mkPen(color=(0, 0, 150, 60) if render_heatmap else (30, 80, 180, 45), width=1)
            max_line_traces = min(traces_drawn, int(render_cfg["max_line_traces"]))
            for i in range(max_line_traces):
                self._plot_widget.plot(time_axis_ui, segment_matrix[i], pen=line_pen)

        half_span_ui = self._eye_span_ui / 2.0
        self._plot_widget.setXRange(-half_span_ui, half_span_ui, padding=0.0)
        self._plot_widget.setYRange(y_min, y_max, padding=0.0)
        selected_phase = start_sample % max(samples_per_ui, 1)
        quality, color = _diagnostic_style_for_score(phase_score)
        score_text = "n/a" if not np.isfinite(phase_score) else f"{phase_score:.3f}"
        self._diagnostics_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: 600; }}")
        self._diagnostics_label.setText(
            "Diagnostics: "
            f"levels={expected_levels}, phase={selected_phase}/{samples_per_ui}, "
            f"score={score_text}, quality={quality}, traces={traces_drawn}, "
            f"mode={self._render_mode}, preset={self._quality_preset}"
        )

        # ── Eye summary ────────────────────────────────────────────────────
        summary = _compute_eye_summary(segment_matrix, samples_per_ui)
        eye_width_ui = summary.get("eye_width_ui", float("nan"))
        eye_width_ps = (
            float(eye_width_ui) * float(ui_s) * 1e12
            if np.isfinite(eye_width_ui)
            else float("nan")
        )
        summary_out = dict(summary)
        summary_out["eye_width_ps"] = eye_width_ps
        summary_out["bit_period_ps"] = float(ui_s) * 1e12
        # Legacy alias
        summary_out["width_ps"] = eye_width_ps
        self._eye_summary = summary_out
        one = summary["one_level"]
        zero = summary["zero_level"]
        amp = summary["eye_amplitude"]
        ht = summary["eye_height"]
        wd_ps = eye_width_ps
        cross = summary["eye_crossing_pct"]
        bit_period_ps = float(ui_s) * 1e12

        def _fmt_v(v: float) -> str:
            if not np.isfinite(v):
                return "n/a"
            if abs(v) >= 1.0:
                return f"{v * 1000:.1f} mV"  # noqa: keep mV
            return f"{v * 1000:.2f} mV"

        def _fmt_mv(v: float) -> str:
            return "n/a" if not np.isfinite(v) else f"{v * 1000:.2f} mV"

        def _fmt_ps(v: float) -> str:
            return "n/a" if not np.isfinite(v) else f"{v:.2f} ps"

        def _fmt_pct(v: float) -> str:
            return "n/a" if not np.isfinite(v) else f"{v:.2f} %"

        self._summary_label.setText(
            "MEASUREMENTS  |  "
            f"One: {_fmt_mv(one)}  |  "
            f"Zero: {_fmt_mv(zero)}  |  "
            f"Amp: {_fmt_mv(amp)}  |  "
            f"Height: {_fmt_mv(ht)}  |  "
            f"Width: {_fmt_ps(wd_ps)}  |  "
            f"Cross: {_fmt_pct(cross)}  |  "
            f"Bit Period: {_fmt_ps(bit_period_ps)}"
        )
