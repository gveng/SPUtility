from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.special import erfcinv as _scipy_erfcinv
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
    """Black → dark-blue → blue → cyan → orange → red palette (jet-like on black bg).

    Tuned to resemble high-end signal-integrity eye displays where the background
    is black, the bulk of the traces are blue/cyan and the high-density crossings
    light up red/orange.
    """
    stops = [
        (0.000,   0,   0,   0),   # black background
        (0.020,   8,  14,  60),   # near-black blue
        (0.080,  25,  60, 180),   # deep blue (sparse traces)
        (0.220,  40, 110, 230),   # bright blue
        (0.380,  80, 180, 245),   # light blue / cyan
        (0.520, 170, 220, 240),   # pale cyan
        (0.640, 240, 200, 130),   # warm transition
        (0.760, 245, 150,  70),   # orange
        (0.880, 235,  80,  50),   # red-orange (high density)
        (1.000, 200,  20,  20),   # deep red (peak crossings)
    ]
    lut = np.zeros((n, 4), dtype=np.uint8)
    for i in range(n):
        t = i / max(n - 1, 1)
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


def _expand_segments_to_lines(
    segments: np.ndarray,
    time_axis_ui: np.ndarray,
    y_min: float,
    y_max: float,
    y_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-trace sample points into adaptively oversampled line points.

    The plain ``histogram2d`` of raw sample points misses the line segment that
    visually connects two consecutive samples — on steep edges this leaves the
    eye contour looking dotted/aliased (see SciPy Cookbook EyeDiagram). For
    each consecutive sample pair we insert intermediate points proportional to
    the vertical distance in *bin units* so that no pixel along the connecting
    segment is skipped, equivalent to anti-aliased Bresenham line counting but
    fully vectorized in NumPy.
    """
    n_traces, n_samples = segments.shape
    if n_samples < 2 or n_traces == 0:
        return np.tile(time_axis_ui, n_traces), segments.ravel()

    y_range = max(y_max - y_min, 1e-30)
    bin_height = y_range / max(y_bins, 1)

    # Vertical jump per segment in bin units, take the global maximum across
    # all traces so the oversampling factor is the same for every segment —
    # required to keep the per-pair fancy indexing rectangular.
    dy = np.abs(np.diff(segments, axis=1))
    max_jump_bins = float(np.max(dy)) / bin_height if dy.size else 0.0
    # +2 → at least the two endpoints; clamp to avoid runaway memory on
    # pathological inputs.
    n_sub = int(np.clip(np.ceil(max_jump_bins) + 2, 2, 64))

    # Linear interpolation along each segment with ``n_sub`` points (endpoints
    # included once, except for the joining sample that would otherwise be
    # duplicated in the next pair).
    alphas = np.linspace(0.0, 1.0, n_sub, endpoint=False, dtype=np.float32)
    # Shape: (n_traces, n_samples-1, n_sub)
    seg_left = segments[:, :-1, None]
    seg_right = segments[:, 1:, None]
    y_lines = seg_left + (seg_right - seg_left) * alphas
    t_left = time_axis_ui[:-1, None]
    t_right = time_axis_ui[1:, None]
    t_lines = t_left + (t_right - t_left) * alphas
    x_values = np.broadcast_to(t_lines, y_lines.shape).reshape(-1)
    y_values = y_lines.reshape(-1)
    # Append the final sample of every trace to close the polyline.
    x_tail = np.full(n_traces, time_axis_ui[-1])
    y_tail = segments[:, -1]
    x_values = np.concatenate([x_values, x_tail])
    y_values = np.concatenate([y_values, y_tail])
    return x_values, y_values


def _build_eye_density(
    segments: np.ndarray,
    time_axis_ui: np.ndarray,
    x_bins: int = DEFAULT_HEATMAP_X_BINS,
    y_bins: int = DEFAULT_HEATMAP_Y_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(float(time_axis_ui[0]), float(time_axis_ui[-1]), x_bins + 1)
    y_min, y_max = _compute_voltage_limits(segments)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)

    x_values, y_values = _expand_segments_to_lines(
        segments, time_axis_ui, y_min, y_max, y_bins
    )
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
        x_vals, y_vals = _expand_segments_to_lines(
            segments, time_axis_ui, y_min, y_max, y_bins
        )
        density, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])
        return density, x_edges, y_edges

    def _worker(chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.zeros((x_bins, y_bins), dtype=float)
        x_vals, y_vals = _expand_segments_to_lines(
            chunk, time_axis_ui, y_min, y_max, y_bins
        )
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


def _q_factor_for_ber(target_ber: float) -> float:
    """Return the Q-factor (sigmas) corresponding to a target NRZ BER.

    For a single-sided gaussian tail, ``BER = 0.5 * erfc(Q / sqrt(2))``.
    Inverting:  ``Q = sqrt(2) * erfcinv(2 * BER)``.
    Examples:  BER=1e-12 → Q≈7.034,  BER=1e-15 → Q≈7.941.
    """
    ber = float(np.clip(target_ber, 1e-30, 0.49))
    return float(np.sqrt(2.0) * _scipy_erfcinv(2.0 * ber))


def _crossing_times_at_column(
    segments: np.ndarray,
    threshold: float,
    target_col: int,
    samples_per_ui: int,
) -> np.ndarray:
    """First threshold-crossing sample index per trace within ±UI/3 of target.

    Returns an array of length ``n_traces`` where each entry is the
    sub-sample index (float) at which the trace crosses ``threshold`` in the
    search window, or NaN when no crossing is found. Linear interpolation is
    used to recover sub-sample resolution.
    """
    n_traces, n_samp = segments.shape
    half = max(2, samples_per_ui // 3)
    lo = int(max(0, target_col - half))
    hi = int(min(n_samp - 1, target_col + half))
    if hi - lo < 2:
        return np.full(n_traces, np.nan, dtype=float)
    sub = segments[:, lo : hi + 1] - threshold
    sign = np.sign(sub)
    sign[sign == 0] = 1.0
    crosses = sign[:, :-1] != sign[:, 1:]
    has_cross = crosses.any(axis=1)
    if not has_cross.any():
        return np.full(n_traces, np.nan, dtype=float)
    first_idx = np.argmax(crosses, axis=1)
    rows = np.arange(n_traces)
    v0 = sub[rows, first_idx]
    v1 = sub[rows, first_idx + 1]
    denom = v0 - v1
    denom = np.where(denom == 0, 1e-30, denom)
    frac = v0 / denom
    crossing_sample = lo + first_idx.astype(float) + frac
    return np.where(has_cross, crossing_sample, np.nan)


def _dual_dirac_tail_fit(
    crossing_samples: np.ndarray,
    dt_s: float,
    target_ber: float,
    cdf_window: tuple[float, float] = (0.005, 0.20),
) -> tuple[float, float, float]:
    """Tail-fit dual-Dirac jitter decomposition on a single crossing.

    Models the crossing-time PDF as the convolution of a bimodal deterministic
    distribution (Δ/2 separation = DJ_pp/2) with a gaussian RJ of σ_RJ. Linear
    regression of the empirical Q-quantile against the sorted crossing times
    in the tail bands gives the two Dirac means and σ_RJ.

    Returns ``(DJ_pp_s, sigma_RJ_s, TJ_pp_s_at_BER)``. Falls back to NaN if
    there are too few crossings to populate the tails.
    """
    t = crossing_samples[~np.isnan(crossing_samples)]
    if t.size < 200:
        if t.size < 10:
            return float("nan"), float("nan"), float("nan")
        # Robust fallback for small samples: use observed pk-pk for DJ and
        # std-after-trimming for σ_RJ. Less accurate but never worse than nothing.
        t_s = np.sort(t)
        trim = max(1, int(0.05 * t_s.size))
        dj_pp = float(t_s[-trim] - t_s[trim - 1])
        sig = float(np.std(t_s[trim - 1 : -trim], ddof=1)) if t_s.size > 4 else 0.0
        q = _q_factor_for_ber(target_ber)
        return dj_pp * dt_s, sig * dt_s, (dj_pp + 2.0 * q * sig) * dt_s

    t_sorted = np.sort(t)
    n = t_sorted.size
    cdf = (np.arange(1, n + 1) - 0.5) / n
    lo_lim, hi_lim = cdf_window
    # Left tail: small CDF → Q_left = sqrt(2)·erfcinv(2·CDF)  (positive)
    # Right tail: large CDF → Q_right = sqrt(2)·erfcinv(2·(1-CDF))  (positive)
    left_mask = (cdf >= lo_lim) & (cdf <= hi_lim)
    right_mask = (cdf >= 1.0 - hi_lim) & (cdf <= 1.0 - lo_lim)
    if int(left_mask.sum()) < 5 or int(right_mask.sum()) < 5:
        # Same fallback as the small-sample branch.
        trim = max(1, int(0.05 * n))
        dj_pp = float(t_sorted[-trim] - t_sorted[trim - 1])
        sig = float(np.std(t_sorted[trim - 1 : -trim], ddof=1))
        q = _q_factor_for_ber(target_ber)
        return dj_pp * dt_s, sig * dt_s, (dj_pp + 2.0 * q * sig) * dt_s

    q_left = np.sqrt(2.0) * _scipy_erfcinv(2.0 * cdf[left_mask])
    q_right = np.sqrt(2.0) * _scipy_erfcinv(2.0 * (1.0 - cdf[right_mask]))
    # Left tail line: t = mu_L - σ_L · Q_left  → slope_L = -σ_L, intercept_L = mu_L
    slope_L, intercept_L = np.polyfit(q_left, t_sorted[left_mask], 1)
    slope_R, intercept_R = np.polyfit(q_right, t_sorted[right_mask], 1)
    sigma_rj_samples = 0.5 * (abs(slope_L) + abs(slope_R))
    dj_pp_samples = max(0.0, float(intercept_R - intercept_L))

    q_ber = _q_factor_for_ber(target_ber)
    tj_pp_samples = dj_pp_samples + 2.0 * q_ber * sigma_rj_samples
    return dj_pp_samples * dt_s, sigma_rj_samples * dt_s, tj_pp_samples * dt_s


def _statistical_eye_envelopes(
    upper_segs: np.ndarray,
    lower_segs: np.ndarray,
    target_ber: float,
    n_voltage_bins: int = 512,
    sigma_rj_v: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column inner-edge envelopes from the Statistical Eye.

    For each time column, build the *conditional* voltage PDF for the "1" and
    "0" clusters via histogram, optionally convolve with a gaussian noise
    kernel (Vrn / random jitter mapped to voltage), then integrate the CDF and
    read the voltage contour at ``target_ber``.

    Returns:
        inner_upper(col): smallest v such that ``P(V|bit=1) <= v) >= BER``
        inner_lower(col): largest  v such that ``P(V|bit=0) >= v) >= BER``

    These are the same contours that ADS Channel Sim's "Statistical Eye"
    reports for Eye Height / Eye Width — they capture multi-modal ISI (where a
    plain ``μ ± Qσ`` gaussian fit would over-/under-estimate the opening).
    """
    n_up = int(upper_segs.shape[0])
    n_lo = int(lower_segs.shape[0])
    n_samp = int(upper_segs.shape[1] if n_up else lower_segs.shape[1])
    if n_up == 0 or n_lo == 0:
        nan_arr = np.full(n_samp, np.nan, dtype=float)
        return nan_arr, nan_arr

    v_min = float(min(upper_segs.min(), lower_segs.min()))
    v_max = float(max(upper_segs.max(), lower_segs.max()))
    pad = 0.05 * max(v_max - v_min, 1e-9)
    edges = np.linspace(v_min - pad, v_max + pad, n_voltage_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])

    def _per_column_pdf(arr: np.ndarray) -> np.ndarray:
        """Vectorized per-column histogram (rows: time, cols: voltage bins)."""
        if arr.size == 0:
            return np.zeros((n_samp, n_voltage_bins), dtype=float)
        # Map voltages → bin indices, clamp to valid range.
        idx = np.clip(((arr - edges[0]) / bin_width).astype(np.int64),
                      0, n_voltage_bins - 1)
        # Encode (column, bin) into a single linear index; one bincount per
        # matrix is much faster than looping per column.
        n_traces = arr.shape[0]
        col_offsets = np.arange(n_samp, dtype=np.int64)[None, :] * n_voltage_bins
        flat = (idx + col_offsets).reshape(-1)
        counts = np.bincount(flat, minlength=n_samp * n_voltage_bins)
        pdf = counts.reshape(n_samp, n_voltage_bins).astype(float)
        norm = pdf.sum(axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        return pdf / norm

    pdf_up = _per_column_pdf(upper_segs)
    pdf_lo = _per_column_pdf(lower_segs)

    # ── Optional random-noise convolution (Vrn / RJ projected on voltage) ──
    if sigma_rj_v > 0.0:
        sigma_bins = max(sigma_rj_v / bin_width, 0.5)
        pdf_up = gaussian_filter1d(pdf_up, sigma=sigma_bins, axis=1, mode="constant")
        pdf_lo = gaussian_filter1d(pdf_lo, sigma=sigma_bins, axis=1, mode="constant")
        norm_up = pdf_up.sum(axis=1, keepdims=True); norm_up[norm_up == 0.0] = 1.0
        norm_lo = pdf_lo.sum(axis=1, keepdims=True); norm_lo[norm_lo == 0.0] = 1.0
        pdf_up /= norm_up
        pdf_lo /= norm_lo

    # ── CDF integration ───────────────────────────────────────────────────
    cdf_up = np.cumsum(pdf_up, axis=1)                 # P(V_1 <= v)
    sf_lo = np.cumsum(pdf_lo[:, ::-1], axis=1)[:, ::-1]  # P(V_0 >= v)

    inner_upper = np.full(n_samp, np.nan, dtype=float)
    inner_lower = np.full(n_samp, np.nan, dtype=float)

    # First index where cdf_up(col) >= BER  → smallest v with P(V_1<=v)>=BER.
    above_up = cdf_up >= target_ber
    has_up = above_up.any(axis=1)
    first_up = np.argmax(above_up, axis=1)
    inner_upper[has_up] = centers[first_up[has_up]]

    # Last index where sf_lo(col) >= BER  → largest v with P(V_0>=v)>=BER.
    above_lo = sf_lo >= target_ber
    has_lo = above_lo.any(axis=1)
    # Trick to find the *last* True per row: argmax on reversed axis.
    last_lo = (n_voltage_bins - 1) - np.argmax(above_lo[:, ::-1], axis=1)
    inner_lower[has_lo] = centers[last_lo[has_lo]]
    return inner_upper, inner_lower


def _compute_eye_summary(
    segment_matrix: np.ndarray,
    samples_per_ui: int,
    target_ber: float = 1e-12,
    sigma_rj_v: float = 0.0,
    ui_s: float = 0.0,
) -> dict[str, float]:
    """Compute NRZ eye summary using Statistical-Eye / dual-Dirac definitions.

    Reported values:
    * One Level / Zero Level: medians of upper/lower clusters at decision center.
    * Eye Amplitude: One Level - Zero Level.
    * Eye Height @ BER: vertical opening at decision center where the Statistical
      Eye contour crosses ``target_ber`` (PDF/CDF method, matches ADS Channel
      Sim's "Statistical Eye" reporting). For low-statistics columns the method
      falls back to a per-cluster gaussian fit ``μ ± Q·σ``.
    * Eye Width @ BER: contiguous open interval around centre, in UI, using the
      same per-column BER contour.
    * Eye Crossing: crossing level at +/-0.5 UI, normalized to amplitude (%).
    * ``sigma_rj_v``: optional gaussian random-noise σ (volt) added to the eye
      contour via convolution; set to 0 for deterministic-only ISI eye.
    """
    n_seg, n_samp = segment_matrix.shape
    q_ber = _q_factor_for_ber(target_ber)

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
            "target_ber": float(target_ber),
            "q_factor": q_ber,
            "sigma_rj_v": float(sigma_rj_v),
            "dj_pp_s": nan,
            "sigma_rj_s": nan,
            "tj_pp_s": nan,
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

    # ── Horizontal/vertical measurement at the BER-extrapolated envelopes ──
    # For each column compute mean ± Q·sigma per cluster (upper/lower) and
    # use those as inner-edge envelopes. Eye is "open" at a column when the
    # upper-cluster lower edge exceeds the lower-cluster upper edge.
    threshold = (one_level + zero_level) / 2.0
    upper_mask = center_vals >= threshold
    lower_mask = ~upper_mask

    upper_segs = segment_matrix[upper_mask]
    lower_segs = segment_matrix[lower_mask]

    if upper_segs.shape[0] < 2 or lower_segs.shape[0] < 2:
        eye_height = float("nan")
        eye_width_ui = float("nan")
    else:
        # ── Statistical Eye contour at target BER (ADS-equivalent) ──────
        # The simulated channel is purely linear/deterministic: all the
        # variability seen in the eye is bounded ISI driven by the PRBS.
        # There is *no* intrinsic random noise to extrapolate, so when
        # Vrn = 0 we report the empirical peak-distortion (matches ADS
        # Channel Sim with all noise sources disabled). When the user
        # provides a receiver noise σ (Vrn), the BER contour is shifted
        # by Q_BER · σ_Vrn — this is the only random component the model
        # accounts for.
        worst_inner_upper = upper_segs.min(axis=0)
        worst_inner_lower = lower_segs.max(axis=0)
        vrn_term = q_ber * float(sigma_rj_v)
        gauss_inner_upper = worst_inner_upper - vrn_term
        gauss_inner_lower = worst_inner_lower + vrn_term

        # Try the histogram contour as well; when the tails reach BER it is
        # tighter than worst-case (rare values get filtered out).
        stat_inner_upper, stat_inner_lower = _statistical_eye_envelopes(
            upper_segs, lower_segs, target_ber, sigma_rj_v=sigma_rj_v
        )
        # Combine: where the histogram tails cannot reach BER (NaN) fall back
        # to the gaussian extrapolation; otherwise take the worst case.
        inner_upper = np.where(
            np.isnan(stat_inner_upper),
            gauss_inner_upper,
            np.minimum(stat_inner_upper, gauss_inner_upper),
        )
        inner_lower = np.where(
            np.isnan(stat_inner_lower),
            gauss_inner_lower,
            np.maximum(stat_inner_lower, gauss_inner_lower),
        )

        center_col = n_samp // 2
        eye_height = float(inner_upper[center_col] - inner_lower[center_col])
        # Eye height cannot be negative — when the BER-extrapolated clouds
        # overlap, the eye is closed at the configured BER.
        if eye_height < 0.0:
            eye_height = 0.0

        open_cols = inner_upper > inner_lower
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

    # ── Dual-Dirac jitter decomposition (DJ_pp / σ_RJ / TJ_pp @ BER) ──────
    dj_pp_s = float("nan")
    sigma_rj_s = float("nan")
    tj_pp_s = float("nan")
    eye_width_jitter_ui = float("nan")
    if (
        ui_s > 0.0
        and np.isfinite(eye_amplitude)
        and eye_amplitude > 0.0
        and n_samp >= 4
    ):
        dt_s = ui_s / float(samples_per_ui)
        threshold_v = (one_level + zero_level) / 2.0
        center_col = n_samp // 2
        half_ui_samples = max(1, int(round(samples_per_ui / 2.0)))
        target_left = max(0, center_col - half_ui_samples)
        target_right = min(n_samp - 1, center_col + half_ui_samples)
        crossings_left = _crossing_times_at_column(
            segment_matrix, threshold_v, target_left, samples_per_ui
        )
        crossings_right = _crossing_times_at_column(
            segment_matrix, threshold_v, target_right, samples_per_ui
        )
        dj_l, sig_l, tj_l = _dual_dirac_tail_fit(crossings_left, dt_s, target_ber)
        dj_r, sig_r, tj_r = _dual_dirac_tail_fit(crossings_right, dt_s, target_ber)
        dj_vals = [v for v in (dj_l, dj_r) if np.isfinite(v)]
        sig_vals = [v for v in (sig_l, sig_r) if np.isfinite(v)]
        tj_vals = [v for v in (tj_l, tj_r) if np.isfinite(v)]
        if dj_vals:
            dj_pp_s = float(np.mean(dj_vals))
        if sig_vals:
            sigma_rj_s = float(np.mean(sig_vals))
        if tj_vals:
            # Worst-case TJ across the two edges (matches ADS reporting).
            tj_pp_s = float(np.max(tj_vals))
            eye_width_jitter_ui = max(0.0, 1.0 - tj_pp_s / ui_s)

    # Final eye-width: the worst between PDF-based and jitter-based opening.
    if np.isfinite(eye_width_jitter_ui) and np.isfinite(eye_width_ui):
        eye_width_ui = float(min(eye_width_ui, eye_width_jitter_ui))
    elif np.isfinite(eye_width_jitter_ui):
        eye_width_ui = eye_width_jitter_ui

    return {
        "one_level": one_level,
        "zero_level": zero_level,
        "eye_amplitude": eye_amplitude,
        "eye_height": eye_height,
        "eye_width_ui": eye_width_ui,
        "eye_crossing_pct": eye_crossing_pct,
        "target_ber": float(target_ber),
        "q_factor": q_ber,
        "sigma_rj_v": float(sigma_rj_v),
        "dj_pp_s": dj_pp_s,
        "sigma_rj_s": sigma_rj_s,
        "tj_pp_s": tj_pp_s,
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
        progress_callback: Callable[[int, str], None] | None = None,
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
        self._progress_callback: Callable[[int, str], None] | None = progress_callback

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        self._eye_span_combo = QComboBox()
        for span_ui in EYE_SPAN_CHOICES:
            self._eye_span_combo.addItem(f"{span_ui} UI", span_ui)
        combo_index = self._eye_span_combo.findData(self._eye_span_ui)
        if combo_index >= 0:
            self._eye_span_combo.setCurrentIndex(combo_index)
        self._eye_span_combo.currentIndexChanged.connect(self._on_eye_span_changed)

        self._render_mode_combo = QComboBox()
        for mode in RENDER_MODE_CHOICES:
            self._render_mode_combo.addItem(mode, mode)
        render_mode_index = self._render_mode_combo.findData(self._render_mode)
        if render_mode_index >= 0:
            self._render_mode_combo.setCurrentIndex(render_mode_index)
        self._render_mode_combo.currentIndexChanged.connect(self._on_render_mode_changed)

        self._quality_preset_combo = QComboBox()
        for preset in QUALITY_PRESET_CHOICES:
            self._quality_preset_combo.addItem(preset, preset)
        quality_index = self._quality_preset_combo.findData(self._quality_preset)
        if quality_index >= 0:
            self._quality_preset_combo.setCurrentIndex(quality_index)
        self._quality_preset_combo.currentIndexChanged.connect(self._on_quality_preset_changed)

        def _summary_box_style(*, text_color: str, background: str, border: str) -> str:
            return (
                "QLabel { "
                "font-family: Consolas, 'Courier New', monospace; "
                "font-weight: 600; "
                "font-size: 11px; "
                f"color: {text_color}; "
                f"background: {background}; "
                f"border: 1px solid {border}; "
                "border-radius: 4px; "
                "padding: 4px 8px; "
                "}"
            )

        self._settings_label = QLabel()
        self._settings_label.setWordWrap(True)
        self._settings_label.setStyleSheet(
            _summary_box_style(
                text_color="#1e3a5f",
                background="#eff6ff",
                border="#3b82f6",
            )
        )
        self._settings_label.setMinimumHeight(24)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet(
            _summary_box_style(
                text_color="#052e16",
                background="#ecfdf5",
                border="#16a34a",
            )
        )
        self._summary_label.setMinimumHeight(24)
        self._eye_summary: dict[str, float] = {}

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("k")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.35)
        self._plot_widget.setLabel("bottom", "time, UI", color="#cc1a00", size="11pt")
        self._plot_widget.setLabel("left", "Density", color="#cc1a00", size="11pt")
        self._plot_widget.setTitle("Eye Diagram")
        layout.addWidget(self._plot_widget)

        layout.addWidget(self._settings_label)
        layout.addWidget(self._summary_label)

        self._controls_panel = QWidget()
        controls_layout = QHBoxLayout(self._controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(QLabel("Span"))
        controls_layout.addWidget(self._eye_span_combo)
        controls_layout.addWidget(QLabel("Render"))
        controls_layout.addWidget(self._render_mode_combo)
        controls_layout.addWidget(QLabel("Quality"))
        controls_layout.addWidget(self._quality_preset_combo)
        controls_layout.addStretch(1)
        layout.addWidget(self._controls_panel)

        self._draw_eye_diagram()
        # Progress callback is only consumed by the very first draw triggered
        # by the host (typically a QProgressDialog from the simulation flow).
        # Subsequent re-draws (span/render/quality changes) run silently.
        self._progress_callback = None

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
        self._update_settings_summary()
        self._emit_draw_progress(2, "Preparing eye diagram...")
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
        start_sample, _phase_score = _find_best_eye_phase_and_score(
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
        self._emit_draw_progress(35, "Collecting eye segments...")

        if traces_drawn == 0:
            self._summary_label.setText("MEASUREMENTS  |  insufficient data to render eye diagram")
            self._eye_summary = {}
            self._plot_widget.setTitle("Eye Diagram (insufficient data)")
            self._emit_draw_progress(100, "Eye diagram ready.")
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
        self._emit_draw_progress(45, "Aligning crossings...")

        if self._statistical_enabled and self._noise_rms_mv > 0.0 and _rng is not None:
            segment_matrix = segment_matrix + _rng.normal(0.0, self._noise_rms_mv * 1e-3, segment_matrix.shape)
        render_heatmap = self._render_mode in {"Heatmap", "Heatmap + Lines"}
        render_lines = self._render_mode in {"Lines", "Heatmap + Lines"}

        y_min: float
        y_max: float
        if render_heatmap:
            self._emit_draw_progress(55, "Building eye density map...")
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
            self._emit_draw_progress(75, "Rendering heatmap...")
        else:
            y_min, y_max = _compute_voltage_limits(segment_matrix)

        if render_lines:
            line_pen = pg.mkPen(color=(180, 210, 255, 70) if render_heatmap else (120, 180, 255, 90), width=1)
            max_line_traces = min(traces_drawn, int(render_cfg["max_line_traces"]))
            for i in range(max_line_traces):
                self._plot_widget.plot(time_axis_ui, segment_matrix[i], pen=line_pen)
            self._emit_draw_progress(88, "Drawing trace overlay...")

        half_span_ui = self._eye_span_ui / 2.0
        self._plot_widget.setXRange(-half_span_ui, half_span_ui, padding=0.0)
        self._plot_widget.setYRange(y_min, y_max, padding=0.0)

        # ── Eye summary ────────────────────────────────────────────────────
        self._emit_draw_progress(92, "Computing eye measurements...")
        sigma_vrn_v = float(getattr(self._result.driver_spec, "random_noise_v", 0.0))
        summary = _compute_eye_summary(
            segment_matrix,
            samples_per_ui,
            ui_s=float(ui_s),
            sigma_rj_v=sigma_vrn_v,
        )
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
            f"DJ(pp): {_fmt_ps(summary.get('dj_pp_s', float('nan')) * 1e12)}  |  "
            f"RJ(σ): {_fmt_ps(summary.get('sigma_rj_s', float('nan')) * 1e12)}  |  "
            f"TJ(pp@BER): {_fmt_ps(summary.get('tj_pp_s', float('nan')) * 1e12)}  |  "
            f"Bit Period: {_fmt_ps(bit_period_ps)}"
        )
        self._emit_draw_progress(100, "Eye diagram ready.")

    def _emit_draw_progress(self, percent: int, label: str) -> None:
        cb = self._progress_callback
        if cb is None:
            return
        try:
            cb(int(percent), str(label))
            QApplication.processEvents()
        except Exception:
            # Progress reporting must never break the draw flow.
            self._progress_callback = None

    def _update_settings_summary(self) -> None:
        spec = self._result.driver_spec
        mode = "Differential" if self._result.is_differential else "Single-ended"
        vrn_mv = float(getattr(spec, "random_noise_v", 0.0)) * 1e3
        self._settings_label.setText(
            "SETTINGS  |  "
            f"Mode: {mode}  |  "
            f"Bitrate: {spec.bitrate_gbps} Gbps  |  "
            f"Pattern: {spec.prbs_pattern}  |  "
            f"Encoding: {spec.encoding}  |  "
            f"Rise: {spec.rise_time_s * 1e12:.1f} ps  |  "
            f"Fall: {spec.fall_time_s * 1e12:.1f} ps  |  "
            f"Vrn: {vrn_mv:.2f} mV  |  "
            f"Bits: {spec.num_bits}"
        )
