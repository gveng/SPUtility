from __future__ import annotations

import numpy as np
import pyqtgraph as pg
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
RENDER_MODE_CHOICES = ("Heatmap", "Lines", "Heatmap + Lines")
DEFAULT_NOISE_RMS_MV: float = 0.0
DEFAULT_JITTER_RMS_PS: float = 0.0


def _make_blue_yellow_lut(n: int = 256) -> np.ndarray:
    """Build an ADS-style heatmap LUT: white → navy → blue → cyan → green → yellow → orange."""
    # Colour stops (t, R, G, B) mirroring the Keysight ADS eye diagram palette.
    stops = [
        (0.000, 255, 255, 255),   # white  – zero density (background)
        (0.040, 255, 255, 255),   # white  – keep background clean for near-zero
        (0.080,  10,  20, 130),   # dark navy
        (0.220,   0,  60, 220),   # blue
        (0.380,   0, 200, 240),   # cyan
        (0.530,   0, 210,  80),   # cyan-green
        (0.660,  80, 220,   0),   # yellow-green
        (0.780, 240, 220,   0),   # yellow
        (0.900, 255, 140,   0),   # orange
        (1.000, 255, 190,   0),   # gold (peak)
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
    x_bins: int = 260,
    y_bins: int = 220,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.tile(time_axis_ui, segments.shape[0])
    y_values = segments.ravel()

    x_edges = np.linspace(float(time_axis_ui[0]), float(time_axis_ui[-1]), x_bins + 1)
    y_min, y_max = _compute_voltage_limits(segments)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)

    density, _, _ = np.histogram2d(x_values, y_values, bins=[x_edges, y_edges])
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
    """Compute NRZ eye summary: Level1, Level0, Height, Width.

    * Level1 / Level0 – median voltage of the upper / lower cluster sampled at
      the centre of the UI window (eye decision point).
    * Height – Level1 − Level0 (eye opening in voltage).
    * Width  – fraction of the UI over which the eye remains open, expressed
                in UI (0 … 1 for NRZ).  Estimated by finding the time range
                around the centre where the upper and lower waveform envelopes
                do not cross.
    """
    n_seg, n_samp = segment_matrix.shape
    if n_seg < 4 or n_samp < 2:
        return {"level1": float("nan"), "level0": float("nan"),
                "height": float("nan"), "width": float("nan")}

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
        return {"level1": float("nan"), "level0": float("nan"),
                "height": float("nan"), "width": float("nan")}

    level1 = float(np.median(cluster1))
    level0 = float(np.median(cluster0))
    height = level1 - level0

    # ── Horizontal measurement (eye width) ─────────────────────────────────
    # For each time sample compute the p5 (low envelope) and p95 (high envelope)
    # Eye is "open" at a column when p5 of upper cluster > p95 of lower cluster
    threshold = (level1 + level0) / 2.0
    upper_mask = center_vals >= threshold
    lower_mask = ~upper_mask

    upper_segs = segment_matrix[upper_mask]
    lower_segs = segment_matrix[lower_mask]

    if upper_segs.shape[0] < 2 or lower_segs.shape[0] < 2:
        width = float("nan")
    else:
        # per-column 5th/95th percentile
        p5_upper = np.percentile(upper_segs, 5.0, axis=0)
        p95_lower = np.percentile(lower_segs, 95.0, axis=0)
        open_cols = p5_upper > p95_lower   # True where eye is open
        # Count consecutive open columns around the centre
        open_count = int(np.sum(open_cols))
        width = float(open_count) / float(n_samp)  # in UI fraction

    return {"level1": level1, "level0": level0, "height": height, "width": width}


class EyeDiagramWindow(QMainWindow):
    span_changed = Signal(int)
    render_mode_changed = Signal(str)

    def __init__(
        self,
        result: ChannelSimResult,
        title: str = "Eye Diagram",
        parent=None,
        initial_span_ui: int = DEFAULT_EYE_SPAN_UI,
        initial_render_mode: str = DEFAULT_RENDER_MODE,
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
        layout.addLayout(info_layout)

        self._diagnostics_label = QLabel()
        self._diagnostics_label.setWordWrap(True)
        layout.addWidget(self._diagnostics_label)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet(
            "QLabel { font-family: monospace; font-weight: 600; "
            "background: #f0fdf4; border: 1px solid #86efac; "
            "border-radius: 4px; padding: 3px 6px; }"
        )
        layout.addWidget(self._summary_label)
        self._eye_summary: dict[str, float] = {}

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=False, y=False)
        self._plot_widget.setLabel("bottom", "Time", units="UI")
        self._plot_widget.setLabel("left", "Voltage", units="V")
        self._plot_widget.setTitle("Eye Diagram")
        layout.addWidget(self._plot_widget)

        self._draw_eye_diagram()

    @property
    def eye_summary(self) -> dict[str, float]:
        """Return the last-computed eye summary dict with keys: level1, level0, height, width."""
        return dict(self._eye_summary)

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

    def _draw_eye_diagram(self) -> None:
        self._plot_widget.clear()
        result = self._result
        ui_s = result.ui_s
        dt = result.time_s[1] - result.time_s[0] if len(result.time_s) > 1 else 1e-12
        samples_per_ui = max(1, int(round(ui_s / dt)))

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

        traces_drawn = 0
        segments: list[np.ndarray] = []
        pos = start_sample
        _use_jitter = self._statistical_enabled and self._jitter_rms_ps > 0.0
        _jitter_sigma_samples = (self._jitter_rms_ps * 1e-12 / dt) if _use_jitter else 0.0
        _rng = np.random.default_rng() if (self._statistical_enabled and (self._jitter_rms_ps > 0.0 or self._noise_rms_mv > 0.0)) else None
        while pos + overlay_samples <= end_sample and traces_drawn < 5000:
            if _use_jitter and _rng is not None and _jitter_sigma_samples > 0.0:
                jitter_shift = int(_rng.normal(0.0, _jitter_sigma_samples))
                actual_pos = max(0, min(pos + jitter_shift, end_sample - overlay_samples))
            else:
                actual_pos = pos
            segment = waveform[actual_pos: actual_pos + overlay_samples]
            segments.append(segment)
            pos += samples_per_ui
            traces_drawn += 1

        if traces_drawn == 0:
            self._diagnostics_label.setText("Diagnostics: insufficient data to render eye diagram.")
            self._summary_label.setText("")
            self._eye_summary = {}
            self._plot_widget.setTitle("Eye Diagram (insufficient data)")
            return

        segment_matrix = np.asarray(segments, dtype=float)
        if self._statistical_enabled and self._noise_rms_mv > 0.0 and _rng is not None:
            segment_matrix = segment_matrix + _rng.normal(0.0, self._noise_rms_mv * 1e-3, segment_matrix.shape)
        render_heatmap = self._render_mode in {"Heatmap", "Heatmap + Lines"}
        render_lines = self._render_mode in {"Lines", "Heatmap + Lines"}

        y_min: float
        y_max: float
        if render_heatmap:
            density, x_edges, y_edges = _build_eye_density(segment_matrix, time_axis_ui)
            density_image = np.log1p(density)

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
            max_level = float(np.max(density_image))
            image_item.setLevels((0.0, max(1e-6, max_level)))
            self._plot_widget.addItem(image_item)
            y_min, y_max = float(y_edges[0]), float(y_edges[-1])
        else:
            y_min, y_max = _compute_voltage_limits(segment_matrix)

        if render_lines:
            line_pen = pg.mkPen(color=(0, 0, 150, 60) if render_heatmap else (30, 80, 180, 45), width=1)
            max_line_traces = min(traces_drawn, 2000)
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
            f"score={score_text}, quality={quality}, traces={traces_drawn}, mode={self._render_mode}"
        )

        # ── Eye summary ────────────────────────────────────────────────────
        summary = _compute_eye_summary(segment_matrix, samples_per_ui)
        self._eye_summary = summary
        lv1 = summary["level1"]
        lv0 = summary["level0"]
        ht = summary["height"]
        wd = summary["width"]

        def _fmt_v(v: float) -> str:
            if not np.isfinite(v):
                return "n/a"
            if abs(v) >= 1.0:
                return f"{v * 1000:.1f} mV"  # noqa: keep mV
            return f"{v * 1000:.2f} mV"

        def _fmt_mv(v: float) -> str:
            return "n/a" if not np.isfinite(v) else f"{v * 1000:.2f} mV"

        def _fmt_ui(v: float) -> str:
            return "n/a" if not np.isfinite(v) else f"{v:.3f} UI"

        self._summary_label.setText(
            f"Eye Summary │  Level1: {_fmt_mv(lv1)}  │  Level0: {_fmt_mv(lv0)}"
            f"  │  Height: {_fmt_mv(ht)}  │  Width: {_fmt_ui(wd)}"
        )
