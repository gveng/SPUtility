from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

from sparams_utility.circuit_solver import ChannelSimResult
from sparams_utility.models.circuit import DriverSpec
from sparams_utility.ui.eye_diagram_window import (
    DEFAULT_RENDER_MODE,
    DEFAULT_EYE_SPAN_UI,
    EyeDiagramWindow,
    _align_to_ui_boundary,
    _build_eye_time_axis,
    _build_eye_density,
    _estimate_crossing_phase_shift_samples,
    _decision_marker_positions,
    _diagnostic_style_for_score,
    _expected_eye_levels_for_encoding,
    _find_best_eye_phase,
    _find_best_eye_phase_and_score,
    _normalize_eye_span_ui,
    _normalize_render_mode,
    _score_eye_phase,
)


class EyeDiagramWindowHelpersTests(unittest.TestCase):
    def test_normalize_render_mode_defaults_invalid_values(self) -> None:
        self.assertEqual(_normalize_render_mode("Heatmap"), "Heatmap")
        self.assertEqual(_normalize_render_mode("Lines"), "Lines")
        self.assertEqual(_normalize_render_mode("Heatmap + Lines"), "Heatmap + Lines")
        self.assertEqual(_normalize_render_mode("invalid"), DEFAULT_RENDER_MODE)

    def test_diagnostic_style_for_score(self) -> None:
        self.assertEqual(_diagnostic_style_for_score(float("-inf"))[0], "LOW")
        self.assertEqual(_diagnostic_style_for_score(2.0)[0], "LOW")
        self.assertEqual(_diagnostic_style_for_score(4.0)[0], "MEDIUM")
        self.assertEqual(_diagnostic_style_for_score(7.0)[0], "HIGH")

    def test_decision_marker_positions(self) -> None:
        self.assertEqual(_decision_marker_positions(1), [0.0])
        self.assertEqual(_decision_marker_positions(2), [-1.0, 0.0, 1.0])
        self.assertEqual(_decision_marker_positions(3), [-1.0, 0.0, 1.0])

    def test_expected_eye_levels_for_encoding(self) -> None:
        self.assertEqual(_expected_eye_levels_for_encoding("None"), 2)
        self.assertEqual(_expected_eye_levels_for_encoding("8b10b"), 2)
        self.assertEqual(_expected_eye_levels_for_encoding("PAM4"), 4)

    def test_normalize_eye_span_defaults_invalid_values(self) -> None:
        self.assertEqual(_normalize_eye_span_ui(1), 1)
        self.assertEqual(_normalize_eye_span_ui(3), 3)
        self.assertEqual(_normalize_eye_span_ui(4), DEFAULT_EYE_SPAN_UI)

    def test_align_to_ui_boundary_snaps_forward(self) -> None:
        self.assertEqual(_align_to_ui_boundary(320, 64), 320)
        self.assertEqual(_align_to_ui_boundary(321, 64), 384)
        self.assertEqual(_align_to_ui_boundary(383, 64), 384)

    def test_build_eye_time_axis_spans_two_ui_centered_on_zero(self) -> None:
        axis = _build_eye_time_axis(8, 2)
        self.assertEqual(len(axis), 16)
        self.assertAlmostEqual(axis[0], -1.0)
        self.assertAlmostEqual(axis[-1], 0.875)
        np.testing.assert_allclose(np.diff(axis), np.full(15, 0.125))

    def test_build_eye_time_axis_supports_three_ui(self) -> None:
        axis = _build_eye_time_axis(8, 3)
        self.assertEqual(len(axis), 24)
        self.assertAlmostEqual(axis[0], -1.5)
        self.assertAlmostEqual(axis[-1], 1.375)

    def test_build_eye_density_returns_expected_shapes(self) -> None:
        samples_per_ui = 8
        time_axis = _build_eye_time_axis(samples_per_ui, 2)
        segments = np.vstack([
            np.linspace(-0.5, 0.5, len(time_axis)),
            np.linspace(0.5, -0.5, len(time_axis)),
        ])

        density, x_edges, y_edges = _build_eye_density(segments, time_axis, x_bins=50, y_bins=40)

        self.assertEqual(density.shape, (50, 40))
        self.assertEqual(x_edges.shape, (51,))
        self.assertEqual(y_edges.shape, (41,))
        self.assertGreater(np.sum(density), 0.0)

    def test_find_best_eye_phase_prefers_widest_opening(self) -> None:
        samples_per_ui = 8
        bits = np.array([0, 1] * 32, dtype=float)
        waveform = np.repeat(bits, samples_per_ui)
        waveform = np.convolve(waveform, np.ones(5, dtype=float) / 5.0, mode="same")
        overlay_samples = len(_build_eye_time_axis(samples_per_ui, 2))

        best_phase = _find_best_eye_phase(waveform, 0, len(waveform), samples_per_ui, overlay_samples)
        best_score = _score_eye_phase(waveform, best_phase, len(waveform), samples_per_ui, overlay_samples)
        edge_score = _score_eye_phase(waveform, 0, len(waveform), samples_per_ui, overlay_samples)

        self.assertGreater(best_score, edge_score)
        self.assertTrue(0 <= best_phase < samples_per_ui * 2)

    def test_find_best_eye_phase_with_score_is_finite_on_nrz(self) -> None:
        samples_per_ui = 8
        bits = np.array([0, 1] * 32, dtype=float)
        waveform = np.repeat(bits, samples_per_ui)
        waveform = np.convolve(waveform, np.ones(5, dtype=float) / 5.0, mode="same")
        overlay_samples = len(_build_eye_time_axis(samples_per_ui, 2))

        best_phase, score = _find_best_eye_phase_and_score(
            waveform,
            0,
            len(waveform),
            samples_per_ui,
            overlay_samples,
            expected_levels=2,
        )

        self.assertTrue(0 <= best_phase < samples_per_ui * 2)
        self.assertTrue(np.isfinite(score))

    def test_find_best_eye_phase_with_score_fallback_when_clusters_fail(self) -> None:
        samples_per_ui = 8
        bits = np.array([1] * 30 + [0] * 2, dtype=float)
        waveform = np.repeat(bits, samples_per_ui)
        overlay_samples = len(_build_eye_time_axis(samples_per_ui, 2))

        best_phase, score = _find_best_eye_phase_and_score(
            waveform,
            0,
            len(waveform),
            samples_per_ui,
            overlay_samples,
            expected_levels=2,
        )

        self.assertTrue(0 <= best_phase < samples_per_ui * 2)
        self.assertTrue(np.isfinite(score))

    def test_score_eye_phase_supports_pam4_multilevel(self) -> None:
        samples_per_ui = 8
        symbols = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0] * 24, dtype=float)
        waveform = np.repeat(symbols, samples_per_ui)
        waveform = np.convolve(waveform, np.ones(3, dtype=float) / 3.0, mode="same")
        overlay_samples = len(_build_eye_time_axis(samples_per_ui, 2))

        score = _score_eye_phase(
            waveform,
            0,
            len(waveform),
            samples_per_ui,
            overlay_samples,
            expected_levels=4,
        )

        self.assertTrue(np.isfinite(score))

    def test_score_eye_phase_pam4_request_on_binary_waveform_is_still_finite(self) -> None:
        samples_per_ui = 8
        bits = np.array([0, 1] * 32, dtype=float)
        waveform = np.repeat(bits, samples_per_ui)
        waveform = np.convolve(waveform, np.ones(5, dtype=float) / 5.0, mode="same")
        overlay_samples = len(_build_eye_time_axis(samples_per_ui, 2))

        score = _score_eye_phase(
            waveform,
            0,
            len(waveform),
            samples_per_ui,
            overlay_samples,
            expected_levels=4,
        )

        self.assertTrue(np.isfinite(score))

    def test_score_eye_phase_rejects_strongly_unbalanced_clusters(self) -> None:
        samples_per_ui = 8
        bits = np.array([1] * 30 + [0] * 2, dtype=float)
        waveform = np.repeat(bits, samples_per_ui)
        overlay_samples = len(_build_eye_time_axis(samples_per_ui, 2))

        score = _score_eye_phase(waveform, 0, len(waveform), samples_per_ui, overlay_samples)

        self.assertTrue(np.isneginf(score))

    def test_estimate_crossing_phase_shift_samples_detects_half_ui_offset(self) -> None:
        samples_per_ui = 16
        n_segments = 256
        x = np.arange(samples_per_ui * 2, dtype=float)
        shift_samples = 3
        segments: list[np.ndarray] = []
        for _ in range(n_segments):
            direction = 1.0 if np.random.rand() > 0.5 else -1.0
            left_cross = samples_per_ui // 2 + shift_samples
            right_cross = samples_per_ui + samples_per_ui // 2 + shift_samples
            left = direction * np.tanh((x - left_cross) / 1.5)
            right = -direction * np.tanh((x - right_cross) / 1.5)
            segments.append(left + right)

        segment_matrix = np.asarray(segments, dtype=float)
        estimated = _estimate_crossing_phase_shift_samples(segment_matrix, samples_per_ui, span_ui=2)
        self.assertLessEqual(abs(estimated - shift_samples), 1)


class EyeDiagramWindowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _sample_result(self) -> ChannelSimResult:
        time_s = np.linspace(0.0, 12e-9, 600, dtype=float)
        waveform_v = np.where(np.sin(np.linspace(0.0, 18.0 * np.pi, 600)) >= 0.0, 0.4, 0.0)
        return ChannelSimResult(
            time_s=time_s,
            waveform_v=waveform_v,
            ui_s=1e-9,
            driver_spec=DriverSpec(bitrate_gbps=20.0, prbs_pattern="PRBS-8", encoding="8b10b", num_bits=4096),
            is_differential=True,
        )

    def test_layout_moves_controls_below_plot_and_removes_diagnostics(self) -> None:
        window = EyeDiagramWindow(self._sample_result())
        self.addCleanup(window.close)

        layout = window.centralWidget().layout()
        self.assertIs(layout.itemAt(0).widget(), window._plot_widget)
        self.assertIs(layout.itemAt(1).widget(), window._settings_label)
        self.assertIs(layout.itemAt(2).widget(), window._summary_label)
        self.assertIs(layout.itemAt(3).widget(), window._controls_panel)
        self.assertFalse(hasattr(window, "_diagnostics_label"))
        self.assertIn("font-size: 11px", window._settings_label.styleSheet())
        self.assertIn("font-size: 11px", window._summary_label.styleSheet())

    def test_settings_summary_contains_run_context_and_controls_remain_below(self) -> None:
        window = EyeDiagramWindow(self._sample_result())
        self.addCleanup(window.close)

        window._eye_span_combo.setCurrentText("3 UI")
        window._render_mode_combo.setCurrentText("Lines")
        window._quality_preset_combo.setCurrentText("Fast")

        text = window._settings_label.text()
        self.assertIn("SETTINGS", text)
        self.assertIn("Mode: Differential", text)
        self.assertIn("Bitrate: 20.0 Gbps", text)
        self.assertIn("Pattern: PRBS-8", text)
        self.assertIn("Encoding: 8b10b", text)
        self.assertIn("Bits: 4096", text)
        self.assertEqual(window._eye_span_combo.currentText(), "3 UI")
        self.assertEqual(window._render_mode_combo.currentText(), "Lines")
        self.assertEqual(window._quality_preset_combo.currentText(), "Fast")


if __name__ == "__main__":
    unittest.main()