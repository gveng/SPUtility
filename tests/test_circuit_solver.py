from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparams_utility.circuit_solver import (
    _analyze_passivity,
    _build_touchstone_cache,
    _fill_missing_frequency_points,
    _generate_prbs,
    _interpolate_y_matrix,
    _s_to_y,
    _y_to_s,
    solve_circuit_network,
    to_touchstone_string_with_format,
)
from sparams_utility.models.circuit import CircuitDocument, CircuitPortRef, PRBS_CHOICES
from sparams_utility.models.state import AppState
from sparams_utility.touchstone_parser import (
    SParameterCell, TouchstoneFile, TouchstoneFormat, TouchstoneOptions,
    TouchstonePoint, MagnitudeTable,
)


class CircuitSolverTests(unittest.TestCase):
    def test_prbs_choices_include_8_and_12(self) -> None:
        self.assertIn("PRBS-8", PRBS_CHOICES)
        self.assertIn("PRBS-12", PRBS_CHOICES)

    def test_generate_prbs_supports_8_and_12(self) -> None:
        for pattern in ("PRBS-8", "PRBS-12"):
            bits = _generate_prbs(pattern, 64)
            self.assertEqual(bits.shape, (64,))
            self.assertTrue(np.all((bits == 0) | (bits == 1)))
            self.assertGreater(np.count_nonzero(bits), 0)
            self.assertLess(np.count_nonzero(bits), 64)

    def test_generate_prbs_rejects_unknown_pattern(self) -> None:
        with self.assertRaises(ValueError):
            _generate_prbs("PRBS-99", 16)

    def test_resistor_between_two_ports(self) -> None:
        doc = CircuitDocument()
        p1 = doc.add_instance(
            source_file_id="__special__:port_ground",
            display_label="P1",
            nports=1,
            position_x=0.0,
            position_y=0.0,
            block_kind="port_ground",
            impedance_ohm=50.0,
        )
        p2 = doc.add_instance(
            source_file_id="__special__:port_ground",
            display_label="P2",
            nports=1,
            position_x=100.0,
            position_y=0.0,
            block_kind="port_ground",
            impedance_ohm=50.0,
        )
        r1 = doc.add_instance(
            source_file_id="__special__:lumped_r",
            display_label="R1",
            nports=2,
            position_x=50.0,
            position_y=0.0,
            block_kind="lumped_r",
            impedance_ohm=100.0,
        )

        doc.add_connection(CircuitPortRef(p1.instance_id, 1), CircuitPortRef(r1.instance_id, 1))
        doc.add_connection(CircuitPortRef(p2.instance_id, 1), CircuitPortRef(r1.instance_id, 2))

        result = solve_circuit_network(doc, AppState())
        self.assertEqual(result.nports, 2)
        self.assertGreater(result.frequencies_hz.size, 0)
        s0 = result.s_matrices[0]
        self.assertAlmostEqual(s0[0, 0].real, 0.5, places=6)
        self.assertAlmostEqual(s0[0, 1].real, 0.5, places=6)
        self.assertAlmostEqual(s0[1, 0].real, 0.5, places=6)
        self.assertAlmostEqual(s0[1, 1].real, 0.5, places=6)

    def test_touchstone_export_formats(self) -> None:
        doc = CircuitDocument()
        p1 = doc.add_instance(
            source_file_id="__special__:port_ground",
            display_label="P1",
            nports=1,
            position_x=0.0,
            position_y=0.0,
            block_kind="port_ground",
            impedance_ohm=50.0,
        )
        p2 = doc.add_instance(
            source_file_id="__special__:port_ground",
            display_label="P2",
            nports=1,
            position_x=100.0,
            position_y=0.0,
            block_kind="port_ground",
            impedance_ohm=50.0,
        )
        r1 = doc.add_instance(
            source_file_id="__special__:lumped_r",
            display_label="R1",
            nports=2,
            position_x=50.0,
            position_y=0.0,
            block_kind="lumped_r",
            impedance_ohm=100.0,
        )

        doc.add_connection(CircuitPortRef(p1.instance_id, 1), CircuitPortRef(r1.instance_id, 1))
        doc.add_connection(CircuitPortRef(p2.instance_id, 1), CircuitPortRef(r1.instance_id, 2))

        result = solve_circuit_network(doc, AppState())

        text_ri = to_touchstone_string_with_format(result, data_format="RI", frequency_unit="GHz")
        text_ma = to_touchstone_string_with_format(result, data_format="MA", frequency_unit="MHz")
        text_db = to_touchstone_string_with_format(result, data_format="DB", frequency_unit="Hz")

        self.assertIn("# GHZ S RI R 50", text_ri)
        self.assertIn("# MHZ S MA R 50", text_ma)
        self.assertIn("# HZ S DB R 50", text_db)


class FrequencyGapInterpolationTests(unittest.TestCase):
    def _make_s_cell(self, val: complex) -> SParameterCell:
        return SParameterCell(row=1, col=1,
                              raw_primary=val.real, raw_secondary=val.imag,
                              complex_value=val, magnitude_db=-30.0)

    def _make_touchstone_file(self, freqs_hz, values) -> TouchstoneFile:
        points = []
        for f, v in zip(freqs_hz, values):
            cell = self._make_s_cell(v)
            points.append(TouchstonePoint(frequency_hz=f, s_matrix=[[cell]]))
        options = TouchstoneOptions(
            frequency_unit="GHZ",
            parameter="S",
            data_format=TouchstoneFormat.RI,
            reference_resistance=50.0,
        )
        mag_table = MagnitudeTable(frequencies_hz=list(freqs_hz), traces_db={"S11": [-30.0] * len(freqs_hz)})
        return TouchstoneFile(
            source_name="test.s1p",
            nports=1,
            options=options,
            trace_names=["S11"],
            points=points,
            magnitude_table=mag_table,
            comments=[],
        )

    def test_no_gap_unchanged(self) -> None:
        """Uniform spacing: no extra points should be added."""
        freqs = np.array([1e9, 2e9, 3e9, 4e9], dtype=float)
        s = np.zeros((4, 1, 1), dtype=np.complex128)
        s[:, 0, 0] = [0.1, 0.2, 0.3, 0.4]
        out_f, out_s = _fill_missing_frequency_points(freqs, s)
        np.testing.assert_array_almost_equal(out_f, freqs)
        np.testing.assert_array_almost_equal(out_s[:, 0, 0], s[:, 0, 0])

    def test_frequency_samples_are_sorted_and_deduplicated(self) -> None:
        """Normalization should not invent points; it only sorts and removes duplicates."""
        freqs = np.array([5e9, 2e9, 2e9, 1e9], dtype=float)
        s = np.zeros((4, 1, 1), dtype=np.complex128)
        s[:, 0, 0] = [0.4, 0.2, 0.25, 0.1]
        out_f, out_s = _fill_missing_frequency_points(freqs, s)
        np.testing.assert_array_equal(out_f, np.array([1e9, 2e9, 5e9], dtype=float))
        np.testing.assert_array_almost_equal(out_s[:, 0, 0], np.array([0.1, 0.2, 0.4], dtype=np.complex128))

    def test_interpolation_matches_measured_points_in_y_domain(self) -> None:
        """Interpolated Y must match the exact converted Y at measured frequencies."""
        freqs_hz = [1e9, 2e9, 5e9, 6e9]
        values = [complex(0.1, 0.0), complex(0.15, 0.05), complex(0.35, 0.1), complex(0.4, 0.05)]
        ts_file = self._make_touchstone_file(freqs_hz, values)

        from pathlib import Path
        from sparams_utility.models.state import LoadedTouchstone

        loaded = LoadedTouchstone(
            file_id="test_interp_file",
            path=Path("test_interp.s1p"),
            display_name="test_interp.s1p",
            data=ts_file,
        )
        state = AppState()
        state._files_by_id[loaded.file_id] = loaded
        state._order.append(loaded.file_id)

        doc = CircuitDocument()
        doc.add_instance(
            source_file_id="test_interp_file",
            display_label="ts",
            nports=1,
            position_x=0.0,
            position_y=0.0,
            block_kind="touchstone",
            impedance_ohm=50.0,
        )

        cache = _build_touchstone_cache(doc, state)
        entry = cache["test_interp_file"]
        for freq_hz, value in zip(freqs_hz, values):
            y_expected = _s_to_y(np.array([[value]], dtype=np.complex128), 50.0)
            y_actual = _interpolate_y_matrix(entry, freq_hz)
            np.testing.assert_allclose(y_actual, y_expected, rtol=1e-10, atol=1e-12)

    def test_gap_interpolation_in_circuit(self) -> None:
        """Circuit solver should produce valid S matrices on sparse Touchstone data."""
        freqs_hz = [1e9, 2e9, 5e9, 6e9]  # 3 GHz gap between 2 and 5 GHz
        values = [complex(0.1, 0.0), complex(0.15, 0.05), complex(0.35, 0.1), complex(0.4, 0.05)]
        ts_file = self._make_touchstone_file(freqs_hz, values)

        from sparams_utility.models.state import LoadedTouchstone
        from pathlib import Path

        loaded = LoadedTouchstone(
            file_id="test_gap_file",
            path=Path("test.s1p"),
            display_name="test.s1p",
            data=ts_file,
        )
        state = AppState()
        state._files_by_id["test_gap_file"] = loaded
        state._order.append("test_gap_file")

        from sparams_utility.models.circuit import FrequencySweepSpec
        doc = CircuitDocument()
        doc.sweep = FrequencySweepSpec(fmin_hz=1e9, fmax_hz=6e9, fstep_hz=0.5e9, display_unit="GHz")
        port = doc.add_instance(source_file_id="__special__:port_ground", display_label="P1",
                                nports=1, position_x=0.0, position_y=0.0, block_kind="port_ground",
                                impedance_ohm=50.0)
        ts_block = doc.add_instance(source_file_id="test_gap_file", display_label="ts",
                                    nports=1, position_x=100.0, position_y=0.0, block_kind="touchstone",
                                    impedance_ohm=50.0)
        doc.add_connection(CircuitPortRef(port.instance_id, 1), CircuitPortRef(ts_block.instance_id, 1))

        result = solve_circuit_network(doc, state)
        self.assertEqual(result.nports, 1)
        # All S values must be finite
        self.assertTrue(np.all(np.isfinite(np.abs(result.s_matrices))))

    def test_out_of_band_touchstone_frequency_raises(self) -> None:
        """The solver should reject extrapolation outside measured Touchstone coverage."""
        freqs_hz = [2e9, 4e9, 6e9]
        values = [complex(0.1, 0.0), complex(0.2, 0.05), complex(0.3, 0.1)]
        ts_file = self._make_touchstone_file(freqs_hz, values)

        from pathlib import Path
        from sparams_utility.models.circuit import FrequencySweepSpec
        from sparams_utility.models.state import LoadedTouchstone

        loaded = LoadedTouchstone(
            file_id="test_oob_file",
            path=Path("test_oob.s1p"),
            display_name="test_oob.s1p",
            data=ts_file,
        )
        state = AppState()
        state._files_by_id[loaded.file_id] = loaded
        state._order.append(loaded.file_id)

        doc = CircuitDocument()
        doc.sweep = FrequencySweepSpec(fmin_hz=1e9, fmax_hz=6e9, fstep_hz=1e9, display_unit="GHz")
        port = doc.add_instance(
            source_file_id="__special__:port_ground",
            display_label="P1",
            nports=1,
            position_x=0.0,
            position_y=0.0,
            block_kind="port_ground",
            impedance_ohm=50.0,
        )
        ts_block = doc.add_instance(
            source_file_id="test_oob_file",
            display_label="ts",
            nports=1,
            position_x=100.0,
            position_y=0.0,
            block_kind="touchstone",
            impedance_ohm=50.0,
        )
        doc.add_connection(CircuitPortRef(port.instance_id, 1), CircuitPortRef(ts_block.instance_id, 1))

        with self.assertRaisesRegex(ValueError, "outside Touchstone data range"):
            solve_circuit_network(doc, state)


class NumericalStabilityTests(unittest.TestCase):
    def test_s_y_round_trip_stays_stable_near_short(self) -> None:
        s_matrix = np.array([[complex(-0.999999, 1e-6)]], dtype=np.complex128)
        y_matrix = _s_to_y(s_matrix, 50.0)
        s_round_trip = _y_to_s(y_matrix, np.array([50.0], dtype=float))
        np.testing.assert_allclose(s_round_trip, s_matrix, rtol=1e-8, atol=1e-10)


class PassivityDiagnosticsTests(unittest.TestCase):
    def test_passive_network_is_reported_as_passive(self) -> None:
        frequencies = np.array([1e9, 2e9], dtype=float)
        s_matrices = np.zeros((2, 1, 1), dtype=np.complex128)
        s_matrices[:, 0, 0] = [0.2 + 0.1j, -0.3 + 0.05j]
        diagnostic = _analyze_passivity(frequencies, s_matrices)
        self.assertEqual(diagnostic.summary.severity, "pass")
        self.assertEqual(diagnostic.summary.points_over_warn, 0)

    def test_non_passive_network_is_reported_as_hard_violation(self) -> None:
        frequencies = np.array([1e9, 2e9, 3e9], dtype=float)
        s_matrices = np.zeros((3, 1, 1), dtype=np.complex128)
        s_matrices[:, 0, 0] = [1.0002 + 0j, 1.002 + 0j, 1.003 + 0j]
        diagnostic = _analyze_passivity(frequencies, s_matrices)
        self.assertEqual(diagnostic.summary.severity, "hard")
        self.assertGreater(diagnostic.summary.worst_sigma_excess, 1e-3)


if __name__ == "__main__":
    unittest.main()
