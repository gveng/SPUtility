from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sparams_utility.circuit_solver import solve_circuit_network, to_touchstone_string_with_format, _fill_missing_frequency_points
from sparams_utility.models.circuit import CircuitDocument, CircuitPortRef
from sparams_utility.models.state import AppState
from sparams_utility.touchstone_parser import (
    SParameterCell, TouchstoneFile, TouchstoneFormat, TouchstoneOptions,
    TouchstonePoint, MagnitudeTable,
)


class CircuitSolverTests(unittest.TestCase):
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

    def test_gap_adds_interpolated_points(self) -> None:
        """A 3× gap should introduce 2 interpolated points in the middle."""
        freqs = np.array([1e9, 2e9, 5e9, 6e9], dtype=float)   # gap between 2 GHz and 5 GHz (3 steps)
        s = np.zeros((4, 1, 1), dtype=np.complex128)
        s[:, 0, 0] = [0.0, 0.1, 0.4, 0.5]
        out_f, out_s = _fill_missing_frequency_points(freqs, s)
        self.assertGreater(out_f.size, freqs.size)
        # 3e9 should appear as a filled point
        diffs = np.abs(out_f - 3e9)
        mid_idx = int(np.argmin(diffs))
        self.assertLess(float(diffs[mid_idx]), 0.1e9, "Expected 3 GHz interpolated point not found")
        self.assertTrue(0.1 <= abs(out_s[mid_idx, 0, 0]) <= 0.4)

    def test_gap_interpolation_in_circuit(self) -> None:
        """Circuit solver should produce valid S matrices when Touchstone has a gap."""
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


if __name__ == "__main__":
    unittest.main()
