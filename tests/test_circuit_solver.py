from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparams_utility.circuit_solver import solve_circuit_network, to_touchstone_string_with_format
from sparams_utility.models.circuit import CircuitDocument, CircuitPortRef
from sparams_utility.models.state import AppState


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


if __name__ == "__main__":
    unittest.main()
