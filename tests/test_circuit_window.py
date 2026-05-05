from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint

from sparams_utility.circuit_solver import ChannelSimResult
from sparams_utility.models.circuit import DriverSpec
from sparams_utility.models.state import AppState
from sparams_utility.ui.circuit_window import BlockPreviewWidget, CircuitBlockItem, CircuitWindow
from sparams_utility.ui.main_window import MainWindow


class CircuitWindowPalettePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_touchstone_preview_grows_for_long_filename_without_spaces(self) -> None:
        short_preview = BlockPreviewWidget(
            "short.s8p",
            8,
            block_kind="touchstone",
        )
        long_preview = BlockPreviewWidget(
            "veep_40GHz_NO_DC_BLOCK_LONG_TOUCHSTONE_FILENAME_WITH_UNDERSCORES.s8p",
            8,
            block_kind="touchstone",
        )

        self.assertGreater(long_preview.sizeHint().height(), short_preview.sizeHint().height())
        self.assertGreaterEqual(long_preview.sizeHint().width(), short_preview.sizeHint().width())


class CircuitWindowEyeWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.main_window = MainWindow(AppState())
        self.circuit_window = CircuitWindow(self.main_window._state, window_number=1)
        self.main_window._windows.present(self.circuit_window)
        self.circuit_window.show()

    def tearDown(self) -> None:
        self.main_window._project_dirty = False
        self.main_window.close()

    def _sample_eye_result(self) -> ChannelSimResult:
        time_s = np.linspace(0.0, 12e-9, 600, dtype=float)
        waveform_v = np.where(np.sin(np.linspace(0.0, 18.0 * np.pi, 600)) >= 0.0, 0.4, 0.0)
        return ChannelSimResult(
            time_s=time_s,
            waveform_v=waveform_v,
            ui_s=1e-9,
            driver_spec=DriverSpec(bitrate_gbps=1.0, num_bits=128),
            is_differential=False,
        )

    def test_open_eye_window_uses_host_window_manager(self) -> None:
        eye_window = self.circuit_window._open_eye_window(self._sample_eye_result())

        self.assertIn(eye_window, self.main_window._windows.list_widgets("eye"))
        eye_widgets = [w for w in self.main_window._windows.list_widgets("eye") if w is eye_window]
        self.assertEqual(len(eye_widgets), 1)
        self.assertIn(eye_window, self.circuit_window.get_open_eye_windows())

    def test_export_equivalent_touchstone_uses_filename_prompt_and_fixed_ri_ghz(self) -> None:
        emitted_payloads: list[dict] = []
        self.circuit_window.sparameter_result_generated.connect(emitted_payloads.append)

        with patch("sparams_utility.ui.circuit_window.solve_circuit_network", return_value=SimpleNamespace(nports=2)) as solve_mock, \
             patch("sparams_utility.ui.circuit_window.to_touchstone_string_with_format", return_value="touchstone-data") as text_mock, \
             patch.object(self.circuit_window, "_confirm_passivity_export", return_value=True), \
             patch.object(self.circuit_window, "_describe_passivity", return_value="Passive"), \
             patch("sparams_utility.ui.circuit_window.QInputDialog.getText", return_value=("My Export", True)) as get_text_mock, \
             patch("sparams_utility.ui.circuit_window.QInputDialog.getItem") as get_item_mock, \
             patch("sparams_utility.ui.circuit_window.QFileDialog.getSaveFileName") as get_save_name_mock, \
               patch("sparams_utility.ui.circuit_window.QMessageBox.warning"), \
             patch("sparams_utility.ui.circuit_window.QMessageBox.information"):
            self.circuit_window._export_equivalent_touchstone()

        solve_mock.assert_called_once()
        text_mock.assert_called_once_with(solve_mock.return_value, data_format="RI", frequency_unit="GHz")
        get_text_mock.assert_called_once()
        get_item_mock.assert_not_called()
        get_save_name_mock.assert_not_called()
        self.assertEqual(len(emitted_payloads), 1)
        self.assertEqual(emitted_payloads[0]["data_format"], "RI")
        self.assertEqual(emitted_payloads[0]["frequency_unit"], "GHz")
        self.assertEqual(emitted_payloads[0]["file_name"], "My Export")


class CircuitWindowTransformActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.main_window = MainWindow(AppState())
        self.circuit_window = CircuitWindow(self.main_window._state, window_number=1)
        self.main_window._windows.present(self.circuit_window)

    def tearDown(self) -> None:
        self.main_window._project_dirty = False
        self.main_window.close()

    def _add_port_ground_block(self, *, rotation_deg: int = 0):
        instance = self.circuit_window._document.add_instance(
            source_file_id="port_ground",
            display_label="P1",
            nports=1,
            position_x=0.0,
            position_y=0.0,
            block_kind="port_ground",
            rotation_deg=rotation_deg,
        )
        block_item = CircuitBlockItem(self.circuit_window._scene, instance)
        self.circuit_window._scene.register_block(block_item)
        return instance

    def _choose_context_action(self, action_text: str):
        def _fake_exec(menu, _global_pos):
            for action in menu.actions():
                if action.text() == action_text:
                    return action
            return None

        return _fake_exec

    def test_mirror_horizontal_toggles_horizontal_flag_at_zero_rotation(self) -> None:
        instance = self._add_port_ground_block(rotation_deg=0)

        with patch("sparams_utility.ui.circuit_window.QMenu.exec", new=self._choose_context_action("Mirror horizontal")):
            self.circuit_window._show_block_context_menu(instance.instance_id, QPoint(0, 0))

        updated = self.circuit_window._document.get_instance(instance.instance_id)
        self.assertIsNotNone(updated)
        self.assertTrue(updated.mirror_horizontal)
        self.assertFalse(updated.mirror_vertical)

    def test_mirror_horizontal_after_90_deg_toggles_vertical_local_flag(self) -> None:
        instance = self._add_port_ground_block(rotation_deg=90)

        with patch("sparams_utility.ui.circuit_window.QMenu.exec", new=self._choose_context_action("Mirror horizontal")):
            self.circuit_window._show_block_context_menu(instance.instance_id, QPoint(0, 0))

        updated = self.circuit_window._document.get_instance(instance.instance_id)
        self.assertIsNotNone(updated)
        self.assertFalse(updated.mirror_horizontal)
        self.assertTrue(updated.mirror_vertical)


if __name__ == "__main__":
    unittest.main()