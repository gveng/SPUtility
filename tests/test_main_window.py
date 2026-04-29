from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from sparams_utility.circuit_solver import ChannelSimResult
from sparams_utility.models.circuit import DriverSpec
from sparams_utility.models.state import AppState
from sparams_utility.ui.eye_diagram_window import EyeDiagramWindow
from sparams_utility.ui.main_window import MainWindow
from sparams_utility.ui.plot_window import PlotWindow


def _icon_png_bytes(icon) -> bytes:
    pixmap = icon.pixmap(32, 32)
    payload = QByteArray()
    buffer = QBuffer(payload)
    buffer.open(QIODevice.WriteOnly)
    pixmap.save(buffer, "PNG")
    return bytes(payload)


def _resource_icon_png_bytes(path: Path) -> bytes:
    return _icon_png_bytes(QIcon(str(path)))


class MainWindowTreeIconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])
        cls._repo_root = Path(__file__).resolve().parents[1]

    def setUp(self) -> None:
        self.window = MainWindow(AppState())

    def tearDown(self) -> None:
        self.window._project_dirty = False
        self.window.close()

    def _file_menu_labels(self) -> list[str]:
        for action in self.window.menuBar().actions():
            menu = action.menu()
            if menu is not None and action.text() == "File":
                return ["<separator>" if item.isSeparator() else item.text() for item in menu.actions()]
        self.fail("File menu not found")

    def _configure_exportable_project(self, project_path: Path) -> tuple[Path, Path, Path]:
        used_path = (self._repo_root / "LC.s2p").resolve()
        unused_path = (self._repo_root / "rlcserie.s2p").resolve()
        added, errors = self.window._state.load_files([str(used_path), str(unused_path)])
        self.assertEqual(errors, [])
        self.assertEqual(added, 2)

        data_dir = project_path.with_name(f"{project_path.stem}_Data")
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "exported_output.txt").write_text("runtime output", encoding="utf-8")

        self.window._project_path = str(project_path)
        self.window._project_data_dir = data_dir
        self.window._window_registry = {"sp": {}, "tdr": {}, "circuit": {}}

        plot_entry_id = "sp:export"
        self.window._window_registry["sp"][plot_entry_id] = {
            "id": plot_entry_id,
            "kind": "sp",
            "title": "Export Plot",
            "state": {
                "window_title": "Export Plot",
                "plot_name": "Export Plot",
                "plot_settings": {},
                "legend_position": [10.0, 10.0],
                "files": [
                    {
                        "file_path": str(used_path),
                        "file_name": used_path.name,
                        "legend_label": used_path.name,
                        "selected_parameters": ["S11"],
                    }
                ],
                "excluded_files": [],
            },
            "window_number": 1,
            "is_open": False,
            "eye_file": None,
            "sparam_file": None,
            "sparam_plot_file": None,
            "output_kind": None,
        }

        circuit_entry_id = "circuit:export"
        self.window._window_registry["circuit"][circuit_entry_id] = {
            "id": circuit_entry_id,
            "kind": "circuit",
            "title": "Circuit Composer #1 - Export Circuit",
            "state": {
                "window_title": "Circuit Composer #1 - Export Circuit",
                "circuit_name": "Export Circuit",
                "simulation_mode": "S-Parameters",
                "instances": [
                    {
                        "instance_id": "inst-1",
                        "source_file_id": str(used_path),
                        "display_label": used_path.name,
                        "nports": 2,
                        "position_x": 0.0,
                        "position_y": 0.0,
                        "block_kind": "touchstone",
                        "impedance_ohm": 50.0,
                        "symbol_scale": 1.0,
                        "rotation_deg": 0,
                        "mirror_horizontal": False,
                        "mirror_vertical": False,
                    }
                ],
                "connections": [],
                "external_ports": [],
                "differential_ports": [],
                "sweep": {
                    "fmin_hz": 1e7,
                    "fmax_hz": 1e10,
                    "fstep_hz": 1e7,
                    "display_unit": "GHz",
                },
            },
            "window_number": 1,
            "is_open": False,
            "eye_file": None,
            "sparam_file": None,
            "sparam_plot_file": None,
            "output_kind": None,
        }

        return used_path, unused_path, data_dir

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

    def _sample_touchstone_text(self, file_name: str = "LC.s2p") -> str:
        return (self._repo_root / file_name).read_text(encoding="utf-8")

    def test_circuit_node_keeps_circuit_icon_when_output_is_sparameter(self) -> None:
        circuit_icon_path = Path(__file__).resolve().parents[1] / "src" / "sparams_utility" / "resources" / "circuit_icon.svg"
        entry_id = "circuit:test"
        self.window._window_registry["circuit"][entry_id] = {
            "id": entry_id,
            "kind": "circuit",
            "title": "Circuit Window - Aries",
            "state": {"simulation_mode": "S-Parameters"},
            "window_number": 1,
            "is_open": False,
            "eye_file": None,
            "sparam_file": "Aries.s2p",
            "sparam_plot_file": "Aries_S.png",
            "output_kind": "sparam",
        }

        parent = QTreeWidgetItem(["Project"])
        self.window._add_tree_window_item(parent, "circuit", entry_id, "Circuit Window - Aries")
        item = parent.child(0)

        self.assertEqual(_icon_png_bytes(item.icon(0)), _icon_png_bytes(self.window._icon_for_kind("circuit")))
        self.assertEqual(_icon_png_bytes(item.icon(0)), _resource_icon_png_bytes(circuit_icon_path))
        self.assertNotEqual(_icon_png_bytes(item.icon(0)), _icon_png_bytes(self.window._icon_for_kind("sparam-file")))

    def test_tree_hides_extensions_for_project_and_output_files(self) -> None:
        self.window._project_path = r"D:\Visual_Studio_Code\SParamsUtility\Aries_vs_Parksville.json"
        entry_id = "circuit:eye"
        self.window._window_registry["circuit"][entry_id] = {
            "id": entry_id,
            "kind": "circuit",
            "title": "Circuit Window - Parksville",
            "state": {},
            "window_number": 1,
            "is_open": False,
            "eye_file": "Parksville_Eye.eye",
            "sparam_file": None,
            "sparam_plot_file": "Parksville_S.png",
            "output_kind": "eye",
        }

        self.window._refresh_project_tree()

        project_item = self.window._project_tree.topLevelItem(0)
        self.assertEqual(project_item.text(0), "Aries_vs_Parksville")

        circuit_item = project_item.child(0)
        self.assertEqual(circuit_item.text(0), "Parksville")
        self.assertEqual(circuit_item.child(0).text(0), "Parksville_Eye")
        self.assertEqual(circuit_item.child(1).text(0), "Parksville_S")

    def test_plot_window_node_uses_dedicated_sp_plot_icon(self) -> None:
        cosine_icon_path = Path(__file__).resolve().parents[1] / "src" / "sparams_utility" / "resources" / "cosine_wave.svg"

        self.assertEqual(_icon_png_bytes(self.window._icon_for_kind("sp")), _resource_icon_png_bytes(cosine_icon_path))
        self.assertEqual(_icon_png_bytes(self.window._icon_for_kind("sp")), _icon_png_bytes(self.window._icon_for_kind("sparam-file")))
        self.assertEqual(_icon_png_bytes(self.window._icon_for_kind("touchstone-file")), _resource_icon_png_bytes(cosine_icon_path))

    def test_eye_output_node_uses_purple_eye_outline_icon(self) -> None:
        eye_icon_path = Path(__file__).resolve().parents[1] / "src" / "sparams_utility" / "resources" / "eye_icon.svg"

        self.assertEqual(_icon_png_bytes(self.window._icon_for_kind("eye-file")), _resource_icon_png_bytes(eye_icon_path))
        self.assertNotEqual(_icon_png_bytes(self.window._icon_for_kind("eye-file")), _icon_png_bytes(self.window._icon_for_kind("sparam-file")))

    def test_recent_touchstone_actions_use_cosine_wave_icon(self) -> None:
        touchstone_path = (self._repo_root / "LC.s2p").resolve()
        self.window._recent_sparams = [str(touchstone_path)]

        self.window._rebuild_recent_file_menus()

        action = self.window._recent_sparams_menu.actions()[0]
        self.assertEqual(
            _icon_png_bytes(action.icon()),
            _icon_png_bytes(self.window._icon_for_kind("touchstone-file")),
        )

    def test_file_menu_separates_project_and_file_actions(self) -> None:
        self.assertEqual(
            self._file_menu_labels(),
            [
                "Open Project",
                "Save Project",
                "Save Project As",
                "Close Project",
                "Export Project",
                "Recent Projects",
                "<separator>",
                "Open File",
                "Recent S-Parameters",
                "<separator>",
                "Exit",
            ],
        )

    def test_close_project_clears_current_project_state(self) -> None:
        touchstone_path = (self._repo_root / "LC.s2p").resolve()
        added, errors = self.window._state.load_files([str(touchstone_path)])
        self.assertEqual(errors, [])
        self.assertEqual(added, 1)

        self.window._project_path = str(self._repo_root / "Project_Example" / "Aries_Meas_vs_Sym.json")
        self.window._project_data_dir = self._repo_root / "Project_Example"
        self.window._plot_counter = 3
        self.window._tdr_counter = 2
        self.window._circuit_counter = 1
        self.window._window_registry["sp"]["sp:test"] = {
            "id": "sp:test",
            "kind": "sp",
            "title": "Plot",
            "state": {},
            "window_number": 3,
            "is_open": False,
            "eye_file": None,
            "sparam_file": None,
            "sparam_plot_file": None,
            "output_kind": None,
        }
        self.window._open_windows[("sp", "sp:test")] = object()

        self.window._close_project()

        self.assertEqual(self.window._state.get_loaded_files(), [])
        self.assertEqual(self.window._window_registry, {"sp": {}, "tdr": {}, "circuit": {}})
        self.assertEqual(self.window._open_windows, {})
        self.assertEqual(self.window._plot_counter, 0)
        self.assertEqual(self.window._tdr_counter, 0)
        self.assertEqual(self.window._circuit_counter, 0)
        self.assertIsNone(self.window._project_path)
        self.assertIsNone(self.window._project_data_dir)
        self.assertFalse(self.window._project_dirty)

        project_item = self.window._project_tree.topLevelItem(0)
        self.assertIsNotNone(project_item)
        self.assertEqual(project_item.text(0), "Untitled Project")

    def test_tables_menu_touchstone_submenu_uses_cosine_wave_icon(self) -> None:
        touchstone_path = (self._repo_root / "LC.s2p").resolve()
        added, errors = self.window._state.load_files([str(touchstone_path)])
        self.assertEqual(errors, [])
        self.assertEqual(added, 1)

        submenu_action = self.window._tables_menu.actions()[0]
        self.assertEqual(
            _icon_png_bytes(submenu_action.icon()),
            _icon_png_bytes(self.window._icon_for_kind("touchstone-file")),
        )

    def test_delete_tree_output_file_removes_saved_file_and_registry_reference(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_name = "Parksville_Eye.eye"
            (data_dir / file_name).write_bytes(b"eye")
            self.window._project_data_dir = data_dir

            entry_id = "circuit:eye"
            self.window._window_registry["circuit"][entry_id] = {
                "id": entry_id,
                "kind": "circuit",
                "title": "Circuit Window - Parksville",
                "state": {},
                "window_number": 1,
                "is_open": False,
                "eye_file": file_name,
                "sparam_file": None,
                "sparam_plot_file": None,
                "output_kind": "eye",
            }

            deleted = self.window._delete_tree_output_file("circuit", entry_id, file_name, "eye")

            self.assertTrue(deleted)
            self.assertFalse((data_dir / file_name).exists())
            self.assertIsNone(self.window._window_registry["circuit"][entry_id]["eye_file"])
            self.assertIsNone(self.window._window_registry["circuit"][entry_id]["output_kind"])

    def test_delete_tree_window_entry_removes_circuit_and_saved_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_names = ["Parksville_Eye.eye", "Parksville.s2p", "Parksville_S.png"]
            for file_name in file_names:
                (data_dir / file_name).write_bytes(b"data")
            self.window._project_data_dir = data_dir

            entry_id = "circuit:delete"
            self.window._window_registry["circuit"][entry_id] = {
                "id": entry_id,
                "kind": "circuit",
                "title": "Circuit Window - Parksville",
                "state": {"simulation_mode": "S-Parameters"},
                "window_number": 1,
                "is_open": False,
                "eye_file": file_names[0],
                "sparam_file": file_names[1],
                "sparam_plot_file": file_names[2],
                "output_kind": "sparam",
            }

            deleted = self.window._delete_tree_window_entry("circuit", entry_id)

            self.assertTrue(deleted)
            self.assertNotIn(entry_id, self.window._window_registry["circuit"])
            for file_name in file_names:
                self.assertFalse((data_dir / file_name).exists())

    def test_duplicate_plot_window_entry_loads_touchstone_and_registers_copy(self) -> None:
        touchstone_path = (self._repo_root / "LC.s2p").resolve()
        entry_id = "sp:duplicate"
        self.window._window_registry["sp"][entry_id] = {
            "id": entry_id,
            "kind": "sp",
            "title": "Baseline Plot",
            "state": {
                "window_title": "Baseline Plot",
                "plot_name": "Baseline Plot",
                "plot_settings": {},
                "legend_position": [10.0, 10.0],
                "files": [
                    {
                        "file_path": str(touchstone_path),
                        "file_name": touchstone_path.name,
                        "legend_label": touchstone_path.name,
                        "selected_parameters": ["S11"],
                    }
                ],
                "excluded_files": [],
            },
            "window_number": 1,
            "is_open": False,
            "eye_file": None,
            "sparam_file": None,
            "sparam_plot_file": None,
            "output_kind": None,
        }

        duplicated = self.window._duplicate_tree_window_entry("sp", entry_id)

        self.assertTrue(duplicated)
        loaded_paths = {str(item.path.resolve()) for item in self.window._state.get_loaded_files()}
        self.assertIn(str(touchstone_path), loaded_paths)
        self.assertEqual(len(self.window._window_registry["sp"]), 2)

        duplicate_entries = [
            entry for key, entry in self.window._window_registry["sp"].items() if key != entry_id
        ]
        self.assertEqual(len(duplicate_entries), 1)
        duplicate_entry = duplicate_entries[0]
        self.assertIn("Copy", duplicate_entry["title"])
        self.assertEqual(duplicate_entry["state"]["files"][0]["file_path"], str(touchstone_path))

    def test_duplicate_circuit_window_entry_loads_touchstone_and_registers_copy(self) -> None:
        touchstone_path = (self._repo_root / "LC.s2p").resolve()
        entry_id = "circuit:duplicate"
        self.window._window_registry["circuit"][entry_id] = {
            "id": entry_id,
            "kind": "circuit",
            "title": "Circuit Composer #1 - Demo Circuit",
            "state": {
                "window_title": "Circuit Composer #1 - Demo Circuit",
                "circuit_name": "Demo Circuit",
                "simulation_mode": "S-Parameters",
                "instances": [
                    {
                        "instance_id": "inst-1",
                        "source_file_id": str(touchstone_path),
                        "display_label": touchstone_path.name,
                        "nports": 2,
                        "position_x": 0.0,
                        "position_y": 0.0,
                        "block_kind": "touchstone",
                        "impedance_ohm": 50.0,
                        "symbol_scale": 1.0,
                        "rotation_deg": 0,
                        "mirror_horizontal": False,
                        "mirror_vertical": False,
                    }
                ],
                "connections": [],
                "external_ports": [],
                "differential_ports": [],
                "sweep": {
                    "fmin_hz": 1e7,
                    "fmax_hz": 1e10,
                    "fstep_hz": 1e7,
                    "display_unit": "GHz",
                },
            },
            "window_number": 1,
            "is_open": False,
            "eye_file": None,
            "sparam_file": None,
            "sparam_plot_file": None,
            "output_kind": None,
        }

        duplicated = self.window._duplicate_tree_window_entry("circuit", entry_id)

        self.assertTrue(duplicated)
        loaded_paths = {str(item.path.resolve()) for item in self.window._state.get_loaded_files()}
        self.assertIn(str(touchstone_path), loaded_paths)
        self.assertEqual(len(self.window._window_registry["circuit"]), 2)

        duplicate_entries = [
            entry for key, entry in self.window._window_registry["circuit"].items() if key != entry_id
        ]
        self.assertEqual(len(duplicate_entries), 1)
        duplicate_entry = duplicate_entries[0]
        self.assertIn("Copy", duplicate_entry["title"])
        self.assertEqual(duplicate_entry["state"]["circuit_name"], "Demo Circuit Copy")

    def test_duplicate_eye_output_file_preserves_open_window_options(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_name = "Parksville_Eye.eye"
            self.window._project_data_dir = data_dir

            result = self._sample_eye_result()
            payload = self.window._build_eye_binary_payload(
                result,
                eye_span_ui=2,
                render_mode="Heatmap",
                quality_preset="Balanced",
                stat_enabled=False,
                noise_rms_mv=0.0,
                jitter_rms_ps=0.0,
            )
            (data_dir / file_name).write_bytes(payload)

            self.window._open_eye_binary_file(data_dir / file_name)
            eye_windows = self.window._find_output_eye_windows(file_name, "eye")
            self.assertEqual(len(eye_windows), 1)
            source_eye = eye_windows[0]

            source_eye._eye_span_combo.setCurrentText("3 UI")
            source_eye._render_mode_combo.setCurrentText("Lines")
            source_eye._quality_preset_combo.setCurrentText("Fast")

            duplicated = self.window._duplicate_tree_output_file(file_name, "eye")

            self.assertTrue(duplicated)
            eye_windows = self.window._find_output_eye_windows(file_name, "eye")
            self.assertEqual(len(eye_windows), 2)
            duplicate_eye = next(win for win in eye_windows if win is not source_eye)
            self.assertTrue(duplicate_eye.windowTitle().endswith("Copy"))
            self.assertEqual(duplicate_eye._eye_span_ui, source_eye._eye_span_ui)
            self.assertEqual(duplicate_eye._render_mode, source_eye._render_mode)
            self.assertEqual(duplicate_eye._quality_preset, source_eye._quality_preset)

    def test_open_eye_binary_file_registers_eye_window(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_name = "Parksville_Eye.eye"
            payload = self.window._build_eye_binary_payload(
                self._sample_eye_result(),
                eye_span_ui=2,
                render_mode="Heatmap",
                quality_preset="Balanced",
                stat_enabled=False,
                noise_rms_mv=0.0,
                jitter_rms_ps=0.0,
            )
            (data_dir / file_name).write_bytes(payload)

            eye_window = self.window._open_eye_binary_file(data_dir / file_name)

            self.assertIsNotNone(eye_window)
            self.assertIn(eye_window, self.window._windows.list_widgets("eye"))
            self.assertIs(self.window._windows.active_widget(), eye_window)

    def test_resize_all_graph_windows_resizes_eye_diagram_subwindows(self) -> None:
        first_eye = self.window._show_eye_diagram_window(
            self._sample_eye_result(),
            title="Eye 1",
            eye_span_ui=2,
            render_mode="Heatmap",
            quality_preset="Balanced",
            statistical_enabled=False,
            noise_rms_mv=0.0,
            jitter_rms_ps=0.0,
        )
        second_eye = self.window._show_eye_diagram_window(
            self._sample_eye_result(),
            title="Eye 2",
            eye_span_ui=2,
            render_mode="Heatmap",
            quality_preset="Balanced",
            statistical_enabled=False,
            noise_rms_mv=0.0,
            jitter_rms_ps=0.0,
        )

        first_eye.resize(820, 540)
        second_eye.resize(1160, 720)
        self.window._windows.set_active_widget(second_eye)

        self.window._resize_all_graph_windows()

        self.assertEqual(first_eye.size(), second_eye.size())

    def test_circuit_sparameter_result_saves_file_loads_tables_and_nests_plot(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self.window._project_data_dir = data_dir

            circuit_entry_id = "circuit:generated"
            self.window._window_registry["circuit"][circuit_entry_id] = {
                "id": circuit_entry_id,
                "kind": "circuit",
                "title": "Circuit Composer #1 - Demo Circuit",
                "state": {"simulation_mode": "S-Parameters"},
                "window_number": 1,
                "is_open": False,
                "eye_file": None,
                "sparam_file": None,
                "sparam_plot_file": None,
                "output_kind": None,
            }

            payload = {
                "circuit_name": "Demo Circuit",
                "nports": 2,
                "touchstone_text": self._sample_touchstone_text(),
                "frequency_unit": "GHz",
                "data_format": "RI",
                "file_name": "Equivalent Network",
            }

            self.window._on_circuit_sparameter_result_generated(circuit_entry_id, payload)

            saved_path = data_dir / "Equivalent_Network.s2p"
            self.assertEqual(payload["saved_file_name"], "Equivalent_Network.s2p")
            self.assertEqual(payload["saved_path"], str(saved_path))
            self.assertTrue(saved_path.exists())
            self.assertEqual(payload.get("error"), None)

            loaded_paths = {str(item.path.resolve()) for item in self.window._state.get_loaded_files()}
            self.assertEqual(loaded_paths, {str(saved_path.resolve())})

            self.assertEqual(len(self.window._window_registry["sp"]), 1)
            plot_entry_id, plot_entry = next(iter(self.window._window_registry["sp"].items()))
            self.assertEqual(plot_entry["parent_circuit_entry_id"], circuit_entry_id)
            self.assertEqual(plot_entry["state"]["files"][0]["file_path"], str(saved_path.resolve()))
            # Generated S-parameter plots open with all parameters
            # deselected so the user can opt in to traces of interest.
            self.assertEqual(plot_entry["state"]["files"][0]["selected_parameters"], [])
            self.assertEqual(plot_entry["state"]["excluded_files"], [])

            open_plot = self.window._open_windows[("sp", plot_entry_id)]
            self.assertIsInstance(open_plot, PlotWindow)
            self.assertEqual(open_plot.export_project_state()["files"][0]["file_path"], str(saved_path.resolve()))

            project_item = self.window._project_tree.topLevelItem(0)
            self.assertEqual(project_item.childCount(), 1)
            circuit_item = project_item.child(0)
            self.assertEqual(circuit_item.text(0), "Demo Circuit")
            self.assertEqual(circuit_item.childCount(), 1)
            self.assertEqual(circuit_item.child(0).text(0), "Equivalent_Network")

            payload["touchstone_text"] = self._sample_touchstone_text("rlcserie.s2p")
            self.window._on_circuit_sparameter_result_generated(circuit_entry_id, payload)

            self.assertEqual(len(self.window._window_registry["sp"]), 1)
            self.assertEqual(
                {str(item.path.resolve()) for item in self.window._state.get_loaded_files()},
                {str(saved_path.resolve())},
            )

    def test_build_export_project_bundle_copies_data_and_rewrites_source_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "DemoProject.json"
            used_path, unused_path, data_dir = self._configure_exportable_project(project_path)

            export_dir, export_project_path, export_zip_path, export_warnings = self.window._build_export_project_bundle(project_path)

            self.assertEqual(export_warnings, [])
            self.assertTrue(export_dir.exists())
            self.assertTrue(export_project_path.exists())
            self.assertTrue(export_zip_path.exists())
            self.assertTrue((export_dir / data_dir.name / "exported_output.txt").exists())
            self.assertTrue((export_dir / "DemoProject_Source" / used_path.name).exists())
            self.assertFalse((export_dir / "DemoProject_Source" / unused_path.name).exists())

            with open(export_project_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)

            self.assertEqual(
                payload["loaded_files"],
                [{"file_path": "DemoProject_Source/LC.s2p", "file_name": "LC.s2p"}],
            )
            self.assertEqual(
                payload["window_registry"]["sp"][0]["state"]["files"][0]["file_path"],
                "DemoProject_Source/LC.s2p",
            )
            self.assertEqual(
                payload["window_registry"]["circuit"][0]["state"]["instances"][0]["source_file_id"],
                "DemoProject_Source/LC.s2p",
            )

            with zipfile.ZipFile(export_zip_path) as archive:
                archive_names = set(archive.namelist())
            self.assertIn("DemoProject_Export/DemoProject.json", archive_names)
            self.assertIn("DemoProject_Export/DemoProject_Source/LC.s2p", archive_names)
            self.assertIn("DemoProject_Export/DemoProject_Data/exported_output.txt", archive_names)

    def test_load_project_from_export_resolves_relative_source_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "DemoProject.json"
            used_path, _unused_path, _data_dir = self._configure_exportable_project(project_path)
            export_dir, export_project_path, _export_zip_path, _warnings = self.window._build_export_project_bundle(project_path)

            loaded_window = MainWindow(AppState())
            try:
                with patch("sparams_utility.ui.main_window.QMessageBox.information", return_value=0):
                    with patch("sparams_utility.ui.main_window.QMessageBox.warning", return_value=0):
                        loaded_window._load_project_from_path(str(export_project_path))

                expected_source_path = str((export_dir / "DemoProject_Source" / used_path.name).resolve())
                loaded_paths = [str(item.path.resolve()) for item in loaded_window._state.get_loaded_files()]
                self.assertEqual(loaded_paths, [expected_source_path])

                plot_entry = next(iter(loaded_window._window_registry["sp"].values()))
                self.assertEqual(plot_entry["state"]["files"][0]["file_path"], expected_source_path)

                circuit_entry = next(iter(loaded_window._window_registry["circuit"].values()))
                self.assertEqual(circuit_entry["state"]["instances"][0]["source_file_id"], expected_source_path)
            finally:
                loaded_window._project_dirty = False
                loaded_window.close()
