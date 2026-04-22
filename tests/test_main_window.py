from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from sparams_utility.models.state import AppState
from sparams_utility.ui.main_window import MainWindow


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

    def setUp(self) -> None:
        self.window = MainWindow(AppState())

    def tearDown(self) -> None:
        self.window.close()

    def test_circuit_node_keeps_circuit_icon_when_output_is_sparameter(self) -> None:
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
        sp_plot_icon_path = Path(__file__).resolve().parents[1] / "src" / "sparams_utility" / "resources" / "SP_Plot.png"

        self.assertEqual(_icon_png_bytes(self.window._icon_for_kind("sp")), _resource_icon_png_bytes(sp_plot_icon_path))
        self.assertNotEqual(_icon_png_bytes(self.window._icon_for_kind("sp")), _icon_png_bytes(self.window._icon_for_kind("sparam-file")))
