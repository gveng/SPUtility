from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

from sparams_utility.models.state import AppState
from sparams_utility.ui.plot_window import PlotWindow


class PlotWindowProjectStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])
        cls._repo_root = Path(__file__).resolve().parents[1]

    def setUp(self) -> None:
        self.state = AppState()
        added, errors = self.state.load_files(
            [
                str(self._repo_root / "LC.s2p"),
                str(self._repo_root / "rlcserie.s2p"),
            ]
        )
        self.assertEqual(errors, [])
        self.assertEqual(added, 2)
        self.window = PlotWindow(self.state)

    def tearDown(self) -> None:
        self.window.close()

    def test_remove_file_from_this_plot_hides_row_and_persists(self) -> None:
        loaded_files = self.state.get_loaded_files()
        removed_id = loaded_files[0].file_id
        kept_id = loaded_files[1].file_id

        self.window._remove_file_from_plot(removed_id)

        self.assertEqual(self.window._selection_table.rowCount(), 1)
        self.assertEqual(self.window._row_to_fid, [kept_id])
        exported = self.window.export_project_state()
        self.assertEqual(len(exported["files"]), 1)
        self.assertIn(str(loaded_files[0].path), exported["excluded_files"])

        restored = PlotWindow(self.state)
        try:
            restored.apply_project_state(exported)
            self.assertEqual(restored._selection_table.rowCount(), 1)
            self.assertEqual(restored._row_to_fid, [kept_id])
            self.assertIn(removed_id, restored._excluded_file_ids)
        finally:
            restored.close()
