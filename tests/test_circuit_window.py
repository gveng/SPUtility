from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

from sparams_utility.ui.circuit_window import BlockPreviewWidget


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


if __name__ == "__main__":
    unittest.main()