from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)


@dataclass(frozen=True)
class PlotSettings:
    x_log: bool = False
    y_log: bool = False
    x_autorange: bool = True
    y_autorange: bool = True
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None


class PlotSettingsDialog(QDialog):
    def __init__(self, current: PlotSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")
        self._settings = current

        layout = QVBoxLayout(self)

        scale_group = QGroupBox("Scale")
        scale_form = QFormLayout(scale_group)
        self.x_scale = QComboBox()
        self.x_scale.addItems(["Linear", "Logarithmic"])
        self.x_scale.setCurrentIndex(1 if current.x_log else 0)

        self.y_scale = QComboBox()
        self.y_scale.addItems(["Linear", "Logarithmic"])
        self.y_scale.setCurrentIndex(1 if current.y_log else 0)

        scale_form.addRow("X Axis", self.x_scale)
        scale_form.addRow("Y Axis", self.y_scale)

        range_group = QGroupBox("Range")
        range_layout = QGridLayout(range_group)

        self.x_mode = QComboBox()
        self.x_mode.addItems(["Auto", "Manual"])
        self.x_mode.setCurrentIndex(0 if current.x_autorange else 1)

        self.y_mode = QComboBox()
        self.y_mode.addItems(["Auto", "Manual"])
        self.y_mode.setCurrentIndex(0 if current.y_autorange else 1)

        self.x_min = QLineEdit("" if current.x_min is None else str(current.x_min))
        self.x_max = QLineEdit("" if current.x_max is None else str(current.x_max))
        self.y_min = QLineEdit("" if current.y_min is None else str(current.y_min))
        self.y_max = QLineEdit("" if current.y_max is None else str(current.y_max))

        range_layout.addWidget(QLabel("X Range"), 0, 0)
        range_layout.addWidget(self.x_mode, 0, 1)
        range_layout.addWidget(QLabel("X min"), 1, 0)
        range_layout.addWidget(self.x_min, 1, 1)
        range_layout.addWidget(QLabel("X max"), 1, 2)
        range_layout.addWidget(self.x_max, 1, 3)

        range_layout.addWidget(QLabel("Y Range"), 2, 0)
        range_layout.addWidget(self.y_mode, 2, 1)
        range_layout.addWidget(QLabel("Y min"), 3, 0)
        range_layout.addWidget(self.y_min, 3, 1)
        range_layout.addWidget(QLabel("Y max"), 3, 2)
        range_layout.addWidget(self.y_max, 3, 3)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(scale_group)
        layout.addWidget(range_group)
        layout.addWidget(buttons)

        self.x_mode.currentIndexChanged.connect(self._update_enabled_state)
        self.y_mode.currentIndexChanged.connect(self._update_enabled_state)
        self._update_enabled_state()

    def _update_enabled_state(self) -> None:
        x_manual = self.x_mode.currentText() == "Manual"
        y_manual = self.y_mode.currentText() == "Manual"
        self.x_min.setEnabled(x_manual)
        self.x_max.setEnabled(x_manual)
        self.y_min.setEnabled(y_manual)
        self.y_max.setEnabled(y_manual)

    def _parse_range(self, axis_name: str, minimum_edit: QLineEdit, maximum_edit: QLineEdit) -> tuple[float, float] | None:
        try:
            min_value = float(minimum_edit.text())
            max_value = float(maximum_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid values", f"Please enter valid numbers for {axis_name}.")
            return None

        if min_value >= max_value:
            QMessageBox.warning(self, "Invalid range", f"{axis_name}: minimum must be less than maximum.")
            return None

        return min_value, max_value

    def _on_accept(self) -> None:
        x_autorange = self.x_mode.currentText() == "Auto"
        y_autorange = self.y_mode.currentText() == "Auto"

        x_min = x_max = y_min = y_max = None

        if not x_autorange:
            x_range = self._parse_range("X Axis", self.x_min, self.x_max)
            if x_range is None:
                return
            x_min, x_max = x_range

        if not y_autorange:
            y_range = self._parse_range("Y Axis", self.y_min, self.y_max)
            if y_range is None:
                return
            y_min, y_max = y_range

        self._settings = PlotSettings(
            x_log=self.x_scale.currentText() == "Logarithmic",
            y_log=self.y_scale.currentText() == "Logarithmic",
            x_autorange=x_autorange,
            y_autorange=y_autorange,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        self.accept()

    @property
    def settings(self) -> PlotSettings:
        return self._settings
