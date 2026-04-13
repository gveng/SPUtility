from __future__ import annotations

import math
from typing import Any, List

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from sparams_utility.touchstone_parser import TouchstoneFile


class RawDataTableModel(QAbstractTableModel):
    _HEADERS = [
        "Frequency [Hz]",
        "Trace",
        "Raw Primary",
        "Raw Secondary",
        "Real",
        "Imag",
        "Magnitude [dB]",
    ]

    def __init__(self, touchstone: TouchstoneFile) -> None:
        super().__init__()
        self._rows: List[List[Any]] = []
        for point in touchstone.points:
            for row in point.s_matrix:
                for cell in row:
                    self._rows.append(
                        [
                            point.frequency_hz,
                            f"S{cell.row}{cell.col}",
                            cell.raw_primary,
                            cell.raw_secondary,
                            cell.complex_value.real,
                            cell.complex_value.imag,
                            cell.magnitude_db,
                        ]
                    )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:  # noqa: N802,E501
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._HEADERS[section]
        return section + 1

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        value = self._rows[index.row()][index.column()]
        if role == Qt.DisplayRole:
            if isinstance(value, float):
                if math.isinf(value):
                    return "-inf"
                return f"{value:.6g}"
            return str(value)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None


class MagnitudeTableModel(QAbstractTableModel):
    def __init__(self, touchstone: TouchstoneFile) -> None:
        super().__init__()
        self._frequencies = touchstone.magnitude_table.frequencies_hz
        self._trace_names = touchstone.trace_names
        self._traces = touchstone.magnitude_table.traces_db

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._frequencies)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return 1 + len(self._trace_names)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:  # noqa: N802,E501
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if section == 0:
                return "Frequency [Hz]"
            return f"{self._trace_names[section - 1]} [dB]"
        return section + 1

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None

        if index.column() == 0:
            value = self._frequencies[index.row()]
        else:
            trace = self._trace_names[index.column() - 1]
            value = self._traces[trace][index.row()]

        if role == Qt.DisplayRole:
            if isinstance(value, float):
                if math.isinf(value):
                    return "-inf"
                return f"{value:.6g}"
            return str(value)

        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)

        return None
