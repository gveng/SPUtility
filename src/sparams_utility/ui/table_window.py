from __future__ import annotations

from PySide6.QtWidgets import QApplication, QTableView, QVBoxLayout, QWidget


class TableWindow(QWidget):
    def __init__(self, title: str, model, file_id: str) -> None:
        super().__init__()
        self.file_id = file_id
        self.setWindowTitle(title)
        app = QApplication.instance()
        if app is not None:
            self.setWindowIcon(app.windowIcon())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        table = QTableView()
        table.setModel(model)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.resizeColumnsToContents()

        layout.addWidget(table)
