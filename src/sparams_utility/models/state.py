from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PySide6.QtCore import QObject, Signal

from sparams_utility.touchstone_parser import TouchstoneFile, parse_touchstone_file


@dataclass(frozen=True)
class LoadedTouchstone:
    file_id: str
    path: Path
    display_name: str
    data: TouchstoneFile


class AppState(QObject):
    file_added = Signal(object)
    files_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._files_by_id: Dict[str, LoadedTouchstone] = {}
        self._order: List[str] = []

    def load_files(self, paths: List[str]) -> Tuple[int, List[str]]:
        added_count = 0
        errors: List[str] = []

        for raw_path in paths:
            path = Path(raw_path).resolve()
            file_id = str(path)
            if file_id in self._files_by_id:
                continue

            try:
                parsed = parse_touchstone_file(path)
            except Exception as exc:  # pragma: no cover - runtime-facing error path
                errors.append(f"{path.name}: {exc}")
                continue

            loaded = LoadedTouchstone(
                file_id=file_id,
                path=path,
                display_name=path.name,
                data=parsed,
            )
            self._files_by_id[file_id] = loaded
            self._order.append(file_id)
            added_count += 1
            self.file_added.emit(loaded)

        if added_count:
            self.files_changed.emit()

        return added_count, errors

    def get_loaded_files(self) -> List[LoadedTouchstone]:
        return [self._files_by_id[file_id] for file_id in self._order]

    def get_file(self, file_id: str) -> LoadedTouchstone | None:
        return self._files_by_id.get(file_id)

    def clear_files(self) -> None:
        if not self._order:
            return
        self._files_by_id.clear()
        self._order.clear()
        self.files_changed.emit()
