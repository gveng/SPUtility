from __future__ import annotations

import json
from typing import Dict

from PySide6.QtCore import QMimeData, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QFont, QFontMetrics, QKeySequence, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsObject,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sparams_utility.circuit_solver import solve_circuit_network, to_touchstone_string_with_format
from sparams_utility.models.circuit import CircuitDocument, CircuitPortRef, FrequencySweepSpec
from sparams_utility.models.state import AppState

_MIME_BLOCK_DEF = "application/x-sparams-block-def"
_BLOCK_WIDTH = 92.0
_PORT_RADIUS = 6.0
_SCHEMATIC_BG = QColor("#f7f7f7")
_FREQUENCY_UNIT_SCALE = {
    "Hz": 1.0,
    "KHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
}


def _contrast_foreground(background: QColor) -> QColor:
    return QColor("#f8fafc") if background.lightnessF() < 0.5 else QColor("#0f172a")


def _hex_port_polygon(center_x: float, center_y: float, radius: float) -> QPolygonF:
    shoulder = radius * 0.62
    half_height = radius * 0.9
    return QPolygonF(
        [
            QPointF(center_x - radius, center_y),
            QPointF(center_x - shoulder, center_y - half_height),
            QPointF(center_x + shoulder, center_y - half_height),
            QPointF(center_x + radius, center_y),
            QPointF(center_x + shoulder, center_y + half_height),
            QPointF(center_x - shoulder, center_y + half_height),
        ]
    )


def _block_value_suffix(block_kind: str) -> str:
    if block_kind == "lumped_r":
        return " Ohm"
    if block_kind == "lumped_l":
        return " H"
    if block_kind == "lumped_c":
        return " F"
    return " Ohm"


def _block_value_label(block_kind: str, value: float) -> str:
    if block_kind == "gnd":
        return "GND"
    if block_kind == "lumped_r":
        return f"R = {value:g} Ohm"
    if block_kind == "lumped_l":
        return f"L = {value:g} H"
    if block_kind == "lumped_c":
        return f"C = {value:g} F"
    return f"{value:g} Ohm"


def _label_band_height_for_text(text: str, width: float, point_size: float, *, minimum: float = 18.0) -> float:
    font = QFont()
    font.setPointSizeF(point_size)
    metrics = QFontMetrics(font)
    text_rect = metrics.boundingRect(
        0,
        0,
        max(24, int(width - 6.0)),
        1000,
        int(Qt.TextWordWrap | Qt.AlignCenter),
        text,
    )
    return max(minimum, float(text_rect.height()) + 8.0)


class BlockPreviewWidget(QWidget):
    def __init__(
        self,
        label: str,
        nports: int,
        *,
        block_kind: str = "touchstone",
        impedance_ohm: float = 50.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._nports = nports
        self._block_kind = block_kind
        self._impedance_ohm = impedance_ohm
        if block_kind == "touchstone":
            body_h = max(44, (max(nports, 2) + 1) * 14)
            self.setMinimumHeight(body_h + 40)
        elif block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            self.setMinimumHeight(84)
        elif block_kind in {"port_diff", "port_ground", "gnd"}:
            self.setMinimumHeight(80)
        else:
            self.setMinimumHeight(max(64, 32 + max(nports, 2) * 10))

    def sizeHint(self) -> QSize:  # noqa: N802
        if self._block_kind == "touchstone":
            body_h = max(44, (max(self._nports, 2) + 1) * 14)
            return QSize(170, body_h + 40)
        if self._block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            return QSize(170, 84)
        if self._block_kind in {"port_diff", "port_ground", "gnd"}:
            return QSize(170, 80)
        return QSize(170, max(64, 32 + max(self._nports, 2) * 10))

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), _SCHEMATIC_BG)

        is_lumped = self._block_kind in {"lumped_r", "lumped_l", "lumped_c"}
        if is_lumped:
            cx = self.rect().center().x()
            rect = QRectF(cx - _BLOCK_WIDTH / 2.0, 4.0, _BLOCK_WIDTH, 44.0)
        elif self._block_kind == "port_diff":
            cx = self.rect().center().x()
            rect = QRectF(cx - _BLOCK_WIDTH / 2.0, 4.0, _BLOCK_WIDTH, 44.0)
        elif self._block_kind == "port_ground":
            cx = self.rect().center().x()
            rect = QRectF(cx - _BLOCK_WIDTH / 2.0, 4.0, _BLOCK_WIDTH, 44.0)
        elif self._block_kind == "gnd":
            cx = self.rect().center().x()
            rect = QRectF(cx - _BLOCK_WIDTH / 2.0, 4.0, _BLOCK_WIDTH, 44.0)
        elif self._block_kind == "touchstone":
            cx = self.rect().center().x()
            body_h = max(44.0, (max(self._nports, 2) + 1) * 14.0)
            rect = QRectF(cx - _BLOCK_WIDTH / 2.0, 4.0, _BLOCK_WIDTH, body_h)
        else:
            rect = self.rect().adjusted(18, 8, -18, -28)
        painter.setPen(QPen(QColor("#1e40af"), 1.6))
        if self._block_kind not in {"lumped_r", "lumped_l", "lumped_c", "gnd", "port_diff", "port_ground", "touchstone"}:
            painter.setBrush(QBrush(QColor("#60a5fa")))
            painter.drawRoundedRect(rect, 6.0, 6.0)

        font = painter.font()
        font.setPointSizeF(max(7.2, font.pointSizeF() - 1.6))
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        secondary = _block_value_label(self._block_kind, self._impedance_ohm)
        if self._block_kind == "touchstone":
            secondary = f"{self._nports} ports"
        if self._block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            painter.drawText(QRectF(rect.left(), rect.bottom() + 2.0, rect.width(), 18.0), Qt.AlignCenter, secondary)
        if self._block_kind != "touchstone":
            painter.drawText(QRectF(rect.left(), rect.bottom() + 18.0, rect.width(), 18.0), Qt.AlignCenter, self._label)

        if self._block_kind == "touchstone":
            self._draw_touchstone_preview(painter, rect)
            return
        if self._block_kind == "lumped_r":
            self._draw_lumped_symbol(
                painter,
                rect,
                foreground_color=QColor("#1e293b"),
            )
            return
        if self._block_kind == "port_diff":
            self._draw_differential_port_symbol(painter, rect)
            return
        if self._block_kind == "port_ground":
            self._draw_ground_port_symbol(painter, rect)
            return
        if self._block_kind == "gnd":
            self._draw_gnd_symbol(painter, rect)
            return
        if self._block_kind in {"lumped_l", "lumped_c"}:
            return

        left_count = (self._nports + 1) // 2
        right_count = self._nports - left_count
        for idx in range(1, self._nports + 1):
            if idx <= left_count:
                slot = idx
                span = max(left_count, 1)
                x = rect.left()
                label_rect = QRectF(rect.left() + 10.0, rect.top(), 28.0, rect.height())
            else:
                slot = idx - left_count
                span = max(right_count, 1)
                x = rect.right()
                label_rect = QRectF(rect.right() - 38.0, rect.top(), 28.0, rect.height())
            y = rect.top() + (rect.height() / (span + 1)) * slot
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawPolygon(_hex_port_polygon(float(x), float(y), _PORT_RADIUS))
            painter.drawText(
                QRectF(label_rect.left(), y - 10.0, label_rect.width(), 20.0),
                Qt.AlignCenter,
                str(idx),
            )

    def _draw_touchstone_preview(self, painter: QPainter, rect: QRectF) -> None:
        fg = QColor("#1e293b")
        # Outlined rounded rect (same style as port_diff)
        painter.setPen(QPen(fg, 2.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 4.0, 4.0)
        # Port hex markers on left/right edges with port numbers inside
        left_count = (self._nports + 1) // 2
        right_count = self._nports - left_count
        painter.setBrush(QBrush(QColor("#ffffff")))
        for idx in range(1, self._nports + 1):
            if idx <= left_count:
                slot = idx
                span = max(left_count, 1)
                x = rect.left()
                label_rect = QRectF(rect.left() + 8.0, 0.0, 20.0, 18.0)
            else:
                slot = idx - left_count
                span = max(right_count, 1)
                x = rect.right()
                label_rect = QRectF(rect.right() - 28.0, 0.0, 20.0, 18.0)
            y = rect.top() + (rect.height() / (span + 1)) * slot
            painter.drawPolygon(_hex_port_polygon(float(x), float(y), _PORT_RADIUS))
            painter.setPen(QPen(fg, 1.0))
            painter.drawText(
                QRectF(label_rect.left(), y - 9.0, label_rect.width(), 18.0),
                Qt.AlignCenter,
                str(idx),
            )
            painter.setPen(QPen(fg, 2.0))
        # File name below rect — use word wrap to ensure full visibility
        font = painter.font()
        font.setPointSizeF(max(6.5, font.pointSizeF() - 0.5))
        painter.setFont(font)
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        label_area = QRectF(rect.left() - 10.0, rect.bottom() + 2.0, rect.width() + 20.0, 36.0)
        painter.drawText(label_area, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self._label)

    def _draw_ground_port_symbol(self, painter: QPainter, rect: QRectF) -> None:
        fg = QColor("#1e293b")
        scale = 1.0
        y = rect.center().y()
        x0 = rect.left()
        total_w = _BLOCK_WIDTH * scale
        right_end = x0 + total_w
        terminal_length = 20.0 * scale
        x1 = x0 + terminal_length
        x2 = right_end - terminal_length
        painter.setPen(QPen(fg, 2.0))
        # Hex port marker on left
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPolygon(_hex_port_polygon(float(x0), float(y), _PORT_RADIUS))
        # Box leaving 50% of terminals outside
        half_term = terminal_length / 2.0
        box = QRectF(x1 - half_term, rect.top() + 2.0, (x2 + half_term) - (x1 - half_term), rect.height() - 4.0)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(box, 3.0, 3.0)
        # Left terminal lead
        painter.drawLine(QPointF(x0, y), QPointF(x1, y))
        # Right terminal lead to GND
        gnd_x = right_end
        painter.drawLine(QPointF(x2, y), QPointF(gnd_x, y))
        # Horizontal zigzag
        zig_segments = 7
        step = (x2 - x1) / zig_segments
        amplitude = abs(step) * 1.7320508076 * 0.75
        points = [
            QPointF(x1, y),
            QPointF(x1 + step, y - amplitude),
            QPointF(x1 + 2 * step, y + amplitude),
            QPointF(x1 + 3 * step, y - amplitude),
            QPointF(x1 + 4 * step, y + amplitude),
            QPointF(x1 + 5 * step, y - amplitude),
            QPointF(x1 + 6 * step, y + amplitude),
            QPointF(x2, y),
        ]
        for idx in range(len(points) - 1):
            painter.drawLine(points[idx], points[idx + 1])
        # GND symbol at right end
        gw = 10.0 * scale
        painter.drawLine(QPointF(gnd_x, y - gw), QPointF(gnd_x, y + gw))
        painter.drawLine(QPointF(gnd_x + 3.0, y - gw * 0.6), QPointF(gnd_x + 3.0, y + gw * 0.6))
        painter.drawLine(QPointF(gnd_x + 6.0, y - gw * 0.25), QPointF(gnd_x + 6.0, y + gw * 0.25))
        # Impedance label below
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        font = painter.font()
        font.setPointSizeF(max(6.8, font.pointSizeF() - 2.0))
        painter.setFont(font)
        label = _block_value_label(self._block_kind, self._impedance_ohm)
        painter.drawText(QRectF(rect.left(), rect.bottom() + 2.0, rect.width(), 18.0), Qt.AlignCenter, label)

    def _draw_differential_port_symbol(self, painter: QPainter, rect: QRectF) -> None:
        fg = QColor("#1e293b")
        y = rect.center().y()
        x0 = rect.left()
        x3 = rect.right()
        terminal_length = 20.0
        x1 = x0 + terminal_length
        x2 = x3 - terminal_length
        painter.setPen(QPen(fg, 2.0))
        # Hex port markers
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPolygon(_hex_port_polygon(float(x0), float(y), _PORT_RADIUS))
        painter.drawPolygon(_hex_port_polygon(float(x3), float(y), _PORT_RADIUS))
        # Box leaving 50% of terminals outside
        half_term = terminal_length / 2.0
        box = QRectF(x1 - half_term, rect.top() + 2.0, (x2 + half_term) - (x1 - half_term), rect.height() - 4.0)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(box, 4.0, 4.0)
        # Terminal leads
        painter.drawLine(QPointF(x0, y), QPointF(x1, y))
        painter.drawLine(QPointF(x2, y), QPointF(x3, y))
        # Horizontal zigzag (same as R)
        zig_segments = 7
        step = (x2 - x1) / zig_segments
        amplitude = abs(step) * 1.7320508076 * 0.75  # tan(60 deg), 75% length
        points = [
            QPointF(x1, y),
            QPointF(x1 + step, y - amplitude),
            QPointF(x1 + 2 * step, y + amplitude),
            QPointF(x1 + 3 * step, y - amplitude),
            QPointF(x1 + 4 * step, y + amplitude),
            QPointF(x1 + 5 * step, y - amplitude),
            QPointF(x1 + 6 * step, y + amplitude),
            QPointF(x2, y),
        ]
        for idx in range(len(points) - 1):
            painter.drawLine(points[idx], points[idx + 1])
        # Impedance label below
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        font = painter.font()
        font.setPointSizeF(max(6.8, font.pointSizeF() - 2.0))
        painter.setFont(font)
        label = _block_value_label(self._block_kind, self._impedance_ohm)
        painter.drawText(QRectF(rect.left(), rect.bottom() + 2.0, rect.width(), 18.0), Qt.AlignCenter, label)

    def _draw_gnd_symbol(self, painter: QPainter, rect: QRectF) -> None:
        fg = QColor("#1e293b")
        y = rect.center().y()
        x0 = rect.left()
        gnd_x = x0 + _BLOCK_WIDTH * 0.3
        painter.setPen(QPen(fg, 2.0))
        # Hex port marker on left
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPolygon(_hex_port_polygon(float(x0), float(y), _PORT_RADIUS))
        # Straight line from port to GND
        painter.drawLine(QPointF(x0, y), QPointF(gnd_x, y))
        # GND symbol at right end
        gw = 10.0
        painter.drawLine(QPointF(gnd_x, y - gw), QPointF(gnd_x, y + gw))
        painter.drawLine(QPointF(gnd_x + 3.0, y - gw * 0.6), QPointF(gnd_x + 3.0, y + gw * 0.6))
        painter.drawLine(QPointF(gnd_x + 6.0, y - gw * 0.25), QPointF(gnd_x + 6.0, y + gw * 0.25))

    def _draw_lumped_symbol(self, painter: QPainter, rect: QRectF, *, foreground_color: QColor) -> None:
        y = rect.center().y()
        x0 = rect.left()
        x3 = rect.right()
        terminal_length = 20.0
        x1 = x0 + terminal_length
        x2 = x3 - terminal_length
        painter.setPen(QPen(foreground_color, 2.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPolygon(_hex_port_polygon(float(x0), float(y), _PORT_RADIUS))
        painter.drawPolygon(_hex_port_polygon(float(x3), float(y), _PORT_RADIUS))
        painter.drawLine(QPointF(x0, y), QPointF(x1, y))
        painter.drawLine(QPointF(x2, y), QPointF(x3, y))
        zig_segments = 7
        step = (x2 - x1) / zig_segments
        amplitude = abs(step) * 1.7320508076 * 0.75  # tan(60 deg), 75% length
        points = [
            QPointF(x1, y),
            QPointF(x1 + step, y - amplitude),
            QPointF(x1 + 2 * step, y + amplitude),
            QPointF(x1 + 3 * step, y - amplitude),
            QPointF(x1 + 4 * step, y + amplitude),
            QPointF(x1 + 5 * step, y - amplitude),
            QPointF(x1 + 6 * step, y + amplitude),
            QPointF(x2, y),
        ]
        for idx in range(len(points) - 1):
            painter.drawLine(points[idx], points[idx + 1])


def _build_palette_payload(
    *,
    block_kind: str,
    label: str,
    nports: int,
    source_file_id: str,
    impedance_ohm: float,
) -> str:
    return json.dumps(
        {
            "block_kind": block_kind,
            "label": label,
            "nports": nports,
            "source_file_id": source_file_id,
            "impedance_ohm": impedance_ohm,
        }
    )


def _special_palette_blocks() -> list[dict]:
    return [
        {
            "block_kind": "gnd",
            "label": "GND",
            "nports": 1,
            "source_file_id": "__special__:gnd",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "port_ground",
            "label": "Port To Ground",
            "nports": 1,
            "source_file_id": "__special__:port_ground",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "port_diff",
            "label": "Differential Port",
            "nports": 2,
            "source_file_id": "__special__:port_diff",
            "impedance_ohm": 100.0,
        },
        {
            "block_kind": "lumped_r",
            "label": "Resistor",
            "nports": 2,
            "source_file_id": "__special__:lumped_r",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "lumped_l",
            "label": "Inductor",
            "nports": 2,
            "source_file_id": "__special__:lumped_l",
            "impedance_ohm": 1e-9,
        },
        {
            "block_kind": "lumped_c",
            "label": "Capacitor",
            "nports": 2,
            "source_file_id": "__special__:lumped_c",
            "impedance_ohm": 1e-12,
        },
    ]


class FilePaletteList(QListWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setSpacing(6)
        self.setUniformItemSizes(False)
        self.setStyleSheet(
            "QListWidget {"
            " background-color: #f7f7f7;"
            " color: #0f172a;"
            " border: 1px solid #d1d5db;"
            "}"
            "QListWidget::item { border: none; padding: 0px; }"
            "QListWidget::item:selected { background-color: #dbeafe; }"
            "QListWidget::item:hover { background-color: #e5e7eb; }"
        )

    def startDrag(self, supported_actions) -> None:  # noqa: ARG002
        item = self.currentItem()
        if item is None:
            return
        payload = item.data(Qt.UserRole)
        if not isinstance(payload, str) or not payload:
            return
        mime = QMimeData()
        mime.setData(_MIME_BLOCK_DEF, payload.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class CircuitCanvasView(QGraphicsView):
    fileDropped = Signal(str, QPointF)
    deletePressed = Signal()
    connectionContextMenuRequested = Signal(str, QPoint)
    blockContextMenuRequested = Signal(str, QPoint)

    def __init__(self, scene: QGraphicsScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.setAcceptDrops(True)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setBackgroundBrush(QColor("#f7f7f7"))
        self.setFrameShape(QFrame.StyledPanel)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(_MIME_BLOCK_DEF):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(_MIME_BLOCK_DEF):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(_MIME_BLOCK_DEF):
            file_id = bytes(event.mimeData().data(_MIME_BLOCK_DEF)).decode("utf-8")
            scene_pos = self.mapToScene(event.position().toPoint())
            self.fileDropped.emit(file_id, scene_pos)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.matches(QKeySequence.Delete):
            self.deletePressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        if isinstance(item, CircuitConnectionItem):
            self.connectionContextMenuRequested.emit(item.connection_id, event.globalPos())
            event.accept()
            return
        if isinstance(item, PortItem):
            self.blockContextMenuRequested.emit(item.owner.instance.instance_id, event.globalPos())
            event.accept()
            return
        if isinstance(item, CircuitBlockItem):
            self.blockContextMenuRequested.emit(item.instance.instance_id, event.globalPos())
            event.accept()
            return
        super().contextMenuEvent(event)


class PortItem(QGraphicsPolygonItem):
    def __init__(self, owner: "CircuitBlockItem", port_number: int, x: float, y: float) -> None:
        super().__init__(_hex_port_polygon(0.0, 0.0, _PORT_RADIUS), owner)
        self.owner = owner
        self.port_number = port_number
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#1f2937"), 1.5))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setToolTip(f"{owner.instance.display_label} - Port {port_number}")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        super().mousePressEvent(event)
        scene = self.scene()
        if isinstance(scene, CircuitScene):
            scene.handle_port_clicked(self)

    def set_pending(self, active: bool) -> None:
        self.setBrush(QBrush(QColor("#f59e0b" if active else "#ffffff")))

    def set_exported(self, active: bool) -> None:
        if active:
            self.setBrush(QBrush(QColor("#22c55e")))
            return
        self.setBrush(QBrush(QColor("#ffffff")))

    @property
    def port_ref(self) -> CircuitPortRef:
        return CircuitPortRef(self.owner.instance.instance_id, self.port_number)


class CircuitConnectionItem(QGraphicsLineItem):
    def __init__(self, connection_id: str, port_a: PortItem, port_b: PortItem) -> None:
        super().__init__()
        self.connection_id = connection_id
        self.port_a = port_a
        self.port_b = port_b
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QColor("#2563eb"), 4.0))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(-1.0)
        self.refresh_geometry()

    def refresh_geometry(self) -> None:
        pos_a = self.port_a.sceneBoundingRect().center()
        pos_b = self.port_b.sceneBoundingRect().center()
        self.setLine(pos_a.x(), pos_a.y(), pos_b.x(), pos_b.y())

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self.setPen(QPen(QColor("#1d4ed8"), 6.0))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        color = "#ef4444" if self.isSelected() else "#2563eb"
        self.setPen(QPen(QColor(color), 4.0))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        super().mousePressEvent(event)
        self.setPen(QPen(QColor("#ef4444"), 4.0))


class CircuitBlockItem(QGraphicsObject):
    def __init__(self, scene: "CircuitScene", instance) -> None:
        super().__init__()
        self._scene_ref = scene
        self.instance = instance
        self._port_items: Dict[int, PortItem] = {}
        self._port_label: str = ""
        self._symbol_scale = max(0.5, min(3.0, float(getattr(instance, "symbol_scale", 1.0))))
        self._is_touchstone = instance.block_kind == "touchstone"
        self._block_width = _BLOCK_WIDTH * self._symbol_scale
        if instance.block_kind in {"lumped_r", "lumped_l", "lumped_c", "port_diff", "port_ground", "gnd"}:
            self._body_height = 44.0 * self._symbol_scale
        elif instance.block_kind == "touchstone":
            self._body_height = max(44.0, (max(instance.nports, 2) + 1) * 14.0) * self._symbol_scale
        else:
            self._body_height = max(48.0 * self._symbol_scale, (18.0 + (max(instance.nports, 2) * 10.0)) * self._symbol_scale)
        label_text = instance.display_label
        if instance.block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            label_text = _block_value_label(instance.block_kind, instance.impedance_ohm)
        self._label_band_height = (
            _label_band_height_for_text(label_text, self._block_width, 7.2, minimum=18.0)
            if instance.block_kind != "gnd"
            else 0.0
        )
        self._height = self._body_height + self._label_band_height
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setPos(instance.position_x, instance.position_y)
        self._build_ports()
        self._apply_visual_transform()

    def _build_ports(self) -> None:
        if self.instance.block_kind == "gnd":
            self._port_items[1] = PortItem(self, 1, 0.0, 0.0)
            self._apply_port_layout()
            return
        if self.instance.block_kind == "port_ground":
            self._port_items[1] = PortItem(self, 1, 0.0, 0.0)
            self._apply_port_layout()
            return
        if self.instance.block_kind == "port_diff":
            self._port_items[1] = PortItem(self, 1, 0.0, 0.0)
            self._port_items[2] = PortItem(self, 2, 0.0, 0.0)
            self._apply_port_layout()
            return
        left_count = (self.instance.nports + 1) // 2
        right_count = self.instance.nports - left_count
        for idx in range(1, self.instance.nports + 1):
            self._port_items[idx] = PortItem(self, idx, 0.0, 0.0)
        self._apply_port_layout()

    def _mx(self, x: float) -> float:
        if self.instance.mirror_horizontal:
            return self._block_width - x
        return x

    def _my(self, y: float) -> float:
        if self.instance.mirror_vertical:
            return self._body_height - y
        return y

    def _base_port_position(self, port_number: int) -> tuple[float, float]:
        if self.instance.block_kind == "gnd":
            return 0.0, self._body_height / 2.0
        if self.instance.block_kind == "port_ground":
            return 0.0, self._body_height / 2.0
        if self.instance.block_kind == "port_diff":
            if port_number == 1:
                return 0.0, self._body_height / 2.0
            return self._block_width, self._body_height / 2.0

        left_count = (self.instance.nports + 1) // 2
        right_count = self.instance.nports - left_count
        if port_number <= left_count:
            slot = port_number
            span = max(left_count, 1)
            x = 0.0
        else:
            slot = port_number - left_count
            span = max(right_count, 1)
            x = self._block_width
        y = (self._body_height / (span + 1)) * slot
        return x, y

    def _apply_port_layout(self) -> None:
        for port_number, port_item in self._port_items.items():
            base_x, base_y = self._base_port_position(port_number)
            port_item.setPos(self._mx(base_x), self._my(base_y))

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(0.0, 0.0, self._block_width, self._height)

    def paint(self, painter, option, widget=None) -> None:  # noqa: ANN001, ARG002
        rect = QRectF(0.0, 0.0, self._block_width, self._body_height)
        symbol_font = painter.font()
        symbol_font.setPointSizeF(max(6.8, symbol_font.pointSizeF() - 2.0))
        symbol_font.setBold(False)
        painter.setFont(symbol_font)
        
        # Handle GND specially
        if self.instance.block_kind == "gnd":
            self._draw_gnd_canvas(painter, rect)
            return
        
        # Draw background for non-lumped, non-GND blocks
        fg = QColor("#1e293b")
        fill = QColor("#3b82f6") if self.isSelected() else QColor("#60a5fa")
        painter.setPen(QPen(QColor("#1e40af"), 1.6))
        if self.instance.block_kind == "touchstone":
            # Outlined rounded rect (same style as port blocks)
            painter.setPen(QPen(fg, 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, 4.0, 4.0)
        elif self.instance.block_kind not in {"lumped_r", "lumped_l", "lumped_c", "gnd", "port_diff", "port_ground"}:
            painter.setBrush(QBrush(fill))
            painter.drawRoundedRect(rect, 6.0, 6.0)
        
        if self.instance.block_kind == "touchstone":
            # Port numbers inside the box
            painter.setPen(QPen(fg, 1.0))
            for port_number, port_item in self._port_items.items():
                point = port_item.pos()
                label_x = 6.0 if point.x() <= 0 else rect.width() - 22.0
                painter.drawText(QRectF(label_x, point.y() - 9.0, 20.0, 18.0), Qt.AlignCenter, str(port_number))
        elif self.instance.block_kind == "lumped_r":
            self._draw_lumped_symbol(painter, rect)
        elif self.instance.block_kind == "port_diff":
            self._draw_diff_symbol(painter, rect)
        elif self.instance.block_kind == "port_ground":
            self._draw_ground_symbol(painter, rect)
        elif self.instance.block_kind in {"lumped_l", "lumped_c"}:
            pass
        else:
            for port_number, port_item in self._port_items.items():
                point = port_item.pos()
                label_x = 6.0 if point.x() <= 0 else rect.width() - 22.0
                painter.drawText(QRectF(label_x, point.y() - 9.0, 20.0, 18.0), Qt.AlignCenter, str(port_number))

        # Draw label text below block
        font = painter.font()
        font.setPointSizeF(max(7.2, font.pointSizeF() - 1.2))
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#1f2937"), 1.0))
        label = self.instance.display_label
        if self.instance.block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            label = _block_value_label(self.instance.block_kind, self.instance.impedance_ohm)
        elif self.instance.block_kind in {"port_diff", "port_ground"} and self._port_label:
            label = self._port_label
        painter.drawText(
            QRectF(0.0, self._body_height + 2.0, self._block_width, self._label_band_height - 4.0),
            Qt.AlignCenter | Qt.TextWordWrap,
            label,
        )

    def _draw_ground_symbol(self, painter: QPainter, rect: QRectF) -> None:
        scale = 1.0 if self._is_touchstone else self._symbol_scale
        y = rect.height() / 2.0
        port_x = self._port_items[1].pos().x()
        total_w = _BLOCK_WIDTH * scale
        right_end = port_x + total_w
        terminal_length = 20.0 * scale
        x1 = port_x + terminal_length
        x2 = right_end - terminal_length
        fg = QColor("#1e293b")
        painter.setPen(QPen(fg, 2.0))
        # Box leaving 50% of terminals outside
        half_term = terminal_length / 2.0
        box = QRectF(x1 - half_term, 2.0, (x2 + half_term) - (x1 - half_term), rect.height() - 4.0)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(box, 3.0, 3.0)
        # Left terminal lead (from port to zigzag)
        painter.drawLine(QPointF(port_x, y), QPointF(x1, y))
        # Right terminal lead (from zigzag to GND symbol)
        gnd_x = right_end
        painter.drawLine(QPointF(x2, y), QPointF(gnd_x, y))
        # Horizontal zigzag
        zig_segments = 7
        step = (x2 - x1) / zig_segments
        amplitude = abs(step) * 1.7320508076 * 0.75
        points = [
            QPointF(x1, y),
            QPointF(x1 + step, y - amplitude),
            QPointF(x1 + 2 * step, y + amplitude),
            QPointF(x1 + 3 * step, y - amplitude),
            QPointF(x1 + 4 * step, y + amplitude),
            QPointF(x1 + 5 * step, y - amplitude),
            QPointF(x1 + 6 * step, y + amplitude),
            QPointF(x2, y),
        ]
        for idx in range(len(points) - 1):
            painter.drawLine(points[idx], points[idx + 1])
        # GND symbol at right end
        gw = 10.0 * scale
        painter.drawLine(QPointF(gnd_x, y - gw), QPointF(gnd_x, y + gw))
        painter.drawLine(QPointF(gnd_x + 3.0, y - gw * 0.6), QPointF(gnd_x + 3.0, y + gw * 0.6))
        painter.drawLine(QPointF(gnd_x + 6.0, y - gw * 0.25), QPointF(gnd_x + 6.0, y + gw * 0.25))

    def _draw_diff_symbol(self, painter: QPainter, rect: QRectF) -> None:
        scale = 1.0 if self._is_touchstone else self._symbol_scale
        y = rect.height() / 2.0
        port_left = self._port_items[1].pos().x()
        port_right = self._port_items[2].pos().x()
        terminal_length = 20.0 * scale
        x1 = port_left + terminal_length if port_left <= port_right else port_left - terminal_length
        x2 = port_right - terminal_length if port_left <= port_right else port_right + terminal_length
        fg = QColor("#1e293b")
        painter.setPen(QPen(fg, 2.0))
        # Box leaving 50% of terminals outside
        half_term = terminal_length / 2.0
        bx0 = min(x1, x2) - half_term
        bx1 = max(x1, x2) + half_term
        box = QRectF(bx0, 2.0, bx1 - bx0, rect.height() - 4.0)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(box, 4.0, 4.0)
        # Terminal leads
        painter.drawLine(QPointF(port_left, y), QPointF(x1, y))
        painter.drawLine(QPointF(x2, y), QPointF(port_right, y))
        # Horizontal zigzag (same as R)
        zig_segments = 7
        step = (x2 - x1) / zig_segments
        amplitude = abs(step) * 1.7320508076 * 0.75  # tan(60 deg), 75% length
        points = [
            QPointF(x1, y),
            QPointF(x1 + step, y - amplitude),
            QPointF(x1 + 2 * step, y + amplitude),
            QPointF(x1 + 3 * step, y - amplitude),
            QPointF(x1 + 4 * step, y + amplitude),
            QPointF(x1 + 5 * step, y - amplitude),
            QPointF(x1 + 6 * step, y + amplitude),
            QPointF(x2, y),
        ]
        for idx in range(len(points) - 1):
            painter.drawLine(points[idx], points[idx + 1])

    def _draw_gnd_symbol(self, painter: QPainter, rect: QRectF) -> None:
        del painter
        del rect

    def _draw_gnd_canvas(self, painter: QPainter, rect: QRectF) -> None:
        scale = 1.0 if self._is_touchstone else self._symbol_scale
        y = rect.height() / 2.0
        port_x = self._port_items[1].pos().x()
        gnd_x = port_x + _BLOCK_WIDTH * 0.3 * scale
        fg = QColor("#1e293b")
        painter.setPen(QPen(fg, 2.0))
        # Straight line from port to GND
        painter.drawLine(QPointF(port_x, y), QPointF(gnd_x, y))
        # GND symbol at right end
        gw = 10.0 * scale
        painter.drawLine(QPointF(gnd_x, y - gw), QPointF(gnd_x, y + gw))
        painter.drawLine(QPointF(gnd_x + 3.0, y - gw * 0.6), QPointF(gnd_x + 3.0, y + gw * 0.6))
        painter.drawLine(QPointF(gnd_x + 6.0, y - gw * 0.25), QPointF(gnd_x + 6.0, y + gw * 0.25))

    def _draw_lumped_symbol(self, painter: QPainter, rect: QRectF) -> None:
        if self.instance.block_kind != "lumped_r":
            return
        scale = 1.0 if self._is_touchstone else self._symbol_scale
        y = (self._port_items[1].pos().y() + self._port_items[2].pos().y()) / 2.0
        port_left = self._port_items[1].pos().x()
        port_right = self._port_items[2].pos().x()
        terminal_length = 20.0 * scale
        x1 = port_left + terminal_length if port_left <= port_right else port_left - terminal_length
        x2 = port_right - terminal_length if port_left <= port_right else port_right + terminal_length
        painter.setPen(QPen(QColor("#1e293b"), 2.0))
        painter.drawLine(QPointF(port_left, y), QPointF(x1, y))
        painter.drawLine(QPointF(x2, y), QPointF(port_right, y))

        zig_segments = 7
        step = (x2 - x1) / zig_segments
        amplitude = abs(step) * 1.7320508076 * 0.75  # tan(60 deg), 75% length
        points = [
            QPointF(x1, y),
            QPointF(x1 + step, y - amplitude),
            QPointF(x1 + 2 * step, y + amplitude),
            QPointF(x1 + 3 * step, y - amplitude),
            QPointF(x1 + 4 * step, y + amplitude),
            QPointF(x1 + 5 * step, y - amplitude),
            QPointF(x1 + 6 * step, y + amplitude),
            QPointF(x2, y),
        ]
        for idx in range(len(points) - 1):
            painter.drawLine(points[idx], points[idx + 1])

    def sync_from_instance(self, instance) -> None:
        self.instance = instance
        self._apply_visual_transform()
        self.update()

    def _apply_visual_transform(self) -> None:
        self.setTransformOriginPoint(self._block_width / 2.0, self._body_height / 2.0)
        self.setTransform(self.transform().fromScale(1.0, 1.0))
        self._apply_port_layout()
        self.setRotation(float(self.instance.rotation_deg % 360))

    def itemChange(self, change, value):  # noqa: N802, ANN001
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._scene_ref.handle_block_moved(self)
        return super().itemChange(change, value)

    def port_item(self, port_number: int) -> PortItem:
        return self._port_items[port_number]


class CircuitScene(QGraphicsScene):
    changedByUser = Signal()
    portSelectionChanged = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSceneRect(0.0, 0.0, 2400.0, 1600.0)
        self._block_items: Dict[str, CircuitBlockItem] = {}
        self._connection_items: Dict[str, CircuitConnectionItem] = {}
        self._pending_port: PortItem | None = None

    def clear_pending_port(self) -> None:
        if self._pending_port is not None:
            self._pending_port.set_pending(False)
        self._pending_port = None
        self.portSelectionChanged.emit(None)

    def register_block(self, block_item: CircuitBlockItem) -> None:
        self._block_items[block_item.instance.instance_id] = block_item
        self.addItem(block_item)

    def register_connection(self, item: CircuitConnectionItem) -> None:
        self._connection_items[item.connection_id] = item
        self.addItem(item)

    def remove_selected_items(self) -> tuple[list[str], list[str]]:
        instance_ids: list[str] = []
        connection_ids: list[str] = []
        for item in list(self.selectedItems()):
            if isinstance(item, CircuitConnectionItem):
                connection_ids.append(item.connection_id)
            elif isinstance(item, CircuitBlockItem):
                instance_ids.append(item.instance.instance_id)
        return instance_ids, connection_ids

    def handle_port_clicked(self, port_item: PortItem) -> None:
        if self._pending_port is None:
            self._pending_port = port_item
            port_item.set_pending(True)
            self.portSelectionChanged.emit(port_item.port_ref)
            return
        if self._pending_port is port_item:
            self.clear_pending_port()
            return
        first = self._pending_port
        self._pending_port.set_pending(False)
        self._pending_port = None
        self.portSelectionChanged.emit(port_item.port_ref)
        self.changedByUser.emit()
        parent = self.parent()
        if isinstance(parent, CircuitWindow):
            parent.create_connection(first.port_ref, port_item.port_ref)

    def handle_block_moved(self, block_item: CircuitBlockItem) -> None:
        parent = self.parent()
        if isinstance(parent, CircuitWindow):
            parent.update_instance_position(block_item.instance.instance_id, block_item.pos())
        for connection_item in self._connection_items.values():
            if (
                connection_item.port_a.owner is block_item
                or connection_item.port_b.owner is block_item
            ):
                connection_item.refresh_geometry()

    def rebuild_export_state(self, document: CircuitDocument) -> None:
        for block_item in self._block_items.values():
            for port_number in range(1, block_item.instance.nports + 1):
                port_item = block_item.port_item(port_number)
                port_item.set_pending(False)
                port_item.set_exported(document.is_port_exported(port_item.port_ref))
            if block_item.instance.block_kind == "port_ground":
                nums = [
                    str(ep.external_port_number)
                    for ep in document.external_ports
                    if ep.port_ref.instance_id == block_item.instance.instance_id
                ]
                block_item._port_label = ", ".join(f"P{n}" for n in nums) if nums else ""
                block_item.update()
            elif block_item.instance.block_kind == "port_diff":
                nums = [
                    str(dp.external_port_number)
                    for dp in document.differential_ports
                    if dp.port_ref_plus.instance_id == block_item.instance.instance_id
                ]
                block_item._port_label = ", ".join(f"Pd{n}" for n in nums) if nums else ""
                block_item.update()


class CircuitWindow(QMainWindow):
    project_modified = Signal()

    def __init__(self, state: AppState, parent=None, window_number: int = 1) -> None:
        super().__init__(parent)
        self.window_number = window_number
        self.setWindowTitle(f"Circuit Composer #{window_number}")
        self.resize(1450, 900)
        app = QApplication.instance()
        if app is not None:
            self.setWindowIcon(app.windowIcon())

        self._state = state
        self._document = CircuitDocument()

        self._file_palette = FilePaletteList()
        self._file_palette.setMinimumWidth(260)

        self._scene = CircuitScene(self)
        self._scene.changedByUser.connect(self._emit_project_modified)
        self._scene.portSelectionChanged.connect(self._on_port_selection_changed)
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)

        self._canvas = CircuitCanvasView(self._scene)
        self._canvas.fileDropped.connect(self._on_file_dropped)
        self._canvas.deletePressed.connect(self._delete_selected_items)
        self._canvas.connectionContextMenuRequested.connect(self._show_connection_context_menu)
        self._canvas.blockContextMenuRequested.connect(self._show_block_context_menu)

        self._status_label = QLabel("Drag files from the left, then click two ports to create a connection.")
        self._status_label.setWordWrap(True)

        self._updating_sweep_controls = False

        self._fmin = QDoubleSpinBox()
        self._fmax = QDoubleSpinBox()
        self._fstep = QDoubleSpinBox()
        for widget in (self._fmin, self._fmax, self._fstep):
            widget.setRange(1e-12, 1e15)
            widget.setDecimals(6)
            widget.valueChanged.connect(self._on_sweep_changed)

        self._frequency_unit = QComboBox()
        self._frequency_unit.addItems(list(_FREQUENCY_UNIT_SCALE.keys()))
        self._frequency_unit.currentTextChanged.connect(self._on_frequency_unit_changed)

        self._impedance_editor = QDoubleSpinBox()
        self._impedance_editor.setRange(0.001, 1e9)
        self._impedance_editor.setDecimals(6)
        self._impedance_editor.setSuffix(" Ohm")
        self._impedance_editor.valueChanged.connect(self._on_impedance_changed)
        self._impedance_editor.setEnabled(False)
        self._symbol_size_editor = QDoubleSpinBox()
        self._symbol_size_editor.setRange(0.50, 3.00)
        self._symbol_size_editor.setSingleStep(0.10)
        self._symbol_size_editor.setDecimals(2)
        self._symbol_size_editor.setSuffix("x")
        self._symbol_size_editor.valueChanged.connect(self._on_symbol_scale_changed)
        self._symbol_size_editor.setEnabled(False)
        self._selected_instance_id: str | None = None
        self._updating_impedance_editor = False
        self._updating_symbol_size_editor = False

        self._export_button = QPushButton("Export equivalent Touchstone")
        self._export_button.clicked.connect(self._export_equivalent_touchstone)

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setMinimumWidth(220)
        left_panel.setStyleSheet("QFrame { background-color: #f7f7f7; border: 1px solid #d1d5db; }")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.addWidget(QLabel("Blocks"))
        left_layout.addWidget(self._file_palette)

        inspector = QFrame()
        inspector.setFrameShape(QFrame.StyledPanel)
        inspector.setMinimumWidth(280)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(8, 8, 8, 8)
        inspector_layout.addWidget(QLabel("Editor Settings"))

        sweep_form = QFormLayout()
        sweep_form.addRow("Unit", self._frequency_unit)
        sweep_form.addRow("Fmin", self._fmin)
        sweep_form.addRow("Fmax", self._fmax)
        sweep_form.addRow("Step", self._fstep)
        inspector_layout.addLayout(sweep_form)

        impedance_form = QFormLayout()
        impedance_form.addRow("Selected impedance", self._impedance_editor)
        impedance_form.addRow("Selected symbol size", self._symbol_size_editor)
        inspector_layout.addLayout(impedance_form)

        inspector_layout.addWidget(self._export_button)
        inspector_layout.addWidget(self._status_label)

        canvas_panel = QFrame()
        canvas_panel.setFrameShape(QFrame.StyledPanel)
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self._canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(canvas_panel)
        splitter.addWidget(inspector)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([280, 840, 330])
        self._splitter = splitter
        self.setCentralWidget(splitter)

        self._state.files_changed.connect(self.refresh_from_state)
        self.refresh_from_state()
        self._sync_sweep_controls_from_document()
        self._refresh_validation_state()

    def refresh_from_state(self) -> None:
        self._file_palette.clear()
        for block in _special_palette_blocks():
            item = QListWidgetItem()
            item.setData(
                Qt.UserRole,
                _build_palette_payload(
                    block_kind=block["block_kind"],
                    label=block["label"],
                    nports=block["nports"],
                    source_file_id=block["source_file_id"],
                    impedance_ohm=block["impedance_ohm"],
                ),
            )
            preview = BlockPreviewWidget(
                block["label"],
                block["nports"],
                block_kind=block["block_kind"],
                impedance_ohm=block["impedance_ohm"],
            )
            item.setSizeHint(preview.sizeHint())
            self._file_palette.addItem(item)
            self._file_palette.setItemWidget(item, preview)
        for loaded in self._state.get_loaded_files():
            item = QListWidgetItem()
            item.setData(
                Qt.UserRole,
                _build_palette_payload(
                    block_kind="touchstone",
                    label=loaded.display_name,
                    nports=loaded.data.nports,
                    source_file_id=loaded.file_id,
                    impedance_ohm=loaded.data.options.reference_resistance,
                ),
            )
            preview = BlockPreviewWidget(
                loaded.display_name,
                loaded.data.nports,
                block_kind="touchstone",
                impedance_ohm=loaded.data.options.reference_resistance,
            )
            item.setSizeHint(preview.sizeHint())
            self._file_palette.addItem(item)
            self._file_palette.setItemWidget(item, preview)
        self._refresh_validation_state()

    def references_file(self, file_id: str) -> bool:
        return self._document.uses_file(file_id)

    def _on_file_dropped(self, file_id: str, scene_pos: QPointF) -> None:
        try:
            payload = json.loads(file_id)
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Invalid block", "Could not read the dragged block definition.")
            return

        block_kind = str(payload.get("block_kind", "touchstone"))
        source_file_id = str(payload.get("source_file_id", ""))
        display_label = str(payload.get("label", "Block"))
        nports = int(payload.get("nports", 0))
        impedance_ohm = float(payload.get("impedance_ohm", 50.0))
        symbol_scale = float(payload.get("symbol_scale", 1.0))

        if block_kind == "touchstone":
            loaded = self._state.get_file(source_file_id)
            if loaded is None:
                QMessageBox.warning(self, "Missing file", "The selected file is no longer loaded.")
                return
            display_label = loaded.display_name
            nports = loaded.data.nports
            impedance_ohm = loaded.data.options.reference_resistance

        instance = self._document.add_instance(
            source_file_id=source_file_id,
            display_label=display_label,
            nports=nports,
            position_x=scene_pos.x(),
            position_y=scene_pos.y(),
            block_kind=block_kind,
            impedance_ohm=impedance_ohm,
            symbol_scale=symbol_scale,
        )
        block_item = CircuitBlockItem(self._scene, instance)
        self._scene.register_block(block_item)
        self._scene.rebuild_export_state(self._document)
        self._emit_project_modified()

    def update_instance_position(self, instance_id: str, pos: QPointF) -> None:
        self._document.update_instance_position(instance_id, pos.x(), pos.y())
        self._emit_project_modified()

    def _on_scene_selection_changed(self) -> None:
        selected = self._scene.selectedItems()
        block_item = next((item for item in selected if isinstance(item, CircuitBlockItem)), None)
        if block_item is None:
            self._selected_instance_id = None
            self._updating_impedance_editor = True
            self._impedance_editor.setEnabled(False)
            self._impedance_editor.setValue(50.0)
            self._updating_impedance_editor = False
            self._updating_symbol_size_editor = True
            self._symbol_size_editor.setEnabled(False)
            self._symbol_size_editor.setValue(1.0)
            self._updating_symbol_size_editor = False
            return
        self._selected_instance_id = block_item.instance.instance_id
        self._updating_impedance_editor = True
        self._impedance_editor.setEnabled(block_item.instance.block_kind in {"port_ground", "port_diff"})
        self._impedance_editor.setValue(block_item.instance.impedance_ohm)
        self._updating_impedance_editor = False
        self._updating_symbol_size_editor = True
        self._symbol_size_editor.setEnabled(block_item.instance.block_kind != "touchstone")
        self._symbol_size_editor.setValue(float(getattr(block_item.instance, "symbol_scale", 1.0)))
        self._updating_symbol_size_editor = False

    def _on_impedance_changed(self, value: float) -> None:
        if self._updating_impedance_editor or self._selected_instance_id is None:
            return
        instance = self._document.get_instance(self._selected_instance_id)
        if instance is None or instance.block_kind not in {"port_ground", "port_diff"}:
            return
        self._document.update_instance_impedance(self._selected_instance_id, value)
        block_item = self._scene._block_items.get(self._selected_instance_id)
        updated_instance = self._document.get_instance(self._selected_instance_id)
        if block_item is not None and updated_instance is not None:
            block_item.sync_from_instance(updated_instance)
        self._status_label.setText(f"Impedance updated to {value:g} Ohm.")
        self._refresh_validation_state()
        self._emit_project_modified()

    def _on_symbol_scale_changed(self, value: float) -> None:
        if self._updating_symbol_size_editor or self._selected_instance_id is None:
            return
        instance = self._document.get_instance(self._selected_instance_id)
        if instance is None or instance.block_kind == "touchstone":
            return
        self._document.update_instance_symbol_scale(self._selected_instance_id, value)
        block_item = self._scene._block_items.get(self._selected_instance_id)
        updated_instance = self._document.get_instance(self._selected_instance_id)
        if block_item is None or updated_instance is None:
            return
        replacement_item = CircuitBlockItem(self._scene, updated_instance)
        replacement_item.setPos(block_item.pos())
        replacement_item.setSelected(block_item.isSelected())
        for connection_item in self._scene._connection_items.values():
            if connection_item.port_a.owner is block_item:
                connection_item.port_a = replacement_item.port_item(connection_item.port_a.port_number)
            if connection_item.port_b.owner is block_item:
                connection_item.port_b = replacement_item.port_item(connection_item.port_b.port_number)
            connection_item.refresh_geometry()
        self._scene.removeItem(block_item)
        self._scene._block_items[self._selected_instance_id] = replacement_item
        self._scene.addItem(replacement_item)
        self._status_label.setText(f"Symbol size updated to {value:.2f}x.")
        self._refresh_validation_state()
        self._emit_project_modified()

    def _apply_instance_transform(
        self,
        instance_id: str,
        *,
        rotation_deg: int,
        mirror_horizontal: bool,
        mirror_vertical: bool,
    ) -> None:
        self._document.update_instance_transform(
            instance_id,
            rotation_deg=rotation_deg,
            mirror_horizontal=mirror_horizontal,
            mirror_vertical=mirror_vertical,
        )
        updated_instance = self._document.get_instance(instance_id)
        block_item = self._scene._block_items.get(instance_id)
        if updated_instance is None or block_item is None:
            return
        block_item.sync_from_instance(updated_instance)
        self._refresh_connection_geometry_for_block(block_item)
        self._status_label.setText("Block transform updated.")
        self._emit_project_modified()

    def create_connection(self, port_a: CircuitPortRef, port_b: CircuitPortRef) -> None:
        if port_a.instance_id == port_b.instance_id and port_a.port_number == port_b.port_number:
            self._status_label.setText("Select two different ports.")
            return
        pair_a = (port_a.instance_id, port_a.port_number, port_b.instance_id, port_b.port_number)
        pair_b = (port_b.instance_id, port_b.port_number, port_a.instance_id, port_a.port_number)
        for connection in self._document.connections:
            existing = (
                connection.port_a.instance_id,
                connection.port_a.port_number,
                connection.port_b.instance_id,
                connection.port_b.port_number,
            )
            if existing == pair_a or existing == pair_b:
                self._status_label.setText("Connection already exists between these two ports.")
                return
        connection = self._document.add_connection(port_a, port_b)
        port_item_a = self._port_item_for_ref(port_a)
        port_item_b = self._port_item_for_ref(port_b)
        if port_item_a is None or port_item_b is None:
            self._document.remove_connection(connection.connection_id)
            self._status_label.setText("Could not resolve one of the selected ports.")
            return
        connection_item = CircuitConnectionItem(connection.connection_id, port_item_a, port_item_b)
        self._scene.register_connection(connection_item)
        self._status_label.setText("Connection created. Select a line and press Delete, or right-click it to remove it.")
        self._refresh_validation_state()
        self._emit_project_modified()

    def _port_item_for_ref(self, port_ref: CircuitPortRef) -> PortItem | None:
        block_item = self._scene._block_items.get(port_ref.instance_id)
        if block_item is None:
            return None
        return block_item.port_item(port_ref.port_number)

    def _on_port_selection_changed(self, port_ref: CircuitPortRef | None) -> None:
        if port_ref is None:
            self._status_label.setText("Drag files from the left, then click two ports to create a connection.")
            return
        instance = self._document.get_instance(port_ref.instance_id)
        if instance is None:
            return
        self._status_label.setText(
            f"Selected {instance.display_label} - port {port_ref.port_number}. Click another port to connect it."
        )

    def _refresh_validation_state(self) -> None:
        issues = self._document.validate()
        if issues:
            self._export_button.setEnabled(False)
            self._status_label.setText(issues[0].message)
            return

        if not self._document.external_ports and not self._document.differential_ports:
            self._export_button.setEnabled(False)
            self._status_label.setText("Add at least one external port block to export an equivalent Touchstone.")
            return

        self._export_button.setEnabled(True)
        warnings = self._collect_preflight_warnings()
        if warnings:
            self._status_label.setText(warnings[0])
            self._status_label.setToolTip("\n".join(warnings))
            return
        self._status_label.setText("Circuit ready for export.")
        self._status_label.setToolTip("")

    def _collect_preflight_warnings(self) -> list[str]:
        warnings: list[str] = []
        sweep = self._document.sweep
        for instance in self._document.instances:
            if instance.block_kind != "touchstone":
                continue
            loaded = self._state.get_file(instance.source_file_id)
            if loaded is None:
                warnings.append(f"Touchstone block '{instance.display_label}' is not loaded.")
                continue
            if not loaded.data.points:
                warnings.append(f"Touchstone block '{instance.display_label}' has no frequency samples.")
                continue
            f_min = loaded.data.points[0].frequency_hz
            f_max = loaded.data.points[-1].frequency_hz
            if sweep.fmin_hz < f_min or sweep.fmax_hz > f_max:
                warnings.append(
                    f"Sweep extends beyond '{instance.display_label}' data range "
                    f"({self._format_frequency_hz(f_min)} to {self._format_frequency_hz(f_max)})."
                )
        return warnings

    def _format_frequency_hz(self, value_hz: float) -> str:
        if value_hz >= 1e9:
            return f"{value_hz / 1e9:.6g} GHz"
        if value_hz >= 1e6:
            return f"{value_hz / 1e6:.6g} MHz"
        if value_hz >= 1e3:
            return f"{value_hz / 1e3:.6g} KHz"
        return f"{value_hz:.6g} Hz"

    def _describe_passivity(self, result) -> str:  # noqa: ANN001
        diagnostic = result.passivity
        if diagnostic is None:
            return "Passivity check unavailable."
        summary = diagnostic.summary
        if summary.worst_frequency_hz is None:
            return "Passivity: OK."
        if summary.severity == "pass":
            return "Passivity: OK."
        if summary.severity == "noise":
            return (
                "Passivity: minor numerical overrun at "
                f"{self._format_frequency_hz(summary.worst_frequency_hz)} "
                f"(max sigma = {summary.worst_sigma_max:.6g})."
            )
        if summary.severity == "borderline":
            return (
                "Passivity: borderline overrun near "
                f"{self._format_frequency_hz(summary.worst_frequency_hz)} "
                f"(max sigma = {summary.worst_sigma_max:.6g})."
            )
        return (
            "Passivity violation near "
            f"{self._format_frequency_hz(summary.worst_frequency_hz)} "
            f"(max sigma = {summary.worst_sigma_max:.6g}, affected points = {summary.points_over_warn})."
        )

    def _confirm_passivity_export(self, result) -> bool:  # noqa: ANN001
        diagnostic = result.passivity
        if diagnostic is None or diagnostic.summary.severity != "hard":
            return True
        summary = diagnostic.summary
        choice = QMessageBox.warning(
            self,
            "Passivity warning",
            "The solved network is not passive.\n\n"
            f"Worst case: sigma = {summary.worst_sigma_max:.6g} at {self._format_frequency_hz(summary.worst_frequency_hz or 0.0)}.\n"
            "Export anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return choice == QMessageBox.Yes

    def _on_sweep_changed(self) -> None:
        if self._updating_sweep_controls:
            return
        scale = _FREQUENCY_UNIT_SCALE[self._frequency_unit.currentText()]
        self._document.sweep = FrequencySweepSpec(
            fmin_hz=self._fmin.value() * scale,
            fmax_hz=self._fmax.value() * scale,
            fstep_hz=self._fstep.value() * scale,
            display_unit=self._frequency_unit.currentText(),
        )
        self._refresh_validation_state()
        self._emit_project_modified()

    def _on_frequency_unit_changed(self, unit: str) -> None:
        if not unit:
            return
        self._document.sweep = FrequencySweepSpec(
            fmin_hz=self._document.sweep.fmin_hz,
            fmax_hz=self._document.sweep.fmax_hz,
            fstep_hz=self._document.sweep.fstep_hz,
            display_unit=unit,
        )
        self._sync_sweep_controls_from_document()
        self._refresh_validation_state()
        self._emit_project_modified()

    def _sync_sweep_controls_from_document(self) -> None:
        unit = self._document.sweep.display_unit
        scale = _FREQUENCY_UNIT_SCALE.get(unit, 1.0)
        suffix = f" {unit}"
        self._updating_sweep_controls = True
        self._frequency_unit.blockSignals(True)
        if self._frequency_unit.findText(unit) >= 0:
            self._frequency_unit.setCurrentText(unit)
        for widget in (self._fmin, self._fmax, self._fstep):
            widget.blockSignals(True)
            widget.setSuffix(suffix)
        self._fmin.setValue(self._document.sweep.fmin_hz / scale)
        self._fmax.setValue(self._document.sweep.fmax_hz / scale)
        self._fstep.setValue(self._document.sweep.fstep_hz / scale)
        for widget in (self._fmin, self._fmax, self._fstep):
            widget.blockSignals(False)
        self._frequency_unit.blockSignals(False)
        self._updating_sweep_controls = False

    def _delete_selected_items(self) -> None:
        instance_ids, connection_ids = self._scene.remove_selected_items()
        if not instance_ids and not connection_ids:
            return
        for connection_id in connection_ids:
            self._remove_connection(connection_id)
        for instance_id in instance_ids:
            block_item = self._scene._block_items.pop(instance_id, None)
            if block_item is not None:
                self._scene.removeItem(block_item)
            removed_connection_ids = [
                conn.connection_id
                for conn in self._document.connections
                if conn.port_a.instance_id == instance_id or conn.port_b.instance_id == instance_id
            ]
            self._document.remove_instance(instance_id)
            for connection_id in removed_connection_ids:
                self._remove_connection(connection_id)
        self._scene.clear_pending_port()
        self._scene.rebuild_export_state(self._document)
        self._refresh_validation_state()
        self._emit_project_modified()

    def _remove_connection(self, connection_id: str) -> None:
        item = self._scene._connection_items.pop(connection_id, None)
        if item is not None:
            self._scene.removeItem(item)
        self._document.remove_connection(connection_id)

    def _refresh_connection_geometry_for_block(self, block_item: CircuitBlockItem) -> None:
        for connection_item in self._scene._connection_items.values():
            if connection_item.port_a.owner is block_item or connection_item.port_b.owner is block_item:
                connection_item.refresh_geometry()

    def _show_connection_context_menu(self, connection_id: str, global_pos: QPoint) -> None:
        menu = QMenu(self)
        delete_action = menu.addAction("Delete connection")
        chosen = menu.exec(global_pos)
        if chosen is delete_action:
            self._remove_connection(connection_id)
            self._scene.clear_pending_port()
            self._scene.rebuild_export_state(self._document)
            self._refresh_validation_state()
            self._status_label.setText("Connection removed.")
            self._emit_project_modified()

    def _show_block_context_menu(self, instance_id: str, global_pos: QPoint) -> None:
        instance = self._document.get_instance(instance_id)
        if instance is None:
            return
        menu = QMenu(self)
        rotate_right_action = menu.addAction("Rotate 90 deg")
        rotate_left_action = menu.addAction("Rotate -90 deg")
        menu.addSeparator()
        mirror_h_action = menu.addAction("Mirror horizontal")
        mirror_h_action.setCheckable(True)
        mirror_h_action.setChecked(instance.mirror_horizontal)
        mirror_v_action = menu.addAction("Mirror vertical")
        mirror_v_action.setCheckable(True)
        mirror_v_action.setChecked(instance.mirror_vertical)

        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen is rotate_right_action:
            self._apply_instance_transform(
                instance_id,
                rotation_deg=(instance.rotation_deg + 90) % 360,
                mirror_horizontal=instance.mirror_horizontal,
                mirror_vertical=instance.mirror_vertical,
            )
            return
        if chosen is rotate_left_action:
            self._apply_instance_transform(
                instance_id,
                rotation_deg=(instance.rotation_deg - 90) % 360,
                mirror_horizontal=instance.mirror_horizontal,
                mirror_vertical=instance.mirror_vertical,
            )
            return
        if chosen is mirror_h_action:
            self._apply_instance_transform(
                instance_id,
                rotation_deg=instance.rotation_deg,
                mirror_horizontal=not instance.mirror_horizontal,
                mirror_vertical=instance.mirror_vertical,
            )
            return
        if chosen is mirror_v_action:
            self._apply_instance_transform(
                instance_id,
                rotation_deg=instance.rotation_deg,
                mirror_horizontal=instance.mirror_horizontal,
                mirror_vertical=not instance.mirror_vertical,
            )

    def export_project_state(self) -> dict:
        return {
            "window_title": self.windowTitle(),
            "splitter_sizes": self._splitter.sizes(),
            **self._document.to_dict(),
        }

    def apply_project_state(self, state: dict) -> None:
        document = CircuitDocument.from_dict(state)
        valid_file_ids = {loaded.file_id for loaded in self._state.get_loaded_files()}
        document.instances = [
            item
            for item in document.instances
            if item.block_kind != "touchstone" or item.source_file_id in valid_file_ids
        ]
        valid_instance_ids = {item.instance_id for item in document.instances}
        document.connections = [
            item
            for item in document.connections
            if item.port_a.instance_id in valid_instance_ids and item.port_b.instance_id in valid_instance_ids
        ]
        document.external_ports = [
            item for item in document.external_ports if item.port_ref.instance_id in valid_instance_ids
        ]
        self._document = document

        self._scene.clear()
        self._scene._block_items.clear()
        self._scene._connection_items.clear()
        self._scene._pending_port = None

        self._fmin.blockSignals(True)
        self._fmax.blockSignals(True)
        self._fstep.blockSignals(True)
        self._frequency_unit.blockSignals(True)
        self._fmin.blockSignals(False)
        self._fmax.blockSignals(False)
        self._fstep.blockSignals(False)
        self._frequency_unit.blockSignals(False)

        self._sync_sweep_controls_from_document()

        for instance in self._document.instances:
            self._scene.register_block(CircuitBlockItem(self._scene, instance))
        for connection in self._document.connections:
            port_item_a = self._port_item_for_ref(connection.port_a)
            port_item_b = self._port_item_for_ref(connection.port_b)
            if port_item_a is None or port_item_b is None:
                continue
            self._scene.register_connection(
                CircuitConnectionItem(connection.connection_id, port_item_a, port_item_b)
            )

        sizes = state.get("splitter_sizes")
        if isinstance(sizes, list) and len(sizes) == 3 and all(isinstance(v, int) for v in sizes):
            self._splitter.setSizes(sizes)
        elif isinstance(sizes, list) and len(sizes) == 2 and all(isinstance(v, int) for v in sizes):
            self._splitter.setSizes([sizes[0], sizes[1], 330])

        self._scene.rebuild_export_state(self._document)
        self._refresh_validation_state()

    def _export_equivalent_touchstone(self) -> None:
        format_choices = ["RI", "MA", "DB"]
        selected_format, ok = QInputDialog.getItem(
            self,
            "Export format",
            "Touchstone data format:",
            format_choices,
            0,
            False,
        )
        if not ok:
            return

        unit_choices = ["Hz", "KHz", "MHz", "GHz"]
        default_unit = self._document.sweep.display_unit
        default_index = unit_choices.index(default_unit) if default_unit in unit_choices else 3
        selected_unit, ok = QInputDialog.getItem(
            self,
            "Frequency unit",
            "Frequency unit in output file:",
            unit_choices,
            default_index,
            False,
        )
        if not ok:
            return

        try:
            self._status_label.setText("Solving equivalent network...")
            result = solve_circuit_network(self._document, self._state)
            self._status_label.setText(self._describe_passivity(result))
            if not self._confirm_passivity_export(result):
                self._status_label.setText("Export cancelled due to passivity warning.")
                return
            text = to_touchstone_string_with_format(
                result,
                data_format=selected_format,
                frequency_unit=selected_unit,
            )
        except Exception as exc:  # pragma: no cover - runtime-facing error path
            self._status_label.setText(f"Export failed: {exc}")
            QMessageBox.warning(self, "Export failed", str(exc))
            return

        suffix = f"s{result.nports}p"
        default_name = f"equivalent.{suffix}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export equivalent Touchstone",
            default_name,
            f"Touchstone (*.{suffix});;All files (*)",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        except Exception as exc:  # pragma: no cover - runtime-facing error path
            self._status_label.setText(f"Export failed: {exc}")
            QMessageBox.warning(self, "Export failed", str(exc))
            return

        self._status_label.setText(f"Equivalent Touchstone exported to {path}. {self._describe_passivity(result)}")
        QMessageBox.information(
            self,
            "Export completed",
            f"Saved {result.nports}-port equivalent network ({selected_format}, {selected_unit}).\n\n{self._describe_passivity(result)}",
        )

    def _emit_project_modified(self) -> None:
        self.project_modified.emit()