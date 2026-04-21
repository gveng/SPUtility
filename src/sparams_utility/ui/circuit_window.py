from __future__ import annotations

import json
from typing import Dict

from PySide6.QtCore import QMimeData, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QFont, QFontMetrics, QKeySequence, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sparams_utility.circuit_solver import solve_circuit_network, to_touchstone_string_with_format, simulate_channel, ChannelSimResult
from sparams_utility.models.circuit import (
    CircuitDocument,
    CircuitPortRef,
    DriverSpec,
    FrequencySweepSpec,
    PRBS_CHOICES,
    ENCODING_CHOICES,
)
from sparams_utility.models.state import AppState
from sparams_utility.ui.eye_diagram_window import (
    DEFAULT_EYE_SPAN_UI,
    DEFAULT_QUALITY_PRESET,
    DEFAULT_RENDER_MODE,
    EYE_SPAN_CHOICES,
    QUALITY_PRESET_CHOICES,
    RENDER_MODE_CHOICES,
    EyeDiagramWindow,
)

_MIME_BLOCK_DEF = "application/x-sparams-block-def"
_BLOCK_WIDTH = 80.0
_PORT_RADIUS = 6.0
_GRID_SIZE = 20.0
_SCHEMATIC_BG = QColor("#f7f7f7")
_FREQUENCY_UNIT_SCALE = {
    "Hz": 1.0,
    "KHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
}


def _contrast_foreground(background: QColor) -> QColor:
    return QColor("#f8fafc") if background.lightnessF() < 0.5 else QColor("#0f172a")


def _draw_eye_on_screen(painter: QPainter, screen: QRectF) -> None:
    """Draw a simplified NRZ eye diagram pattern inside `screen`."""
    x0 = screen.left() + 1.0
    x1 = screen.right() - 1.0
    cx = (x0 + x1) / 2.0
    cy = screen.center().y()
    h = screen.height() * 0.40
    pen = QPen(QColor("#a78bfa"), 1.5)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    # Upper arc: left-cross → up → right-cross
    path = QPainterPath()
    path.moveTo(x0, cy)
    path.cubicTo(QPointF(x0 + (cx - x0) * 0.6, cy - h),
                 QPointF(cx + (x1 - cx) * 0.4, cy - h),
                 QPointF(x1, cy))
    painter.drawPath(path)
    # Lower arc: left-cross → down → right-cross
    path = QPainterPath()
    path.moveTo(x0, cy)
    path.cubicTo(QPointF(x0 + (cx - x0) * 0.6, cy + h),
                 QPointF(cx + (x1 - cx) * 0.4, cy + h),
                 QPointF(x1, cy))
    painter.drawPath(path)
    # Crossing diagonals (X at left and right ends)
    dh = h * 0.55
    painter.drawLine(QPointF(x0 - 3.0, cy - dh), QPointF(x0 + 3.0, cy + dh))
    painter.drawLine(QPointF(x0 - 3.0, cy + dh), QPointF(x0 + 3.0, cy - dh))
    painter.drawLine(QPointF(x1 - 3.0, cy - dh), QPointF(x1 + 3.0, cy + dh))
    painter.drawLine(QPointF(x1 - 3.0, cy + dh), QPointF(x1 + 3.0, cy - dh))


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
    if block_kind == "driver_se":
        return "SE Driver"
    if block_kind == "driver_diff":
        return "Diff Driver"
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
            body_h = max(26, (max(nports, 2) + 1) * 9)
            self.setMinimumHeight(body_h + 26)
        elif block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            self.setMinimumHeight(56)
        elif block_kind in {"port_diff", "port_ground", "gnd"}:
            self.setMinimumHeight(52)
        elif block_kind in {"driver_se", "driver_diff"}:
            self.setMinimumHeight(56)
        elif block_kind in {"eyescope_se", "eyescope_diff"}:
            self.setMinimumHeight(56)
        elif block_kind == "net_node":
            self.setMinimumHeight(42)
        else:
            self.setMinimumHeight(max(48, 20 + max(nports, 2) * 6))

    def sizeHint(self) -> QSize:  # noqa: N802
        if self._block_kind == "touchstone":
            body_h = max(26, (max(self._nports, 2) + 1) * 9)
            return QSize(170, body_h + 26)
        if self._block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            return QSize(170, 56)
        if self._block_kind in {"port_diff", "port_ground", "gnd"}:
            return QSize(170, 52)
        if self._block_kind in {"driver_se", "driver_diff"}:
            return QSize(170, 56)
        if self._block_kind in {"eyescope_se", "eyescope_diff"}:
            return QSize(170, 56)
        if self._block_kind == "net_node":
            return QSize(170, 42)
        return QSize(170, max(48, 20 + max(self._nports, 2) * 6))

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), _SCHEMATIC_BG)

        preview_w = _BLOCK_WIDTH * 0.65
        preview_h = 44.0 * 0.65

        is_lumped = self._block_kind in {"lumped_r", "lumped_l", "lumped_c"}
        if is_lumped:
            cx = self.rect().center().x()
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, preview_h)
        elif self._block_kind == "port_diff":
            cx = self.rect().center().x()
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, preview_h)
        elif self._block_kind == "port_ground":
            cx = self.rect().center().x()
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, preview_h)
        elif self._block_kind == "gnd":
            cx = self.rect().center().x()
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, preview_h)
        elif self._block_kind in {"driver_se", "driver_diff"}:
            cx = self.rect().center().x()
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, preview_h)
        elif self._block_kind in {"eyescope_se", "eyescope_diff"}:
            cx = self.rect().center().x()
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, preview_h)
        elif self._block_kind == "net_node":
            cx = self.rect().center().x()
            cy = self.rect().center().y() - 8.0
            rect = QRectF(cx - 8.0, cy - 8.0, 16.0, 16.0)
        elif self._block_kind == "touchstone":
            cx = self.rect().center().x()
            body_h = max(22.0, (max(self._nports, 2) + 1) * 7.0)
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, body_h)
        else:
            rect = self.rect().adjusted(40, 8, -40, -28)
        painter.setPen(QPen(QColor("#1e40af"), 1.6))
        if self._block_kind not in {"lumped_r", "lumped_l", "lumped_c", "gnd", "port_diff", "port_ground", "touchstone", "driver_se", "driver_diff", "eyescope_se", "eyescope_diff", "net_node"}:
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
        if self._block_kind in {"driver_se", "driver_diff"}:
            self._draw_driver_symbol(painter, rect)
            return
        if self._block_kind in {"eyescope_se", "eyescope_diff"}:
            self._draw_eyescope_symbol(painter, rect)
            return
        if self._block_kind == "net_node":
            self._draw_net_node_symbol(painter, rect)
            return
        if self._block_kind == "lumped_l":
            self._draw_inductor_symbol(
                painter,
                rect,
                foreground_color=QColor("#1e293b"),
            )
            return
        if self._block_kind == "lumped_c":
            self._draw_capacitor_symbol(
                painter,
                rect,
                foreground_color=QColor("#1e293b"),
            )
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
        total_w = rect.width() * scale
        right_end = x0 + total_w
        terminal_length = min(20.0 * scale, rect.width() * 0.24)
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
        terminal_length = min(20.0, rect.width() * 0.24)
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

    def _draw_inductor_symbol(self, painter: QPainter, rect: QRectF, *, foreground_color: QColor) -> None:
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
        # Draw coil arcs (4 humps centred on wire)
        painter.setBrush(Qt.NoBrush)
        n_humps = 4
        hump_w = (x2 - x1) / n_humps
        hump_h = hump_w * 0.8
        for i in range(n_humps):
            arc_rect = QRectF(x1 + i * hump_w, y - hump_h, hump_w, hump_h * 2.0)
            painter.drawArc(arc_rect, 0, 180 * 16)

    def _draw_capacitor_symbol(self, painter: QPainter, rect: QRectF, *, foreground_color: QColor) -> None:
        y = rect.center().y()
        x0 = rect.left()
        x3 = rect.right()
        terminal_length = 20.0
        cx = (x0 + x3) / 2.0
        painter.setPen(QPen(foreground_color, 2.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPolygon(_hex_port_polygon(float(x0), float(y), _PORT_RADIUS))
        painter.drawPolygon(_hex_port_polygon(float(x3), float(y), _PORT_RADIUS))
        # Horizontal leads
        gap = 4.0
        painter.drawLine(QPointF(x0, y), QPointF(cx - gap, y))
        painter.drawLine(QPointF(cx + gap, y), QPointF(x3, y))
        # Two vertical plates
        plate_h = 16.0
        painter.drawLine(QPointF(cx - gap, y - plate_h), QPointF(cx - gap, y + plate_h))
        painter.drawLine(QPointF(cx + gap, y - plate_h), QPointF(cx + gap, y + plate_h))

    def _draw_eyescope_symbol(self, painter: QPainter, rect: QRectF) -> None:
        """Draw oscilloscope-style palette preview for EyeScope blocks."""
        fg = QColor("#4c1d95")
        is_diff = self._block_kind == "eyescope_diff"
        cy = rect.center().y()
        lead_w = rect.width() * 0.18
        body = QRectF(rect.left() + lead_w, rect.top() + 2.0, rect.width() - lead_w - 2.0, rect.height() - 4.0)
        # Port markers on left
        painter.setPen(QPen(fg, 1.6))
        painter.setBrush(QBrush(QColor("#ffffff")))
        if is_diff:
            dy = min(_GRID_SIZE / 2.0, rect.height() * 0.24)
            p1y = cy - dy
            p2y = cy + dy
            painter.drawPolygon(_hex_port_polygon(float(rect.left()), float(p1y), _PORT_RADIUS))
            painter.drawPolygon(_hex_port_polygon(float(rect.left()), float(p2y), _PORT_RADIUS))
        else:
            painter.drawPolygon(_hex_port_polygon(float(rect.left()), float(cy), _PORT_RADIUS))
        # Terminal lead(s)
        painter.setPen(QPen(fg, 2.0))
        if is_diff:
            painter.drawLine(QPointF(rect.left(), p1y), QPointF(body.left(), p1y))
            painter.drawLine(QPointF(rect.left(), p2y), QPointF(body.left(), p2y))
        else:
            painter.drawLine(QPointF(rect.left(), cy), QPointF(body.left(), cy))
        # Body
        painter.setPen(QPen(fg, 2.0))
        painter.setBrush(QBrush(QColor("#ede9fe")))
        painter.drawRoundedRect(body, 4.0, 4.0)
        # Screen
        sw = body.width() * 0.72
        sh = body.height() * 0.52
        screen = QRectF(body.center().x() - sw / 2, body.center().y() - sh / 2, sw, sh)
        painter.setPen(QPen(QColor("#4c1d95"), 1.0))
        painter.setBrush(QBrush(QColor("#1e1b4b")))
        painter.drawRect(screen)
        # Eye diagram trace
        _draw_eye_on_screen(painter, screen)

    def _draw_net_node_symbol(self, painter: QPainter, rect: QRectF) -> None:
        """Draw a filled dot representing a wire junction node."""
        cx = rect.center().x()
        cy = rect.center().y()
        r = 8.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#1d4ed8")))
        painter.drawEllipse(QPointF(cx, cy), r, r)
        # Short stub lines in all 4 directions to hint at T-junction capability
        painter.setPen(QPen(QColor("#1d4ed8"), 2.5))
        stub = 14.0
        painter.drawLine(QPointF(cx - stub, cy), QPointF(cx - r, cy))
        painter.drawLine(QPointF(cx + r, cy), QPointF(cx + stub, cy))
        painter.drawLine(QPointF(cx, cy - stub), QPointF(cx, cy - r))
        painter.drawLine(QPointF(cx, cy + r), QPointF(cx, cy + stub))

    def _draw_driver_symbol(self, painter: QPainter, rect: QRectF) -> None:
        fg = QColor("#1e293b")
        is_diff = self._block_kind == "driver_diff"
        painter.setPen(QPen(fg, 2.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 4.0, 4.0)
        # Draw a small pulse waveform inside
        cx = rect.center().x()
        cy = rect.center().y()
        w = rect.width() * 0.5
        h = rect.height() * 0.35
        painter.setPen(QPen(QColor("#2563eb"), 2.0))
        pts = [
            QPointF(cx - w / 2, cy + h / 2),
            QPointF(cx - w / 4, cy + h / 2),
            QPointF(cx - w / 4, cy - h / 2),
            QPointF(cx, cy - h / 2),
            QPointF(cx, cy + h / 2),
            QPointF(cx + w / 4, cy + h / 2),
            QPointF(cx + w / 4, cy - h / 2),
            QPointF(cx + w / 2, cy - h / 2),
        ]
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])
        # Port markers
        painter.setPen(QPen(fg, 1.6))
        painter.setBrush(QBrush(QColor("#ffffff")))
        if is_diff:
            dy = min(_GRID_SIZE / 2.0, rect.height() * 0.24)
            y1 = cy - dy
            y2 = cy + dy
            painter.drawPolygon(_hex_port_polygon(float(rect.right()), float(y1), _PORT_RADIUS))
            painter.drawPolygon(_hex_port_polygon(float(rect.right()), float(y2), _PORT_RADIUS))
        else:
            painter.drawPolygon(_hex_port_polygon(float(rect.right()), float(cy), _PORT_RADIUS))


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
        {
            "block_kind": "driver_se",
            "label": "SE Driver",
            "nports": 1,
            "source_file_id": "__special__:driver_se",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "driver_diff",
            "label": "Diff Driver",
            "nports": 2,
            "source_file_id": "__special__:driver_diff",
            "impedance_ohm": 100.0,
        },
        {
            "block_kind": "eyescope_se",
            "label": "EyeScope SE",
            "nports": 1,
            "source_file_id": "__special__:eyescope_se",
            "impedance_ohm": 1e6,
        },
        {
            "block_kind": "eyescope_diff",
            "label": "EyeScope Diff",
            "nports": 2,
            "source_file_id": "__special__:eyescope_diff",
            "impedance_ohm": 2e6,
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
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

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

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        # Forward mouse moves to scene for rubber-band routing preview
        s = self.scene()
        if isinstance(s, CircuitScene) and s._routing_active:
            scene_pos = self.mapToScene(event.pos())
            s._update_routing_preview(scene_pos)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # Disable rubber-band selection drag while routing is active
        s = self.scene()
        if isinstance(s, CircuitScene) and s._routing_active:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        # Restore rubber-band after releasing
        s = self.scene()
        if not (isinstance(s, CircuitScene) and s._routing_active):
            self.setDragMode(QGraphicsView.RubberBandDrag)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            s = self.scene()
            if isinstance(s, CircuitScene) and s._routing_active:
                s.cancel_routing()
                event.accept()
                return
        if event.matches(QKeySequence.Delete):
            self.deletePressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        # Cancel routing on right-click instead of showing context menus
        s = self.scene()
        if isinstance(s, CircuitScene) and s._routing_active:
            s.cancel_routing()
            event.accept()
            return
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
        self._is_pending = False
        self._is_exported = False
        self._is_snap_hover = False
        self.setPos(x, y)
        self._refresh_brush()
        self.setPen(QPen(QColor("#1f2937"), 1.5))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setToolTip(f"{owner.instance.display_label} - Port {port_number}")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        super().mousePressEvent(event)
        scene = self.scene()
        if isinstance(scene, CircuitScene):
            scene.handle_port_clicked(self)

    def shape(self) -> QPainterPath:  # noqa: N802
        # Enlarge hit target for easier EDA-style pin capture.
        path = QPainterPath()
        r = max(_PORT_RADIUS + 5.0, _GRID_SIZE * 0.45)
        path.addEllipse(QPointF(0.0, 0.0), r, r)
        return path

    def set_pending(self, active: bool) -> None:
        self._is_pending = active
        self._refresh_brush()

    def set_exported(self, active: bool) -> None:
        self._is_exported = active
        self._refresh_brush()

    def set_snap_hover(self, active: bool) -> None:
        self._is_snap_hover = active
        self._refresh_brush()

    def _refresh_brush(self) -> None:
        if self._is_snap_hover:
            color = QColor("#f43f5e")
        elif self._is_pending:
            color = QColor("#f59e0b")
        elif self._is_exported:
            color = QColor("#22c55e")
        else:
            color = QColor("#ffffff")
        self.setBrush(QBrush(color))

    @property
    def port_ref(self) -> CircuitPortRef:
        return CircuitPortRef(self.owner.instance.instance_id, self.port_number)


class _WaypointHandle(QGraphicsRectItem):
    """Small draggable square handle for repositioning a waypoint on a wire."""

    _SIZE = 10.0

    def __init__(self, index: int, owner: "CircuitConnectionItem") -> None:
        super().__init__(-self._SIZE / 2, -self._SIZE / 2, self._SIZE, self._SIZE, owner)
        self._index = index
        self._owner = owner
        self._suspend_move_callback = False
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.SizeAllCursor)
        self.setPen(QPen(QColor("#1d4ed8"), 1.5))
        self.setBrush(QBrush(QColor("#bfdbfe")))
        self.setZValue(2.0)
        self.setVisible(False)

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.ItemPositionChange:
            snapped_x = round(value.x() / _GRID_SIZE) * _GRID_SIZE
            snapped_y = round(value.y() / _GRID_SIZE) * _GRID_SIZE
            return QPointF(snapped_x, snapped_y)
        if change == QGraphicsItem.ItemPositionHasChanged:
            if not self._suspend_move_callback:
                self._owner._on_handle_moved(self._index, self.pos())
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self.setBrush(QBrush(QColor("#3b82f6")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self.setBrush(QBrush(QColor("#bfdbfe")))
        super().hoverLeaveEvent(event)


class CircuitConnectionItem(QGraphicsPathItem):
    def __init__(
        self,
        connection_id: str,
        port_a: PortItem,
        port_b: PortItem,
        waypoints: tuple[tuple[float, float], ...] = (),
    ) -> None:
        super().__init__()
        self.connection_id = connection_id
        self.port_a = port_a
        self.port_b = port_b
        # Waypoints in scene coordinates (list so we can mutate)
        self._waypoints: list[QPointF] = [QPointF(x, y) for x, y in waypoints]
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QColor("#2563eb"), 4.0))
        self.setBrush(Qt.NoBrush)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        # Keep nets above symbol bodies so new connections are always visible.
        self.setZValue(1.0)
        # Build handles for each waypoint (initially hidden)
        self._handles: list[_WaypointHandle] = []
        for i in range(len(self._waypoints)):
            self._add_handle(i)
        self.refresh_geometry()

    # ── handle management ─────────────────────────────────────────────────
    def _add_handle(self, index: int) -> _WaypointHandle:
        h = _WaypointHandle(index, self)
        if index < len(self._waypoints):
            h.setPos(self._waypoints[index])
        self._handles.append(h)
        return h

    def waypoints_as_tuples(self) -> tuple[tuple[float, float], ...]:
        return tuple((wp.x(), wp.y()) for wp in self._waypoints)

    def _on_handle_moved(self, index: int, new_pos: QPointF) -> None:
        if index < len(self._waypoints):
            self._waypoints[index] = new_pos
        self.refresh_geometry()
        # Persist to document
        scene = self.scene()
        if scene is not None:
            parent = scene.parent()
            if isinstance(parent, CircuitWindow):
                parent.update_connection_waypoints(
                    self.connection_id, self.waypoints_as_tuples()
                )

    def _show_handles(self, visible: bool) -> None:
        for h in self._handles:
            h.setVisible(visible)

    # ── geometry ──────────────────────────────────────────────────────────
    def refresh_geometry(self) -> None:
        pa = self.port_a.sceneBoundingRect().center()
        pb = self.port_b.sceneBoundingRect().center()

        path = QPainterPath()
        path.moveTo(pa)

        if self._waypoints:
            # User-defined waypoints: route orthogonally through each point
            all_pts = [pa] + list(self._waypoints) + [pb]
            for i in range(1, len(all_pts)):
                prev = all_pts[i - 1]
                curr = all_pts[i]
                # Always go horizontal first, then vertical (L-route)
                path.lineTo(QPointF(curr.x(), prev.y()))
                path.lineTo(curr)
        else:
            # Auto-route when no waypoints
            ax, ay = pa.x(), pa.y()
            bx, by = pb.x(), pb.y()

            a_right = self._port_exits_right(self.port_a)
            b_right = self._port_exits_right(self.port_b)
            if a_right is None:
                a_right = bx > ax
            if b_right is None:
                b_right = ax > bx

            def _snap(v: float) -> float:
                return round(v / _GRID_SIZE) * _GRID_SIZE

            can_z = (a_right and not b_right and ax < bx) or (
                not a_right and b_right and ax > bx
            )

            if can_z:
                if abs(ay - by) < 1.0:
                    pass  # straight horizontal
                else:
                    mid_x = _snap((ax + bx) / 2.0)
                    path.lineTo(mid_x, ay)
                    path.lineTo(mid_x, by)
            else:
                a_dir = 1.0 if a_right else -1.0
                b_dir = 1.0 if b_right else -1.0
                a_ext = _snap(ax + a_dir * _GRID_SIZE * 2)
                b_ext = _snap(bx + b_dir * _GRID_SIZE * 2)
                mid_y = _snap((ay + by) / 2.0)
                if abs(mid_y - ay) < _GRID_SIZE and abs(mid_y - by) < _GRID_SIZE:
                    mid_y = _snap(min(ay, by) - _GRID_SIZE * 3)
                path.lineTo(a_ext, ay)
                path.lineTo(a_ext, mid_y)
                path.lineTo(b_ext, mid_y)
                path.lineTo(b_ext, by)

        path.lineTo(pb)
        self.setPath(path)

        # Sync handle positions (handles are children so coords are in item space)
        for i, (handle, wp) in enumerate(zip(self._handles, self._waypoints)):
            handle._suspend_move_callback = True
            handle.setPos(wp)
            handle._suspend_move_callback = False

    @staticmethod
    def _port_exits_right(port_item: PortItem) -> bool | None:
        owner = port_item.owner
        if not isinstance(owner, CircuitBlockItem):
            return None
        local_x = port_item.pos().x()
        return local_x >= owner._block_width / 2.0

    # ── context menu: add / remove waypoint ───────────────────────────────
    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu()
        add_act = menu.addAction("Add waypoint here")
        clear_act = menu.addAction("Clear all waypoints")
        chosen = menu.exec(event.screenPos())
        if chosen == add_act:
            self._insert_waypoint_near(event.scenePos())
        elif chosen == clear_act:
            self._clear_waypoints()

    def _insert_waypoint_near(self, scene_pos: QPointF) -> None:
        sx = round(scene_pos.x() / _GRID_SIZE) * _GRID_SIZE
        sy = round(scene_pos.y() / _GRID_SIZE) * _GRID_SIZE
        new_wp = QPointF(sx, sy)
        # Insert at the index of the nearest segment
        pa = self.port_a.sceneBoundingRect().center()
        pb = self.port_b.sceneBoundingRect().center()
        all_pts = [pa] + list(self._waypoints) + [pb]
        best_idx = 1
        best_dist = float("inf")
        for i in range(len(all_pts) - 1):
            p0 = all_pts[i]
            p1 = all_pts[i + 1]
            cx = (p0.x() + p1.x()) / 2.0
            cy = (p0.y() + p1.y()) / 2.0
            d = (cx - sx) ** 2 + (cy - sy) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i + 1
        self._waypoints.insert(best_idx - 1 if best_idx > 0 else 0, new_wp)
        # Rebuild handles list
        for h in self._handles:
            if h.scene():
                h.scene().removeItem(h)
        self._handles.clear()
        for i in range(len(self._waypoints)):
            self._add_handle(i)
        self._show_handles(self.isSelected())
        self.refresh_geometry()
        self._persist_waypoints()

    def _clear_waypoints(self) -> None:
        for h in self._handles:
            if h.scene():
                h.scene().removeItem(h)
        self._handles.clear()
        self._waypoints.clear()
        self.refresh_geometry()
        self._persist_waypoints()

    def _persist_waypoints(self) -> None:
        scene = self.scene()
        if scene is not None:
            parent = scene.parent()
            if isinstance(parent, CircuitWindow):
                parent.update_connection_waypoints(
                    self.connection_id, self.waypoints_as_tuples()
                )

    # ── selection: show/hide handles ─────────────────────────────────────
    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.ItemSelectedChange:
            self._show_handles(bool(value))
        return super().itemChange(change, value)

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
        if instance.block_kind in {"lumped_r", "lumped_l", "lumped_c", "port_ground", "gnd", "driver_se", "eyescope_se"}:
            self._body_height = (2.0 * _GRID_SIZE) * self._symbol_scale
        elif instance.block_kind in {"port_diff", "driver_diff", "eyescope_diff"}:
            # Taller differential symbols keep +/- pins clearly separated and clickable.
            self._body_height = (3.0 * _GRID_SIZE) * self._symbol_scale
        elif instance.block_kind == "net_node":
            self._body_height = 20.0 * self._symbol_scale  # tiny dot block
        elif instance.block_kind == "touchstone":
            left_count = (instance.nports + 1) // 2
            right_count = instance.nports - left_count
            max_side_ports = max(left_count, right_count, 2)
            self._body_height = max(2.0 * _GRID_SIZE, (max_side_ports + 1) * _GRID_SIZE) * self._symbol_scale
        else:
            self._body_height = max(48.0 * self._symbol_scale, (18.0 + (max(instance.nports, 2) * 10.0)) * self._symbol_scale)
        label_text = instance.display_label
        if instance.block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            label_text = _block_value_label(instance.block_kind, instance.impedance_ohm)
        self._label_band_height = (
            _label_band_height_for_text(label_text, self._block_width, 7.2, minimum=18.0)
            if instance.block_kind not in {"gnd", "net_node"}
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
        if self.instance.block_kind == "driver_se":
            self._port_items[1] = PortItem(self, 1, 0.0, 0.0)
            self._apply_port_layout()
            return
        if self.instance.block_kind == "driver_diff":
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
            cy = self._body_height / 2.0
            if port_number == 1:
                return 0.0, cy - (_GRID_SIZE / 2.0)
            return self._block_width, cy + (_GRID_SIZE / 2.0)
        if self.instance.block_kind == "driver_se":
            return self._block_width, self._body_height / 2.0
        if self.instance.block_kind == "driver_diff":
            cy = self._body_height / 2.0
            if port_number == 1:
                return self._block_width, cy - (_GRID_SIZE / 2.0)
            return self._block_width, cy + (_GRID_SIZE / 2.0)
        if self.instance.block_kind == "eyescope_se":
            return 0.0, self._body_height / 2.0
        if self.instance.block_kind == "eyescope_diff":
            cy = self._body_height / 2.0
            if port_number == 1:
                return 0.0, cy - (_GRID_SIZE / 2.0)
            return 0.0, cy + (_GRID_SIZE / 2.0)
        if self.instance.block_kind == "net_node":
            # Port is at centre of the dot
            return self._block_width / 2.0, self._body_height / 2.0
        if self.instance.block_kind == "touchstone":
            # Keep Touchstone side pin spacing on a fixed grid pitch.
            left_count = (self.instance.nports + 1) // 2
            right_count = self.instance.nports - left_count
            if port_number <= left_count:
                slot = port_number
                x = 0.0
            else:
                slot = port_number - left_count
                x = self._block_width
            y = slot * _GRID_SIZE
            return x, y

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
        elif self.instance.block_kind not in {"lumped_r", "lumped_l", "lumped_c", "gnd", "port_diff", "port_ground", "driver_se", "driver_diff", "eyescope_se", "eyescope_diff", "net_node"}:
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
        elif self.instance.block_kind == "lumped_l":
            self._draw_inductor_symbol(painter, rect)
        elif self.instance.block_kind == "lumped_c":
            self._draw_capacitor_symbol(painter, rect)
        elif self.instance.block_kind in {"driver_se", "driver_diff"}:
            self._draw_driver_canvas(painter, rect)
        elif self.instance.block_kind in {"eyescope_se", "eyescope_diff"}:
            self._draw_eyescope_canvas(painter, rect)
        elif self.instance.block_kind == "net_node":
            self._draw_net_node_canvas(painter, rect)
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
        if self.instance.block_kind == "net_node":
            return  # no label for junction nodes
        label = self.instance.display_label
        if self.instance.block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            label = _block_value_label(self.instance.block_kind, self.instance.impedance_ohm)
        elif self.instance.block_kind in {"port_diff", "port_ground", "eyescope_se", "eyescope_diff"} and self._port_label:
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

    def _draw_inductor_symbol(self, painter: QPainter, rect: QRectF) -> None:
        scale = 1.0 if self._is_touchstone else self._symbol_scale
        y = (self._port_items[1].pos().y() + self._port_items[2].pos().y()) / 2.0
        port_left = self._port_items[1].pos().x()
        port_right = self._port_items[2].pos().x()
        terminal_length = 20.0 * scale
        x1 = port_left + terminal_length if port_left <= port_right else port_left - terminal_length
        x2 = port_right - terminal_length if port_left <= port_right else port_right + terminal_length
        painter.setPen(QPen(QColor("#1e293b"), 2.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(QPointF(port_left, y), QPointF(x1, y))
        painter.drawLine(QPointF(x2, y), QPointF(port_right, y))
        # Draw coil arcs (4 humps centred on wire)
        n_humps = 4
        hump_w = (x2 - x1) / n_humps
        hump_h = abs(hump_w) * 0.8
        start_x = min(x1, x2)
        for i in range(n_humps):
            arc_rect = QRectF(start_x + i * abs(hump_w), y - hump_h, abs(hump_w), hump_h * 2.0)
            painter.drawArc(arc_rect, 0, 180 * 16)

    def _draw_capacitor_symbol(self, painter: QPainter, rect: QRectF) -> None:
        scale = 1.0 if self._is_touchstone else self._symbol_scale
        y = (self._port_items[1].pos().y() + self._port_items[2].pos().y()) / 2.0
        port_left = self._port_items[1].pos().x()
        port_right = self._port_items[2].pos().x()
        cx = (port_left + port_right) / 2.0
        gap = 4.0 * scale
        painter.setPen(QPen(QColor("#1e293b"), 2.0))
        painter.setBrush(Qt.NoBrush)
        # Horizontal leads
        painter.drawLine(QPointF(port_left, y), QPointF(cx - gap, y))
        painter.drawLine(QPointF(cx + gap, y), QPointF(port_right, y))
        # Two vertical plates
        plate_h = 12.0 * scale
        painter.drawLine(QPointF(cx - gap, y - plate_h), QPointF(cx - gap, y + plate_h))
        painter.drawLine(QPointF(cx + gap, y - plate_h), QPointF(cx + gap, y + plate_h))

    def _draw_driver_canvas(self, painter: QPainter, rect: QRectF) -> None:
        fg = QColor("#1e293b")
        is_diff = self.instance.block_kind == "driver_diff"
        painter.setPen(QPen(fg, 2.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 4.0, 4.0)
        # Draw pulse waveform inside
        cx = rect.center().x()
        cy = rect.center().y()
        w = rect.width() * 0.45
        h = rect.height() * 0.3
        painter.setPen(QPen(QColor("#2563eb"), 2.0))
        pts = [
            QPointF(cx - w / 2, cy + h / 2),
            QPointF(cx - w / 4, cy + h / 2),
            QPointF(cx - w / 4, cy - h / 2),
            QPointF(cx, cy - h / 2),
            QPointF(cx, cy + h / 2),
            QPointF(cx + w / 4, cy + h / 2),
            QPointF(cx + w / 4, cy - h / 2),
            QPointF(cx + w / 2, cy - h / 2),
        ]
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

        # Draw terminal lead(s) so ports remain visually attached to body side.
        if is_diff:
            p1 = self._port_items[1].pos()
            p2 = self._port_items[2].pos()
            lead = min(16.0, rect.width() * 0.18)
            painter.setPen(QPen(fg, 2.0))
            painter.drawLine(QPointF(rect.right() - lead, p1.y()), QPointF(p1.x(), p1.y()))
            painter.drawLine(QPointF(rect.right() - lead, p2.y()), QPointF(p2.x(), p2.y()))
        else:
            p1 = self._port_items[1].pos()
            lead = min(16.0, rect.width() * 0.18)
            painter.setPen(QPen(fg, 2.0))
            painter.drawLine(QPointF(rect.right() - lead, p1.y()), QPointF(p1.x(), p1.y()))

    def _draw_eyescope_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """Draw oscilloscope-style block on the circuit canvas."""
        is_diff = self.instance.block_kind == "eyescope_diff"
        fg = QColor("#4c1d95")
        lead_w = rect.width() * 0.18
        body = QRectF(lead_w, rect.top() + 2.0, rect.width() - lead_w - 2.0, rect.height() - 4.0)
        # Terminal lead(s)
        painter.setPen(QPen(QColor("#1e293b"), 2.0))
        if is_diff:
            p1y = self._port_items[1].pos().y()
            p2y = self._port_items[2].pos().y()
            painter.drawLine(QPointF(0.0, p1y), QPointF(lead_w, p1y))
            painter.drawLine(QPointF(0.0, p2y), QPointF(lead_w, p2y))
        else:
            p1y = self._port_items[1].pos().y()
            painter.drawLine(QPointF(0.0, p1y), QPointF(lead_w, p1y))
        # Body
        painter.setPen(QPen(fg, 2.0))
        painter.setBrush(QBrush(QColor("#ede9fe")))
        painter.drawRoundedRect(body, 4.0, 4.0)
        # Screen
        sw = body.width() * 0.72
        sh = body.height() * 0.52
        screen = QRectF(body.center().x() - sw / 2, body.center().y() - sh / 2, sw, sh)
        painter.setPen(QPen(QColor("#4c1d95"), 1.0))
        painter.setBrush(QBrush(QColor("#1e1b4b")))
        painter.drawRect(screen)
        # Eye diagram trace
        _draw_eye_on_screen(painter, screen)

    def _draw_net_node_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """Draw a filled junction dot on the circuit canvas."""
        cx = rect.center().x()
        cy = rect.center().y()
        r = 7.0 * self._symbol_scale
        color = QColor("#1d4ed8") if not self.isSelected() else QColor("#ef4444")
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(cx, cy), r, r)

    def sync_from_instance(self, instance) -> None:
        self.instance = instance
        self._apply_visual_transform()
        self.update()

    def _apply_visual_transform(self) -> None:
        self.setTransformOriginPoint(self._block_width / 2.0, self._body_height / 2.0)
        self.setTransform(self.transform().fromScale(1.0, 1.0))
        self.setRotation(float(self.instance.rotation_deg % 360))
        self._apply_port_layout()

    def itemChange(self, change, value):  # noqa: N802, ANN001
        if change == QGraphicsItem.ItemPositionChange:
            # Snap to grid
            new_pos = value
            snapped_x = round(new_pos.x() / _GRID_SIZE) * _GRID_SIZE
            snapped_y = round(new_pos.y() / _GRID_SIZE) * _GRID_SIZE
            return QPointF(snapped_x, snapped_y)
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
        # ── interactive routing state ─────────────────────────────────────
        self._routing_active: bool = False
        self._routing_start_port: PortItem | None = None
        self._routing_hover_port: PortItem | None = None
        self._routing_waypoints: list[QPointF] = []
        self._routing_preview: QGraphicsPathItem | None = None

    # ── background grid ───────────────────────────────────────────────────
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        super().drawBackground(painter, rect)
        pen = QPen(QColor("#d0d0d0"), 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        left = int(rect.left() / _GRID_SIZE) * _GRID_SIZE
        top = int(rect.top() / _GRID_SIZE) * _GRID_SIZE
        x = left
        while x <= rect.right():
            y = top
            while y <= rect.bottom():
                painter.drawPoint(QPointF(x, y))
                y += _GRID_SIZE
            x += _GRID_SIZE

    # ── routing: start / complete / cancel ────────────────────────────────
    def handle_port_clicked(self, port_item: PortItem) -> None:
        if not self._routing_active:
            # Start routing from this port
            self._routing_active = True
            self._routing_start_port = port_item
            self._routing_waypoints = []
            port_item.set_pending(True)
            self.portSelectionChanged.emit(port_item.port_ref)
            # Create rubber-band preview line
            self._routing_preview = QGraphicsPathItem()
            self._routing_preview.setPen(QPen(QColor("#f59e0b"), 2.5, Qt.DashLine))
            self._routing_preview.setBrush(Qt.NoBrush)
            self._routing_preview.setZValue(10.0)
            self.addItem(self._routing_preview)
            # Cross-cursor on all views
            for v in self.views():
                v.viewport().setCursor(Qt.CrossCursor)
        else:
            if port_item is self._routing_start_port:
                self.cancel_routing()
                return
            self._complete_routing_to_port(port_item)

    def _complete_routing_to_port(self, port_b: PortItem) -> None:
        start = self._routing_start_port
        waypoints = list(self._routing_waypoints)

        # EDA-style pin entry: approach port from one of the three allowed
        # directions (excluding the symbol-facing direction).
        if start is not None:
            last_point = (
                waypoints[-1]
                if waypoints
                else start.sceneBoundingRect().center()
            )
            entry = self._choose_port_entry_point(port_b, last_point)
            if entry is not None:
                waypoints.append(entry)

        self.cancel_routing()
        parent = self.parent()
        if isinstance(parent, CircuitWindow) and start is not None:
            parent.create_connection(start.port_ref, port_b.port_ref, waypoints)

    def _complete_routing_to_junction(self, junction_port: PortItem) -> None:
        """Finish routing to the port of a newly created junction node."""
        start = self._routing_start_port
        waypoints = list(self._routing_waypoints)
        self.cancel_routing()
        parent = self.parent()
        if isinstance(parent, CircuitWindow) and start is not None:
            parent.create_connection(start.port_ref, junction_port.port_ref, waypoints)

    def cancel_routing(self) -> None:
        if self._routing_preview is not None:
            self.removeItem(self._routing_preview)
            self._routing_preview = None
        if self._routing_start_port is not None:
            self._routing_start_port.set_pending(False)
            self._routing_start_port = None
        self._set_routing_hover_port(None)
        self._routing_waypoints = []
        self._routing_active = False
        for v in self.views():
            v.viewport().setCursor(Qt.ArrowCursor)
        self.portSelectionChanged.emit(None)

    # backward-compat alias (used from CircuitWindow)
    def clear_pending_port(self) -> None:
        self.cancel_routing()

    # ── rubber-band preview update ────────────────────────────────────────
    def _update_routing_preview(self, mouse_pos: QPointF) -> None:
        if self._routing_preview is None or self._routing_start_port is None:
            return
        near_port = self._find_port_near(mouse_pos, radius=max(12.0, _GRID_SIZE * 0.7))
        self._set_routing_hover_port(near_port)
        start_scene = self._routing_start_port.sceneBoundingRect().center()
        if near_port is not None:
            target = near_port.sceneBoundingRect().center()
            sx = round(target.x() / _GRID_SIZE) * _GRID_SIZE
            sy = round(target.y() / _GRID_SIZE) * _GRID_SIZE
        else:
            sx = round(mouse_pos.x() / _GRID_SIZE) * _GRID_SIZE
            sy = round(mouse_pos.y() / _GRID_SIZE) * _GRID_SIZE

        path = QPainterPath()
        path.moveTo(start_scene)

        # Draw fixed waypoint segments (horizontal-first L)
        all_fixed = [start_scene] + self._routing_waypoints
        for i in range(1, len(all_fixed)):
            prev = all_fixed[i - 1]
            curr = all_fixed[i]
            path.lineTo(QPointF(curr.x(), prev.y()))
            path.lineTo(curr)

        # Rubber-band segment from last fixed point to mouse
        last = all_fixed[-1]
        path.lineTo(QPointF(sx, last.y()))
        path.lineTo(QPointF(sx, sy))

        self._routing_preview.setPath(path)

    def _set_routing_hover_port(self, port_item: PortItem | None) -> None:
        if self._routing_hover_port is port_item:
            return
        if self._routing_hover_port is not None:
            self._routing_hover_port.set_snap_hover(False)
        self._routing_hover_port = port_item
        if self._routing_hover_port is not None:
            self._routing_hover_port.set_snap_hover(True)

    def _find_port_near(self, scene_pos: QPointF, radius: float = 14.0) -> PortItem | None:
        best: PortItem | None = None
        best_d2 = radius * radius
        for block_item in self._block_items.values():
            for port_number in range(1, block_item.instance.nports + 1):
                p = block_item.port_item(port_number).sceneBoundingRect().center()
                dx = p.x() - scene_pos.x()
                dy = p.y() - scene_pos.y()
                d2 = dx * dx + dy * dy
                if d2 <= best_d2:
                    best_d2 = d2
                    best = block_item.port_item(port_number)
        return best

    @staticmethod
    def _snap_grid(v: float) -> float:
        return round(v / _GRID_SIZE) * _GRID_SIZE

    def _blocked_direction_for_port(self, port_item: PortItem) -> str | None:
        owner = port_item.owner
        port_scene = port_item.sceneBoundingRect().center()
        center_scene = owner.mapToScene(QPointF(owner._block_width / 2.0, owner._body_height / 2.0))
        dx = center_scene.x() - port_scene.x()
        dy = center_scene.y() - port_scene.y()
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    def _choose_port_entry_point(self, port_item: PortItem, last_point: QPointF) -> QPointF | None:
        port_scene = port_item.sceneBoundingRect().center()
        blocked = self._blocked_direction_for_port(port_item)
        candidates: list[QPointF] = []
        dirs = {
            "left": (-_GRID_SIZE, 0.0),
            "right": (_GRID_SIZE, 0.0),
            "up": (0.0, -_GRID_SIZE),
            "down": (0.0, _GRID_SIZE),
        }
        for name, (dx, dy) in dirs.items():
            if name == blocked:
                continue
            cp = QPointF(self._snap_grid(port_scene.x() + dx), self._snap_grid(port_scene.y() + dy))
            candidates.append(cp)
        if not candidates:
            return None

        def _manhattan(a: QPointF, b: QPointF) -> float:
            return abs(a.x() - b.x()) + abs(a.y() - b.y())

        best = min(candidates, key=lambda cp: _manhattan(last_point, cp) + _manhattan(cp, port_scene))
        if _manhattan(best, port_scene) < 1.0:
            return None
        return best

    # ── scene mouse events ────────────────────────────────────────────────
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._routing_active:
            self._update_routing_preview(event.scenePos())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._routing_active:
            pos = event.scenePos()
            if event.button() == Qt.RightButton:
                self.cancel_routing()
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                near_port = self._find_port_near(pos, radius=max(12.0, _GRID_SIZE * 0.7))
                if near_port is not None:
                    self.handle_port_clicked(near_port)
                    event.accept()
                    return
                items_at = self.items(pos)
                # Port click → let super dispatch → PortItem.mousePressEvent → handle_port_clicked
                if any(isinstance(i, PortItem) for i in items_at):
                    super().mousePressEvent(event)
                    return
                # Wire click → create junction
                wire = next((i for i in items_at if isinstance(i, CircuitConnectionItem)), None)
                if wire is not None:
                    self._route_onto_wire(wire, pos)
                    event.accept()
                    return
                # Empty space → add waypoint
                snapped = QPointF(
                    round(pos.x() / _GRID_SIZE) * _GRID_SIZE,
                    round(pos.y() / _GRID_SIZE) * _GRID_SIZE,
                )
                self._routing_waypoints.append(snapped)
                self._update_routing_preview(pos)
                event.accept()
                return
        super().mousePressEvent(event)

    # ── wire junction ─────────────────────────────────────────────────────
    def _route_onto_wire(self, wire: CircuitConnectionItem, pos: QPointF) -> None:
        """Split existing wire at pos, create junction node, complete routing there."""
        snapped = QPointF(
            round(pos.x() / _GRID_SIZE) * _GRID_SIZE,
            round(pos.y() / _GRID_SIZE) * _GRID_SIZE,
        )
        parent = self.parent()
        if not isinstance(parent, CircuitWindow):
            return
        junction_port = parent.create_net_junction_on_wire(wire.connection_id, snapped)
        if junction_port is not None:
            self._complete_routing_to_junction(junction_port)

    # ── scene registry ────────────────────────────────────────────────────
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
            elif block_item.instance.block_kind == "driver_se":
                nums = [
                    str(ep.external_port_number)
                    for ep in document.external_ports
                    if ep.port_ref.instance_id == block_item.instance.instance_id
                ]
                block_item._port_label = ", ".join(f"D{n}" for n in nums) if nums else "DRV"
                block_item.update()
            elif block_item.instance.block_kind == "driver_diff":
                nums = [
                    str(dp.external_port_number)
                    for dp in document.differential_ports
                    if dp.port_ref_plus.instance_id == block_item.instance.instance_id
                ]
                block_item._port_label = ", ".join(f"Dd{n}" for n in nums) if nums else "DRV"
                block_item.update()
            elif block_item.instance.block_kind == "eyescope_se":
                nums = [
                    str(ep.external_port_number)
                    for ep in document.external_ports
                    if ep.port_ref.instance_id == block_item.instance.instance_id
                ]
                block_item._port_label = ", ".join(f"ES{n}" for n in nums) if nums else ""
                block_item.update()
            elif block_item.instance.block_kind == "eyescope_diff":
                nums = [
                    str(dp.external_port_number)
                    for dp in document.differential_ports
                    if dp.port_ref_plus.instance_id == block_item.instance.instance_id
                ]
                block_item._port_label = ", ".join(f"ESD{n}" for n in nums) if nums else ""
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
        self._impedance_label = QLabel("Value")
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

        # --- Simulation mode selector ---
        self._sim_mode = QComboBox()
        self._sim_mode.addItems(["S-Parameters", "Channel Sim"])
        self._sim_mode.currentTextChanged.connect(self._on_sim_mode_changed)

        # --- Channel sim controls ---
        self._channel_sim_button = QPushButton("Run Channel Simulation")
        self._channel_sim_button.clicked.connect(self._run_channel_simulation)
        self._channel_sim_button.setVisible(False)

        self._driver_settings_group = QGroupBox("Driver Settings")
        self._driver_settings_group.setVisible(False)
        drv_layout = QFormLayout(self._driver_settings_group)

        self._drv_v_high = QDoubleSpinBox()
        self._drv_v_high.setRange(-10.0, 10.0)
        self._drv_v_high.setDecimals(4)
        self._drv_v_high.setValue(0.4)
        self._drv_v_high.setSuffix(" V")
        drv_layout.addRow("V high", self._drv_v_high)

        self._drv_v_low = QDoubleSpinBox()
        self._drv_v_low.setRange(-10.0, 10.0)
        self._drv_v_low.setDecimals(4)
        self._drv_v_low.setValue(-0.4)
        self._drv_v_low.setSuffix(" V")
        drv_layout.addRow("V low", self._drv_v_low)

        self._drv_rise_time = QDoubleSpinBox()
        self._drv_rise_time.setRange(0.1, 10000.0)
        self._drv_rise_time.setDecimals(1)
        self._drv_rise_time.setValue(25.0)
        self._drv_rise_time.setSuffix(" ps")
        drv_layout.addRow("Rise time", self._drv_rise_time)

        self._drv_fall_time = QDoubleSpinBox()
        self._drv_fall_time.setRange(0.1, 10000.0)
        self._drv_fall_time.setDecimals(1)
        self._drv_fall_time.setValue(25.0)
        self._drv_fall_time.setSuffix(" ps")
        drv_layout.addRow("Fall time", self._drv_fall_time)

        self._drv_bitrate = QDoubleSpinBox()
        self._drv_bitrate.setRange(0.001, 200.0)
        self._drv_bitrate.setDecimals(3)
        self._drv_bitrate.setValue(10.0)
        self._drv_bitrate.setSuffix(" Gbps")
        drv_layout.addRow("Bitrate", self._drv_bitrate)

        self._drv_prbs = QComboBox()
        self._drv_prbs.addItems(PRBS_CHOICES)
        self._drv_prbs.setCurrentText("PRBS-8")
        drv_layout.addRow("PRBS pattern", self._drv_prbs)

        self._drv_encoding = QComboBox()
        self._drv_encoding.addItems(ENCODING_CHOICES)
        self._drv_encoding.setCurrentText("8b10b")
        drv_layout.addRow("Encoding", self._drv_encoding)

        self._drv_num_bits = QSpinBox()
        self._drv_num_bits.setRange(128, 2**20)
        self._drv_num_bits.setValue(2**13)
        self._drv_num_bits.setSingleStep(1024)
        drv_layout.addRow("Num bits", self._drv_num_bits)

        self._drv_output_port_instance = QComboBox()
        drv_layout.addRow("Output port", self._drv_output_port_instance)

        self._drv_eye_span = QComboBox()
        for span_ui in EYE_SPAN_CHOICES:
            self._drv_eye_span.addItem(f"{span_ui} UI", span_ui)
        default_eye_span_index = self._drv_eye_span.findData(DEFAULT_EYE_SPAN_UI)
        if default_eye_span_index >= 0:
            self._drv_eye_span.setCurrentIndex(default_eye_span_index)
        self._drv_eye_span.currentIndexChanged.connect(self._emit_project_modified)
        drv_layout.addRow("Eye span", self._drv_eye_span)

        self._drv_eye_render_mode = QComboBox()
        for mode in RENDER_MODE_CHOICES:
            self._drv_eye_render_mode.addItem(mode, mode)
        default_render_mode_index = self._drv_eye_render_mode.findData(DEFAULT_RENDER_MODE)
        if default_render_mode_index >= 0:
            self._drv_eye_render_mode.setCurrentIndex(default_render_mode_index)
        self._drv_eye_render_mode.currentIndexChanged.connect(self._emit_project_modified)
        drv_layout.addRow("Eye render", self._drv_eye_render_mode)

        self._drv_eye_quality_preset = QComboBox()
        for preset in QUALITY_PRESET_CHOICES:
            self._drv_eye_quality_preset.addItem(preset, preset)
        default_quality_index = self._drv_eye_quality_preset.findData(DEFAULT_QUALITY_PRESET)
        if default_quality_index >= 0:
            self._drv_eye_quality_preset.setCurrentIndex(default_quality_index)
        self._drv_eye_quality_preset.currentIndexChanged.connect(self._emit_project_modified)
        drv_layout.addRow("Eye quality", self._drv_eye_quality_preset)

        self._stat_group = QGroupBox("Simulazione Statistica")
        self._stat_group.setVisible(False)
        stat_layout = QFormLayout(self._stat_group)

        self._stat_enabled = QCheckBox("Abilita simulazione statistica")
        self._stat_enabled.setChecked(False)
        self._stat_enabled.toggled.connect(self._on_stat_enabled_changed)
        self._stat_enabled.toggled.connect(self._emit_project_modified)
        stat_layout.addRow(self._stat_enabled)

        self._stat_noise = QDoubleSpinBox()
        self._stat_noise.setRange(0.0, 500.0)
        self._stat_noise.setSingleStep(1.0)
        self._stat_noise.setDecimals(1)
        self._stat_noise.setValue(0.0)
        self._stat_noise.setSuffix(" mV")
        self._stat_noise.setEnabled(False)
        self._stat_noise.valueChanged.connect(self._emit_project_modified)
        stat_layout.addRow("Noise RMS", self._stat_noise)

        self._stat_jitter = QDoubleSpinBox()
        self._stat_jitter.setRange(0.0, 100.0)
        self._stat_jitter.setSingleStep(0.5)
        self._stat_jitter.setDecimals(1)
        self._stat_jitter.setValue(0.0)
        self._stat_jitter.setSuffix(" ps")
        self._stat_jitter.setEnabled(False)
        self._stat_jitter.valueChanged.connect(self._emit_project_modified)
        stat_layout.addRow("Jitter RMS", self._stat_jitter)

        self._eye_windows: list = []

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

        sim_form = QFormLayout()
        sim_form.addRow("Simulation mode", self._sim_mode)
        inspector_layout.addLayout(sim_form)

        sweep_form = QFormLayout()
        sweep_form.addRow("Unit", self._frequency_unit)
        sweep_form.addRow("Fmin", self._fmin)
        sweep_form.addRow("Fmax", self._fmax)
        sweep_form.addRow("Step", self._fstep)
        inspector_layout.addLayout(sweep_form)

        impedance_form = QFormLayout()
        impedance_form.addRow(self._impedance_label, self._impedance_editor)
        impedance_form.addRow("Selected symbol size", self._symbol_size_editor)
        inspector_layout.addLayout(impedance_form)

        inspector_layout.addWidget(self._export_button)
        inspector_layout.addWidget(self._driver_settings_group)
        inspector_layout.addWidget(self._stat_group)
        inspector_layout.addWidget(self._channel_sim_button)
        inspector_layout.addWidget(self._status_label)
        inspector_layout.addStretch(1)

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

        driver_spec = None
        if block_kind in {"driver_se", "driver_diff"}:
            driver_spec = DriverSpec(
                voltage_high_v=self._drv_v_high.value(),
                voltage_low_v=self._drv_v_low.value(),
                rise_time_s=self._drv_rise_time.value() * 1e-12,
                fall_time_s=self._drv_fall_time.value() * 1e-12,
                bitrate_gbps=self._drv_bitrate.value(),
                prbs_pattern=self._drv_prbs.currentText(),
                encoding=self._drv_encoding.currentText(),
                num_bits=self._drv_num_bits.value(),
            )

        instance = self._document.add_instance(
            source_file_id=source_file_id,
            display_label=display_label,
            nports=nports,
            position_x=scene_pos.x(),
            position_y=scene_pos.y(),
            block_kind=block_kind,
            impedance_ohm=impedance_ohm,
            symbol_scale=symbol_scale,
            driver_spec=driver_spec,
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
            self._impedance_editor.setSuffix(" Ohm")
            self._impedance_editor.setRange(0.001, 1e9)
            self._impedance_editor.setValue(50.0)
            self._impedance_label.setText("Value")
            self._updating_impedance_editor = False
            self._updating_symbol_size_editor = True
            self._symbol_size_editor.setEnabled(False)
            self._symbol_size_editor.setValue(1.0)
            self._updating_symbol_size_editor = False
            return
        self._selected_instance_id = block_item.instance.instance_id
        kind = block_item.instance.block_kind
        self._updating_impedance_editor = True
        editable_kinds = {"port_ground", "port_diff", "lumped_r", "lumped_l", "lumped_c", "eyescope_se", "eyescope_diff"}
        self._impedance_editor.setEnabled(kind in editable_kinds)
        if kind == "lumped_r":
            self._impedance_editor.setSuffix(" Ohm")
            self._impedance_editor.setRange(1e-6, 1e12)
            self._impedance_editor.setDecimals(6)
            self._impedance_label.setText("Resistance")
        elif kind == "lumped_l":
            self._impedance_editor.setSuffix(" H")
            self._impedance_editor.setRange(1e-15, 1e3)
            self._impedance_editor.setDecimals(15)
            self._impedance_label.setText("Inductance")
        elif kind == "lumped_c":
            self._impedance_editor.setSuffix(" F")
            self._impedance_editor.setRange(1e-18, 1e3)
            self._impedance_editor.setDecimals(18)
            self._impedance_label.setText("Capacitance")
        elif kind in {"eyescope_se", "eyescope_diff"}:
            self._impedance_editor.setSuffix(" Ohm")
            self._impedance_editor.setRange(1.0, 1e12)
            self._impedance_editor.setDecimals(0)
            self._impedance_label.setText("Probe Impedance")
        else:
            self._impedance_editor.setSuffix(" Ohm")
            self._impedance_editor.setRange(0.001, 1e9)
            self._impedance_editor.setDecimals(6)
            self._impedance_label.setText("Impedance")
        self._impedance_editor.setValue(block_item.instance.impedance_ohm)
        self._updating_impedance_editor = False
        self._updating_symbol_size_editor = True
        self._symbol_size_editor.setEnabled(kind != "touchstone")
        self._symbol_size_editor.setValue(float(getattr(block_item.instance, "symbol_scale", 1.0)))
        self._updating_symbol_size_editor = False

    def _on_impedance_changed(self, value: float) -> None:
        if self._updating_impedance_editor or self._selected_instance_id is None:
            return
        instance = self._document.get_instance(self._selected_instance_id)
        if instance is None or instance.block_kind not in {"port_ground", "port_diff", "lumped_r", "lumped_l", "lumped_c", "eyescope_se", "eyescope_diff"}:
            return
        self._document.update_instance_impedance(self._selected_instance_id, value)
        block_item = self._scene._block_items.get(self._selected_instance_id)
        updated_instance = self._document.get_instance(self._selected_instance_id)
        if block_item is not None and updated_instance is not None:
            block_item.sync_from_instance(updated_instance)
        suffix = _block_value_suffix(instance.block_kind)
        self._status_label.setText(f"Value updated to {value:g}{suffix}.")
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

    def create_connection(
        self,
        port_a: CircuitPortRef,
        port_b: CircuitPortRef,
        waypoints: list[QPointF] | None = None,
    ) -> None:
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
                existing_item = self._scene._connection_items.get(connection.connection_id)
                if existing_item is not None:
                    self._scene.clearSelection()
                    existing_item.setSelected(True)
                    center = existing_item.path().boundingRect().center()
                    self._canvas.centerOn(center)
                self._status_label.setText("Connection already exists and has been selected.")
                return
        wp_tuples: tuple[tuple[float, float], ...] = ()
        if waypoints:
            wp_tuples = tuple((p.x(), p.y()) for p in waypoints)
        connection = self._document.add_connection(port_a, port_b, waypoints=wp_tuples)
        port_item_a = self._port_item_for_ref(port_a)
        port_item_b = self._port_item_for_ref(port_b)
        if port_item_a is None or port_item_b is None:
            self._document.remove_connection(connection.connection_id)
            self._status_label.setText("Could not resolve one of the selected ports.")
            return
        connection_item = CircuitConnectionItem(
            connection.connection_id, port_item_a, port_item_b, connection.waypoints
        )
        self._scene.register_connection(connection_item)
        self._status_label.setText("Net created.")
        self._refresh_validation_state()
        self._emit_project_modified()

    def create_net_junction_on_wire(
        self, connection_id: str, pos: QPointF
    ) -> "PortItem | None":
        """Insert a junction net_node at pos, split the wire, return the node port item."""
        old_conn = next(
            (c for c in self._document.connections if c.connection_id == connection_id), None
        )
        if old_conn is None:
            return None

        # Create the net_node block at snapped position
        node_instance = self._document.add_instance(
            source_file_id="__special__:net_node",
            display_label="N",
            nports=1,
            position_x=pos.x(),
            position_y=pos.y(),
            block_kind="net_node",
            impedance_ohm=0.0,
        )
        node_block = CircuitBlockItem(self._scene, node_instance)
        self._scene.register_block(node_block)

        node_port_ref = CircuitPortRef(node_instance.instance_id, 1)

        # Remove old connection (both from document and scene)
        self._remove_connection(connection_id)

        # Re-connect: old_port_a → node and old_port_b → node (auto-routed, no waypoints)
        self.create_connection(old_conn.port_a, node_port_ref)
        self.create_connection(old_conn.port_b, node_port_ref)

        self._scene.rebuild_export_state(self._document)
        self._refresh_validation_state()
        self._emit_project_modified()

        return node_block.port_item(1)

    def _port_item_for_ref(self, port_ref: CircuitPortRef) -> "PortItem | None":
        block_item = self._scene._block_items.get(port_ref.instance_id)
        if block_item is None:
            return None
        return block_item.port_item(port_ref.port_number)

    def update_connection_waypoints(
        self, connection_id: str, waypoints: tuple[tuple[float, float], ...]
    ) -> None:
        self._document.update_connection_waypoints(connection_id, waypoints)
        self._emit_project_modified()

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
        if self._sim_mode.currentText() == "Channel Sim":
            self._refresh_output_port_list()
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

    # ------------------------------------------------------------------ #
    #  Channel simulation helpers                                         #
    # ------------------------------------------------------------------ #

    def _on_stat_enabled_changed(self, checked: bool) -> None:
        self._stat_noise.setEnabled(checked)
        self._stat_jitter.setEnabled(checked)

    def _on_sim_mode_changed(self, mode: str) -> None:  # noqa: F811
        is_channel = mode == "Channel Sim"
        self._export_button.setVisible(not is_channel)
        self._channel_sim_button.setVisible(is_channel)
        self._driver_settings_group.setVisible(is_channel)
        self._stat_group.setVisible(is_channel)
        if is_channel:
            self._refresh_output_port_list()

    def _refresh_output_port_list(self) -> None:
        self._drv_output_port_instance.clear()
        for inst in self._document.instances:
            if inst.block_kind in {"port_ground", "port_diff", "eyescope_se", "eyescope_diff"}:
                label = f"{inst.block_kind} ({inst.instance_id[:8]})"
                self._drv_output_port_instance.addItem(label, inst.instance_id)
        pending = getattr(self, "_pending_output_port_instance_id", None)
        if pending is not None:
            idx = self._drv_output_port_instance.findData(pending)
            if idx >= 0:
                self._drv_output_port_instance.setCurrentIndex(idx)
            self._pending_output_port_instance_id = None

    def _run_channel_simulation(self) -> None:
        # Find the driver instance in the document
        driver_inst = None
        for inst in self._document.instances:
            if inst.block_kind in {"driver_se", "driver_diff"}:
                driver_inst = inst
                break
        if driver_inst is None:
            QMessageBox.warning(self, "Channel Sim", "No driver block found in the circuit.")
            return

        # Update driver spec from UI controls
        spec = DriverSpec(
            voltage_high_v=self._drv_v_high.value(),
            voltage_low_v=self._drv_v_low.value(),
            rise_time_s=self._drv_rise_time.value() * 1e-12,
            fall_time_s=self._drv_fall_time.value() * 1e-12,
            bitrate_gbps=self._drv_bitrate.value(),
            prbs_pattern=self._drv_prbs.currentText(),
            encoding=self._drv_encoding.currentText(),
            num_bits=self._drv_num_bits.value(),
        )
        self._document.update_instance_driver_spec(driver_inst.instance_id, spec)

        # Get selected output port
        out_idx = self._drv_output_port_instance.currentIndex()
        if out_idx < 0:
            QMessageBox.warning(self, "Channel Sim", "No output port selected.")
            return
        out_instance_id = self._drv_output_port_instance.itemData(out_idx)

        progress = QProgressDialog("Running channel simulation...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Channel Simulation")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)

        def _on_progress(percent: int, label: str) -> None:
            if progress.wasCanceled():
                raise InterruptedError("Simulation cancelled by user.")
            progress.setLabelText(label)
            progress.setValue(percent)
            QApplication.processEvents()

        try:
            self._status_label.setText("Running channel simulation...")
            result = simulate_channel(
                self._document,
                self._state,
                driver_instance_id=driver_inst.instance_id,
                output_port_instance_id=out_instance_id,
                progress_callback=_on_progress,
            )
        except InterruptedError:
            self._status_label.setText("Channel simulation cancelled.")
            progress.close()
            return
        except Exception as exc:
            progress.close()
            self._status_label.setText(f"Channel simulation failed: {exc}")
            QMessageBox.warning(self, "Channel Sim failed", str(exc))
            return

        progress.setValue(100)
        progress.close()
        win = EyeDiagramWindow(
            result,
            title=f"Eye Diagram – Circuit #{self.window_number}",
            parent=self,
            initial_span_ui=self._selected_eye_span_ui(),
            initial_render_mode=self._selected_eye_render_mode(),
            initial_quality_preset=self._selected_eye_quality_preset(),
            statistical_enabled=self._stat_enabled.isChecked(),
            noise_rms_mv=self._stat_noise.value(),
            jitter_rms_ps=self._stat_jitter.value(),
        )
        win.span_changed.connect(self._on_eye_span_window_changed)
        win.render_mode_changed.connect(self._on_eye_render_mode_window_changed)
        win.quality_preset_changed.connect(self._on_eye_quality_preset_window_changed)
        win.setAttribute(Qt.WA_DeleteOnClose)
        win.show()
        self._eye_windows.append(win)

        # Show eye summary in the status bar
        s = win.eye_summary
        def _fmv(v: float) -> str:
            import math as _math
            return "n/a" if not _math.isfinite(v) else f"{v * 1000:.2f} mV"
        def _fps(v: float) -> str:
            import math as _math
            return "n/a" if not _math.isfinite(v) else f"{v:.2f} ps"
        width_ps = s.get("width_ps", float("nan"))
        self._status_label.setText(
            f"Channel sim done │ "
            f"Level1: {_fmv(s.get('level1', float('nan')))}  "
            f"Level0: {_fmv(s.get('level0', float('nan')))}  "
            f"Height: {_fmv(s.get('height', float('nan')))}  "
            f"Width: {_fps(width_ps)}"
        )

    def export_project_state(self) -> dict:
        return {
            "window_title": self.windowTitle(),
            "splitter_sizes": self._splitter.sizes(),
            "simulation_mode": self._sim_mode.currentText(),
            "eye_span_ui": self._selected_eye_span_ui(),
            "eye_render_mode": self._selected_eye_render_mode(),
            "eye_quality_preset": self._selected_eye_quality_preset(),
            "stat_enabled": self._stat_enabled.isChecked(),
            "stat_noise_mv": self._stat_noise.value(),
            "stat_jitter_ps": self._stat_jitter.value(),
            "drv_v_high": self._drv_v_high.value(),
            "drv_v_low": self._drv_v_low.value(),
            "drv_rise_time_ps": self._drv_rise_time.value(),
            "drv_fall_time_ps": self._drv_fall_time.value(),
            "drv_bitrate_gbps": self._drv_bitrate.value(),
            "drv_prbs": self._drv_prbs.currentText(),
            "drv_encoding": self._drv_encoding.currentText(),
            "drv_num_bits": self._drv_num_bits.value(),
            "drv_output_port_instance_id": self._drv_output_port_instance.currentData(),
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
        self._scene.cancel_routing()

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
                CircuitConnectionItem(
                    connection.connection_id, port_item_a, port_item_b, connection.waypoints
                )
            )

        sizes = state.get("splitter_sizes")
        if isinstance(sizes, list) and len(sizes) == 3 and all(isinstance(v, int) for v in sizes):
            self._splitter.setSizes(sizes)
        elif isinstance(sizes, list) and len(sizes) == 2 and all(isinstance(v, int) for v in sizes):
            self._splitter.setSizes([sizes[0], sizes[1], 330])

        simulation_mode = str(state.get("simulation_mode", "S-Parameters"))
        if self._sim_mode.findText(simulation_mode) >= 0:
            self._sim_mode.blockSignals(True)
            self._sim_mode.setCurrentText(simulation_mode)
            self._sim_mode.blockSignals(False)
        self._on_sim_mode_changed(self._sim_mode.currentText())

        eye_span_ui = state.get("eye_span_ui", DEFAULT_EYE_SPAN_UI)
        eye_span_index = self._drv_eye_span.findData(int(eye_span_ui))
        if eye_span_index >= 0:
            self._drv_eye_span.blockSignals(True)
            self._drv_eye_span.setCurrentIndex(eye_span_index)
            self._drv_eye_span.blockSignals(False)

        eye_render_mode = str(state.get("eye_render_mode", DEFAULT_RENDER_MODE))
        eye_render_index = self._drv_eye_render_mode.findData(eye_render_mode)
        if eye_render_index >= 0:
            self._drv_eye_render_mode.blockSignals(True)
            self._drv_eye_render_mode.setCurrentIndex(eye_render_index)
            self._drv_eye_render_mode.blockSignals(False)

        eye_quality_preset = str(state.get("eye_quality_preset", DEFAULT_QUALITY_PRESET))
        eye_quality_index = self._drv_eye_quality_preset.findData(eye_quality_preset)
        if eye_quality_index >= 0:
            self._drv_eye_quality_preset.blockSignals(True)
            self._drv_eye_quality_preset.setCurrentIndex(eye_quality_index)
            self._drv_eye_quality_preset.blockSignals(False)

        self._scene.rebuild_export_state(self._document)
        self._refresh_validation_state()

        stat_enabled = bool(state.get("stat_enabled", False))
        self._stat_enabled.blockSignals(True)
        self._stat_enabled.setChecked(stat_enabled)
        self._stat_enabled.blockSignals(False)
        self._on_stat_enabled_changed(stat_enabled)

        stat_noise_mv = float(state.get("stat_noise_mv", 0.0))
        self._stat_noise.blockSignals(True)
        self._stat_noise.setValue(stat_noise_mv)
        self._stat_noise.blockSignals(False)

        stat_jitter_ps = float(state.get("stat_jitter_ps", 0.0))
        self._stat_jitter.blockSignals(True)
        self._stat_jitter.setValue(stat_jitter_ps)
        self._stat_jitter.blockSignals(False)

        self._drv_v_high.blockSignals(True)
        self._drv_v_high.setValue(float(state.get("drv_v_high", 0.4)))
        self._drv_v_high.blockSignals(False)

        self._drv_v_low.blockSignals(True)
        self._drv_v_low.setValue(float(state.get("drv_v_low", -0.4)))
        self._drv_v_low.blockSignals(False)

        self._drv_rise_time.blockSignals(True)
        self._drv_rise_time.setValue(float(state.get("drv_rise_time_ps", 25.0)))
        self._drv_rise_time.blockSignals(False)

        self._drv_fall_time.blockSignals(True)
        self._drv_fall_time.setValue(float(state.get("drv_fall_time_ps", 25.0)))
        self._drv_fall_time.blockSignals(False)

        self._drv_bitrate.blockSignals(True)
        self._drv_bitrate.setValue(float(state.get("drv_bitrate_gbps", 10.0)))
        self._drv_bitrate.blockSignals(False)

        drv_prbs = str(state.get("drv_prbs", "PRBS-8"))
        if self._drv_prbs.findText(drv_prbs) >= 0:
            self._drv_prbs.blockSignals(True)
            self._drv_prbs.setCurrentText(drv_prbs)
            self._drv_prbs.blockSignals(False)

        drv_encoding = str(state.get("drv_encoding", "8b10b"))
        if self._drv_encoding.findText(drv_encoding) >= 0:
            self._drv_encoding.blockSignals(True)
            self._drv_encoding.setCurrentText(drv_encoding)
            self._drv_encoding.blockSignals(False)

        self._drv_num_bits.blockSignals(True)
        self._drv_num_bits.setValue(int(state.get("drv_num_bits", 2**13)))
        self._drv_num_bits.blockSignals(False)

        self._pending_output_port_instance_id = state.get("drv_output_port_instance_id")
        if self._pending_output_port_instance_id is not None:
            self._refresh_output_port_list()

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

    def _selected_eye_span_ui(self) -> int:
        span_ui = self._drv_eye_span.currentData()
        return int(span_ui) if span_ui is not None else DEFAULT_EYE_SPAN_UI

    def _selected_eye_render_mode(self) -> str:
        render_mode = self._drv_eye_render_mode.currentData()
        return str(render_mode) if render_mode is not None else DEFAULT_RENDER_MODE

    def _selected_eye_quality_preset(self) -> str:
        preset = self._drv_eye_quality_preset.currentData()
        return str(preset) if preset is not None else DEFAULT_QUALITY_PRESET

    def _on_eye_span_window_changed(self, span_ui: int) -> None:
        eye_span_index = self._drv_eye_span.findData(int(span_ui))
        if eye_span_index < 0 or eye_span_index == self._drv_eye_span.currentIndex():
            return
        self._drv_eye_span.setCurrentIndex(eye_span_index)

    def _on_eye_render_mode_window_changed(self, render_mode: str) -> None:
        eye_render_index = self._drv_eye_render_mode.findData(str(render_mode))
        if eye_render_index < 0 or eye_render_index == self._drv_eye_render_mode.currentIndex():
            return
        self._drv_eye_render_mode.setCurrentIndex(eye_render_index)

    def _on_eye_quality_preset_window_changed(self, quality_preset: str) -> None:
        quality_index = self._drv_eye_quality_preset.findData(str(quality_preset))
        if quality_index < 0 or quality_index == self._drv_eye_quality_preset.currentIndex():
            return
        self._drv_eye_quality_preset.setCurrentIndex(quality_index)