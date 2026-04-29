from __future__ import annotations

import json
from typing import Dict, Optional

from PySide6.QtCore import QMimeData, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QFont, QFontMetrics, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QShortcut
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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sparams_utility.circuit_solver import (
    ChannelSimResult,
    TransientSimResult,
    simulate_channel,
    simulate_transient,
    solve_circuit_network,
    to_touchstone_string_with_format,
)
from sparams_utility.models.circuit import (
    AttenuatorSpec,
    CirculatorSpec,
    CouplerSpec,
    CircuitDocument,
    CircuitPortRef,
    DriverSpec,
    FrequencySweepSpec,
    PRBS_CHOICES,
    ENCODING_CHOICES,
    TRANSIENT_POLARITY_CHOICES,
    TransientSourceSpec,
    SubstrateSpec,
    TransmissionLineSpec,
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
from sparams_utility.ui.transient_window import TransientResultWindow

_MIME_BLOCK_DEF = "application/x-sparams-block-def"
_BLOCK_WIDTH = 80.0
_PORT_RADIUS = 6.0
_GRID_SIZE = 20.0
_PALETTE_PREVIEW_WIDTH = 170.0
_TOUCHSTONE_PALETTE_PREVIEW_WIDTH = 220.0
_SCHEMATIC_BG = QColor("#f7f7f7")
_FREQUENCY_UNIT_SCALE = {
    "Hz": 1.0,
    "KHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
}
_TRANSIENT_SOURCE_KINDS = {"transient_step_se", "transient_pulse_se"}
_SUBSTRATE_KINDS = {"substrate", "substrate_stripline"}
_TLINE_KINDS = {
    "tline_microstrip",
    "tline_stripline",
    "tline_microstrip_coupled",
    "tline_stripline_coupled",
    "tline_cpw",
    "tline_cpw_coupled",
    "taper",
}
_TLINE_COUPLED_KINDS = {
    "tline_microstrip_coupled",
    "tline_stripline_coupled",
    "tline_cpw_coupled",
}
_TLINE_TAG = {
    "tline_microstrip": "Microstrip",
    "tline_stripline": "Stripline",
    "tline_microstrip_coupled": "Coupled Microstrip",
    "tline_stripline_coupled": "Coupled Stripline",
    "tline_cpw": "CPW",
    "tline_cpw_coupled": "Coupled CPW",
    "taper": "Taper (beta)",
}
# "taper" is intentionally absent: it accepts either substrate kind and
# is checked explicitly by the validation code.
_TLINE_REQUIRED_SUBSTRATE_KIND = {
    "tline_microstrip": "substrate",
    "tline_microstrip_coupled": "substrate",
    "tline_stripline": "substrate_stripline",
    "tline_stripline_coupled": "substrate_stripline",
    "tline_cpw": "substrate",
    "tline_cpw_coupled": "substrate",
}
_COMPONENT_KINDS = {"attenuator", "circulator", "coupler"}
_TRANSIENT_OUTPUT_KINDS = {"port_ground", "port_diff", "eyescope_se", "eyescope_diff", "scope_se", "scope_diff"}
_SCOPE_PROBE_KINDS = {"scope_se", "scope_diff"}


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
    if block_kind == "transient_step_se":
        return "V-Step"
    if block_kind == "transient_pulse_se":
        return "V-Pulse"
    if block_kind == "scope_se":
        return "Scope"
    if block_kind == "scope_diff":
        return "Scope Diff"
    return f"{value:g} Ohm"


def _label_band_height_for_text(
    text: str,
    width: float,
    point_size: float,
    *,
    minimum: float = 18.0,
    wrap_mode=Qt.TextWordWrap,
) -> float:
    font = QFont()
    font.setPointSizeF(point_size)
    metrics = QFontMetrics(font)
    text_rect = metrics.boundingRect(
        0,
        0,
        max(24, int(width - 6.0)),
        1000,
        int(wrap_mode | Qt.AlignCenter),
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
        self._preview_width = (
            _TOUCHSTONE_PALETTE_PREVIEW_WIDTH
            if block_kind == "touchstone"
            else _PALETTE_PREVIEW_WIDTH
        )
        self._touchstone_label_band_height = 0.0
        self.setMinimumWidth(int(self._preview_width))
        if block_kind == "touchstone":
            body_h = max(26, (max(nports, 2) + 1) * 9)
            self._touchstone_label_band_height = _label_band_height_for_text(
                self._label,
                self._preview_width - 12.0,
                6.5,
                minimum=18.0,
                wrap_mode=Qt.TextWrapAnywhere,
            )
            self.setMinimumHeight(int(body_h + self._touchstone_label_band_height + 8.0))
        elif block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            self.setMinimumHeight(56)
        elif block_kind in {"port_diff", "port_ground", "gnd"}:
            self.setMinimumHeight(52)
        elif block_kind in {"driver_se", "driver_diff"}:
            self.setMinimumHeight(56)
        elif block_kind in _TRANSIENT_SOURCE_KINDS:
            self.setMinimumHeight(56)
        elif block_kind in {"eyescope_se", "eyescope_diff", "scope_se", "scope_diff"}:
            self.setMinimumHeight(56)
        elif block_kind in _TLINE_KINDS:
            # Single-bar pill for SE, two-bar pill for coupled, with the
            # kind tag below it. Coupled needs a touch more vertical room.
            self.setMinimumHeight(56 if block_kind not in _TLINE_COUPLED_KINDS else 64)
        elif block_kind == "attenuator":
            self.setMinimumHeight(64)
        elif block_kind == "circulator":
            self.setMinimumHeight(82)
        elif block_kind == "coupler":
            self.setMinimumHeight(78)
        elif block_kind == "net_node":
            self.setMinimumHeight(42)
        else:
            self.setMinimumHeight(max(48, 20 + max(nports, 2) * 6))

    def sizeHint(self) -> QSize:  # noqa: N802
        if self._block_kind == "touchstone":
            body_h = max(26, (max(self._nports, 2) + 1) * 9)
            return QSize(
                int(self._preview_width),
                int(body_h + self._touchstone_label_band_height + 8.0),
            )
        if self._block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
            return QSize(int(self._preview_width), 56)
        if self._block_kind in {"port_diff", "port_ground", "gnd"}:
            return QSize(int(self._preview_width), 52)
        if self._block_kind in {"driver_se", "driver_diff"}:
            return QSize(int(self._preview_width), 56)
        if self._block_kind in _TRANSIENT_SOURCE_KINDS:
            return QSize(int(self._preview_width), 56)
        if self._block_kind in {"eyescope_se", "eyescope_diff", "scope_se", "scope_diff"}:
            return QSize(int(self._preview_width), 56)
        if self._block_kind in _TLINE_KINDS:
            return QSize(
                int(self._preview_width),
                56 if self._block_kind not in _TLINE_COUPLED_KINDS else 64,
            )
        if self._block_kind == "attenuator":
            return QSize(int(self._preview_width), 64)
        if self._block_kind == "circulator":
            return QSize(int(self._preview_width), 82)
        if self._block_kind == "coupler":
            return QSize(int(self._preview_width), 78)
        if self._block_kind == "net_node":
            return QSize(int(self._preview_width), 42)
        return QSize(int(self._preview_width), max(48, 20 + max(self._nports, 2) * 6))

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
        elif self._block_kind in _TRANSIENT_SOURCE_KINDS:
            cx = self.rect().center().x()
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, preview_h)
        elif self._block_kind in {"eyescope_se", "eyescope_diff", "scope_se", "scope_diff"}:
            cx = self.rect().center().x()
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, preview_h)
        elif self._block_kind == "net_node":
            cx = self.rect().center().x()
            cy = self.rect().center().y() - 8.0
            rect = QRectF(cx - 8.0, cy - 8.0, 16.0, 16.0)
        elif self._block_kind in _SUBSTRATE_KINDS:
            cx = self.rect().center().x()
            sub_w = max(preview_w, 70.0)
            sub_h = max(preview_h, 30.0)
            rect = QRectF(cx - sub_w / 2.0, 4.0, sub_w, sub_h)
        elif self._block_kind in _TLINE_KINDS:
            # Match the on-canvas tline body proportions: short bar with
            # long external pin stubs, label below.
            cx = self.rect().center().x()
            tl_w = max(preview_w, 80.0)
            tl_h = 26.0 if self._block_kind in _TLINE_COUPLED_KINDS else 16.0
            rect = QRectF(cx - tl_w / 2.0, 4.0, tl_w, tl_h)
        elif self._block_kind == "attenuator":
            cx = self.rect().center().x()
            rect = QRectF(cx - 50.0, 6.0, 100.0, 32.0)
        elif self._block_kind == "circulator":
            cx = self.rect().center().x()
            rect = QRectF(cx - 28.0, 4.0, 56.0, 56.0)
        elif self._block_kind == "coupler":
            cx = self.rect().center().x()
            rect = QRectF(cx - 55.0, 6.0, 110.0, 44.0)
        elif self._block_kind == "touchstone":
            cx = self.rect().center().x()
            body_h = max(22.0, (max(self._nports, 2) + 1) * 7.0)
            rect = QRectF(cx - preview_w / 2.0, 4.0, preview_w, body_h)
        else:
            rect = self.rect().adjusted(40, 8, -40, -28)
        painter.setPen(QPen(QColor("#1e40af"), 1.6))
        if self._block_kind not in {"lumped_r", "lumped_l", "lumped_c", "gnd", "port_diff", "port_ground", "touchstone", "driver_se", "driver_diff", "eyescope_se", "eyescope_diff", "scope_se", "scope_diff", "net_node", *_SUBSTRATE_KINDS, *_TRANSIENT_SOURCE_KINDS, *_TLINE_KINDS, *_COMPONENT_KINDS}:
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
        if self._block_kind != "touchstone":
            # Use the full widget width for the text label so longer block
            # names (e.g. "Coupled Microstrip") are not clipped by the
            # narrow body rectangle. Keep a small gap so the label sits
            # close to the symbol (matching the touchstone preview spacing).
            label_rect = QRectF(0.0, rect.bottom() + 2.0, float(self.width()), 18.0)
            painter.drawText(label_rect, Qt.AlignCenter, self._label)

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
        if self._block_kind in _TRANSIENT_SOURCE_KINDS:
            self._draw_transient_source_symbol(painter, rect)
            return
        if self._block_kind in {"eyescope_se", "eyescope_diff"}:
            self._draw_eyescope_symbol(painter, rect)
            return
        if self._block_kind in {"scope_se", "scope_diff"}:
            self._draw_scope_symbol(painter, rect)
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
        if self._block_kind == "substrate":
            self._draw_substrate_preview_symbol(painter, rect, stripline=False)
            return
        if self._block_kind == "substrate_stripline":
            self._draw_substrate_preview_symbol(painter, rect, stripline=True)
            return
        if self._block_kind in _TLINE_KINDS:
            self._draw_tline_preview(painter, rect)
            return
        if self._block_kind == "attenuator":
            self._draw_attenuator_preview(painter, rect)
            return
        if self._block_kind == "circulator":
            self._draw_circulator_preview(painter, rect)
            return
        if self._block_kind == "coupler":
            self._draw_coupler_preview(painter, rect)
            return
            self._draw_substrate_preview_symbol(painter, rect)
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
        label_area = QRectF(
            6.0,
            rect.bottom() + 2.0,
            max(24.0, self.rect().width() - 12.0),
            max(18.0, self.rect().height() - rect.bottom() - 4.0),
        )
        painter.drawText(label_area, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWrapAnywhere, self._label)

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

    def _draw_substrate_preview_symbol(self, painter: QPainter, rect: QRectF, *, stripline: bool = False) -> None:
        """Compact PCB-stackup glyph for the palette preview."""
        outline = QColor("#1e293b")
        copper = QColor("#b45309")
        ground = QColor("#374151")
        dielectric = QColor("#fde68a")
        margin_x = 14.0
        body_x0 = rect.left() + margin_x
        body_w = max(40.0, rect.width() - 2.0 * margin_x)
        cu_h = 4.0
        gnd_h = 4.0

        painter.setPen(QPen(outline, 1.0))
        if stripline:
            die_h = max(12.0, rect.height() - 2.0 * gnd_h - 4.0)
            top_gnd = QRectF(body_x0, rect.top() + 2.0, body_w, gnd_h)
            die_rect = QRectF(body_x0, top_gnd.bottom(), body_w, die_h)
            bot_gnd = QRectF(body_x0, die_rect.bottom(), body_w, gnd_h)
            painter.setBrush(QBrush(ground))
            painter.drawRect(top_gnd)
            painter.setBrush(QBrush(dielectric))
            painter.drawRect(die_rect)
            painter.setBrush(QBrush(ground))
            painter.drawRect(bot_gnd)
            painter.setBrush(QBrush(copper))
            painter.drawRect(QRectF(body_x0 + body_w * 0.30,
                                    die_rect.center().y() - cu_h / 2.0,
                                    body_w * 0.40, cu_h))
        else:
            die_h = max(12.0, rect.height() - cu_h - gnd_h - 6.0)
            cy_top = rect.top() + 2.0
            die_top = cy_top + cu_h
            gnd_top = die_top + die_h
            painter.setBrush(QBrush(dielectric))
            painter.drawRect(QRectF(body_x0, die_top, body_w, die_h))
            painter.setBrush(QBrush(copper))
            painter.drawRect(QRectF(body_x0 + body_w * 0.25, cy_top, body_w * 0.5, cu_h))
            painter.setBrush(QBrush(ground))
            painter.drawRect(QRectF(body_x0, gnd_top, body_w, gnd_h))

    def _draw_tline_preview(self, painter: QPainter, rect: QRectF) -> None:
        """Compact transmission-line glyph for the palette preview.

        Mirrors the on-canvas drawing: bar(s) ≈⅓ of the available width
        flanked by long pin stubs, with port hexagons at the outer edges
        and a kind tag drawn underneath. For coupled lines two bars are
        stacked one grid pitch apart.
        """
        is_coupled = self._block_kind in _TLINE_COUPLED_KINDS
        is_stripline = self._block_kind in {"tline_stripline", "tline_stripline_coupled"}
        is_cpw = self._block_kind in {"tline_cpw", "tline_cpw_coupled"}
        if is_stripline:
            body_fill = QColor("#dbeafe")
            bar_outline = QColor("#1d4ed8")
        elif is_cpw:
            body_fill = QColor("#dcfce7")
            bar_outline = QColor("#15803d")
        else:
            body_fill = QColor("#fef3c7")
            bar_outline = QColor("#b45309")

        bar_h = 8.0
        bar_len = max(16.0, rect.width() / 3.0)
        bar_cx = rect.center().x()
        bar_x0 = bar_cx - bar_len / 2.0
        bar_x1 = bar_cx + bar_len / 2.0

        if is_coupled:
            cy = rect.center().y()
            top_y = cy - 5.0
            bot_y = cy + 5.0
            bars = [
                QRectF(bar_x0, top_y - bar_h / 2.0, bar_len, bar_h),
                QRectF(bar_x0, bot_y - bar_h / 2.0, bar_len, bar_h),
            ]
            port_ys = [top_y, top_y, bot_y, bot_y]
            port_xs = [rect.left(), rect.right(), rect.left(), rect.right()]
        else:
            cy = rect.center().y()
            bars = [QRectF(bar_x0, cy - bar_h / 2.0, bar_len, bar_h)]
            port_ys = [cy, cy]
            port_xs = [rect.left(), rect.right()]

        # Pin stubs from each port to the closest bar end.
        painter.setPen(QPen(QColor("#1e293b"), 1.4))
        for px, py in zip(port_xs, port_ys):
            stub_x = bar_x0 if px <= bar_cx else bar_x1
            painter.drawLine(QPointF(float(px), float(py)), QPointF(stub_x, float(py)))

        # Bars themselves.
        painter.setPen(QPen(bar_outline, 1.4))
        painter.setBrush(QBrush(body_fill))
        for bar in bars:
            painter.drawRoundedRect(bar, 3.0, 3.0)

        # Port hexagons at the outer edges.
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        for px, py in zip(port_xs, port_ys):
            painter.drawPolygon(_hex_port_polygon(float(px), float(py), _PORT_RADIUS))

    def _draw_attenuator_preview(self, painter: QPainter, rect: QRectF) -> None:
        """Industry-standard attenuator: rectangle with bold 'dB' triangle marker."""
        outline = QColor("#334155")
        body_h = min(rect.height(), 26.0)
        body = QRectF(
            rect.left() + 14.0,
            rect.center().y() - body_h / 2.0,
            rect.width() - 28.0,
            body_h,
        )
        cy = body.center().y()
        # Pin leads
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        painter.drawLine(QPointF(rect.left(), cy), QPointF(body.left(), cy))
        painter.drawLine(QPointF(body.right(), cy), QPointF(rect.right(), cy))
        # Body
        painter.setPen(QPen(outline, 1.6))
        painter.setBrush(QBrush(QColor("#fef9c3")))
        painter.drawRoundedRect(body, 4.0, 4.0)
        # Two big slashes "//" — the IEEE attenuator marker
        painter.setPen(QPen(outline, 2.0))
        slash_h = body.height() * 0.55
        slash_w = body.height() * 0.30
        cx_b = body.center().x()
        for off in (-slash_w * 0.9, slash_w * 0.9):
            painter.drawLine(
                QPointF(cx_b + off - slash_w / 2.0, cy + slash_h / 2.0),
                QPointF(cx_b + off + slash_w / 2.0, cy - slash_h / 2.0),
            )
        # Bold "dB" label below the symbol body
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        font = painter.font()
        old_pt = font.pointSizeF()
        font.setBold(True)
        font.setPointSizeF(max(7.5, old_pt))
        painter.setFont(font)
        painter.drawText(
            QRectF(body.left(), body.bottom() + 1.0, body.width(), 14.0),
            Qt.AlignCenter,
            "dB",
        )
        font.setBold(False)
        font.setPointSizeF(old_pt)
        painter.setFont(font)
        # Port hexagons
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPolygon(_hex_port_polygon(float(rect.left()), float(cy), _PORT_RADIUS))
        painter.drawPolygon(_hex_port_polygon(float(rect.right()), float(cy), _PORT_RADIUS))

    def _draw_circulator_preview(self, painter: QPainter, rect: QRectF) -> None:
        """Industry-standard 3-port circulator: circle with curved arrow inside."""
        from math import cos, sin, radians, pi as _pi

        outline = QColor("#334155")
        # Body = inset square so we leave room for the three port leads.
        side = min(rect.width(), rect.height()) - 12.0
        cx = rect.center().x()
        cy = rect.top() + side / 2.0 + 4.0
        radius = side / 2.0
        # Three port positions: 9 o'clock, 3 o'clock, 6 o'clock.
        port_pts = [
            (cx - radius - 6.0, cy),
            (cx + radius + 6.0, cy),
            (cx, cy + radius + 6.0),
        ]
        # Pin leads from circle edge to port hex.
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        for px, py in port_pts:
            dx = px - cx
            dy = py - cy
            length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            ex = cx + dx * radius / length
            ey = cy + dy * radius / length
            painter.drawLine(QPointF(px, py), QPointF(ex, ey))
        # Body circle with light pink fill.
        painter.setPen(QPen(outline, 1.8))
        painter.setBrush(QBrush(QColor("#fce7f3")))
        painter.drawEllipse(QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius))
        # Inner circular arrow (~280° sweep, CW) — thicker stroke for visibility.
        arc_r = radius * 0.62
        arc_rect = QRectF(cx - arc_r, cy - arc_r, 2.0 * arc_r, 2.0 * arc_r)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(outline, 2.0))
        start_deg = 60.0
        sweep_deg = -280.0  # Qt: negative = CW
        painter.drawArc(arc_rect, int(start_deg * 16), int(sweep_deg * 16))
        # Arrowhead at the end of the arc.
        end_deg = start_deg + sweep_deg
        end_rad = radians(end_deg)
        ax = cx + arc_r * cos(end_rad)
        ay = cy - arc_r * sin(end_rad)
        # Tangent direction at end (CW so we point along -tangent of CCW).
        tx = sin(end_rad)
        ty = cos(end_rad)
        ah_len = radius * 0.32
        for sign in (-1.0, 1.0):
            theta = sign * (28.0 * _pi / 180.0)
            rx = tx * cos(theta) - ty * sin(theta)
            ry = tx * sin(theta) + ty * cos(theta)
            painter.drawLine(QPointF(ax, ay), QPointF(ax - rx * ah_len, ay - ry * ah_len))
        # Port hexagons.
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        for px, py in port_pts:
            painter.drawPolygon(_hex_port_polygon(float(px), float(py), _PORT_RADIUS))

    def _draw_coupler_preview(self, painter: QPainter, rect: QRectF) -> None:
        """Industry-standard directional coupler: two parallel transmission\n        lines with a coupling region between them.\n        """
        outline = QColor("#334155")
        # Body inset to leave port leads.
        body = QRectF(
            rect.left() + 12.0,
            rect.top() + 6.0,
            rect.width() - 24.0,
            rect.height() - 12.0,
        )
        # Two horizontal lines (the coupled tracks).
        bar_y_top = body.top() + body.height() * 0.28
        bar_y_bot = body.top() + body.height() * 0.72
        # Pin stubs from each port hex to the corresponding bar end.
        port_pts = [
            (rect.left(), bar_y_top),
            (rect.right(), bar_y_top),
            (rect.left(), bar_y_bot),
            (rect.right(), bar_y_bot),
        ]
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        for px, py in port_pts:
            painter.drawLine(QPointF(px, py), QPointF(body.left() if px < body.left() else body.right(), py))
        # Body rectangle (light fill so it reads as a discrete element).
        painter.setPen(QPen(outline, 1.6))
        painter.setBrush(QBrush(QColor("#dcfce7")))
        painter.drawRoundedRect(body, 4.0, 4.0)
        # Two thicker horizontal bars (the lines).
        painter.setPen(QPen(outline, 2.2))
        painter.drawLine(QPointF(body.left() + 4.0, bar_y_top), QPointF(body.right() - 4.0, bar_y_top))
        painter.drawLine(QPointF(body.left() + 4.0, bar_y_bot), QPointF(body.right() - 4.0, bar_y_bot))
        # Coupling region: short vertical segment + arrows indicating coupled energy.
        cx_b = body.center().x()
        coup_w = body.width() * 0.18
        painter.setPen(QPen(outline, 1.4))
        painter.drawLine(QPointF(cx_b - coup_w / 2.0, bar_y_top), QPointF(cx_b - coup_w / 2.0, bar_y_bot))
        painter.drawLine(QPointF(cx_b + coup_w / 2.0, bar_y_top), QPointF(cx_b + coup_w / 2.0, bar_y_bot))
        # Bold "dB" label centred between the two bars.
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        font = painter.font()
        old_pt = font.pointSizeF()
        font.setBold(True)
        font.setPointSizeF(max(7.5, old_pt))
        painter.setFont(font)
        painter.drawText(
            QRectF(body.left(), (bar_y_top + bar_y_bot) / 2.0 - 8.0, body.width(), 16.0),
            Qt.AlignCenter,
            "dB",
        )
        font.setBold(False)
        font.setPointSizeF(old_pt)
        painter.setFont(font)
        # Port hexagons at the four "corners" (aligned to bar y).
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        for px, py in port_pts:
            painter.drawPolygon(_hex_port_polygon(float(px), float(py), _PORT_RADIUS))

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

    def _draw_scope_symbol(self, painter: QPainter, rect: QRectF) -> None:
        """Draw an oscilloscope-style probe preview for time-domain Scope blocks."""
        fg = QColor("#92400e")
        accent = QColor("#facc15")
        is_diff = self._block_kind == "scope_diff"
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
        painter.setBrush(QBrush(QColor("#fef3c7")))
        painter.drawRoundedRect(body, 4.0, 4.0)
        # Screen
        sw = body.width() * 0.72
        sh = body.height() * 0.52
        screen = QRectF(body.center().x() - sw / 2, body.center().y() - sh / 2, sw, sh)
        painter.setPen(QPen(fg, 1.0))
        painter.setBrush(QBrush(QColor("#111827")))
        painter.drawRect(screen)
        # Time-domain waveform: pulse with rise/fall
        painter.setPen(QPen(accent, 1.6))
        margin_x = screen.width() * 0.10
        margin_y = screen.height() * 0.18
        x0 = screen.left() + margin_x
        x1 = screen.right() - margin_x
        y_low = screen.bottom() - margin_y
        y_high = screen.top() + margin_y
        seg = (x1 - x0) / 5.0
        pts = [
            QPointF(x0, y_low),
            QPointF(x0 + seg, y_low),
            QPointF(x0 + seg * 1.5, y_high),
            QPointF(x0 + seg * 3.5, y_high),
            QPointF(x0 + seg * 4.0, y_low),
            QPointF(x1, y_low),
        ]
        for start, end in zip(pts, pts[1:], strict=False):
            painter.drawLine(start, end)

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

    def _draw_transient_source_symbol(self, painter: QPainter, rect: QRectF) -> None:
        is_step = self._block_kind == "transient_step_se"
        fg = QColor("#14532d") if is_step else QColor("#9a3412")
        accent = QColor("#16a34a") if is_step else QColor("#f97316")
        fill = QColor("#ecfdf5") if is_step else QColor("#fff7ed")
        cy = rect.center().y()

        painter.setPen(QPen(fg, 2.0))
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(rect, 4.0, 4.0)

        lead = min(16.0, rect.width() * 0.18)
        painter.setPen(QPen(fg, 2.0))
        painter.drawLine(QPointF(rect.right() - lead, cy), QPointF(rect.right(), cy))

        painter.setPen(QPen(accent, 2.0))
        left = rect.left() + rect.width() * 0.18
        right = rect.right() - rect.width() * 0.14
        low = cy + rect.height() * 0.20
        high = cy - rect.height() * 0.20
        if is_step:
            points = [
                QPointF(left, low),
                QPointF(left + rect.width() * 0.18, low),
                QPointF(left + rect.width() * 0.18, high),
                QPointF(right, high),
            ]
        else:
            points = [
                QPointF(left, low),
                QPointF(left + rect.width() * 0.12, low),
                QPointF(left + rect.width() * 0.12, high),
                QPointF(left + rect.width() * 0.40, high),
                QPointF(left + rect.width() * 0.40, low),
                QPointF(right, low),
            ]
        for start, end in zip(points, points[1:], strict=False):
            painter.drawLine(start, end)

        painter.setPen(QPen(fg, 1.6))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPolygon(_hex_port_polygon(float(rect.right()), float(cy), _PORT_RADIUS))


# Ordered palette categories shown in the left-side block library.
# Touchstone files are appended to "SP Blocks" dynamically.
_PALETTE_CATEGORIES: tuple[str, ...] = (
    "Lumped",
    "Ports",
    "Drivers & Generators",
    "Scope & Probe",
    "Substrates",
    "Transmission Lines",
    "Components",
    "SP Blocks",
)

# Map block_kind -> palette category.
_PALETTE_CATEGORY_BY_KIND: dict[str, str] = {
    "lumped_r": "Lumped",
    "lumped_l": "Lumped",
    "lumped_c": "Lumped",
    "gnd": "Ports",
    "port_ground": "Ports",
    "port_diff": "Ports",
    "driver_se": "Drivers & Generators",
    "driver_diff": "Drivers & Generators",
    "transient_step_se": "Drivers & Generators",
    "transient_pulse_se": "Drivers & Generators",
    "eyescope_se": "Scope & Probe",
    "eyescope_diff": "Scope & Probe",
    "scope_se": "Scope & Probe",
    "scope_diff": "Scope & Probe",
    "substrate": "Substrates",
    "substrate_stripline": "Substrates",
    "tline_microstrip": "Transmission Lines",
    "tline_stripline": "Transmission Lines",
    "tline_microstrip_coupled": "Transmission Lines",
    "tline_stripline_coupled": "Transmission Lines",
    "tline_cpw": "Transmission Lines",
    "tline_cpw_coupled": "Transmission Lines",
    "taper": "Transmission Lines",
    "attenuator": "Components",
    "circulator": "Components",
    "coupler": "Components",
    "touchstone": "SP Blocks",
}


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
            "block_kind": "transient_step_se",
            "label": "V-Step",
            "nports": 1,
            "source_file_id": "__special__:transient_step_se",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "transient_pulse_se",
            "label": "V-Pulse",
            "nports": 1,
            "source_file_id": "__special__:transient_pulse_se",
            "impedance_ohm": 50.0,
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
        {
            "block_kind": "scope_se",
            "label": "Scope",
            "nports": 1,
            "source_file_id": "__special__:scope_se",
            "impedance_ohm": 1e6,
        },
        {
            "block_kind": "scope_diff",
            "label": "Scope Diff",
            "nports": 2,
            "source_file_id": "__special__:scope_diff",
            "impedance_ohm": 2e6,
        },
        {
            "block_kind": "substrate",
            "label": "Substrate (Microstrip)",
            "nports": 0,
            "source_file_id": "__special__:substrate",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "substrate_stripline",
            "label": "Substrate (Stripline)",
            "nports": 0,
            "source_file_id": "__special__:substrate_stripline",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "tline_microstrip",
            "label": "Microstrip Line",
            "nports": 2,
            "source_file_id": "__special__:tline_microstrip",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "tline_stripline",
            "label": "Stripline Line",
            "nports": 2,
            "source_file_id": "__special__:tline_stripline",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "tline_microstrip_coupled",
            "label": "Coupled Microstrip",
            "nports": 4,
            "source_file_id": "__special__:tline_microstrip_coupled",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "tline_stripline_coupled",
            "label": "Coupled Stripline",
            "nports": 4,
            "source_file_id": "__special__:tline_stripline_coupled",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "tline_cpw",
            "label": "CPW Line",
            "nports": 2,
            "source_file_id": "__special__:tline_cpw",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "tline_cpw_coupled",
            "label": "Coupled CPW",
            "nports": 4,
            "source_file_id": "__special__:tline_cpw_coupled",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "taper",
            "label": "Taper (beta)",
            "nports": 2,
            "source_file_id": "__special__:taper",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "attenuator",
            "label": "Attenuator",
            "nports": 2,
            "source_file_id": "__special__:attenuator",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "circulator",
            "label": "Circulator",
            "nports": 3,
            "source_file_id": "__special__:circulator",
            "impedance_ohm": 50.0,
        },
        {
            "block_kind": "coupler",
            "label": "Coupler",
            "nports": 4,
            "source_file_id": "__special__:coupler",
            "impedance_ohm": 50.0,
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

    def wheelEvent(self, event) -> None:  # noqa: N802
        if bool(event.modifiers() & Qt.ControlModifier):
            delta_y = event.angleDelta().y()
            if delta_y == 0:
                event.accept()
                return
            factor = 1.15 if delta_y > 0 else (1.0 / 1.15)
            current_scale = float(self.transform().m11())
            target_scale = current_scale * factor
            if target_scale < 0.10 or target_scale > 10.0:
                event.accept()
                return
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


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
        if instance.block_kind in {"lumped_r", "lumped_l", "lumped_c", "port_ground", "gnd", "driver_se", "eyescope_se", "scope_se", * _TRANSIENT_SOURCE_KINDS}:
            self._body_height = (2.0 * _GRID_SIZE) * self._symbol_scale
        elif instance.block_kind in {"port_diff", "driver_diff", "eyescope_diff", "scope_diff"}:
            # Taller differential symbols keep +/- pins clearly separated and clickable.
            self._body_height = (3.0 * _GRID_SIZE) * self._symbol_scale
        elif instance.block_kind == "net_node":
            self._body_height = 20.0 * self._symbol_scale  # tiny dot block
        elif instance.block_kind in _SUBSTRATE_KINDS:
            # 0-port physical token — wider/taller body so the stackup glyph,
            # title and parameter list (εr, tan δ, h, t) stay legible.
            self._block_width = max(self._block_width, 220.0 * self._symbol_scale)
            self._body_height = (110.0 if instance.block_kind == "substrate_stripline" else 100.0) * self._symbol_scale
        elif instance.block_kind in _TLINE_KINDS:
            # Compact pill (single bar, or two close-stacked bars for the
            # coupled kinds) flanked by long pin stubs. The body width
            # and height are kept as integer multiples of the grid pitch
            # so the ports always land on grid intersections regardless
            # of where the block is placed in the schematic.
            self._block_width = max(self._block_width, 4.0 * _GRID_SIZE * self._symbol_scale)  # 4 grid pitches = 80 px
            if instance.block_kind in _TLINE_COUPLED_KINDS:
                # Two bars near the top, plus a stack of 5 parameter lines
                # below (kind tag + sub + W + L + S). Make the body tall
                # enough so the instance label drawn under the body does
                # not overlap the last parameter line.
                self._body_height = 7.0 * _GRID_SIZE * self._symbol_scale  # 140 px
            else:
                # Single bar at y = GRID, 4 parameter lines (kind, sub,
                # W, L) below. 6 grid pitches keeps the label clear of
                # the parameter stack.
                self._body_height = 6.0 * _GRID_SIZE * self._symbol_scale  # 120 px
        elif instance.block_kind in _COMPONENT_KINDS:
            # Schematic symbols for attenuator (2-port), circulator
            # (3-port), coupler (4-port). Sizes are integer multiples of
            # the grid pitch so all ports land on grid intersections.
            if instance.block_kind == "attenuator":
                self._block_width = 4.0 * _GRID_SIZE * self._symbol_scale  # 80 px
                self._body_height = 2.0 * _GRID_SIZE * self._symbol_scale  # 40 px
            elif instance.block_kind == "circulator":
                self._block_width = 4.0 * _GRID_SIZE * self._symbol_scale  # 80 px
                self._body_height = 4.0 * _GRID_SIZE * self._symbol_scale  # 80 px
            else:  # coupler
                self._block_width = 6.0 * _GRID_SIZE * self._symbol_scale  # 120 px
                self._body_height = 4.0 * _GRID_SIZE * self._symbol_scale  # 80 px
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
            if instance.block_kind not in {"gnd", "net_node", *_SUBSTRATE_KINDS}
            else 0.0
        )
        # Allow long single-line labels (typical for Touchstone file names with
        # underscores) to extend horizontally beyond the block body so they are
        # not clipped. Capped at 4x the block width to avoid runaway layouts.
        if instance.block_kind in {"gnd", "net_node"}:
            self._label_draw_width = self._block_width
        else:
            _label_font = QFont()
            _label_font.setPointSizeF(7.2)
            _label_metrics = QFontMetrics(_label_font)
            text_advance = float(_label_metrics.horizontalAdvance(label_text)) + 8.0
            # For transmission-line blocks we also paint the kind tag
            # ("Microstrip", "Coupled Microstrip", …) and a stack of
            # parameter lines (sub=…, W=…, L=…, S=…) inside the body
            # area. Their widest line is the lower bound for the draw
            # width so the in-body text is not clipped either.
            if instance.block_kind in _TLINE_KINDS:
                tag_label = _TLINE_TAG.get(instance.block_kind, instance.block_kind)
                spec = instance.transmission_line_spec or TransmissionLineSpec()
                sub_label = spec.substrate_name or "<no sub>"
                w_um = float(spec.width_m) * 1e6
                l_mm = float(spec.length_m) * 1e3
                tline_lines = [
                    tag_label,
                    f"sub = {sub_label}",
                    f"W = {w_um:g} µm",
                    f"L = {l_mm:g} mm",
                ]
                if instance.block_kind in _TLINE_COUPLED_KINDS:
                    tline_lines.append(f"S = {float(spec.spacing_m) * 1e6:g} µm")
                widest = max(
                    float(_label_metrics.horizontalAdvance(line)) + 8.0
                    for line in tline_lines
                )
                text_advance = max(text_advance, widest)
            self._label_draw_width = min(
                max(self._block_width, text_advance),
                self._block_width * 4.0,
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
        if self.instance.block_kind in _TRANSIENT_SOURCE_KINDS:
            self._port_items[1] = PortItem(self, 1, 0.0, 0.0)
            self._apply_port_layout()
            return
        if self.instance.block_kind == "eyescope_se":
            self._port_items[1] = PortItem(self, 1, 0.0, 0.0)
            self._apply_port_layout()
            return
        if self.instance.block_kind == "eyescope_diff":
            self._port_items[1] = PortItem(self, 1, 0.0, 0.0)
            self._port_items[2] = PortItem(self, 2, 0.0, 0.0)
            self._apply_port_layout()
            return
        if self.instance.block_kind == "scope_se":
            self._port_items[1] = PortItem(self, 1, 0.0, 0.0)
            self._apply_port_layout()
            return
        if self.instance.block_kind == "scope_diff":
            self._port_items[1] = PortItem(self, 1, 0.0, 0.0)
            self._port_items[2] = PortItem(self, 2, 0.0, 0.0)
            self._apply_port_layout()
            return
        if self.instance.block_kind in _TLINE_KINDS:
            # Single-ended: 2 ports (P1 left, P2 right).
            # Coupled    : 4 ports (P1 left-top, P2 right-top, P3 left-bot,
            #              P4 right-bot) following the synthesis convention
            #              P1=A near, P2=A far, P3=B near, P4=B far.
            for idx in range(1, self.instance.nports + 1):
                self._port_items[idx] = PortItem(self, idx, 0.0, 0.0)
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
        if self.instance.block_kind in _TRANSIENT_SOURCE_KINDS:
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
        if self.instance.block_kind == "scope_se":
            return 0.0, self._body_height / 2.0
        if self.instance.block_kind == "scope_diff":
            cy = self._body_height / 2.0
            if port_number == 1:
                return 0.0, cy - (_GRID_SIZE / 2.0)
            return 0.0, cy + (_GRID_SIZE / 2.0)
        if self.instance.block_kind == "net_node":
            # Port is at centre of the dot
            return self._block_width / 2.0, self._body_height / 2.0
        if self.instance.block_kind in _TLINE_KINDS:
            if self.instance.block_kind in _TLINE_COUPLED_KINDS:
                # 4-port coupled line: bars one grid pitch apart, both
                # rows at integer-grid y so connection wires snap cleanly.
                top_y = 1.0 * _GRID_SIZE
                bot_y = 2.0 * _GRID_SIZE
                if port_number == 1:
                    return 0.0, top_y
                if port_number == 2:
                    return self._block_width, top_y
                if port_number == 3:
                    return 0.0, bot_y
                return self._block_width, bot_y
            # Single-ended 2-port line: port y on the first grid row.
            bar_cy = 1.0 * _GRID_SIZE
            if port_number == 1:
                return 0.0, bar_cy
            return self._block_width, bar_cy
        if self.instance.block_kind == "attenuator":
            # 2 ports at left/right midline.
            return (0.0 if port_number == 1 else self._block_width), self._body_height / 2.0
        if self.instance.block_kind == "circulator":
            # 3 ports: P1 left-mid, P2 right-mid, P3 bottom-mid.
            if port_number == 1:
                return 0.0, self._body_height / 2.0
            if port_number == 2:
                return self._block_width, self._body_height / 2.0
            return self._block_width / 2.0, self._body_height
        if self.instance.block_kind == "coupler":
            # 4 ports at the four corners (P1 TL, P2 TR, P3 BL, P4 BR).
            if port_number == 1:
                return 0.0, 0.0
            if port_number == 2:
                return self._block_width, 0.0
            if port_number == 3:
                return 0.0, self._body_height
            return self._block_width, self._body_height
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
        # Extend horizontal bounds when the label is wider than the block body
        # so long names paint without clipping and selection/repaint regions
        # cover the visible text.
        extra = max(0.0, self._label_draw_width - self._block_width) / 2.0
        return QRectF(-extra, 0.0, self._block_width + 2.0 * extra, self._height)

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
        elif self.instance.block_kind not in {"lumped_r", "lumped_l", "lumped_c", "gnd", "port_diff", "port_ground", "driver_se", "driver_diff", "eyescope_se", "eyescope_diff", "scope_se", "scope_diff", "net_node", *_SUBSTRATE_KINDS, *_TRANSIENT_SOURCE_KINDS, *_TLINE_KINDS, *_COMPONENT_KINDS}:
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
        elif self.instance.block_kind in _TRANSIENT_SOURCE_KINDS:
            self._draw_transient_source_canvas(painter, rect)
        elif self.instance.block_kind in {"eyescope_se", "eyescope_diff"}:
            self._draw_eyescope_canvas(painter, rect)
        elif self.instance.block_kind in {"scope_se", "scope_diff"}:
            self._draw_scope_canvas(painter, rect)
        elif self.instance.block_kind == "net_node":
            self._draw_net_node_canvas(painter, rect)
        elif self.instance.block_kind == "substrate":
            self._draw_substrate_canvas(painter, rect)
        elif self.instance.block_kind == "substrate_stripline":
            self._draw_substrate_stripline_canvas(painter, rect)
        elif self.instance.block_kind in _TLINE_KINDS:
            self._draw_tline_canvas(painter, rect)
        elif self.instance.block_kind == "attenuator":
            self._draw_attenuator_canvas(painter, rect)
        elif self.instance.block_kind == "circulator":
            self._draw_circulator_canvas(painter, rect)
        elif self.instance.block_kind == "coupler":
            self._draw_coupler_canvas(painter, rect)
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
        elif self.instance.block_kind in {"port_diff", "port_ground", "eyescope_se", "eyescope_diff", "scope_se", "scope_diff"} and self._port_label:
            label = self._port_label
        label_extra = max(0.0, self._label_draw_width - self._block_width) / 2.0
        painter.drawText(
            QRectF(
                -label_extra,
                self._body_height + 2.0,
                self._block_width + 2.0 * label_extra,
                self._label_band_height - 4.0,
            ),
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

    def _draw_transient_source_canvas(self, painter: QPainter, rect: QRectF) -> None:
        is_step = self.instance.block_kind == "transient_step_se"
        fg = QColor("#14532d") if is_step else QColor("#9a3412")
        accent = QColor("#16a34a") if is_step else QColor("#f97316")
        fill = QColor("#ecfdf5") if is_step else QColor("#fff7ed")
        painter.setPen(QPen(fg, 2.0))
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(rect, 4.0, 4.0)

        p1 = self._port_items[1].pos()
        lead = min(16.0, rect.width() * 0.18)
        painter.setPen(QPen(fg, 2.0))
        painter.drawLine(QPointF(rect.right() - lead, p1.y()), QPointF(p1.x(), p1.y()))

        cy = rect.center().y()
        left = rect.left() + rect.width() * 0.18
        right = rect.right() - rect.width() * 0.14
        low = cy + rect.height() * 0.20
        high = cy - rect.height() * 0.20
        painter.setPen(QPen(accent, 2.0))
        if is_step:
            points = [
                QPointF(left, low),
                QPointF(left + rect.width() * 0.18, low),
                QPointF(left + rect.width() * 0.18, high),
                QPointF(right, high),
            ]
        else:
            points = [
                QPointF(left, low),
                QPointF(left + rect.width() * 0.12, low),
                QPointF(left + rect.width() * 0.12, high),
                QPointF(left + rect.width() * 0.40, high),
                QPointF(left + rect.width() * 0.40, low),
                QPointF(right, low),
            ]
        for start, end in zip(points, points[1:], strict=False):
            painter.drawLine(start, end)

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

    def _draw_scope_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """Draw an oscilloscope-style time-domain probe on the circuit canvas."""
        is_diff = self.instance.block_kind == "scope_diff"
        fg = QColor("#92400e")
        accent = QColor("#facc15")
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
        painter.setBrush(QBrush(QColor("#fef3c7")))
        painter.drawRoundedRect(body, 4.0, 4.0)
        # Screen
        sw = body.width() * 0.72
        sh = body.height() * 0.52
        screen = QRectF(body.center().x() - sw / 2, body.center().y() - sh / 2, sw, sh)
        painter.setPen(QPen(fg, 1.0))
        painter.setBrush(QBrush(QColor("#111827")))
        painter.drawRect(screen)
        # Time-domain pulse waveform
        painter.setPen(QPen(accent, 1.6))
        margin_x = screen.width() * 0.10
        margin_y = screen.height() * 0.18
        x0 = screen.left() + margin_x
        x1 = screen.right() - margin_x
        y_low = screen.bottom() - margin_y
        y_high = screen.top() + margin_y
        seg = (x1 - x0) / 5.0
        pts = [
            QPointF(x0, y_low),
            QPointF(x0 + seg, y_low),
            QPointF(x0 + seg * 1.5, y_high),
            QPointF(x0 + seg * 3.5, y_high),
            QPointF(x0 + seg * 4.0, y_low),
            QPointF(x1, y_low),
        ]
        for start, end in zip(pts, pts[1:], strict=False):
            painter.drawLine(start, end)

    def _draw_net_node_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """Draw a filled junction dot on the circuit canvas."""
        cx = rect.center().x()
        cy = rect.center().y()
        r = 7.0 * self._symbol_scale
        color = QColor("#1d4ed8") if not self.isSelected() else QColor("#ef4444")
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(cx, cy), r, r)

    def _draw_substrate_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """Draw a microstrip PCB stack-up glyph for substrate blocks.

        Layout:
          • Top band  : substrate name (display_label), bold and centered.
          • Left half : stack-up glyph (top conductor + dielectric + bottom GND).
          • Right half: vertical list of all parameters (εr, tan δ, h, t).
        """
        from PySide6.QtGui import QFont as _QFont  # local import to keep top scope tidy

        spec = self.instance.substrate_spec or SubstrateSpec()
        outline = QColor("#ef4444") if self.isSelected() else QColor("#0f172a")
        copper = QColor("#b45309")
        ground = QColor("#374151")
        dielectric = QColor("#fde68a")
        text_color = QColor("#0f172a")

        # ---- Title band (substrate name, on top) -----------------------
        title_h = 16.0
        title_rect = QRectF(rect.left(), rect.top(), rect.width(), title_h)
        title_font = _QFont(painter.font())
        title_font.setBold(True)
        title_font.setPointSizeF(max(8.5, title_font.pointSizeF() + 0.5))
        painter.setFont(title_font)
        painter.setPen(QPen(text_color, 1.0))
        painter.drawText(title_rect, Qt.AlignCenter, self.instance.display_label)

        # ---- Working area below the title ------------------------------
        work_top = rect.top() + title_h + 2.0
        work_h = max(20.0, rect.bottom() - work_top - 2.0)
        # Split: left ~55% glyph, right ~45% parameter list.
        glyph_w = rect.width() * 0.55
        glyph_rect = QRectF(rect.left() + 4.0, work_top, glyph_w - 8.0, work_h)
        params_rect = QRectF(rect.left() + glyph_w, work_top, rect.width() - glyph_w - 4.0, work_h)

        # ---- Stack-up glyph (left): top conductor + dielectric + bottom GND
        body_x0 = glyph_rect.left()
        body_w = glyph_rect.width()
        cu_h = 4.0
        gnd_h = 5.0
        die_h = max(14.0, glyph_rect.height() - cu_h - gnd_h - 2.0)
        cu_top = glyph_rect.top()
        die_top = cu_top + cu_h
        gnd_top = die_top + die_h

        die_rect = QRectF(body_x0, die_top, body_w, die_h)
        painter.setPen(QPen(outline, 1.0))
        painter.setBrush(QBrush(dielectric))
        painter.drawRect(die_rect)
        painter.setPen(QPen(QColor(180, 130, 50, 110), 0.7))
        step = 6.0
        x = die_rect.left() - die_rect.height()
        while x < die_rect.right():
            painter.drawLine(
                QPointF(max(die_rect.left(), x),
                        die_rect.top() + max(0.0, die_rect.left() - x)),
                QPointF(min(die_rect.right(), x + die_rect.height()),
                        die_rect.top() + min(die_rect.height(), die_rect.right() - x)),
            )
            x += step
        cu_rect = QRectF(body_x0 + body_w * 0.25, cu_top, body_w * 0.5, cu_h)
        painter.setPen(QPen(outline, 1.0))
        painter.setBrush(QBrush(copper))
        painter.drawRect(cu_rect)
        gnd_rect = QRectF(body_x0, gnd_top, body_w, gnd_h)
        painter.setBrush(QBrush(ground))
        painter.drawRect(gnd_rect)

        # ---- Parameters list (right) -----------------------------------
        param_font = _QFont(painter.font())
        param_font.setBold(True)
        painter.setFont(param_font)
        painter.setPen(QPen(text_color, 1.0))
        h_um = float(spec.height_m) * 1e6
        t_um = float(spec.conductor_thickness_m) * 1e6
        lines = [
            f"εr = {spec.epsilon_r:g}",
            f"tan δ = {spec.loss_tangent:g}",
            f"h = {h_um:g} µm",
            f"t = {t_um:g} µm",
        ]
        painter.drawText(params_rect, Qt.AlignVCenter | Qt.AlignLeft, "\n".join(lines))

    def _draw_substrate_stripline_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """Draw a stripline PCB stack-up glyph with asymmetric h_top / h_bottom.

        The vertical position of the embedded conductor is proportional to
        the ratio of the configured h_top and h_bottom distances, so the
        canvas always reflects the real asymmetry of the stripline. Each
        gap is annotated with its own dimension line and value.
        """
        from PySide6.QtGui import QFont as _QFont  # local import to keep top scope tidy

        spec = self.instance.substrate_spec or SubstrateSpec()
        outline = QColor("#ef4444") if self.isSelected() else QColor("#0f172a")
        copper = QColor("#b45309")
        ground = QColor("#374151")
        dielectric = QColor("#fde68a")
        text_color = QColor("#0f172a")
        dim_color = QColor("#1e3a8a")

        # ---- Title band (substrate name, on top) -----------------------
        title_h = 16.0
        title_rect = QRectF(rect.left(), rect.top(), rect.width(), title_h)
        title_font = _QFont(painter.font())
        title_font.setBold(True)
        title_font.setPointSizeF(max(8.5, title_font.pointSizeF() + 0.5))
        painter.setFont(title_font)
        painter.setPen(QPen(text_color, 1.0))
        painter.drawText(title_rect, Qt.AlignCenter, self.instance.display_label)

        # ---- Working area below the title ------------------------------
        work_top = rect.top() + title_h + 2.0
        work_h = max(20.0, rect.bottom() - work_top - 2.0)
        # Split: left ~55% glyph (with extra room for dim. annotations on
        # the right edge of the glyph), right ~45% parameter list.
        glyph_w = rect.width() * 0.55
        glyph_rect = QRectF(rect.left() + 4.0, work_top, glyph_w - 8.0, work_h)
        params_rect = QRectF(rect.left() + glyph_w, work_top, rect.width() - glyph_w - 4.0, work_h)

        # ---- Stack-up glyph: top GND + dielectric + bottom GND ---------
        # Reserve a small dimension-line column inside the glyph rect so
        # h_top / h_bottom can be marked next to the stack.
        dim_col_w = 22.0
        body_x0 = glyph_rect.left()
        body_w = max(20.0, glyph_rect.width() - dim_col_w)
        cu_h = 4.0
        gnd_h = 5.0

        die_h = max(20.0, glyph_rect.height() - 2.0 * gnd_h - 2.0)
        top_gnd = QRectF(body_x0, glyph_rect.top(), body_w, gnd_h)
        die_rect = QRectF(body_x0, top_gnd.bottom(), body_w, die_h)
        bot_gnd = QRectF(body_x0, die_rect.bottom(), body_w, gnd_h)

        painter.setPen(QPen(outline, 1.0))
        painter.setBrush(QBrush(ground))
        painter.drawRect(top_gnd)
        painter.setBrush(QBrush(dielectric))
        painter.drawRect(die_rect)
        painter.setBrush(QBrush(ground))
        painter.drawRect(bot_gnd)
        # Diagonal hatch on dielectric.
        painter.setPen(QPen(QColor(180, 130, 50, 110), 0.7))
        step = 6.0
        x = die_rect.left() - die_rect.height()
        while x < die_rect.right():
            painter.drawLine(
                QPointF(max(die_rect.left(), x),
                        die_rect.top() + max(0.0, die_rect.left() - x)),
                QPointF(min(die_rect.right(), x + die_rect.height()),
                        die_rect.top() + min(die_rect.height(), die_rect.right() - x)),
            )
            x += step

        # Conductor position: scale h_top : t : h_bottom into the dielectric.
        h_top_m = max(0.0, float(spec.stripline_h_top_m))
        h_bot_m = max(0.0, float(spec.stripline_h_bottom_m))
        t_m = max(1e-12, float(spec.conductor_thickness_m))
        total_m = h_top_m + t_m + h_bot_m
        # Pixels-per-meter for this glyph.
        ppm = die_rect.height() / total_m
        # Conductor pixel thickness has a visual minimum so it stays
        # readable, but its position uses the true scaled top/bottom.
        cu_h_px = max(cu_h, t_m * ppm)
        cu_top_y = die_rect.top() + h_top_m * ppm
        cu_rect = QRectF(body_x0 + body_w * 0.30,
                          cu_top_y,
                          body_w * 0.40, cu_h_px)
        painter.setPen(QPen(outline, 1.0))
        painter.setBrush(QBrush(copper))
        painter.drawRect(cu_rect)

        # ---- Dimension lines for h_top and h_bottom --------------------
        dim_x = body_x0 + body_w + 4.0
        tick = 3.0
        painter.setPen(QPen(dim_color, 1.0))
        # h_top: from top GND bottom edge down to conductor top.
        y0 = die_rect.top()
        y1 = cu_rect.top()
        if y1 - y0 >= 2.0:
            painter.drawLine(QPointF(dim_x, y0), QPointF(dim_x, y1))
            painter.drawLine(QPointF(dim_x - tick, y0), QPointF(dim_x + tick, y0))
            painter.drawLine(QPointF(dim_x - tick, y1), QPointF(dim_x + tick, y1))
        # h_bottom: from conductor bottom down to bottom GND top edge.
        y2 = cu_rect.bottom()
        y3 = die_rect.bottom()
        if y3 - y2 >= 2.0:
            painter.drawLine(QPointF(dim_x, y2), QPointF(dim_x, y3))
            painter.drawLine(QPointF(dim_x - tick, y2), QPointF(dim_x + tick, y2))
            painter.drawLine(QPointF(dim_x - tick, y3), QPointF(dim_x + tick, y3))

        # Tiny labels on the dimension lines (h_t / h_b).
        small_font = _QFont(painter.font())
        small_font.setBold(False)
        small_font.setPointSizeF(max(6.5, small_font.pointSizeF() - 1.5))
        painter.setFont(small_font)
        painter.drawText(QRectF(dim_x + tick + 1.0, (y0 + y1) / 2.0 - 6.0, 18.0, 12.0),
                          Qt.AlignVCenter | Qt.AlignLeft, "h_t")
        painter.drawText(QRectF(dim_x + tick + 1.0, (y2 + y3) / 2.0 - 6.0, 18.0, 12.0),
                          Qt.AlignVCenter | Qt.AlignLeft, "h_b")

        # ---- Parameters list (right) -----------------------------------
        param_font = _QFont(painter.font())
        param_font.setBold(True)
        painter.setFont(param_font)
        painter.setPen(QPen(text_color, 1.0))
        h_top_um = h_top_m * 1e6
        h_bot_um = h_bot_m * 1e6
        t_um = float(spec.conductor_thickness_m) * 1e6
        lines = [
            f"εr = {spec.epsilon_r:g}",
            f"tan δ = {spec.loss_tangent:g}",
            f"h_top = {h_top_um:g} µm",
            f"h_bot = {h_bot_um:g} µm",
            f"t = {t_um:g} µm",
        ]
        painter.drawText(params_rect, Qt.AlignVCenter | Qt.AlignLeft, "\n".join(lines))

    def _draw_tline_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """Draw a transmission-line block.

        Layout (single-ended): a short coloured pill of length ≈⅓ of the
        block width, centred horizontally on the bar y-position, with two
        long pin stubs running outward to the port hexagons. The kind
        tag and all parameter lines are stacked vertically below the bar.

        Coupled lines: same idea, but two short bars one grid pitch apart
        near the top of the body, with text underneath both.
        """
        from PySide6.QtGui import QFont as _QFont, QFontMetrics as _QFontMetrics

        kind = self.instance.block_kind
        spec = getattr(self.instance, "transmission_line_spec", None)
        is_coupled = kind in _TLINE_COUPLED_KINDS
        is_stripline = kind in {"tline_stripline", "tline_stripline_coupled"}
        is_cpw = kind in {"tline_cpw", "tline_cpw_coupled"}

        # Match the substrate palette: yellow for microstrip, blue for
        # stripline, green for CPW. Selection state shows a purple outline.
        if is_stripline:
            body_fill = QColor("#dbeafe")
            bar_outline = QColor("#1d4ed8")
        elif is_cpw:
            body_fill = QColor("#dcfce7")
            bar_outline = QColor("#15803d")
        else:
            body_fill = QColor("#fef3c7")
            bar_outline = QColor("#b45309")
        outline = QColor("#7c3aed") if self.isSelected() else bar_outline

        # ---- Bar geometry: ~1/3 of the block width, centred ----------
        bar_h = 10.0
        bar_len = max(18.0, rect.width() / 3.0)
        bar_cx = rect.width() / 2.0
        bar_x0 = bar_cx - bar_len / 2.0
        bar_x1 = bar_cx + bar_len / 2.0

        if is_coupled:
            top_cy = self._port_items[1].pos().y()
            bot_cy = self._port_items[3].pos().y()
            bars = [
                QRectF(bar_x0, top_cy - bar_h / 2.0, bar_len, bar_h),
                QRectF(bar_x0, bot_cy - bar_h / 2.0, bar_len, bar_h),
            ]
        else:
            cy = self._port_items[1].pos().y()
            bars = [QRectF(bar_x0, cy - bar_h / 2.0, bar_len, bar_h)]

        # ---- Long pin stubs from the ports to the bar ends -----------
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        for _port_no, port_item in self._port_items.items():
            ppt = port_item.pos()
            stub_x = bar_x0 if ppt.x() <= rect.width() / 2.0 else bar_x1
            painter.drawLine(QPointF(ppt.x(), ppt.y()), QPointF(stub_x, ppt.y()))

        # ---- Bars (the line(s) themselves) ---------------------------
        painter.setPen(QPen(outline, 1.6))
        painter.setBrush(QBrush(body_fill))
        for bar in bars:
            painter.drawRoundedRect(bar, 3.0, 3.0)

        # ---- Stacked text below: kind tag + parameter lines ----------
        if is_coupled:
            text_top = max(b.bottom() for b in bars) + 3.0
        else:
            text_top = bars[0].bottom() + 3.0

        kind_label = _TLINE_TAG.get(kind, kind)
        if spec is not None:
            sub_label = spec.substrate_name if spec.substrate_name else "<no sub>"
            w_um = float(spec.width_m) * 1e6
            l_mm = float(spec.length_m) * 1e3
            param_lines = [
                f"sub = {sub_label}",
                f"W = {w_um:g} µm",
                f"L = {l_mm:g} mm",
            ]
            if is_coupled:
                param_lines.append(f"S = {float(spec.spacing_m) * 1e6:g} µm")
        else:
            param_lines = ["<no parameters>"]

        # Kind tag, bold and slightly larger than the params.
        tag_font = _QFont(painter.font())
        tag_font.setBold(True)
        tag_font.setPointSizeF(max(7.4, tag_font.pointSizeF() - 0.6))
        painter.setFont(tag_font)
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        tag_h = float(_QFontMetrics(tag_font).height())
        # Allow text to extend horizontally beyond the (narrow) bar so
        # long labels like "Coupled Microstrip" are not clipped. The
        # extra width is centred on the bar.
        tag_metrics = _QFontMetrics(tag_font)
        tag_w_needed = float(tag_metrics.horizontalAdvance(kind_label)) + 8.0
        tag_w = max(rect.width(), tag_w_needed)
        tag_x = rect.left() + (rect.width() - tag_w) / 2.0
        painter.drawText(
            QRectF(tag_x, text_top, tag_w, tag_h),
            Qt.AlignHCenter | Qt.AlignTop,
            kind_label,
        )
        line_top = text_top + tag_h + 1.0

        # Parameter lines, regular weight, all centred and stacked.
        param_font = _QFont(painter.font())
        param_font.setBold(False)
        param_font.setPointSizeF(max(6.8, param_font.pointSizeF() - 1.0))
        painter.setFont(param_font)
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        line_h = float(_QFontMetrics(param_font).height())
        param_metrics = _QFontMetrics(param_font)
        # Compute the widest needed width across all stacked parameter
        # lines so they share a single wide draw rectangle, allowing long
        # substrate names or values to render in full.
        param_w_needed = max(
            (float(param_metrics.horizontalAdvance(line)) + 8.0 for line in param_lines),
            default=rect.width(),
        )
        param_w = max(rect.width(), param_w_needed)
        param_x = rect.left() + (rect.width() - param_w) / 2.0
        for i, line in enumerate(param_lines):
            r = QRectF(param_x, line_top + i * line_h, param_w, line_h)
            painter.drawText(r, Qt.AlignHCenter | Qt.AlignTop, line)

    # ------------------------------------------------------------------
    # Component schematic symbols (attenuator / circulator / coupler)
    # ------------------------------------------------------------------
    def _draw_attenuator_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """2-port attenuator: rounded box with 'ATT' + value, two slash
        lines suggesting the resistive divider, and horizontal pin leads
        running to the side port hexagons.
        """
        from PySide6.QtGui import QFont as _QFont

        outline = QColor("#7c3aed") if self.isSelected() else QColor("#334155")
        spec = getattr(self.instance, "attenuator_spec", None)
        att_db = float(spec.attenuation_db) if spec is not None else 0.0

        # Body inset so the port hexagons sit on the rect edges with a
        # short pin lead drawn horizontally inwards.
        inset = _GRID_SIZE * 0.5
        body = QRectF(
            rect.left() + inset,
            rect.top() + 2.0,
            rect.width() - 2.0 * inset,
            rect.height() - 4.0,
        )

        cy = rect.center().y()
        # Pin leads from the side ports to the body edge.
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        painter.drawLine(QPointF(rect.left(), cy), QPointF(body.left(), cy))
        painter.drawLine(QPointF(body.right(), cy), QPointF(rect.right(), cy))

        # Body box.
        painter.setPen(QPen(outline, 1.6))
        painter.setBrush(QBrush(QColor("#fef9c3")))
        painter.drawRoundedRect(body, 4.0, 4.0)

        # Two parallel slash lines (the resistive-divider hint).
        painter.setPen(QPen(outline, 1.2))
        slash_dx = body.width() * 0.10
        slash_dy = body.height() * 0.55
        cx_body = body.center().x()
        cy_body = body.center().y()
        for offset in (-slash_dx * 0.7, slash_dx * 0.7):
            painter.drawLine(
                QPointF(cx_body + offset - slash_dx * 0.5, cy_body + slash_dy * 0.5),
                QPointF(cx_body + offset + slash_dx * 0.5, cy_body - slash_dy * 0.5),
            )

        # Centred text: ATT on the upper half, "<N> dB" on the lower half.
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        label_font = _QFont(painter.font())
        label_font.setBold(True)
        label_font.setPointSizeF(max(6.8, label_font.pointSizeF() - 0.5))
        painter.setFont(label_font)
        painter.drawText(
            QRectF(body.left(), body.top(), body.width(), body.height() / 2.0),
            Qt.AlignCenter,
            "ATT",
        )
        val_font = _QFont(painter.font())
        val_font.setBold(False)
        painter.setFont(val_font)
        painter.drawText(
            QRectF(body.left(), body.center().y(), body.width(), body.height() / 2.0),
            Qt.AlignCenter,
            f"{att_db:g} dB",
        )

        # Port hexagons at the side edges.
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        for port_no, port_item in self._port_items.items():
            ppt = port_item.pos()
            painter.drawPolygon(_hex_port_polygon(float(ppt.x()), float(ppt.y()), _PORT_RADIUS))

    def _draw_circulator_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """3-port circulator: outer circle with a curved arrow whose
        direction reflects ``circulator_spec.direction`` (cw / ccw).
        Three pin stubs run from each port to the circle edge.
        """
        from math import cos, sin, radians

        outline = QColor("#7c3aed") if self.isSelected() else QColor("#334155")
        spec = getattr(self.instance, "circulator_spec", None)
        direction = (spec.direction if spec is not None else "cw").lower()

        cx = rect.center().x()
        cy = rect.center().y()
        radius = 0.4 * min(rect.width(), rect.height())
        circle_rect = QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius)

        # Pin stubs from each port to the closest point on the circle.
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        for port_no, port_item in self._port_items.items():
            ppt = port_item.pos()
            # Direction from circle centre to port.
            dx = float(ppt.x()) - cx
            dy = float(ppt.y()) - cy
            length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            ex = cx + dx * radius / length
            ey = cy + dy * radius / length
            painter.drawLine(QPointF(float(ppt.x()), float(ppt.y())), QPointF(ex, ey))

        # Outer body circle.
        painter.setPen(QPen(outline, 1.6))
        painter.setBrush(QBrush(QColor("#fce7f3")))
        painter.drawEllipse(circle_rect)

        # Inner arc: ~300 deg sweep with an arrowhead at one end.
        arc_rect = QRectF(
            circle_rect.left() + radius * 0.30,
            circle_rect.top() + radius * 0.30,
            circle_rect.width() - 2.0 * radius * 0.30,
            circle_rect.height() - 2.0 * radius * 0.30,
        )
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(outline, 1.6))
        # In Qt drawArc: positive sweep is counter-clockwise.
        # cw direction => negative sweep so the arrow tip rotates the cw way.
        sweep_deg = 300.0 if direction == "ccw" else -300.0
        start_deg = 30.0
        # Qt drawArc uses 16ths of a degree.
        painter.drawArc(arc_rect, int(start_deg * 16), int(sweep_deg * 16))

        # Arrowhead at the end of the arc.
        end_angle_deg = start_deg + sweep_deg
        end_rad = radians(end_angle_deg)
        ax = arc_rect.center().x() + (arc_rect.width() / 2.0) * cos(end_rad)
        ay = arc_rect.center().y() - (arc_rect.height() / 2.0) * sin(end_rad)
        # Tangent direction at end of arc (perpendicular to the radius,
        # rotated by the sweep sign so it points along the rotation).
        tangent_sign = 1.0 if sweep_deg > 0 else -1.0
        tdx = -sin(end_rad) * tangent_sign
        tdy = -cos(end_rad) * tangent_sign
        ah_len = radius * 0.25
        # Two short lines forming the arrowhead.
        from math import pi as _pi
        # Rotate tangent by ±25° to draw the head.
        for sign in (-1.0, 1.0):
            theta = sign * (25.0 * _pi / 180.0)
            rdx = tdx * cos(theta) - tdy * sin(theta)
            rdy = tdx * sin(theta) + tdy * cos(theta)
            painter.drawLine(
                QPointF(ax, ay),
                QPointF(ax - rdx * ah_len, ay - rdy * ah_len),
            )

        # Port hexagons.
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        for port_no, port_item in self._port_items.items():
            ppt = port_item.pos()
            painter.drawPolygon(_hex_port_polygon(float(ppt.x()), float(ppt.y()), _PORT_RADIUS))

    def _draw_coupler_canvas(self, painter: QPainter, rect: QRectF) -> None:
        """4-port directional / hybrid coupler: outer rect, two parallel
        horizontal bars (through + coupled lines), two short diagonals
        between them in the middle, centred kind label and dB value.
        """
        from PySide6.QtGui import QFont as _QFont

        outline = QColor("#7c3aed") if self.isSelected() else QColor("#334155")
        spec = getattr(self.instance, "coupler_spec", None)
        kind_label = "DC"
        if spec is not None:
            sk = (spec.kind or "").lower()
            if sk == "branch_line_90":
                kind_label = "BL 90°"
            elif sk == "rat_race_180":
                kind_label = "RR 180°"
            else:
                kind_label = "DC"
        coupling_db = float(spec.coupling_db) if spec is not None else 0.0

        # Outer body rect (small inset to leave port hexagons at corners).
        inset = _GRID_SIZE * 0.5
        body = QRectF(
            rect.left() + inset,
            rect.top() + inset,
            rect.width() - 2.0 * inset,
            rect.height() - 2.0 * inset,
        )

        # Pin leads from each corner port to the corresponding body corner.
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        corners = {
            1: (rect.left(), rect.top(), body.left(), body.top()),
            2: (rect.right(), rect.top(), body.right(), body.top()),
            3: (rect.left(), rect.bottom(), body.left(), body.bottom()),
            4: (rect.right(), rect.bottom(), body.right(), body.bottom()),
        }
        for port_no, _port_item in self._port_items.items():
            if port_no not in corners:
                continue
            px, py, bx, by = corners[port_no]
            painter.drawLine(QPointF(px, py), QPointF(bx, by))

        # Body rectangle.
        painter.setPen(QPen(outline, 1.6))
        painter.setBrush(QBrush(QColor("#dcfce7")))
        painter.drawRect(body)

        # Two horizontal bars: through (top) and coupled (bottom).
        bar_pad = body.width() * 0.08
        bar_y_top = body.top() + body.height() * 0.28
        bar_y_bot = body.top() + body.height() * 0.72
        bar_x0 = body.left() + bar_pad
        bar_x1 = body.right() - bar_pad
        painter.setPen(QPen(outline, 2.0))
        painter.drawLine(QPointF(bar_x0, bar_y_top), QPointF(bar_x1, bar_y_top))
        painter.drawLine(QPointF(bar_x0, bar_y_bot), QPointF(bar_x1, bar_y_bot))

        # Two short diagonal lines in the middle (the coupling region).
        mid_cx = body.center().x()
        coupling_dx = body.width() * 0.12
        painter.setPen(QPen(outline, 1.4))
        painter.drawLine(
            QPointF(mid_cx - coupling_dx, bar_y_top),
            QPointF(mid_cx + coupling_dx, bar_y_bot),
        )
        painter.drawLine(
            QPointF(mid_cx + coupling_dx, bar_y_top),
            QPointF(mid_cx - coupling_dx, bar_y_bot),
        )

        # Centre labels: kind on top, "<N> dB" below it.
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        kind_font = _QFont(painter.font())
        kind_font.setBold(True)
        kind_font.setPointSizeF(max(6.8, kind_font.pointSizeF() - 0.4))
        painter.setFont(kind_font)
        mid_h = body.height() * 0.25
        painter.drawText(
            QRectF(body.left(), body.center().y() - mid_h, body.width(), mid_h),
            Qt.AlignCenter,
            kind_label,
        )
        val_font = _QFont(painter.font())
        val_font.setBold(False)
        painter.setFont(val_font)
        painter.drawText(
            QRectF(body.left(), body.center().y(), body.width(), mid_h),
            Qt.AlignCenter,
            f"{coupling_db:g} dB",
        )

        # Port hexagons at the four corners.
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        for port_no, port_item in self._port_items.items():
            ppt = port_item.pos()
            painter.drawPolygon(_hex_port_polygon(float(ppt.x()), float(ppt.y()), _PORT_RADIUS))

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
        # ── multi-block drag state ────────────────────────────────────────
        # Captured at mouse-press, used at mouse-release to translate the
        # waypoints of nets that connect any of the selected blocks and to
        # commit a single undo entry per drag.
        self._block_drag_active: bool = False
        self._block_drag_initial_positions: Dict[str, QPointF] = {}
        self._block_drag_initial_waypoints: Dict[str, list[QPointF]] = {}

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
        # After Qt has resolved the click (incl. selection updates), detect
        # whether the user is starting to drag a block. If so, snapshot the
        # initial geometry of every selected block and every wire so we can
        # translate connected waypoints en masse on release and commit a
        # single undo entry.
        if (
            event.button() == Qt.LeftButton
            and not self._routing_active
            and not self._block_drag_active
        ):
            grabber = self.mouseGrabberItem()
            if isinstance(grabber, CircuitBlockItem):
                self._begin_block_drag()

    def _begin_block_drag(self) -> None:
        self._block_drag_active = True
        self._block_drag_initial_positions = {
            bid: QPointF(bi.pos())
            for bid, bi in self._block_items.items()
            if bi.isSelected()
        }
        self._block_drag_initial_waypoints = {
            cid: [QPointF(wp) for wp in ci._waypoints]
            for cid, ci in self._connection_items.items()
        }
        # Suspend per-step undo capture; we commit a single snapshot on
        # release for the whole drag.
        parent = self.parent()
        if isinstance(parent, CircuitWindow):
            parent._suspend_undo_capture = True

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        was_dragging = self._block_drag_active
        super().mouseReleaseEvent(event)
        if was_dragging and event.button() == Qt.LeftButton:
            self._finish_block_drag()

    def _finish_block_drag(self) -> None:
        if not self._block_drag_active:
            return
        self._block_drag_active = False
        parent = self.parent()
        moved_block_ids: set[str] = set()
        delta = QPointF(0.0, 0.0)
        for bid, start_pos in self._block_drag_initial_positions.items():
            block_item = self._block_items.get(bid)
            if block_item is None:
                continue
            current = block_item.pos()
            d = current - start_pos
            if abs(d.x()) > 0.5 or abs(d.y()) > 0.5:
                moved_block_ids.add(bid)
                if delta.isNull():
                    delta = QPointF(d)

        # Always restore undo capture before any document mutation that
        # should land on the undo stack.
        if isinstance(parent, CircuitWindow):
            parent._suspend_undo_capture = False

        if not moved_block_ids or not isinstance(parent, CircuitWindow):
            self._block_drag_initial_positions.clear()
            self._block_drag_initial_waypoints.clear()
            return

        # Translate waypoints of fully-selected wires (both endpoints moved
        # by the same delta), and reset waypoints of partially-selected
        # wires so the auto-router can re-draw a clean orthogonal path that
        # doesn't tangle through the old positions.
        for cid, conn_item in self._connection_items.items():
            a_in = conn_item.port_a.owner.instance.instance_id in moved_block_ids
            b_in = conn_item.port_b.owner.instance.instance_id in moved_block_ids
            if a_in and b_in:
                initial_wps = self._block_drag_initial_waypoints.get(cid, [])
                if initial_wps:
                    new_wps = [
                        QPointF(
                            self._snap_grid(wp.x() + delta.x()),
                            self._snap_grid(wp.y() + delta.y()),
                        )
                        for wp in initial_wps
                    ]
                    self._set_connection_waypoints(conn_item, new_wps)
            elif a_in or b_in:
                if conn_item._waypoints:
                    self._set_connection_waypoints(conn_item, [])

        # Persist the entire batch (positions + waypoints) into the
        # document, then refresh geometries and emit one project_modified
        # so the undo stack records exactly one entry for this drag.
        document = parent._document
        for bid in moved_block_ids:
            block_item = self._block_items[bid]
            document.update_instance_position(bid, block_item.pos().x(), block_item.pos().y())
        for cid, conn_item in self._connection_items.items():
            document.update_connection_waypoints(cid, conn_item.waypoints_as_tuples())
        for connection_item in self._connection_items.values():
            connection_item.refresh_geometry()

        self._block_drag_initial_positions.clear()
        self._block_drag_initial_waypoints.clear()
        parent._emit_project_modified()

    def _set_connection_waypoints(
        self, conn_item: "CircuitConnectionItem", new_waypoints: list[QPointF]
    ) -> None:
        for h in conn_item._handles:
            if h.scene():
                h.scene().removeItem(h)
        conn_item._handles.clear()
        conn_item._waypoints = [QPointF(wp) for wp in new_waypoints]
        for i in range(len(conn_item._waypoints)):
            conn_item._add_handle(i)
        conn_item._show_handles(conn_item.isSelected())
        conn_item.refresh_geometry()

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
        # During an interactive drag we batch the document update and the
        # undo entry until mouse-release (see _finish_block_drag); only
        # refresh wire geometry here for live visual feedback.
        if not self._block_drag_active:
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
    eye_result_generated = Signal(object)
    sparameter_result_generated = Signal(object)
    transient_result_generated = Signal(object)
    transient_windows_changed = Signal()

    def _fit_circuit_in_view(self):
        # Centra e zooma il circuito per farlo entrare nella finestra
        if self._scene.items():
            rect = self._scene.itemsBoundingRect()
            if rect.isValid() and rect.width() > 0 and rect.height() > 0:
                self._canvas.fitInView(rect, Qt.KeepAspectRatio)

    def __init__(self, state: AppState, parent=None, window_number: int = 1) -> None:
        super().__init__(parent)
        self.window_number = window_number
        self._circuit_name: str = f"Circuit #{window_number}"
        self._refresh_window_title()
        self.resize(1450, 900)
        app = QApplication.instance()
        if app is not None:
            self.setWindowIcon(app.windowIcon())

        self._state = state
        self._document = CircuitDocument()

        # ── Undo (Ctrl+Z) infrastructure ─────────────────────────────────
        # Stack of previous CircuitDocument snapshots (most recent on top).
        self._undo_stack: list[dict] = []
        self._UNDO_LIMIT = 100
        # Snapshot of the document captured *after* the most recent change,
        # used as the previous-state to push onto the undo stack on the
        # next mutation.
        self._last_doc_snapshot: dict = self._document.to_dict()
        # When True, _emit_project_modified will not capture undo entries
        # (used while restoring a snapshot or applying a project state).
        self._suspend_undo_capture: bool = False

        self._file_palette = FilePaletteList()
        self._file_palette.setMinimumWidth(220)
        self._palette_filter = QLineEdit()
        self._palette_filter.setPlaceholderText("Filter blocks/files")
        self._palette_filter.textChanged.connect(self._apply_palette_filter)
        # Category dropdown filter — "All" plus the 5 category names.
        self._palette_category_filter = QComboBox()
        self._palette_category_filter.addItem("All categories", "")
        for category in _PALETTE_CATEGORIES:
            self._palette_category_filter.addItem(category, category)
        self._palette_category_filter.currentIndexChanged.connect(self._apply_palette_filter)

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
        self._scope_name_editor = QLineEdit()
        self._scope_name_editor.setPlaceholderText("Scope name")
        self._scope_name_editor.editingFinished.connect(self._on_scope_name_editing_finished)
        self._scope_name_editor.setEnabled(False)
        self._scope_name_label = QLabel("Scope name")
        self._updating_scope_name_editor = False
        self._selected_instance_id: str | None = None
        self._updating_impedance_editor = False

        self._export_button = QPushButton("Export equivalent Touchstone")
        self._export_button.clicked.connect(self._export_equivalent_touchstone)

        # --- Simulation mode selector ---
        self._sim_mode = QComboBox()
        self._sim_mode.addItems(["S-Parameters", "Channel Sim", "Transient"])
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
        self._drv_v_high.setToolTip(
            "High logic level (open-circuit Thevenin voltage):\n"
            "  • SE Driver: voltage on the single line.\n"
            "  • Diff Driver: differential voltage V_+ − V_− between the two "
            "legs (e.g. 0.4 V here means +0.2 V / −0.2 V per leg, i.e. ADS "
            "Tx_Diff with Vhigh=0.2, Vlow=−0.2)."
        )
        drv_layout.addRow("V high", self._drv_v_high)

        self._drv_v_low = QDoubleSpinBox()
        self._drv_v_low.setRange(-10.0, 10.0)
        self._drv_v_low.setDecimals(4)
        self._drv_v_low.setValue(-0.4)
        self._drv_v_low.setSuffix(" V")
        self._drv_v_low.setToolTip(
            "Low logic level (open-circuit Thevenin voltage):\n"
            "  • SE Driver: voltage on the single line.\n"
            "  • Diff Driver: differential voltage V_+ − V_− between the two "
            "legs (negative for a logic 0)."
        )
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

        # Random voltage noise σ (Vrn) at the receiver — enables BER
        # extrapolation of the Statistical Eye to the configured target BER.
        self._drv_random_noise_mv = QDoubleSpinBox()
        self._drv_random_noise_mv.setRange(0.0, 200.0)
        self._drv_random_noise_mv.setDecimals(3)
        self._drv_random_noise_mv.setValue(0.0)
        self._drv_random_noise_mv.setSuffix(" mV")
        self._drv_random_noise_mv.setToolTip(
            "Random voltage noise σ at the receiver. Set > 0 to extrapolate "
            "Eye Height/Width to the target BER (typ. 1e-12)."
        )
        drv_layout.addRow("Random noise σ (Vrn)", self._drv_random_noise_mv)

        # Driver Thevenin source impedance — controls the launch voltage:
        # V_pad = V_src · Z0/(Z_src+Z0). 50 Ω matches the ADS "matched bit
        # source" convention (factor ½). 0 Ω is an ideal voltage source.
        self._drv_source_impedance_ohm = QDoubleSpinBox()
        self._drv_source_impedance_ohm.setRange(0.0, 1000.0)
        self._drv_source_impedance_ohm.setDecimals(2)
        self._drv_source_impedance_ohm.setValue(0.0)
        self._drv_source_impedance_ohm.setSuffix(" Ω")
        self._drv_source_impedance_ohm.setToolTip(
            "Driver Thevenin source impedance, expressed in the same domain "
            "as the driver port reference impedance:\n"
            "  • SE Driver (Z0 = 50 Ω): typically 50 Ω for a matched source, "
            "0 Ω for an ideal voltage source.\n"
            "  • Diff Driver (Z0 = 100 Ω differential): typically 100 Ω for a "
            "back-terminated differential source (= 50 Ω per leg), 0 Ω for "
            "an ideal differential voltage source.\n"
            "0 Ω matches the ADS Bit Source with RSource = 0; the matched "
            "value (= driver Z0) reproduces the standard \"½ V_src incident "
            "wave\" convention."
        )
        drv_layout.addRow("Source impedance", self._drv_source_impedance_ohm)

        self._drv_bitrate = QDoubleSpinBox()
        self._drv_bitrate.setRange(0.001, 200.0)
        self._drv_bitrate.setDecimals(3)
        self._drv_bitrate.setValue(10.0)
        self._drv_bitrate.setSuffix(" Gbps")
        drv_layout.addRow("Bitrate", self._drv_bitrate)

        # When enabled, the bitstream contains exactly one full LFSR period
        # (2^N − 1 bits, where N is the order of the selected PRBS pattern).
        # Mirrors the ADS "Maximal Length LFSR" mode and overrides Num bits.
        self._drv_max_length_lfsr = QCheckBox("Maximal Length LFSR (repeat full period)")
        self._drv_max_length_lfsr.setToolTip(
            "When checked, the bitstream is built by tiling exactly one full "
            "LFSR period (2^N − 1 bits) of the selected PRBS-N up to 'Num "
            "bits'. Matches the ADS 'Maximal Length LFSR' mode while keeping "
            "enough samples for a clean eye diagram."
        )
        drv_layout.addRow("", self._drv_max_length_lfsr)

        # --- Substrate Settings (per-block, visible when a Substrate is selected) ---
        # A Substrate block is a 0-port physical token used by transmission-
        # line blocks (microstrip / stripline / …) to compute Z0, εeff and
        # losses. Its parameters do not contribute to MNA stamping; they are
        # consumed by line synthesis at simulation time.
        self._substrate_settings_group = QGroupBox("Substrate Settings")
        self._substrate_settings_group.setVisible(False)
        sub_layout = QFormLayout(self._substrate_settings_group)

        # Substrate name — this is the identifier transmission-line blocks
        # use to reference the substrate, so it must be unique within the
        # schematic. Uniqueness is enforced on commit by appending a numeric
        # suffix when needed.
        self._sub_name_edit = QLineEdit()
        self._sub_name_edit.setPlaceholderText("e.g. FR4_Top")
        self._sub_name_edit.setToolTip(
            "Substrate name. Transmission-line blocks reference the substrate "
            "by this name. Must be unique within the schematic."
        )
        self._sub_name_edit.editingFinished.connect(self._on_substrate_name_committed)
        sub_layout.addRow("Name", self._sub_name_edit)

        self._sub_epsilon_r = QDoubleSpinBox()
        self._sub_epsilon_r.setRange(1.0, 100.0)
        self._sub_epsilon_r.setDecimals(3)
        self._sub_epsilon_r.setValue(4.3)
        self._sub_epsilon_r.setToolTip("Relative dielectric constant εr.")
        self._sub_epsilon_r.valueChanged.connect(self._on_substrate_param_changed)
        sub_layout.addRow("εr", self._sub_epsilon_r)

        self._sub_loss_tangent = QDoubleSpinBox()
        self._sub_loss_tangent.setRange(0.0, 1.0)
        self._sub_loss_tangent.setDecimals(5)
        self._sub_loss_tangent.setValue(0.02)
        self._sub_loss_tangent.setToolTip("Dielectric loss tangent tan δ.")
        self._sub_loss_tangent.valueChanged.connect(self._on_substrate_param_changed)
        sub_layout.addRow("tan δ", self._sub_loss_tangent)

        self._sub_height_um = QDoubleSpinBox()
        self._sub_height_um.setRange(1.0, 100000.0)
        self._sub_height_um.setDecimals(2)
        self._sub_height_um.setValue(200.0)
        self._sub_height_um.setSuffix(" µm")
        self._sub_height_um.setToolTip("Dielectric thickness h between conductor and reference plane.")
        self._sub_height_um.valueChanged.connect(self._on_substrate_param_changed)
        sub_layout.addRow("Height (h)", self._sub_height_um)

        self._sub_thickness_um = QDoubleSpinBox()
        self._sub_thickness_um.setRange(0.1, 1000.0)
        self._sub_thickness_um.setDecimals(2)
        self._sub_thickness_um.setValue(35.0)
        self._sub_thickness_um.setSuffix(" µm")
        self._sub_thickness_um.setToolTip("Conductor (copper) thickness t.")
        self._sub_thickness_um.valueChanged.connect(self._on_substrate_param_changed)
        sub_layout.addRow("Conductor t", self._sub_thickness_um)

        self._sub_conductivity = QDoubleSpinBox()
        self._sub_conductivity.setRange(1e3, 1e9)
        self._sub_conductivity.setDecimals(0)
        self._sub_conductivity.setValue(5.8e7)
        self._sub_conductivity.setSuffix(" S/m")
        self._sub_conductivity.setToolTip("Conductor electrical conductivity σ (Cu = 5.8e7 S/m).")
        self._sub_conductivity.valueChanged.connect(self._on_substrate_param_changed)
        sub_layout.addRow("σ", self._sub_conductivity)

        self._sub_roughness_um = QDoubleSpinBox()
        self._sub_roughness_um.setRange(0.0, 50.0)
        self._sub_roughness_um.setDecimals(3)
        self._sub_roughness_um.setValue(0.0)
        self._sub_roughness_um.setSuffix(" µm")
        self._sub_roughness_um.setToolTip("Surface roughness Rq (RMS). 0 = smooth.")
        self._sub_roughness_um.valueChanged.connect(self._on_substrate_param_changed)
        sub_layout.addRow("Roughness Rq", self._sub_roughness_um)

        # Stripline geometry: distances from the conductor surfaces to the
        # top and bottom ground planes. These two editors replace the
        # microstrip "Height (h)" row when a stripline substrate is
        # selected, and together they fully describe the stack-up
        # asymmetry. Total dielectric thickness is h_top + t + h_bottom.
        self._sub_stripline_htop_um = QDoubleSpinBox()
        self._sub_stripline_htop_um.setRange(0.0, 50000.0)
        self._sub_stripline_htop_um.setDecimals(2)
        self._sub_stripline_htop_um.setValue(82.5)
        self._sub_stripline_htop_um.setSuffix(" µm")
        self._sub_stripline_htop_um.setToolTip(
            "Stripline: distance between the top of the embedded conductor "
            "and the top ground plane."
        )
        self._sub_stripline_htop_um.valueChanged.connect(self._on_substrate_param_changed)
        sub_layout.addRow("h_top (stripline)", self._sub_stripline_htop_um)

        self._sub_stripline_hbottom_um = QDoubleSpinBox()
        self._sub_stripline_hbottom_um.setRange(0.0, 50000.0)
        self._sub_stripline_hbottom_um.setDecimals(2)
        self._sub_stripline_hbottom_um.setValue(82.5)
        self._sub_stripline_hbottom_um.setSuffix(" µm")
        self._sub_stripline_hbottom_um.setToolTip(
            "Stripline: distance between the bottom of the embedded conductor "
            "and the bottom ground plane."
        )
        self._sub_stripline_hbottom_um.valueChanged.connect(self._on_substrate_param_changed)
        sub_layout.addRow("h_bottom (stripline)", self._sub_stripline_hbottom_um)
        # Track the form rows so we can switch h ↔ (h_top, h_bottom)
        # depending on the selected substrate kind.
        self._sub_form_layout = sub_layout

        self._updating_substrate_controls = False

        # --- Transmission Line Settings -------------------------------
        # Per-instance geometry of a transmission-line block. The block
        # references a substrate by name (combobox is filtered by the
        # required substrate kind: microstrip vs stripline) and exposes
        # W / L / S editors.
        self._tline_settings_group = QGroupBox("Transmission Line Settings")
        self._tline_settings_group.setVisible(False)
        tline_layout = QFormLayout(self._tline_settings_group)

        self._tline_substrate_combo = QComboBox()
        self._tline_substrate_combo.setToolTip(
            "Substrate stack-up referenced by this line. Only substrates of "
            "the matching kind (microstrip ↔ Substrate, stripline ↔ Stripline "
            "Substrate) are listed."
        )
        self._tline_substrate_combo.currentIndexChanged.connect(
            self._on_tline_param_changed
        )
        tline_layout.addRow("Substrate", self._tline_substrate_combo)

        self._tline_width_um = QDoubleSpinBox()
        self._tline_width_um.setRange(1.0, 50000.0)
        self._tline_width_um.setDecimals(2)
        self._tline_width_um.setValue(200.0)
        self._tline_width_um.setSuffix(" µm")
        self._tline_width_um.setToolTip("Conductor width W (per line for coupled).")
        self._tline_width_um.valueChanged.connect(self._on_tline_param_changed)
        tline_layout.addRow("Width (W)", self._tline_width_um)

        self._tline_length_mm = QDoubleSpinBox()
        self._tline_length_mm.setRange(0.001, 100000.0)
        self._tline_length_mm.setDecimals(3)
        self._tline_length_mm.setValue(10.0)
        self._tline_length_mm.setSuffix(" mm")
        self._tline_length_mm.setToolTip("Physical length L of the line.")
        self._tline_length_mm.valueChanged.connect(self._on_tline_param_changed)
        tline_layout.addRow("Length (L)", self._tline_length_mm)

        self._tline_spacing_um = QDoubleSpinBox()
        self._tline_spacing_um.setRange(1.0, 50000.0)
        self._tline_spacing_um.setDecimals(2)
        self._tline_spacing_um.setValue(200.0)
        self._tline_spacing_um.setSuffix(" µm")
        self._tline_spacing_um.setToolTip(
            "Edge-to-edge spacing S between the two conductors (coupled lines only)."
        )
        self._tline_spacing_um.valueChanged.connect(self._on_tline_param_changed)
        tline_layout.addRow("Spacing (S)", self._tline_spacing_um)

        self._tline_z0_ref = QDoubleSpinBox()
        self._tline_z0_ref.setRange(1.0, 1000.0)
        self._tline_z0_ref.setDecimals(2)
        self._tline_z0_ref.setValue(50.0)
        self._tline_z0_ref.setSuffix(" Ω")
        self._tline_z0_ref.setToolTip(
            "Per-port reference impedance used to render the line as an "
            "S-matrix (50 Ω is the convention for both single-ended and "
            "coupled lines; combine with a 100 Ω differential port for "
            "differential signalling)."
        )
        self._tline_z0_ref.valueChanged.connect(self._on_tline_param_changed)
        tline_layout.addRow("Z₀ ref", self._tline_z0_ref)

        self._tline_width_end_um = QDoubleSpinBox()
        self._tline_width_end_um.setRange(1.0, 50000.0)
        self._tline_width_end_um.setDecimals(2)
        self._tline_width_end_um.setValue(400.0)
        self._tline_width_end_um.setSuffix(" µm")
        self._tline_width_end_um.setToolTip(
            "Final conductor width W_end (Taper only)."
        )
        self._tline_width_end_um.valueChanged.connect(self._on_tline_param_changed)
        tline_layout.addRow("End Width (W_end)", self._tline_width_end_um)

        self._tline_taper_profile = QComboBox()
        self._tline_taper_profile.addItems(["linear", "exponential", "klopfenstein"])
        self._tline_taper_profile.setToolTip(
            "Width profile along the taper length."
        )
        self._tline_taper_profile.currentIndexChanged.connect(
            self._on_tline_param_changed
        )
        tline_layout.addRow("Profile", self._tline_taper_profile)

        self._tline_cpw_slot_um = QDoubleSpinBox()
        self._tline_cpw_slot_um.setRange(1.0, 50000.0)
        self._tline_cpw_slot_um.setDecimals(2)
        self._tline_cpw_slot_um.setValue(150.0)
        self._tline_cpw_slot_um.setSuffix(" µm")
        self._tline_cpw_slot_um.setToolTip(
            "CPW slot width S between center conductor and ground."
        )
        self._tline_cpw_slot_um.valueChanged.connect(self._on_tline_param_changed)
        tline_layout.addRow("Slot (S_cpw)", self._tline_cpw_slot_um)

        self._tline_form_layout = tline_layout
        self._updating_tline_controls = False

        # --- Component Settings (Attenuator / Circulator / Coupler) ---
        self._component_settings_group = QGroupBox("Component Settings")
        self._component_settings_group.setVisible(False)
        comp_layout = QFormLayout(self._component_settings_group)

        self._comp_attn_db = QDoubleSpinBox()
        self._comp_attn_db.setRange(0.0, 100.0)
        self._comp_attn_db.setDecimals(3)
        self._comp_attn_db.setValue(6.0)
        self._comp_attn_db.setSuffix(" dB")
        self._comp_attn_db.setToolTip("Total attenuation.")
        self._comp_attn_db.valueChanged.connect(self._on_component_param_changed)
        comp_layout.addRow("Attenuation", self._comp_attn_db)

        self._comp_kind_combo = QComboBox()
        self._comp_kind_combo.addItems(["branch_line_90", "rat_race_180", "directional"])
        self._comp_kind_combo.currentIndexChanged.connect(self._on_component_param_changed)
        comp_layout.addRow("Coupler Kind", self._comp_kind_combo)

        self._comp_coupling_db = QDoubleSpinBox()
        self._comp_coupling_db.setRange(0.1, 60.0)
        self._comp_coupling_db.setDecimals(3)
        self._comp_coupling_db.setValue(3.0)
        self._comp_coupling_db.setSuffix(" dB")
        self._comp_coupling_db.valueChanged.connect(self._on_component_param_changed)
        comp_layout.addRow("Coupling", self._comp_coupling_db)

        self._comp_il_db = QDoubleSpinBox()
        self._comp_il_db.setRange(0.0, 10.0)
        self._comp_il_db.setDecimals(3)
        self._comp_il_db.setValue(0.3)
        self._comp_il_db.setSuffix(" dB")
        self._comp_il_db.valueChanged.connect(self._on_component_param_changed)
        comp_layout.addRow("Insertion Loss", self._comp_il_db)

        self._comp_iso_db = QDoubleSpinBox()
        self._comp_iso_db.setRange(5.0, 100.0)
        self._comp_iso_db.setDecimals(3)
        self._comp_iso_db.setValue(30.0)
        self._comp_iso_db.setSuffix(" dB")
        self._comp_iso_db.valueChanged.connect(self._on_component_param_changed)
        comp_layout.addRow("Isolation", self._comp_iso_db)

        self._comp_rl_db = QDoubleSpinBox()
        self._comp_rl_db.setRange(5.0, 60.0)
        self._comp_rl_db.setDecimals(3)
        self._comp_rl_db.setValue(25.0)
        self._comp_rl_db.setSuffix(" dB")
        self._comp_rl_db.valueChanged.connect(self._on_component_param_changed)
        comp_layout.addRow("Return Loss", self._comp_rl_db)

        self._comp_dir_combo = QComboBox()
        self._comp_dir_combo.addItems(["cw", "ccw"])
        self._comp_dir_combo.currentIndexChanged.connect(self._on_component_param_changed)
        comp_layout.addRow("Direction", self._comp_dir_combo)

        self._comp_z0_ref = QDoubleSpinBox()
        self._comp_z0_ref.setRange(1.0, 1000.0)
        self._comp_z0_ref.setDecimals(2)
        self._comp_z0_ref.setValue(50.0)
        self._comp_z0_ref.setSuffix(" Ω")
        self._comp_z0_ref.valueChanged.connect(self._on_component_param_changed)
        comp_layout.addRow("Z₀ ref", self._comp_z0_ref)

        self._component_form_layout = comp_layout
        self._updating_component_controls = False

        # --- Channel Sim Settings (separate group from Driver Settings) ---
        # These parameters belong to the channel/eye analysis itself, not to
        # any specific driver block, so they live in their own group whose
        # visibility is tied to the simulation mode rather than to a block
        # selection.
        self._channel_sim_settings_group = QGroupBox("Channel Sim Settings")
        self._channel_sim_settings_group.setVisible(False)
        chsim_layout = QFormLayout(self._channel_sim_settings_group)

        self._drv_prbs = QComboBox()
        self._drv_prbs.addItems(PRBS_CHOICES)
        self._drv_prbs.setCurrentText("PRBS-8")
        self._drv_prbs.currentIndexChanged.connect(self._emit_project_modified)
        chsim_layout.addRow("PRBS pattern", self._drv_prbs)

        self._drv_encoding = QComboBox()
        self._drv_encoding.addItems(ENCODING_CHOICES)
        self._drv_encoding.setCurrentText("8b10b")
        self._drv_encoding.currentIndexChanged.connect(self._emit_project_modified)
        chsim_layout.addRow("Encoding", self._drv_encoding)

        self._drv_num_bits = QSpinBox()
        self._drv_num_bits.setRange(128, 2**20)
        self._drv_num_bits.setValue(2**13)
        self._drv_num_bits.setSingleStep(1024)
        self._drv_num_bits.valueChanged.connect(self._emit_project_modified)
        chsim_layout.addRow("Num bits", self._drv_num_bits)

        self._drv_output_port_instance = QComboBox()
        chsim_layout.addRow("Output port", self._drv_output_port_instance)

        self._transient_settings_group = QGroupBox("Transient Settings")
        self._transient_settings_group.setVisible(False)
        transient_layout = QFormLayout(self._transient_settings_group)

        self._updating_transient_controls = False
        self._transient_source_instance = QComboBox()
        self._transient_source_instance.currentIndexChanged.connect(self._on_transient_source_selection_changed)
        transient_layout.addRow("Source", self._transient_source_instance)

        self._transient_amplitude = QDoubleSpinBox()
        self._transient_amplitude.setRange(0.0, 10.0)
        self._transient_amplitude.setDecimals(4)
        self._transient_amplitude.setValue(0.4)
        self._transient_amplitude.setSuffix(" V")
        self._transient_amplitude.valueChanged.connect(self._on_transient_controls_changed)
        transient_layout.addRow("Amplitude", self._transient_amplitude)

        self._transient_polarity = QComboBox()
        self._transient_polarity.addItems(TRANSIENT_POLARITY_CHOICES)
        self._transient_polarity.currentTextChanged.connect(self._on_transient_controls_changed)
        transient_layout.addRow("Polarity", self._transient_polarity)

        self._transient_rise_time = QDoubleSpinBox()
        self._transient_rise_time.setRange(0.0, 10000.0)
        self._transient_rise_time.setDecimals(2)
        self._transient_rise_time.setValue(35.0)
        self._transient_rise_time.setSuffix(" ps")
        self._transient_rise_time.valueChanged.connect(self._on_transient_controls_changed)
        self._transient_rise_time_label = QLabel("Rise time")
        transient_layout.addRow(self._transient_rise_time_label, self._transient_rise_time)

        self._transient_fall_time = QDoubleSpinBox()
        self._transient_fall_time.setRange(0.0, 10000.0)
        self._transient_fall_time.setDecimals(2)
        self._transient_fall_time.setValue(35.0)
        self._transient_fall_time.setSuffix(" ps")
        self._transient_fall_time.valueChanged.connect(self._on_transient_controls_changed)
        self._transient_fall_time_label = QLabel("Fall time")
        transient_layout.addRow(self._transient_fall_time_label, self._transient_fall_time)

        self._transient_delay = QDoubleSpinBox()
        self._transient_delay.setRange(0.0, 1e9)
        self._transient_delay.setDecimals(3)
        self._transient_delay.setValue(0.0)
        self._transient_delay.setSuffix(" ps")
        self._transient_delay.valueChanged.connect(self._on_transient_controls_changed)
        transient_layout.addRow("Delay", self._transient_delay)

        self._transient_pulse_width = QDoubleSpinBox()
        self._transient_pulse_width.setRange(0.0, 1e9)
        self._transient_pulse_width.setDecimals(3)
        self._transient_pulse_width.setValue(250.0)
        self._transient_pulse_width.setSuffix(" ps")
        self._transient_pulse_width.valueChanged.connect(self._on_transient_controls_changed)
        self._transient_pulse_width_label = QLabel("Pulse width")
        transient_layout.addRow(self._transient_pulse_width_label, self._transient_pulse_width)

        self._transient_stop_time = QDoubleSpinBox()
        self._transient_stop_time.setRange(0.001, 1e9)
        self._transient_stop_time.setDecimals(3)
        self._transient_stop_time.setValue(5.0)
        self._transient_stop_time.setSuffix(" ns")
        self._transient_stop_time.valueChanged.connect(self._emit_project_modified)
        transient_layout.addRow("Stop time", self._transient_stop_time)

        self._transient_output_list = QListWidget()
        self._transient_output_list.setSelectionMode(QListWidget.NoSelection)
        self._transient_output_list.setMinimumHeight(90)
        self._transient_output_list.setEnabled(False)
        transient_layout.addRow("Scopes", self._transient_output_list)

        self._transient_sim_button = QPushButton("Run Transient Simulation")
        self._transient_sim_button.setVisible(False)
        self._transient_sim_button.clicked.connect(self._run_transient_simulation)

        self._circuit_name_edit = QLineEdit(self._circuit_name)
        self._circuit_name_edit.textChanged.connect(self._on_circuit_name_changed)

        self._drv_eye_span = QComboBox()
        for span_ui in EYE_SPAN_CHOICES:
            self._drv_eye_span.addItem(f"{span_ui} UI", span_ui)
        default_eye_span_index = self._drv_eye_span.findData(DEFAULT_EYE_SPAN_UI)
        if default_eye_span_index >= 0:
            self._drv_eye_span.setCurrentIndex(default_eye_span_index)
        self._drv_eye_span.currentIndexChanged.connect(self._emit_project_modified)
        chsim_layout.addRow("Eye span", self._drv_eye_span)

        self._drv_eye_render_mode = QComboBox()
        for mode in RENDER_MODE_CHOICES:
            self._drv_eye_render_mode.addItem(mode, mode)
        default_render_mode_index = self._drv_eye_render_mode.findData(DEFAULT_RENDER_MODE)
        if default_render_mode_index >= 0:
            self._drv_eye_render_mode.setCurrentIndex(default_render_mode_index)
        self._drv_eye_render_mode.currentIndexChanged.connect(self._emit_project_modified)
        chsim_layout.addRow("Eye render", self._drv_eye_render_mode)

        self._drv_eye_quality_preset = QComboBox()
        for preset in QUALITY_PRESET_CHOICES:
            self._drv_eye_quality_preset.addItem(preset, preset)
        default_quality_index = self._drv_eye_quality_preset.findData(DEFAULT_QUALITY_PRESET)
        if default_quality_index >= 0:
            self._drv_eye_quality_preset.setCurrentIndex(default_quality_index)
        self._drv_eye_quality_preset.currentIndexChanged.connect(self._emit_project_modified)
        chsim_layout.addRow("Eye quality", self._drv_eye_quality_preset)

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
        self._transient_windows: list = []
        self._eye_run_counter: int = 0

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(280)
        # Force a light look-and-feel for the palette panel and its child
        # widgets (label, dropdown, filter) so they remain readable when the
        # OS is in dark mode (otherwise the header label and category combo
        # blend into the panel background).
        left_panel.setStyleSheet(
            "QFrame { background-color: #f7f7f7; border: 1px solid #d1d5db; }"
            " QLabel { color: #0f172a; background: transparent; border: none; }"
            " QComboBox, QLineEdit {"
            "   color: #0f172a;"
            "   background-color: #ffffff;"
            "   border: 1px solid #94a3b8;"
            "   border-radius: 3px;"
            "   padding: 2px 4px;"
            " }"
            " QComboBox QAbstractItemView {"
            "   color: #0f172a;"
            "   background-color: #ffffff;"
            "   selection-background-color: #dbeafe;"
            "   selection-color: #0f172a;"
            " }"
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        blocks_header = QLabel("Blocks")
        blocks_font = blocks_header.font()
        blocks_font.setBold(True)
        blocks_header.setFont(blocks_font)
        left_layout.addWidget(blocks_header)
        category_label = QLabel("Category")
        left_layout.addWidget(category_label)
        left_layout.addWidget(self._palette_category_filter)
        left_layout.addWidget(self._palette_filter)
        left_layout.addWidget(self._file_palette)

        inspector = QFrame()
        inspector.setFrameShape(QFrame.StyledPanel)
        inspector.setMinimumWidth(280)
        inspector.setMaximumWidth(360)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(8, 8, 8, 8)
        inspector_layout.addWidget(QLabel("Editor Settings"))

        sim_form = QFormLayout()
        sim_form.addRow("Circuit name", self._circuit_name_edit)
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
        impedance_form.addRow(self._scope_name_label, self._scope_name_editor)
        self._selected_block_form = impedance_form
        self._selected_block_group = QGroupBox("Settings")
        self._selected_block_group.setLayout(impedance_form)
        # Hidden until a block is selected.
        self._selected_block_group.setVisible(False)
        inspector_layout.addWidget(self._selected_block_group)

        inspector_layout.addWidget(self._channel_sim_settings_group)
        inspector_layout.addWidget(self._driver_settings_group)
        inspector_layout.addWidget(self._substrate_settings_group)
        inspector_layout.addWidget(self._tline_settings_group)
        inspector_layout.addWidget(self._component_settings_group)
        inspector_layout.addWidget(self._stat_group)
        inspector_layout.addWidget(self._channel_sim_button)
        inspector_layout.addWidget(self._transient_settings_group)
        inspector_layout.addWidget(self._transient_sim_button)
        inspector_layout.addWidget(self._status_label)
        inspector_layout.addStretch(1)
        # Export button is anchored to the bottom of the inspector panel.
        inspector_layout.addWidget(self._export_button)

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
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self._splitter = splitter
        self.setCentralWidget(splitter)

        # Ctrl+Z → undo last circuit edit (block move, add, delete, wiring,
        # impedance/label changes, …). Implemented as a snapshot stack on
        # the CircuitDocument; non-document changes are ignored.
        undo_shortcut = QShortcut(QKeySequence.Undo, self)
        undo_shortcut.setContext(Qt.WindowShortcut)
        undo_shortcut.activated.connect(self._undo_last_change)

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
            item.setData(Qt.UserRole + 1, block["label"].lower())
            # Stash category for the dropdown filter.
            item.setData(Qt.UserRole + 2, _PALETTE_CATEGORY_BY_KIND.get(block["block_kind"], ""))
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
            item.setData(Qt.UserRole + 1, loaded.display_name.lower())
            item.setData(Qt.UserRole + 2, "SP Blocks")
            preview = BlockPreviewWidget(
                loaded.display_name,
                loaded.data.nports,
                block_kind="touchstone",
                impedance_ohm=loaded.data.options.reference_resistance,
            )
            item.setSizeHint(preview.sizeHint())
            self._file_palette.addItem(item)
            self._file_palette.setItemWidget(item, preview)
        self._apply_palette_filter()
        self._refresh_validation_state()
        self._fit_circuit_in_view()

    def _apply_palette_filter(self) -> None:
        needle = self._palette_filter.text().strip().lower()
        category = self._palette_category_filter.currentData() or ""
        for row in range(self._file_palette.count()):
            item = self._file_palette.item(row)
            haystack = str(item.data(Qt.UserRole + 1) or "")
            item_category = str(item.data(Qt.UserRole + 2) or "")
            text_match = (not needle) or (needle in haystack)
            category_match = (not category) or (item_category == category)
            item.setHidden(not (text_match and category_match))

    def references_file(self, file_id: str) -> bool:
        return self._document.uses_file(file_id)

    def report_touchstone_file_names(self) -> list[str]:
        """Return unique Touchstone file names used by the current circuit."""
        names: list[str] = []
        seen: set[str] = set()
        for instance in self._document.instances:
            if instance.block_kind != "touchstone":
                continue
            loaded = self._state.get_file(instance.source_file_id)
            name = loaded.display_name if loaded is not None else instance.display_label
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def circuit_display_name(self) -> str:
        return self._circuit_name.strip() or f"Circuit #{self.window_number}"

    def _refresh_window_title(self) -> None:
        self.setWindowTitle(self.circuit_display_name())

    def _eye_window_title(self, run_index: int | None = None) -> str:
        if run_index is None or run_index <= 1:
            return self.circuit_display_name()
        return f"{self.circuit_display_name()} #{run_index}"

    def _transient_window_title(self) -> str:
        return self.circuit_display_name()

    def _sync_eye_window_titles(self) -> None:
        for win in self.get_open_eye_windows():
            run_index = win.property("eye_run_index")
            try:
                idx = int(run_index) if run_index is not None else None
            except (TypeError, ValueError):
                idx = None
            win.setWindowTitle(self._eye_window_title(idx))

    def _sync_transient_window_titles(self) -> None:
        for win in self.get_open_transient_windows():
            win.setWindowTitle(self._transient_window_title())

    def _prompt_export_file_name(self, nports: int) -> str | None:
        default_name = f"{self.circuit_display_name()}.s{nports}p"
        file_name, ok = QInputDialog.getText(
            self,
            "Save equivalent Touchstone",
            "File name:",
            QLineEdit.Normal,
            default_name,
        )
        if not ok:
            return None

        value = str(file_name).strip()
        if not value:
            QMessageBox.warning(self, "Export failed", "Enter a file name to continue.")
            return None
        return value

    def _host_window_manager(self):
        # Direct stash set by ChildWindowManager.present() — most reliable since
        # CategoryWindows are parent-less top-levels (so they appear in the OS
        # taskbar) and the parentWidget() chain doesn't reach the main window.
        host = getattr(self, "_host_main_window", None)
        if host is not None and hasattr(host, "child_window_manager"):
            try:
                return host.child_window_manager()
            except Exception:
                pass
        parent = self.parentWidget()
        while parent is not None:
            if hasattr(parent, "child_window_manager"):
                try:
                    return parent.child_window_manager()
                except Exception:
                    pass
            parent = parent.parentWidget() if hasattr(parent, "parentWidget") else None
        for tlw in QApplication.topLevelWidgets():
            if hasattr(tlw, "child_window_manager"):
                try:
                    return tlw.child_window_manager()
                except Exception:
                    pass
        return None

    def _present_eye_window(self, eye_window: EyeDiagramWindow) -> EyeDiagramWindow:
        eye_window.span_changed.connect(self._on_eye_span_window_changed)
        eye_window.render_mode_changed.connect(self._on_eye_render_mode_window_changed)
        eye_window.quality_preset_changed.connect(self._on_eye_quality_preset_window_changed)
        eye_window.setAttribute(Qt.WA_DeleteOnClose)
        eye_window.destroyed.connect(
            lambda *_: self._eye_windows.remove(eye_window) if eye_window in self._eye_windows else None
        )

        mgr = self._host_window_manager()
        if mgr is not None:
            mgr.present(eye_window)
            mgr.bring_to_front(eye_window)
        else:
            eye_window.show()
            eye_window.raise_()
            eye_window.activateWindow()

        QApplication.processEvents()
        self._eye_windows.append(eye_window)
        return eye_window

    def _present_transient_window(self, transient_window: TransientResultWindow) -> TransientResultWindow:
        transient_window.setAttribute(Qt.WA_DeleteOnClose)
        transient_window.destroyed.connect(
            lambda *_: (
                self._transient_windows.remove(transient_window)
                if transient_window in self._transient_windows
                else None,
                self.transient_windows_changed.emit(),
            )
        )

        mgr = self._host_window_manager()
        if mgr is not None:
            mgr.present(transient_window)
            mgr.bring_to_front(transient_window)
        else:
            transient_window.show()
            transient_window.raise_()
            transient_window.activateWindow()

        QApplication.processEvents()
        self._transient_windows.append(transient_window)
        self.transient_windows_changed.emit()
        return transient_window

    def _open_eye_window(
        self,
        result: ChannelSimResult,
        *,
        title: str | None = None,
        initial_span_ui: int | None = None,
        initial_render_mode: str | None = None,
        initial_quality_preset: str | None = None,
        statistical_enabled: bool | None = None,
        noise_rms_mv: float | None = None,
        jitter_rms_ps: float | None = None,
        progress_callback=None,
    ) -> EyeDiagramWindow:
        eye_window = EyeDiagramWindow(
            result,
            title=title or self._eye_window_title(),
            initial_span_ui=self._selected_eye_span_ui() if initial_span_ui is None else initial_span_ui,
            initial_render_mode=self._selected_eye_render_mode() if initial_render_mode is None else initial_render_mode,
            initial_quality_preset=self._selected_eye_quality_preset() if initial_quality_preset is None else initial_quality_preset,
            statistical_enabled=self._stat_enabled.isChecked() if statistical_enabled is None else statistical_enabled,
            noise_rms_mv=self._stat_noise.value() if noise_rms_mv is None else noise_rms_mv,
            jitter_rms_ps=self._stat_jitter.value() if jitter_rms_ps is None else jitter_rms_ps,
            progress_callback=progress_callback,
        )
        self._eye_run_counter += 1
        eye_window.setProperty("eye_run_index", self._eye_run_counter)
        if title is None:
            eye_window.setWindowTitle(self._eye_window_title(self._eye_run_counter))
        return self._present_eye_window(eye_window)

    def _open_transient_window(
        self,
        result: TransientSimResult,
        *,
        title: str | None = None,
    ) -> TransientResultWindow:
        # Reuse an already open transient window so successive runs refresh the
        # same plot rather than spawning new ones.
        existing = next(iter(self.get_open_transient_windows()), None)
        if existing is not None:
            existing.update_result(result)
            existing.setWindowTitle(title or self._transient_window_title())
            mgr = self._host_window_manager()
            if mgr is not None:
                mgr.bring_to_front(existing)
            else:
                existing.raise_()
                existing.activateWindow()
            QApplication.processEvents()
            self.transient_windows_changed.emit()
            return existing
        transient_window = TransientResultWindow(
            result,
            title=title or self._transient_window_title(),
        )
        return self._present_transient_window(transient_window)

    def _on_circuit_name_changed(self, value: str) -> None:
        self._circuit_name = value.strip() or f"Circuit #{self.window_number}"
        self._refresh_window_title()
        self._sync_eye_window_titles()
        self._sync_transient_window_titles()
        self._emit_project_modified()

    def get_open_eye_windows(self) -> list[EyeDiagramWindow]:
        """Return currently alive eye diagram windows opened from this circuit."""
        alive: list[EyeDiagramWindow] = []
        for win in self._eye_windows:
            try:
                _ = win.windowTitle()
            except RuntimeError:
                continue
            alive.append(win)
        self._eye_windows = alive
        return list(alive)

    def get_open_transient_windows(self) -> list[TransientResultWindow]:
        alive: list[TransientResultWindow] = []
        for win in self._transient_windows:
            try:
                _ = win.windowTitle()
            except RuntimeError:
                continue
            alive.append(win)
        self._transient_windows = alive
        return list(alive)

    def generate_eye_window_for_report(self) -> tuple[EyeDiagramWindow | None, str | None]:
        """Generate an eye window for reports without interactive dialogs."""
        driver_inst = None
        for inst in self._document.instances:
            if inst.block_kind in {"driver_se", "driver_diff"}:
                driver_inst = inst
                break
        if driver_inst is None:
            return None, "No driver block found in circuit."

        spec = DriverSpec(
            voltage_high_v=self._drv_v_high.value(),
            voltage_low_v=self._drv_v_low.value(),
            rise_time_s=self._drv_rise_time.value() * 1e-12,
            fall_time_s=self._drv_fall_time.value() * 1e-12,
            bitrate_gbps=self._drv_bitrate.value(),
            prbs_pattern=self._drv_prbs.currentText(),
            encoding=self._drv_encoding.currentText(),
            num_bits=self._drv_num_bits.value(),
            random_noise_v=self._drv_random_noise_mv.value() * 1e-3,
            source_impedance_ohm=self._drv_source_impedance_ohm.value(),
            maximal_length_lfsr=self._drv_max_length_lfsr.isChecked(),
        )
        self._document.update_instance_driver_spec(driver_inst.instance_id, spec)

        self._refresh_output_port_list()
        out_idx = self._drv_output_port_instance.currentIndex()
        if out_idx < 0 and self._drv_output_port_instance.count() > 0:
            out_idx = 0
            self._drv_output_port_instance.setCurrentIndex(0)
        if out_idx < 0:
            return None, "No output port selected."
        out_instance_id = self._drv_output_port_instance.itemData(out_idx)

        try:
            self._status_label.setText("Running channel simulation for report...")
            result = simulate_channel(
                self._document,
                self._state,
                driver_instance_id=driver_inst.instance_id,
                output_port_instance_id=out_instance_id,
            )
        except Exception as exc:
            return None, f"Channel simulation failed: {exc}"

        win = self._open_eye_window(result)
        return win, None

    def generate_eye_snapshot_for_project(self) -> tuple[QPixmap | None, str | None]:
        """Return eye snapshot pixmap for project save; auto-simulate if needed."""
        open_eyes = self.get_open_eye_windows()
        if open_eyes:
            return open_eyes[0].grab_eye_plot_pixmap(), None

        result, err = self.generate_eye_result_for_project()
        if result is None:
            return None, err

        eye_window = EyeDiagramWindow(
            result,
            title=self._eye_window_title(),
            parent=None,
            initial_span_ui=self._selected_eye_span_ui(),
            initial_render_mode=self._selected_eye_render_mode(),
            initial_quality_preset=self._selected_eye_quality_preset(),
            statistical_enabled=self._stat_enabled.isChecked(),
            noise_rms_mv=self._stat_noise.value(),
            jitter_rms_ps=self._stat_jitter.value(),
        )
        pixmap = eye_window.grab_eye_plot_pixmap()
        eye_window.close()
        eye_window.deleteLater()
        return pixmap, None

    def generate_eye_result_for_project(self) -> tuple[ChannelSimResult | None, str | None]:
        """Run channel simulation and return the raw result for project-side exports."""

        driver_inst = None
        for inst in self._document.instances:
            if inst.block_kind in {"driver_se", "driver_diff"}:
                driver_inst = inst
                break
        if driver_inst is None:
            return None, "No driver block found in circuit."

        spec = DriverSpec(
            voltage_high_v=self._drv_v_high.value(),
            voltage_low_v=self._drv_v_low.value(),
            rise_time_s=self._drv_rise_time.value() * 1e-12,
            fall_time_s=self._drv_fall_time.value() * 1e-12,
            bitrate_gbps=self._drv_bitrate.value(),
            prbs_pattern=self._drv_prbs.currentText(),
            encoding=self._drv_encoding.currentText(),
            num_bits=self._drv_num_bits.value(),
            random_noise_v=self._drv_random_noise_mv.value() * 1e-3,
            source_impedance_ohm=self._drv_source_impedance_ohm.value(),
            maximal_length_lfsr=self._drv_max_length_lfsr.isChecked(),
        )
        self._document.update_instance_driver_spec(driver_inst.instance_id, spec)

        self._refresh_output_port_list()
        out_idx = self._drv_output_port_instance.currentIndex()
        if out_idx < 0 and self._drv_output_port_instance.count() > 0:
            out_idx = 0
            self._drv_output_port_instance.setCurrentIndex(0)
        if out_idx < 0:
            return None, "No output port selected."
        out_instance_id = self._drv_output_port_instance.itemData(out_idx)

        try:
            result = simulate_channel(
                self._document,
                self._state,
                driver_instance_id=driver_inst.instance_id,
                output_port_instance_id=out_instance_id,
            )
        except Exception as exc:
            return None, f"Channel simulation failed: {exc}"
        return result, None

    def export_equivalent_touchstone_for_project(self) -> tuple[str | None, str | None, str | None]:
        """Return filename and Touchstone content for project-side export."""
        try:
            result = solve_circuit_network(self._document, self._state)
            text = to_touchstone_string_with_format(
                result,
                data_format="DB",
                frequency_unit=self._document.sweep.display_unit,
            )
        except Exception as exc:
            return None, None, str(exc)

        safe_name = "".join(
            ch if (ch.isalnum() or ch in {"-", "_"}) else "_"
            for ch in self.circuit_display_name()
        ).strip("_")
        if not safe_name:
            safe_name = f"Circuit_{self.window_number}"
        file_name = f"{safe_name}.s{result.nports}p"
        return file_name, text, None

    def grab_circuit_snapshot_pixmap(self):
        """Return a full-scene snapshot of the circuit for reports."""
        bounds = self._scene.itemsBoundingRect()
        if bounds.isNull() or bounds.width() <= 0.0 or bounds.height() <= 0.0:
            return self._canvas.grab()

        pad = 40.0
        source_rect = bounds.adjusted(-pad, -pad, pad, pad)
        max_w = 2200.0
        max_h = 1400.0
        scale = min(max_w / source_rect.width(), max_h / source_rect.height())
        scale = max(scale, 0.10)
        target_w = max(1, int(round(source_rect.width() * scale)))
        target_h = max(1, int(round(source_rect.height() * scale)))

        pixmap = QPixmap(target_w, target_h)
        pixmap.fill(QColor("#f7f7f7"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._scene.render(
            painter,
            QRectF(0.0, 0.0, float(target_w), float(target_h)),
            source_rect,
        )
        painter.end()
        return pixmap

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

        if block_kind in _SCOPE_PROBE_KINDS:
            display_label = self._next_scope_default_name()

        driver_spec = None
        transient_source_spec = None
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
                random_noise_v=self._drv_random_noise_mv.value() * 1e-3,
                source_impedance_ohm=self._drv_source_impedance_ohm.value(),
                maximal_length_lfsr=self._drv_max_length_lfsr.isChecked(),
            )
        if block_kind in _TRANSIENT_SOURCE_KINDS:
            transient_source_spec = self._transient_source_spec_from_controls()

        substrate_spec = None
        if block_kind in _SUBSTRATE_KINDS:
            substrate_spec = SubstrateSpec()
            base_name = "Substrate" if block_kind == "substrate" else "Stripline_Substrate"
            display_label = self._unique_substrate_name(base_name)
            nports = 0

        transmission_line_spec = None
        if block_kind in _TLINE_KINDS:
            # A transmission line must reference an existing substrate of
            # the matching kind. Open a picker dialog (or auto-select if
            # only one is available); abort the drop if none exist.
            tline_spec = self._prompt_tline_substrate(block_kind)
            if tline_spec is None:
                return
            transmission_line_spec = tline_spec
            display_label = self._next_tline_default_name(block_kind)
            nports = 4 if block_kind in _TLINE_COUPLED_KINDS else 2

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
            transient_source_spec=transient_source_spec,
            substrate_spec=substrate_spec,
            transmission_line_spec=transmission_line_spec,
        )
        block_item = CircuitBlockItem(self._scene, instance)
        self._scene.register_block(block_item)
        self._scene.rebuild_export_state(self._document)
        if block_kind in _TRANSIENT_SOURCE_KINDS:
            self._refresh_transient_source_list(select_instance_id=instance.instance_id)
        elif block_kind in {"driver_se", "driver_diff"}:
            self._refresh_transient_source_list()
        if block_kind in _SCOPE_PROBE_KINDS:
            self._refresh_transient_output_list()
        if block_kind == "touchstone":
            self._auto_clamp_sweep_to_touchstone_blocks()
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
            self._updating_scope_name_editor = True
            self._scope_name_editor.setEnabled(False)
            self._scope_name_editor.clear()
            self._updating_scope_name_editor = False
            # No block selected → hide the per-element parameters group.
            self._selected_block_group.setVisible(False)
            # No driver selected → hide the Driver Settings group too.
            self._driver_settings_group.setVisible(False)
            self._substrate_settings_group.setVisible(False)
            self._tline_settings_group.setVisible(False)
            self._component_settings_group.setVisible(False)
            return
        self._selected_instance_id = block_item.instance.instance_id
        kind = block_item.instance.block_kind
        # Show the per-element parameters group only when at least one of
        # its rows applies to the selected block kind. For blocks like
        # substrates and transmission lines that have their own dedicated
        # settings group, the generic "Settings" rectangle would otherwise
        # be empty.
        editable_impedance_kinds = {"port_ground", "port_diff", "lumped_r", "lumped_l", "lumped_c", "eyescope_se", "eyescope_diff", "scope_se", "scope_diff"}
        is_scope_kind = kind in _SCOPE_PROBE_KINDS
        self._selected_block_group.setVisible(
            kind in editable_impedance_kinds or is_scope_kind
        )
        # Driver Settings group follows the block selection: only visible
        # when a driver block (single-ended or differential) is selected.
        is_driver_kind = kind in {"driver_se", "driver_diff"}
        self._driver_settings_group.setVisible(is_driver_kind)
        if is_driver_kind:
            self._sync_driver_controls_from_instance(block_item.instance)
        is_substrate_kind = (kind in _SUBSTRATE_KINDS)
        self._substrate_settings_group.setVisible(is_substrate_kind)
        if is_substrate_kind:
            self._sync_substrate_controls_from_instance(block_item.instance)
        is_tline_kind = (kind in _TLINE_KINDS)
        self._tline_settings_group.setVisible(is_tline_kind)
        if is_tline_kind:
            self._sync_tline_controls_from_instance(block_item.instance)
        is_component_kind = (kind in _COMPONENT_KINDS)
        self._component_settings_group.setVisible(is_component_kind)
        if is_component_kind:
            self._sync_component_controls_from_instance(block_item.instance)
        self._updating_impedance_editor = True
        editable_kinds = {"port_ground", "port_diff", "lumped_r", "lumped_l", "lumped_c", "eyescope_se", "eyescope_diff", "scope_se", "scope_diff"}
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
        elif kind in {"eyescope_se", "eyescope_diff", "scope_se", "scope_diff"}:
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
        self._updating_scope_name_editor = True
        is_scope = kind in _SCOPE_PROBE_KINDS
        self._scope_name_editor.setEnabled(is_scope)
        self._scope_name_editor.setText(block_item.instance.display_label if is_scope else "")
        self._updating_scope_name_editor = False
        # Hide form rows that don't apply to the selected block kind so the
        # inspector only shows parameters relevant to the current selection.
        editable_impedance_kinds = {"port_ground", "port_diff", "lumped_r", "lumped_l", "lumped_c", "eyescope_se", "eyescope_diff", "scope_se", "scope_diff"}
        self._selected_block_form.setRowVisible(self._impedance_editor, kind in editable_impedance_kinds)
        self._selected_block_form.setRowVisible(self._scope_name_editor, is_scope)
        if kind in _TRANSIENT_SOURCE_KINDS:
            self._bind_transient_source_to_instance(block_item.instance.instance_id)

    def _on_impedance_changed(self, value: float) -> None:
        if self._updating_impedance_editor or self._selected_instance_id is None:
            return
        instance = self._document.get_instance(self._selected_instance_id)
        if instance is None or instance.block_kind not in {"port_ground", "port_diff", "lumped_r", "lumped_l", "lumped_c", "eyescope_se", "eyescope_diff", "scope_se", "scope_diff"}:
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

    def _next_scope_default_name(self) -> str:
        used = {
            inst.display_label
            for inst in self._document.instances
            if inst.block_kind in _SCOPE_PROBE_KINDS
        }
        index = 1
        while f"Scope {index}" in used:
            index += 1
        return f"Scope {index}"

    def _on_scope_name_editing_finished(self) -> None:
        if self._updating_scope_name_editor or self._selected_instance_id is None:
            return
        instance = self._document.get_instance(self._selected_instance_id)
        if instance is None or instance.block_kind not in _SCOPE_PROBE_KINDS:
            return
        new_label = self._scope_name_editor.text().strip()
        if not new_label:
            new_label = self._next_scope_default_name()
            self._updating_scope_name_editor = True
            self._scope_name_editor.setText(new_label)
            self._updating_scope_name_editor = False
        if new_label == instance.display_label:
            return
        self._document.update_instance_display_label(self._selected_instance_id, new_label)
        block_item = self._scene._block_items.get(self._selected_instance_id)
        updated_instance = self._document.get_instance(self._selected_instance_id)
        if block_item is not None and updated_instance is not None:
            block_item.sync_from_instance(updated_instance)
        self._refresh_transient_output_list()
        self._status_label.setText(f"Scope renamed to '{new_label}'.")
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
        mode = self._sim_mode.currentText()
        if mode == "Channel Sim":
            self._refresh_output_port_list()
        elif mode == "Transient":
            self._refresh_transient_source_list()
            self._refresh_transient_output_list()
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

    def _touchstone_blocks_min_fmax_hz(self) -> float | None:
        """Return the lowest f_max across all loaded touchstone blocks (Hz),
        or None if there are no loaded touchstone blocks with frequency data."""
        candidates: list[float] = []
        for instance in self._document.instances:
            if instance.block_kind != "touchstone":
                continue
            loaded = self._state.get_file(instance.source_file_id)
            if loaded is None or not loaded.data.points:
                continue
            candidates.append(float(loaded.data.points[-1].frequency_hz))
        if not candidates:
            return None
        return min(candidates)

    def _auto_clamp_sweep_to_touchstone_blocks(self) -> bool:
        """Clamp document sweep Fmax to the lowest Fmax among loaded
        touchstone blocks. Returns True if the sweep was modified."""
        target_fmax = self._touchstone_blocks_min_fmax_hz()
        if target_fmax is None:
            return False
        sweep = self._document.sweep
        if abs(sweep.fmax_hz - target_fmax) <= max(1.0, 1e-6 * target_fmax):
            return False
        new_fmin = min(sweep.fmin_hz, target_fmax)
        new_fstep = sweep.fstep_hz
        if new_fstep > 0 and new_fstep > (target_fmax - new_fmin):
            new_fstep = max((target_fmax - new_fmin), 0.0)
        self._document.sweep = FrequencySweepSpec(
            fmin_hz=new_fmin,
            fmax_hz=target_fmax,
            fstep_hz=new_fstep,
            display_unit=sweep.display_unit,
        )
        self._sync_sweep_controls_from_document()
        return True

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
        self._auto_clamp_sweep_to_touchstone_blocks()
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

        # Substrate-only entry: open the trace dimension calculator
        # seeded with this substrate's parameters.
        tline_calc_action = None
        if instance.block_kind in _SUBSTRATE_KINDS:
            menu.addSeparator()
            tline_calc_action = menu.addAction("T Line Calculator…")
        tline_calc_for_tline_action = None
        if instance.block_kind in _TLINE_KINDS:
            menu.addSeparator()
            tline_calc_for_tline_action = menu.addAction("T Line Calculator…")

        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if tline_calc_action is not None and chosen is tline_calc_action:
            self._open_tline_calculator(instance_id)
            return
        if tline_calc_for_tline_action is not None and chosen is tline_calc_for_tline_action:
            self._open_tline_calculator_for_tline(instance_id)
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

    def _open_tline_calculator(self, instance_id: str) -> None:
        """Open the trace-dimension calculator seeded with this substrate."""
        instance = self._document.get_instance(instance_id)
        if instance is None or instance.block_kind not in _SUBSTRATE_KINDS:
            return
        spec = instance.substrate_spec or SubstrateSpec()
        from sparams_utility.ui.tline_calculator_dialog import TLineCalculatorDialog

        def _apply(new_spec: SubstrateSpec) -> None:
            target = self._document.get_instance(instance_id)
            if target is None or target.block_kind not in _SUBSTRATE_KINDS:
                return
            self._document.update_instance_substrate_spec(instance_id, new_spec)
            # Refresh the canvas item so the on-symbol εr/h/tan δ labels
            # update immediately.
            for item in self._scene.items():
                if isinstance(item, CircuitBlockItem) and item.instance.instance_id == instance_id:
                    refreshed = self._document.get_instance(instance_id)
                    if refreshed is not None:
                        item.sync_from_instance(refreshed)
                    item.update()
                    break
            # If this substrate is currently selected, refresh the
            # inspector spin boxes to reflect the new values.
            if self._selected_instance_id == instance_id:
                refreshed = self._document.get_instance(instance_id)
                if refreshed is not None:
                    self._sync_substrate_controls_from_instance(refreshed)
            self._emit_project_modified()

        dlg = TLineCalculatorDialog(
            substrate=spec,
            substrate_name=instance.display_label or "Substrate",
            substrate_kind=instance.block_kind,
            parent=self,
            apply_callback=_apply,
        )
        dlg.exec()

    def _open_tline_calculator_for_tline(self, instance_id: str) -> None:
        """Open the calculator seeded from a TLine block; apply writes the TLine."""
        instance = self._document.get_instance(instance_id)
        if instance is None or instance.block_kind not in _TLINE_KINDS:
            return
        tspec = instance.transmission_line_spec or TransmissionLineSpec()
        # Locate the substrate referenced by this tline (by display label
        # AND matching substrate kind for the chosen line family).
        expected_sub_kind = _TLINE_REQUIRED_SUBSTRATE_KIND.get(instance.block_kind, "substrate")
        sub_spec: Optional[SubstrateSpec] = None
        sub_name: str = tspec.substrate_name or ""
        for inst in self._document.instances:
            if (inst.block_kind == expected_sub_kind
                    and inst.display_label == tspec.substrate_name
                    and inst.substrate_spec is not None):
                sub_spec = inst.substrate_spec
                sub_name = inst.display_label
                break
        if sub_spec is None:
            QMessageBox.warning(
                self, "T Line Calculator",
                f"Substrate '{tspec.substrate_name}' not found in the schematic.\n"
                "Add or assign a compatible substrate first.",
            )
            return

        from sparams_utility.ui.tline_calculator_dialog import TLineCalculatorDialog

        def _apply_to_tline(new_tspec: TransmissionLineSpec) -> None:
            target = self._document.get_instance(instance_id)
            if target is None or target.block_kind not in _TLINE_KINDS:
                return
            self._document.update_instance_transmission_line_spec(instance_id, new_tspec)
            for item in self._scene.items():
                if isinstance(item, CircuitBlockItem) and item.instance.instance_id == instance_id:
                    refreshed = self._document.get_instance(instance_id)
                    if refreshed is not None:
                        item.sync_from_instance(refreshed)
                    item.update()
                    break
            if self._selected_instance_id == instance_id:
                refreshed = self._document.get_instance(instance_id)
                if refreshed is not None:
                    self._sync_tline_controls_from_instance(refreshed)
            self._emit_project_modified()

        dlg = TLineCalculatorDialog(
            substrate=sub_spec,
            substrate_name=sub_name or "Substrate",
            substrate_kind=expected_sub_kind,
            parent=self,
            tline_spec=tspec,
            tline_label=instance.display_label or "T Line",
            tline_callback=_apply_to_tline,
        )
        dlg.exec()

    # ------------------------------------------------------------------ #
    #  Channel simulation helpers                                         #
    # ------------------------------------------------------------------ #

    def _on_stat_enabled_changed(self, checked: bool) -> None:
        self._stat_noise.setEnabled(checked)
        self._stat_jitter.setEnabled(checked)

    def _on_sim_mode_changed(self, mode: str) -> None:  # noqa: F811
        is_channel = mode == "Channel Sim"
        is_transient = mode == "Transient"
        self._export_button.setVisible(mode == "S-Parameters")
        self._channel_sim_button.setVisible(is_channel)
        # Channel-sim-wide settings (PRBS/encoding/num_bits/output port/eye)
        # follow the simulation mode. The Driver Settings group instead
        # follows the selected block (a driver_se/driver_diff instance).
        self._channel_sim_settings_group.setVisible(is_channel)
        self._stat_group.setVisible(is_channel)
        self._transient_settings_group.setVisible(is_transient)
        self._transient_sim_button.setVisible(is_transient)
        if is_channel:
            self._refresh_output_port_list()
        elif is_transient:
            self._refresh_transient_source_list()
            self._refresh_transient_output_list()
            # Driver settings panel is also reused for driver-as-transient sources;
            # _on_transient_source_selection_changed will toggle visibility based
            # on the currently selected source kind.
            self._on_transient_source_selection_changed()

    def _bind_transient_source_to_instance(self, instance_id: str) -> None:
        """Make the Transient Settings panel visible and bound to the given source."""
        self._transient_settings_group.setVisible(True)
        self._transient_sim_button.setVisible(self._sim_mode.currentText() == "Transient")
        self._refresh_transient_source_list(select_instance_id=instance_id)

    def _set_driver_channel_sim_rows_visible(self, visible: bool) -> None:
        """Deprecated: channel-sim-only rows now live in their own group.

        Kept as a no-op for backward compatibility with any external caller.
        """
        return None

    def _sync_driver_controls_from_instance(self, instance) -> None:
        """Populate the Channel Sim driver editors from a driver block instance.

        Used when a driver block is selected as the Transient source so that
        the user edits the same DriverSpec exposed in Channel Sim mode.
        """
        spec = instance.driver_spec or DriverSpec()
        widgets = (
            self._drv_v_high,
            self._drv_v_low,
            self._drv_rise_time,
            self._drv_fall_time,
            self._drv_bitrate,
            self._drv_prbs,
            self._drv_encoding,
            self._drv_num_bits,
            self._drv_random_noise_mv,
            self._drv_source_impedance_ohm,
            self._drv_max_length_lfsr,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            self._drv_v_high.setValue(float(spec.voltage_high_v))
            self._drv_v_low.setValue(float(spec.voltage_low_v))
            self._drv_rise_time.setValue(float(spec.rise_time_s) * 1e12)
            self._drv_fall_time.setValue(float(spec.fall_time_s) * 1e12)
            self._drv_bitrate.setValue(float(spec.bitrate_gbps))
            idx = self._drv_prbs.findText(spec.prbs_pattern)
            if idx >= 0:
                self._drv_prbs.setCurrentIndex(idx)
            idx = self._drv_encoding.findText(spec.encoding)
            if idx >= 0:
                self._drv_encoding.setCurrentIndex(idx)
            self._drv_num_bits.setValue(int(spec.num_bits))
            self._drv_random_noise_mv.setValue(float(getattr(spec, "random_noise_v", 0.0)) * 1e3)
            self._drv_source_impedance_ohm.setValue(float(getattr(spec, "source_impedance_ohm", 0.0)))
            self._drv_max_length_lfsr.setChecked(bool(getattr(spec, "maximal_length_lfsr", False)))
        finally:
            for w in widgets:
                w.blockSignals(False)

    # ------------------------------------------------------------------
    # Substrate helpers
    # ------------------------------------------------------------------
    def _sync_substrate_controls_from_instance(self, instance) -> None:
        spec = instance.substrate_spec or SubstrateSpec()
        self._updating_substrate_controls = True
        widgets = (
            self._sub_name_edit,
            self._sub_epsilon_r,
            self._sub_loss_tangent,
            self._sub_height_um,
            self._sub_thickness_um,
            self._sub_conductivity,
            self._sub_roughness_um,
            self._sub_stripline_htop_um,
            self._sub_stripline_hbottom_um,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            self._sub_name_edit.setText(instance.display_label)
            self._sub_epsilon_r.setValue(float(spec.epsilon_r))
            self._sub_loss_tangent.setValue(float(spec.loss_tangent))
            self._sub_height_um.setValue(float(spec.height_m) * 1e6)
            self._sub_thickness_um.setValue(float(spec.conductor_thickness_m) * 1e6)
            self._sub_conductivity.setValue(float(spec.conductivity_s_per_m))
            self._sub_roughness_um.setValue(float(spec.roughness_rq_m) * 1e6)
            self._sub_stripline_htop_um.setValue(float(spec.stripline_h_top_m) * 1e6)
            self._sub_stripline_hbottom_um.setValue(float(spec.stripline_h_bottom_m) * 1e6)
        finally:
            for w in widgets:
                w.blockSignals(False)
            self._updating_substrate_controls = False
        # Microstrip uses a single dielectric height; stripline uses the two
        # gap distances instead. Toggle row visibility accordingly.
        is_stripline = (instance.block_kind == "substrate_stripline")
        self._sub_form_layout.setRowVisible(self._sub_height_um, not is_stripline)
        self._sub_form_layout.setRowVisible(self._sub_stripline_htop_um, is_stripline)
        self._sub_form_layout.setRowVisible(self._sub_stripline_hbottom_um, is_stripline)

    def _substrate_spec_from_controls(self) -> SubstrateSpec:
        t_m = float(self._sub_thickness_um.value()) * 1e-6
        # For microstrip the height comes directly from the editor; for
        # stripline we derive the total dielectric stack from the two gap
        # distances plus the conductor thickness.
        is_stripline = False
        if self._selected_instance_id is not None:
            inst = self._document.get_instance(self._selected_instance_id)
            is_stripline = bool(inst and inst.block_kind == "substrate_stripline")
        h_top_m = max(0.0, float(self._sub_stripline_htop_um.value()) * 1e-6)
        h_bottom_m = max(0.0, float(self._sub_stripline_hbottom_um.value()) * 1e-6)
        if is_stripline:
            h_m = h_top_m + t_m + h_bottom_m
        else:
            h_m = float(self._sub_height_um.value()) * 1e-6
        return SubstrateSpec(
            epsilon_r=float(self._sub_epsilon_r.value()),
            loss_tangent=float(self._sub_loss_tangent.value()),
            height_m=h_m,
            conductor_thickness_m=t_m,
            conductivity_s_per_m=float(self._sub_conductivity.value()),
            roughness_rq_m=float(self._sub_roughness_um.value()) * 1e-6,
            stripline_h_top_m=h_top_m,
            stripline_h_bottom_m=h_bottom_m,
        )

    def _on_substrate_param_changed(self, *_: object) -> None:
        if self._updating_substrate_controls:
            return
        if self._selected_instance_id is None:
            return
        instance = self._document.get_instance(self._selected_instance_id)
        if instance is None or instance.block_kind not in _SUBSTRATE_KINDS:
            return
        new_spec = self._substrate_spec_from_controls()
        self._document.update_instance_substrate_spec(self._selected_instance_id, new_spec)
        # Refresh the canvas so the on-symbol parameter labels stay in sync.
        for item in self._scene.items():
            if isinstance(item, CircuitBlockItem) and item.instance.instance_id == self._selected_instance_id:
                refreshed = self._document.get_instance(self._selected_instance_id)
                if refreshed is not None:
                    item.sync_from_instance(refreshed)
                item.update()
                break
        self._emit_project_modified()

    # ------------------------------------------------------------------
    # Transmission-line inspector synchronisation
    # ------------------------------------------------------------------
    def _sync_tline_controls_from_instance(self, instance) -> None:
        spec = instance.transmission_line_spec or TransmissionLineSpec()
        self._updating_tline_controls = True
        widgets = (
            self._tline_substrate_combo,
            self._tline_width_um,
            self._tline_length_mm,
            self._tline_spacing_um,
            self._tline_z0_ref,
            self._tline_width_end_um,
            self._tline_taper_profile,
            self._tline_cpw_slot_um,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            # Repopulate the substrate combobox with the substrates in the
            # document that match the line's required kind.
            self._tline_substrate_combo.clear()
            available = self._available_substrates_for_tline(instance.block_kind)
            names = [sub.display_label for sub in available]
            self._tline_substrate_combo.addItems(names)
            if spec.substrate_name in names:
                self._tline_substrate_combo.setCurrentIndex(names.index(spec.substrate_name))
            elif names:
                self._tline_substrate_combo.setCurrentIndex(0)
            self._tline_width_um.setValue(float(spec.width_m) * 1e6)
            self._tline_length_mm.setValue(float(spec.length_m) * 1e3)
            self._tline_spacing_um.setValue(float(spec.spacing_m) * 1e6)
            self._tline_z0_ref.setValue(float(spec.z0_ref_ohm))
            width_end_um = float(spec.width_end_m) * 1e6 if spec.width_end_m and spec.width_end_m > 0.0 else 400.0
            self._tline_width_end_um.setValue(width_end_um)
            self._tline_taper_profile.setCurrentText(spec.taper_profile or "linear")
            self._tline_cpw_slot_um.setValue(float(spec.cpw_slot_m) * 1e6)
        finally:
            for w in widgets:
                w.blockSignals(False)
            self._updating_tline_controls = False
        kind = instance.block_kind
        is_coupled = kind in _TLINE_COUPLED_KINDS
        is_taper = kind == "taper"
        is_cpw = kind in {"tline_cpw", "tline_cpw_coupled"}
        # Spacing meaningful for coupled (incl. coupled CPW); hide for taper.
        self._tline_form_layout.setRowVisible(
            self._tline_spacing_um, is_coupled and not is_taper
        )
        self._tline_form_layout.setRowVisible(self._tline_width_end_um, is_taper)
        self._tline_form_layout.setRowVisible(self._tline_taper_profile, is_taper)
        self._tline_form_layout.setRowVisible(self._tline_cpw_slot_um, is_cpw)

    def _tline_spec_from_controls(self, instance) -> TransmissionLineSpec:
        line_kind = {
            "tline_microstrip": "microstrip",
            "tline_stripline": "stripline",
            "tline_microstrip_coupled": "microstrip_coupled",
            "tline_stripline_coupled": "stripline_coupled",
            "tline_cpw": "cpw",
            "tline_cpw_coupled": "cpw_coupled",
            "taper": "taper",
        }.get(instance.block_kind, "microstrip")
        substrate_name = self._tline_substrate_combo.currentText() or ""
        width_m = float(self._tline_width_um.value()) * 1e-6
        return TransmissionLineSpec(
            line_kind=line_kind,
            substrate_name=substrate_name,
            width_m=width_m,
            length_m=float(self._tline_length_mm.value()) * 1e-3,
            spacing_m=float(self._tline_spacing_um.value()) * 1e-6,
            z0_ref_ohm=float(self._tline_z0_ref.value()),
            width_end_m=float(self._tline_width_end_um.value()) * 1e-6,
            taper_profile=str(self._tline_taper_profile.currentText()) or "linear",
            cpw_slot_m=float(self._tline_cpw_slot_um.value()) * 1e-6,
        )

    def _on_tline_param_changed(self, *_: object) -> None:
        if self._updating_tline_controls:
            return
        if self._selected_instance_id is None:
            return
        instance = self._document.get_instance(self._selected_instance_id)
        if instance is None or instance.block_kind not in _TLINE_KINDS:
            return
        new_spec = self._tline_spec_from_controls(instance)
        self._document.update_instance_transmission_line_spec(
            self._selected_instance_id, new_spec
        )
        for item in self._scene.items():
            if isinstance(item, CircuitBlockItem) and item.instance.instance_id == self._selected_instance_id:
                refreshed = self._document.get_instance(self._selected_instance_id)
                if refreshed is not None:
                    item.sync_from_instance(refreshed)
                item.update()
                break
        self._emit_project_modified()

    # ------------------------------------------------------------------
    # Component (Attenuator / Circulator / Coupler) inspector
    # ------------------------------------------------------------------
    def _sync_component_controls_from_instance(self, instance) -> None:
        self._updating_component_controls = True
        kind = instance.block_kind
        widgets = (
            self._comp_attn_db,
            self._comp_kind_combo,
            self._comp_coupling_db,
            self._comp_il_db,
            self._comp_iso_db,
            self._comp_rl_db,
            self._comp_dir_combo,
            self._comp_z0_ref,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            if kind == "attenuator":
                spec = instance.attenuator_spec or AttenuatorSpec()
                self._comp_attn_db.setValue(float(spec.attenuation_db))
                self._comp_z0_ref.setValue(float(spec.z0_ref_ohm))
            elif kind == "circulator":
                spec = instance.circulator_spec or CirculatorSpec()
                self._comp_il_db.setValue(float(spec.insertion_loss_db))
                self._comp_iso_db.setValue(float(spec.isolation_db))
                self._comp_rl_db.setValue(float(spec.return_loss_db))
                self._comp_dir_combo.setCurrentText(spec.direction)
                self._comp_z0_ref.setValue(float(spec.z0_ref_ohm))
            elif kind == "coupler":
                spec = instance.coupler_spec or CouplerSpec()
                self._comp_kind_combo.setCurrentText(spec.kind)
                self._comp_coupling_db.setValue(float(spec.coupling_db))
                self._comp_il_db.setValue(float(spec.insertion_loss_db))
                self._comp_iso_db.setValue(float(spec.isolation_db))
                self._comp_rl_db.setValue(float(spec.return_loss_db))
                self._comp_z0_ref.setValue(float(spec.z0_ref_ohm))
        finally:
            for w in widgets:
                w.blockSignals(False)
            self._updating_component_controls = False
        layout = self._component_form_layout
        layout.setRowVisible(self._comp_attn_db, kind == "attenuator")
        layout.setRowVisible(self._comp_kind_combo, kind == "coupler")
        layout.setRowVisible(self._comp_coupling_db, kind == "coupler")
        layout.setRowVisible(self._comp_il_db, kind in ("circulator", "coupler"))
        layout.setRowVisible(self._comp_iso_db, kind in ("circulator", "coupler"))
        layout.setRowVisible(self._comp_rl_db, kind in ("circulator", "coupler"))
        layout.setRowVisible(self._comp_dir_combo, kind == "circulator")
        layout.setRowVisible(self._comp_z0_ref, True)

    def _on_component_param_changed(self, *_):
        if self._updating_component_controls:
            return
        if self._selected_instance_id is None:
            return
        instance = self._document.get_instance(self._selected_instance_id)
        if instance is None:
            return
        kind = instance.block_kind
        if kind == "attenuator":
            new_spec = AttenuatorSpec(
                attenuation_db=float(self._comp_attn_db.value()),
                z0_ref_ohm=float(self._comp_z0_ref.value()),
            )
            self._document.update_instance_attenuator_spec(
                self._selected_instance_id, new_spec
            )
        elif kind == "circulator":
            new_spec = CirculatorSpec(
                insertion_loss_db=float(self._comp_il_db.value()),
                isolation_db=float(self._comp_iso_db.value()),
                return_loss_db=float(self._comp_rl_db.value()),
                direction=str(self._comp_dir_combo.currentText()),
                z0_ref_ohm=float(self._comp_z0_ref.value()),
            )
            self._document.update_instance_circulator_spec(
                self._selected_instance_id, new_spec
            )
        elif kind == "coupler":
            new_spec = CouplerSpec(
                kind=str(self._comp_kind_combo.currentText()),
                coupling_db=float(self._comp_coupling_db.value()),
                insertion_loss_db=float(self._comp_il_db.value()),
                isolation_db=float(self._comp_iso_db.value()),
                return_loss_db=float(self._comp_rl_db.value()),
                z0_ref_ohm=float(self._comp_z0_ref.value()),
            )
            self._document.update_instance_coupler_spec(
                self._selected_instance_id, new_spec
            )
        else:
            return
        for item in self._scene.items():
            if isinstance(item, CircuitBlockItem) and item.instance.instance_id == self._selected_instance_id:
                refreshed = self._document.get_instance(self._selected_instance_id)
                if refreshed is not None:
                    item.sync_from_instance(refreshed)
                item.update()
                break
        self._emit_project_modified()

    def _unique_substrate_name(self, desired: str, exclude_instance_id: Optional[str] = None) -> str:
        """Return a substrate name guaranteed to be unique among substrates.

        If ``desired`` already collides with another substrate's display
        label, a numeric suffix is appended (``_2``, ``_3``, …). Empty input
        falls back to ``"Substrate"``.
        """
        base = (desired or "").strip() or "Substrate"
        existing = {
            inst.display_label
            for inst in self._document.instances
            if inst.block_kind in _SUBSTRATE_KINDS and inst.instance_id != exclude_instance_id
        }
        if base not in existing:
            return base
        idx = 2
        while f"{base}_{idx}" in existing:
            idx += 1
        return f"{base}_{idx}"

    # ------------------------------------------------------------------
    # Transmission-line helpers
    # ------------------------------------------------------------------
    def _available_substrates_for_tline(self, tline_kind: str) -> list:
        """Return the document's substrate instances compatible with `tline_kind`."""
        if tline_kind == "taper":
            # Taper accepts ANY substrate kind in the document.
            return [
                inst for inst in self._document.instances
                if inst.block_kind in _SUBSTRATE_KINDS
            ]
        required = _TLINE_REQUIRED_SUBSTRATE_KIND.get(tline_kind)
        if required is None:
            return []
        return [
            inst for inst in self._document.instances
            if inst.block_kind == required
        ]

    def _prompt_tline_substrate(self, tline_kind: str) -> Optional[TransmissionLineSpec]:
        """Pick a substrate by name; build a default `TransmissionLineSpec`.

        Returns ``None`` if the user cancels or no compatible substrate
        exists in the schematic. In the latter case a helpful warning is
        shown so the user knows which substrate kind to add first.
        """
        candidates = self._available_substrates_for_tline(tline_kind)
        if not candidates:
            if tline_kind == "taper":
                human_kind = "any (Microstrip or Stripline)"
            else:
                required = _TLINE_REQUIRED_SUBSTRATE_KIND.get(tline_kind, "substrate")
                human_kind = "Microstrip" if required == "substrate" else "Stripline"
            QMessageBox.warning(
                self,
                "No substrate available",
                f"This transmission line requires a '{human_kind}' substrate "
                f"block in the schematic. Add one from the Substrates "
                f"category first, then drop the line again.",
            )
            return None

        names = [inst.display_label for inst in candidates]
        if len(names) == 1:
            chosen = names[0]
        else:
            chosen, ok = QInputDialog.getItem(
                self,
                "Pick substrate",
                "Reference substrate for this transmission line:",
                names,
                0,
                False,
            )
            if not ok:
                return None

        line_kind = {
            "tline_microstrip": "microstrip",
            "tline_stripline": "stripline",
            "tline_microstrip_coupled": "microstrip_coupled",
            "tline_stripline_coupled": "stripline_coupled",
            "tline_cpw": "cpw",
            "tline_cpw_coupled": "cpw_coupled",
            "taper": "taper",
        }[tline_kind]
        if tline_kind == "taper":
            # Sensible defaults for a fresh taper: end width = 2× start
            # width, linear profile. Allows the spec to be valid out of
            # the box; user can edit later from the parameters panel.
            spec = TransmissionLineSpec(
                line_kind=line_kind,
                substrate_name=str(chosen),
            )
            return TransmissionLineSpec(
                line_kind=line_kind,
                substrate_name=str(chosen),
                width_m=spec.width_m,
                length_m=spec.length_m,
                spacing_m=spec.spacing_m,
                z0_ref_ohm=spec.z0_ref_ohm,
                width_end_m=spec.width_m * 2.0,
                taper_profile="linear",
                cpw_slot_m=spec.cpw_slot_m,
            )
        return TransmissionLineSpec(
            line_kind=line_kind,
            substrate_name=str(chosen),
        )

    def _next_tline_default_name(self, tline_kind: str) -> str:
        """Generate a unique default label like 'TL1', 'TL2', … (or 'CTL1' for coupled)."""
        prefix = "CTL" if tline_kind in _TLINE_COUPLED_KINDS else "TL"
        existing = {
            inst.display_label for inst in self._document.instances
            if inst.block_kind in _TLINE_KINDS
        }
        idx = 1
        while f"{prefix}{idx}" in existing:
            idx += 1
        return f"{prefix}{idx}"

    def _on_substrate_name_committed(self) -> None:
        if self._updating_substrate_controls:
            return
        if self._selected_instance_id is None:
            return
        instance = self._document.get_instance(self._selected_instance_id)
        if instance is None or instance.block_kind not in _SUBSTRATE_KINDS:
            return
        new_name = self._unique_substrate_name(
            self._sub_name_edit.text(),
            exclude_instance_id=self._selected_instance_id,
        )
        if new_name == instance.display_label:
            # Still normalize the edit field (e.g. trim whitespace).
            self._updating_substrate_controls = True
            try:
                self._sub_name_edit.setText(new_name)
            finally:
                self._updating_substrate_controls = False
            return
        self._document.update_instance_display_label(self._selected_instance_id, new_name)
        self._updating_substrate_controls = True
        try:
            self._sub_name_edit.setText(new_name)
        finally:
            self._updating_substrate_controls = False
        for item in self._scene.items():
            if isinstance(item, CircuitBlockItem) and item.instance.instance_id == self._selected_instance_id:
                refreshed = self._document.get_instance(self._selected_instance_id)
                if refreshed is not None:
                    item.sync_from_instance(refreshed)
                item.update()
                break
        self._emit_project_modified()

    def _on_max_length_lfsr_toggled(self, checked: bool) -> None:
        # No-op kept for backward compatibility (project state load calls it).
        # Num bits remains editable: it now controls the *total* bitstream
        # length while Maximal Length LFSR controls the *pattern* used.
        return

    def _transient_source_spec_from_controls(self) -> TransientSourceSpec:
        rise_time_ps = self._transient_rise_time.value()
        fall_time_ps = self._transient_fall_time.value()
        # For V-Step the fall time is not user-editable; mirror the rise time so
        # the solver behaves consistently for both polarities.
        if not self._transient_fall_time.isVisible():
            fall_time_ps = rise_time_ps
        return TransientSourceSpec(
            amplitude_v=self._transient_amplitude.value(),
            polarity=self._transient_polarity.currentText(),
            rise_time_s=rise_time_ps * 1e-12,
            fall_time_s=fall_time_ps * 1e-12,
            delay_s=self._transient_delay.value() * 1e-12,
            pulse_width_s=self._transient_pulse_width.value() * 1e-12,
        )

    def _set_transient_control_state(
        self,
        *,
        enabled: bool,
        pulse_visible: bool,
        fall_visible: bool = True,
        rise_visible: bool = True,
    ) -> None:
        for widget in (
            self._transient_amplitude,
            self._transient_polarity,
            self._transient_rise_time,
            self._transient_fall_time,
            self._transient_delay,
            self._transient_pulse_width,
        ):
            widget.setEnabled(enabled)
        self._transient_pulse_width_label.setVisible(pulse_visible)
        self._transient_pulse_width.setVisible(pulse_visible)
        self._transient_fall_time_label.setVisible(fall_visible)
        self._transient_fall_time.setVisible(fall_visible)
        self._transient_rise_time_label.setVisible(rise_visible)
        self._transient_rise_time.setVisible(rise_visible)

    def _refresh_transient_source_list(self, select_instance_id: str | None = None) -> None:
        current_instance_id = select_instance_id or self._transient_source_instance.currentData()
        self._transient_source_instance.blockSignals(True)
        self._transient_source_instance.clear()
        for inst in self._document.instances:
            if inst.block_kind in _TRANSIENT_SOURCE_KINDS:
                kind_tag = "V-Pulse" if inst.block_kind == "transient_pulse_se" else "V-Step"
            elif inst.block_kind == "driver_se":
                kind_tag = "Driver SE"
            elif inst.block_kind == "driver_diff":
                kind_tag = "Driver Diff"
            else:
                continue
            label = f"{inst.display_label} [{kind_tag}] ({inst.instance_id[:8]})"
            self._transient_source_instance.addItem(label, inst.instance_id)
        self._transient_source_instance.blockSignals(False)
        if current_instance_id is not None:
            idx = self._transient_source_instance.findData(current_instance_id)
            if idx >= 0:
                self._transient_source_instance.setCurrentIndex(idx)
        self._on_transient_source_selection_changed()

    def _refresh_transient_output_list(self) -> None:
        self._transient_output_list.clear()
        for inst in self._document.instances:
            if inst.block_kind not in _SCOPE_PROBE_KINDS:
                continue
            suffix = "Diff" if inst.block_kind == "scope_diff" else "SE"
            item = QListWidgetItem(f"{inst.display_label} ({suffix})")
            item.setData(Qt.UserRole, inst.instance_id)
            self._transient_output_list.addItem(item)
        if self._transient_output_list.count() == 0:
            placeholder = QListWidgetItem("No Scope probes placed yet.")
            placeholder.setFlags(Qt.NoItemFlags)
            self._transient_output_list.addItem(placeholder)

    def _on_transient_source_selection_changed(self, *_args) -> None:
        source_instance_id = self._transient_source_instance.currentData()
        source_inst = self._document.get_instance(source_instance_id) if source_instance_id else None
        if source_inst is None:
            self._updating_transient_controls = True
            self._set_transient_control_state(
                enabled=False, pulse_visible=False, fall_visible=False, rise_visible=False
            )
            self._updating_transient_controls = False
            if self._sim_mode.currentText() == "Transient":
                self._driver_settings_group.setVisible(False)
            return
        if source_inst.block_kind in {"driver_se", "driver_diff"}:
            # Driver source: parameters come from the Channel Sim driver spec,
            # so the pulse/step editors are hidden and read-only. The driver
            # settings panel is shown so the user can edit bitrate / PRBS /
            # encoding / num_bits / voltages / edge times exactly as in
            # Channel Sim mode.
            self._updating_transient_controls = True
            self._set_transient_control_state(
                enabled=False,
                pulse_visible=False,
                fall_visible=False,
                rise_visible=False,
            )
            self._updating_transient_controls = False
            self._driver_settings_group.setVisible(True)
            self._sync_driver_controls_from_instance(source_inst)
            return
        # Non-driver source in transient mode: hide the driver settings panel.
        if self._sim_mode.currentText() == "Transient":
            self._driver_settings_group.setVisible(False)
        spec = source_inst.transient_source_spec or TransientSourceSpec()
        if source_inst.transient_source_spec is None:
            self._document.update_instance_transient_source_spec(source_inst.instance_id, spec)
        self._updating_transient_controls = True
        self._transient_amplitude.setValue(spec.amplitude_v)
        self._transient_polarity.setCurrentText(spec.polarity)
        self._transient_rise_time.setValue(spec.rise_time_s * 1e12)
        self._transient_fall_time.setValue(spec.fall_time_s * 1e12)
        self._transient_delay.setValue(spec.delay_s * 1e12)
        self._transient_pulse_width.setValue(spec.pulse_width_s * 1e12)
        is_pulse = source_inst.block_kind == "transient_pulse_se"
        self._set_transient_control_state(
            enabled=True,
            pulse_visible=is_pulse,
            fall_visible=is_pulse,
            rise_visible=True,
        )
        self._updating_transient_controls = False

    def _on_transient_controls_changed(self, *_args) -> None:
        if self._updating_transient_controls:
            return
        source_instance_id = self._transient_source_instance.currentData()
        if not source_instance_id:
            return
        self._document.update_instance_transient_source_spec(
            source_instance_id,
            self._transient_source_spec_from_controls(),
        )
        self._emit_project_modified()

    def _selected_transient_output_refs(self) -> list[CircuitPortRef]:
        refs: list[CircuitPortRef] = []
        for item in self._transient_output_list.selectedItems():
            instance_id = item.data(Qt.UserRole)
            if not instance_id:
                continue
            refs.append(CircuitPortRef(str(instance_id), 1))
        return refs

    def _all_scope_output_refs(self) -> list[CircuitPortRef]:
        """Return one CircuitPortRef per Scope probe placed in the circuit."""
        refs: list[CircuitPortRef] = []
        for inst in self._document.instances:
            if inst.block_kind in _SCOPE_PROBE_KINDS:
                refs.append(CircuitPortRef(inst.instance_id, 1))
        return refs

    def _run_transient_simulation(self) -> None:
        source_instance_id = self._transient_source_instance.currentData()
        source_inst = self._document.get_instance(source_instance_id) if source_instance_id else None
        if source_inst is None:
            QMessageBox.warning(self, "Transient", "No transient source block found in the circuit.")
            return

        if source_inst.block_kind in _TRANSIENT_SOURCE_KINDS:
            spec = self._transient_source_spec_from_controls()
            self._document.update_instance_transient_source_spec(source_inst.instance_id, spec)
        elif source_inst.block_kind in {"driver_se", "driver_diff"}:
            driver_spec = DriverSpec(
                voltage_high_v=self._drv_v_high.value(),
                voltage_low_v=self._drv_v_low.value(),
                rise_time_s=self._drv_rise_time.value() * 1e-12,
                fall_time_s=self._drv_fall_time.value() * 1e-12,
                bitrate_gbps=self._drv_bitrate.value(),
                prbs_pattern=self._drv_prbs.currentText(),
                encoding=self._drv_encoding.currentText(),
                num_bits=self._drv_num_bits.value(),
                random_noise_v=self._drv_random_noise_mv.value() * 1e-3,
                source_impedance_ohm=self._drv_source_impedance_ohm.value(),
                maximal_length_lfsr=self._drv_max_length_lfsr.isChecked(),
            )
            self._document.update_instance_driver_spec(source_inst.instance_id, driver_spec)
        else:
            QMessageBox.warning(
                self,
                "Transient",
                "Selected source must be a V-Step/V-Pulse block or a Channel Sim driver.",
            )
            return
        output_refs = self._all_scope_output_refs()
        if not output_refs:
            QMessageBox.warning(
                self,
                "Transient",
                "Place at least one Scope probe in the circuit to capture a transient waveform.",
            )
            return

        progress = QProgressDialog("Running transient simulation...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Transient Simulation")
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
            self._status_label.setText("Running transient simulation...")
            result = simulate_transient(
                self._document,
                self._state,
                source_instance_id=source_inst.instance_id,
                output_refs=output_refs,
                stop_time_s=self._transient_stop_time.value() * 1e-9,
                progress_callback=_on_progress,
            )
        except InterruptedError:
            self._status_label.setText("Transient simulation cancelled.")
            progress.close()
            return
        except Exception as exc:
            progress.close()
            self._status_label.setText(f"Transient simulation failed: {exc}")
            QMessageBox.warning(self, "Transient failed", str(exc))
            return

        progress.setValue(100)
        progress.close()
        self._open_transient_window(result)
        self.transient_result_generated.emit(
            {
                "result": result,
                "circuit_name": self.circuit_display_name(),
            }
        )
        trace_count = len(result.traces)
        if result.warnings:
            self._status_label.setText(
                f"Transient sim done │ {trace_count} trace(s) │ {result.warnings[0]}"
            )
            self._status_label.setToolTip("\n".join(result.warnings))
            return
        self._status_label.setText(f"Transient sim done │ {trace_count} trace(s) generated.")
        self._status_label.setToolTip("")

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
            random_noise_v=self._drv_random_noise_mv.value() * 1e-3,
            source_impedance_ohm=self._drv_source_impedance_ohm.value(),
            maximal_length_lfsr=self._drv_max_length_lfsr.isChecked(),
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

        # Channel simulation occupies 0-70% of the bar; the eye diagram draw
        # consumes the remaining 70-100% so the user sees a single continuous
        # progress for "simulation + plot generation".
        def _on_progress(percent: int, label: str) -> None:
            if progress.wasCanceled():
                raise InterruptedError("Simulation cancelled by user.")
            progress.setLabelText(label)
            progress.setValue(int(percent * 0.70))
            QApplication.processEvents()

        def _on_eye_progress(percent: int, label: str) -> None:
            try:
                progress.setLabelText(label)
                progress.setValue(70 + int(percent * 0.30))
                QApplication.processEvents()
            except RuntimeError:
                pass

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

        progress.setLabelText("Drawing eye diagram...")
        progress.setValue(70)
        QApplication.processEvents()

        # Notify host window so the binary eye artifact can be persisted immediately.
        self.eye_result_generated.emit(
            {
                "result": result,
                "circuit_name": self.circuit_display_name(),
                "eye_span_ui": int(self._selected_eye_span_ui()),
                "render_mode": str(self._selected_eye_render_mode()),
                "quality_preset": str(self._selected_eye_quality_preset()),
                "stat_enabled": bool(self._stat_enabled.isChecked()),
                "noise_rms_mv": float(self._stat_noise.value()),
                "jitter_rms_ps": float(self._stat_jitter.value()),
            }
        )

        win = self._open_eye_window(result, progress_callback=_on_eye_progress)
        progress.setValue(100)
        progress.close()

        # Show eye summary in the status bar
        s = win.eye_summary
        def _fmv(v: float) -> str:
            import math as _math
            return "n/a" if not _math.isfinite(v) else f"{v * 1000:.2f} mV"
        def _fps(v: float) -> str:
            import math as _math
            return "n/a" if not _math.isfinite(v) else f"{v:.2f} ps"
        def _fpc(v: float) -> str:
            import math as _math
            return "n/a" if not _math.isfinite(v) else f"{v:.2f} %"
        width_ps = s.get("eye_width_ps", s.get("width_ps", float("nan")))
        bit_period_ps = s.get("bit_period_ps", float("nan"))
        dj_pp_ps = s.get("dj_pp_s", float("nan")) * 1e12
        rj_sigma_ps = s.get("sigma_rj_s", float("nan")) * 1e12
        tj_pp_ps = s.get("tj_pp_s", float("nan")) * 1e12
        self._status_label.setText(
            f"Channel sim done │ "
            f"One: {_fmv(s.get('one_level', s.get('level1', float('nan'))))}  "
            f"Zero: {_fmv(s.get('zero_level', s.get('level0', float('nan'))))}  "
            f"Amp: {_fmv(s.get('eye_amplitude', float('nan')))}  "
            f"Height: {_fmv(s.get('eye_height', s.get('height', float('nan'))))}  "
            f"Width: {_fps(width_ps)}  "
            f"Cross: {_fpc(s.get('eye_crossing_pct', float('nan')))}  "
            f"DJ(pp): {_fps(dj_pp_ps)}  "
            f"RJ(σ): {_fps(rj_sigma_ps)}  "
            f"TJ(pp@BER): {_fps(tj_pp_ps)}  "
            f"Bit: {_fps(bit_period_ps)}"
        )

    def export_project_state(self) -> dict:
        return {
            "window_title": self.windowTitle(),
            "circuit_name": self.circuit_display_name(),
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
            "drv_random_noise_mv": self._drv_random_noise_mv.value(),
            "drv_source_impedance_ohm": self._drv_source_impedance_ohm.value(),
            "drv_maximal_length_lfsr": self._drv_max_length_lfsr.isChecked(),
            "drv_output_port_instance_id": self._drv_output_port_instance.currentData(),
            **self._document.to_dict(),
        }

    def apply_project_state(self, state: dict) -> None:
        # Loading a project replaces the entire document; undo history from
        # any previous session is not meaningful and would be confusing.
        self._undo_stack.clear()
        self._suspend_undo_capture = True
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

        saved_name = str(state.get("circuit_name", "")).strip()
        if not saved_name:
            raw_title = str(state.get("window_title", "")).strip()
            if " - " in raw_title:
                saved_name = raw_title.split(" - ", 1)[1].strip()
        if not saved_name:
            saved_name = f"Circuit #{self.window_number}"
        self._circuit_name = saved_name
        self._circuit_name_edit.blockSignals(True)
        self._circuit_name_edit.setText(saved_name)
        self._circuit_name_edit.blockSignals(False)
        self._refresh_window_title()

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

        # Once all touchstone instances are in place, clamp Fmax to the
        # lowest f_max across loaded blocks (so the sweep can't extend
        # beyond any block's data range).
        self._auto_clamp_sweep_to_touchstone_blocks()

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

        self._drv_random_noise_mv.blockSignals(True)
        self._drv_random_noise_mv.setValue(float(state.get("drv_random_noise_mv", 0.0)))
        self._drv_random_noise_mv.blockSignals(False)

        self._drv_source_impedance_ohm.blockSignals(True)
        self._drv_source_impedance_ohm.setValue(float(state.get("drv_source_impedance_ohm", 0.0)))
        self._drv_source_impedance_ohm.blockSignals(False)

        self._drv_max_length_lfsr.blockSignals(True)
        self._drv_max_length_lfsr.setChecked(bool(state.get("drv_maximal_length_lfsr", False)))
        self._drv_max_length_lfsr.blockSignals(False)

        self._pending_output_port_instance_id = state.get("drv_output_port_instance_id")
        if self._pending_output_port_instance_id is not None:
            self._refresh_output_port_list()
        self._sync_eye_window_titles()

        # Re-arm undo capture and adopt the loaded document as the new
        # baseline snapshot.
        self._last_doc_snapshot = self._document.to_dict()
        self._suspend_undo_capture = False

    def _export_equivalent_touchstone(self) -> None:
        selected_format = "RI"
        selected_unit = "GHz"

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

        file_name = self._prompt_export_file_name(int(result.nports))
        if not file_name:
            return

        payload = {
            "circuit_name": self.circuit_display_name(),
            "nports": int(result.nports),
            "touchstone_text": text,
            "frequency_unit": selected_unit,
            "data_format": selected_format,
            "file_name": file_name,
        }
        self.sparameter_result_generated.emit(payload)

        error = str(payload.get("error") or "").strip()
        if error:
            self._status_label.setText(f"Export failed: {error}")
            QMessageBox.warning(self, "Export failed", error)
            return

        saved_path = str(payload.get("saved_path") or "").strip()
        saved_file_name = str(payload.get("saved_file_name") or file_name).strip()
        if not saved_path:
            fallback_error = "The host window did not save the generated Touchstone file."
            self._status_label.setText(f"Export failed: {fallback_error}")
            QMessageBox.warning(self, "Export failed", fallback_error)
            return

        self._status_label.setText(
            f"Equivalent Touchstone exported to {saved_path}. {self._describe_passivity(result)}"
        )

    def _emit_project_modified(self) -> None:
        # Capture an undo snapshot whenever the document changes.
        if not self._suspend_undo_capture:
            try:
                current = self._document.to_dict()
            except Exception:
                current = None
            if current is not None and current != self._last_doc_snapshot:
                self._undo_stack.append(self._last_doc_snapshot)
                if len(self._undo_stack) > self._UNDO_LIMIT:
                    del self._undo_stack[: len(self._undo_stack) - self._UNDO_LIMIT]
                self._last_doc_snapshot = current
        self.project_modified.emit()

    def _undo_last_change(self) -> None:
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        self._suspend_undo_capture = True
        try:
            self._restore_document_from_snapshot(snapshot)
            self._last_doc_snapshot = self._document.to_dict()
        finally:
            self._suspend_undo_capture = False
        self.project_modified.emit()

    def _restore_document_from_snapshot(self, snapshot: dict) -> None:
        """Replace the current CircuitDocument with one rebuilt from a
        snapshot dict, then refresh the scene and dependent UI lists."""
        document = CircuitDocument.from_dict(snapshot)
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
            if item.port_a.instance_id in valid_instance_ids
            and item.port_b.instance_id in valid_instance_ids
        ]
        document.external_ports = [
            item for item in document.external_ports
            if item.port_ref.instance_id in valid_instance_ids
        ]
        document.differential_ports = [
            item for item in document.differential_ports
            if item.port_ref_plus.instance_id in valid_instance_ids
            and item.port_ref_minus.instance_id in valid_instance_ids
        ]
        self._document = document
        self._rebuild_scene_from_document()
        self._sync_sweep_controls_from_document()
        self._scene.rebuild_export_state(self._document)
        # Refresh inspector / dependent lists that mirror document state.
        try:
            self._refresh_transient_source_list()
        except Exception:
            pass
        try:
            self._refresh_transient_output_list()
        except Exception:
            pass
        try:
            self._refresh_output_port_list()
        except Exception:
            pass
        self._refresh_validation_state()
        self._on_scene_selection_changed()

    def _rebuild_scene_from_document(self) -> None:
        self._scene.clear()
        self._scene._block_items.clear()
        self._scene._connection_items.clear()
        self._scene.cancel_routing()
        for instance in self._document.instances:
            self._scene.register_block(CircuitBlockItem(self._scene, instance))
        for connection in self._document.connections:
            port_item_a = self._port_item_for_ref(connection.port_a)
            port_item_b = self._port_item_for_ref(connection.port_b)
            if port_item_a is None or port_item_b is None:
                continue
            self._scene.register_connection(
                CircuitConnectionItem(
                    connection.connection_id,
                    port_item_a,
                    port_item_b,
                    connection.waypoints,
                )
            )

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