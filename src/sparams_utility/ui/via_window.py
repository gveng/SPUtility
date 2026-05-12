"""Via Analysis Window for S-Params Studio — 3D EMerge-script edition.

Provides a tabbed parameter UI (left panel) and an interactive 3D via
visualisation using pyqtgraph.opengl (right panel).  The window also
generates complete EMerge EM-simulation Python scripts.

Follows the same architectural conventions as the rest of the app:
  - emits project_modified on any parameter change
  - implements export_project_state / import_project_state
  - does NOT depend on sparams_utility.via_analysis
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QMessageBox

# ── Optional OpenGL ───────────────────────────────────────────────────────────
try:
    import pyqtgraph.opengl as gl
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False

# ── Default stackup ───────────────────────────────────────────────────────────
_DEFAULT_STACKUP = [
    {"name": "L1 (Signal)",  "thickness_um": 35.0,   "is_copper": True,  "role": "Signal",     "net": "",    "er": 1.0,  "tand": 0.0},
    {"name": "Core",         "thickness_um": 200.0,  "is_copper": False, "role": "Dielectric", "net": "",    "er": 3.76, "tand": 0.009},
    {"name": "L2 (GND)",     "thickness_um": 35.0,   "is_copper": True,  "role": "Plane",      "net": "GND", "er": 1.0,  "tand": 0.0},
    {"name": "Prepreg",      "thickness_um": 1000.0, "is_copper": False, "role": "Dielectric", "net": "",    "er": 4.2,  "tand": 0.02},
    {"name": "L3 (Power)",   "thickness_um": 35.0,   "is_copper": True,  "role": "Plane",      "net": "PWR", "er": 1.0,  "tand": 0.0},
    {"name": "Core",         "thickness_um": 200.0,  "is_copper": False, "role": "Dielectric", "net": "",    "er": 3.76, "tand": 0.009},
    {"name": "L4 (Signal)",  "thickness_um": 35.0,   "is_copper": True,  "role": "Signal",     "net": "",    "er": 1.0,  "tand": 0.0},
]

# ─────────────────────────────────────────────────────────────────────────────
#  Module-level 3-D mesh helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cyl_mesh(r: float, h: float, z0: float = 0.0, n: int = 32):
    """Solid cylinder mesh.  Returns (verts: float32 (N,3), faces: int32 (M,3))."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)

    # bottom ring (indices 0..n-1), top ring (n..2n-1)
    # bottom centre (2n), top centre (2n+1)
    bot_ring = np.column_stack([r * cos_a, r * sin_a, np.full(n, z0, np.float32)])
    top_ring = np.column_stack([r * cos_a, r * sin_a, np.full(n, z0 + h, np.float32)])
    bot_ctr  = np.array([[0.0, 0.0, z0]], dtype=np.float32)
    top_ctr  = np.array([[0.0, 0.0, z0 + h]], dtype=np.float32)
    verts    = np.vstack([bot_ring, top_ring, bot_ctr, top_ctr])

    faces = []
    for i in range(n):
        j = (i + 1) % n
        # side quad → 2 triangles
        faces.append([i,     j,     n + j])
        faces.append([i,     n + j, n + i])
        # bottom cap (fan from 2n)
        faces.append([2 * n, j,     i    ])
        # top cap   (fan from 2n+1)
        faces.append([2 * n + 1, n + i, n + j])

    return verts, np.array(faces, dtype=np.int32)


def _disc_mesh(r: float, z: float, n: int = 32):
    """Full disc at height z.  Returns (verts, faces)."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.column_stack([
        r * np.cos(angles).astype(np.float32),
        r * np.sin(angles).astype(np.float32),
        np.full(n, z, dtype=np.float32),
    ])
    ctr   = np.array([[0.0, 0.0, z]], dtype=np.float32)
    verts = np.vstack([ring, ctr])   # n+1 verts, centre = index n
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([n, i, j])
    return verts, np.array(faces, dtype=np.int32)


def _annulus_mesh(r_in: float, r_out: float, z: float, n: int = 48):
    """Flat annular ring.  Returns (verts, faces) or (None, None) if r_in >= r_out."""
    if r_in >= r_out:
        return None, None
    angles  = np.linspace(0, 2 * math.pi, n, endpoint=False)
    inner   = np.column_stack([
        r_in  * np.cos(angles).astype(np.float32),
        r_in  * np.sin(angles).astype(np.float32),
        np.full(n, z, dtype=np.float32),
    ])
    outer   = np.column_stack([
        r_out * np.cos(angles).astype(np.float32),
        r_out * np.sin(angles).astype(np.float32),
        np.full(n, z, dtype=np.float32),
    ])
    verts   = np.vstack([inner, outer])   # inner: 0..n-1, outer: n..2n-1
    faces   = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i,     n + i, n + j])
        faces.append([i,     n + j, j    ])
    return verts, np.array(faces, dtype=np.int32)


def _annular_cyl_mesh(r_in: float, r_out: float, h: float, z0: float, n: int = 48):
    """Hollow cylinder (annular column / tube).
    Returns (verts: float32, faces: int32) or (None, None) if geometry is degenerate."""
    if r_in >= r_out or h <= 0:
        return None, None
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    # Ring indices: inner-bottom=0..n-1, outer-bottom=n..2n-1, inner-top=2n..3n-1, outer-top=3n..4n-1
    ib = np.column_stack([r_in  * cos_a, r_in  * sin_a, np.full(n, z0,     np.float32)])
    ob = np.column_stack([r_out * cos_a, r_out * sin_a, np.full(n, z0,     np.float32)])
    it = np.column_stack([r_in  * cos_a, r_in  * sin_a, np.full(n, z0 + h, np.float32)])
    ot = np.column_stack([r_out * cos_a, r_out * sin_a, np.full(n, z0 + h, np.float32)])
    verts = np.vstack([ib, ob, it, ot])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        # bottom annulus
        faces += [[i, n + i, n + j], [i, n + j, j]]
        # top annulus
        faces += [[2*n + i, 2*n + j, 3*n + j], [2*n + i, 3*n + j, 3*n + i]]
        # inner wall
        faces += [[i, j, 2*n + j], [i, 2*n + j, 2*n + i]]
        # outer wall
        faces += [[n + i, 3*n + i, 3*n + j], [n + i, 3*n + j, n + j]]
    return verts.astype(np.float32), np.array(faces, dtype=np.int32)


def _cyl_wall_mesh(r: float, h: float, z0: float = 0.0, n: int = 48):
    """Cylinder side wall only (no top/bottom caps)."""
    if r <= 0 or h <= 0:
        return None, None
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    bot = np.column_stack([r * cos_a, r * sin_a, np.full(n, z0, np.float32)])
    top = np.column_stack([r * cos_a, r * sin_a, np.full(n, z0 + h, np.float32)])
    verts = np.vstack([bot, top])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    return verts, np.array(faces, dtype=np.int32)


def _box_mesh(hw: float, hd: float, z0: float, z1: float):
    """Rectangular box (±hw in X, ±hd in Y).  Returns (verts, faces)."""
    verts = np.array([
        [-hw, -hd, z0], [ hw, -hd, z0], [ hw,  hd, z0], [-hw,  hd, z0],
        [-hw, -hd, z1], [ hw, -hd, z1], [ hw,  hd, z1], [-hw,  hd, z1],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],   # bottom
        [4, 6, 5], [4, 7, 6],   # top
        [0, 1, 5], [0, 5, 4],   # front
        [2, 3, 7], [2, 7, 6],   # back
        [1, 2, 6], [1, 6, 5],   # right
        [3, 0, 4], [3, 4, 7],   # left
    ], dtype=np.int32)
    return verts, faces


def _perforated_plane_mesh(
    hw: float,
    hd: float,
    z0: float,
    z1: float,
    holes: list[tuple[float, float, float]],
    target_cell_mm: float = 0.04,
):
    """Rectangular copper slab with true cylindrical clearances approximated on XY grid.

    The resulting mesh is a real solid volume with removed material in hole regions,
    not an overlay marker. Hole sidewalls are generated by exposed side faces.
    """
    if z1 <= z0:
        return None, None
    if not holes:
        return _box_mesh(hw, hd, z0, z1)

    valid_r = [float(r) for _, _, r in holes if float(r) > 0.0]
    if valid_r:
        cell = max(0.01, min(0.08, min(min(valid_r) / 3.0, target_cell_mm)))
    else:
        cell = max(0.01, min(0.08, target_cell_mm))

    nx = max(16, int(math.ceil((2.0 * hw) / cell)))
    ny = max(16, int(math.ceil((2.0 * hd) / cell)))

    x_edges = np.linspace(-hw, hw, nx + 1, dtype=np.float32)
    y_edges = np.linspace(-hd, hd, ny + 1, dtype=np.float32)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xc, yc = np.meshgrid(x_centers, y_centers, indexing="ij")

    solid = np.ones((nx, ny), dtype=bool)
    for cx, cy, rr in holes:
        r = max(float(rr), 0.0)
        if r <= 0.0:
            continue
        solid &= ((xc - float(cx)) ** 2 + (yc - float(cy)) ** 2) >= (r * r)

    if not np.any(solid):
        return None, None

    verts: list[list[float]] = []
    faces: list[list[int]] = []

    def _add_quad(v0, v1, v2, v3):
        b = len(verts)
        verts.extend([v0, v1, v2, v3])
        faces.append([b, b + 1, b + 2])
        faces.append([b, b + 2, b + 3])

    for i in range(nx):
        x0 = float(x_edges[i])
        x1 = float(x_edges[i + 1])
        for j in range(ny):
            if not solid[i, j]:
                continue
            y0 = float(y_edges[j])
            y1 = float(y_edges[j + 1])

            # Top face (+Z)
            _add_quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1])
            # Bottom face (-Z)
            _add_quad([x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0])

            # X- side
            if i == 0 or not solid[i - 1, j]:
                _add_quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0])
            # X+ side
            if i == nx - 1 or not solid[i + 1, j]:
                _add_quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1])
            # Y- side
            if j == 0 or not solid[i, j - 1]:
                _add_quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1])
            # Y+ side
            if j == ny - 1 or not solid[i, j + 1]:
                _add_quad([x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0])

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _trace_box_mesh(length: float, width: float, z0: float, z1: float, angle_deg: float, start_offset: float = 0.0):
    """Oriented rectangular trace prism starting at origin (optionally offset) and extending along angle in XY."""
    l = max(0.001, float(length))
    w = max(0.001, float(width))
    s0 = float(start_offset)
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))

    local = np.array([
        [s0,       -w / 2.0],
        [s0 + l,   -w / 2.0],
        [s0 + l,    w / 2.0],
        [s0,        w / 2.0],
    ], dtype=np.float32)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = local @ rot.T

    verts = np.array([
        [xy[0, 0], xy[0, 1], z0], [xy[1, 0], xy[1, 1], z0],
        [xy[2, 0], xy[2, 1], z0], [xy[3, 0], xy[3, 1], z0],
        [xy[0, 0], xy[0, 1], z1], [xy[1, 0], xy[1, 1], z1],
        [xy[2, 0], xy[2, 1], z1], [xy[3, 0], xy[3, 1], z1],
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    return verts, faces


# ─────────────────────────────────────────────────────────────────────────────
#  ViaWindow
# ─────────────────────────────────────────────────────────────────────────────

class ViaWindow(QMainWindow):
    """Parametric via 3-D visualiser and EMerge script generator."""

    project_modified = Signal()
    simulation_completed = Signal(str)  # emitted with result .sNp path on success

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self, parent=None, window_number: int = 1) -> None:
        super().__init__(parent)
        self.window_number = window_number
        self.setWindowTitle(f"Via Analysis #{window_number}")
        self.resize(1500, 860)

        app = QApplication.instance()
        if app is not None:
            self.setWindowIcon(app.windowIcon())

        self._suppress_signals = False   # guard against feedback loops
        self._stitch_deleted_indices: set[int] = set()
        self._stitch_visible_base_indices: list[int] = []
        self._stitch_selected_row: int = -1
        self._hidden_3d_keys: set[str] = set()
        self._scene_group_items: dict[str, QTreeWidgetItem] = {}
        self._scene_object_items: dict[str, QTreeWidgetItem] = {}
        self._scene_gl_items: dict[str, list] = {}

        # ── build UI ──────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root_vbox = QVBoxLayout(central)
        root_vbox.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root_vbox.addWidget(splitter, 1)

        # ---- Left panel ----
        left_container = QWidget()
        left_container.setMinimumWidth(460)
        left_vbox = QVBoxLayout(left_container)
        left_vbox.setContentsMargins(0, 0, 0, 0)

        # ── Name row ─────────────────────────────────────────────────────
        name_row = QHBoxLayout()
        name_row.setContentsMargins(4, 4, 4, 2)
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit(f"Via Analysis #{window_number}")
        self._name_edit.setPlaceholderText("Via Analysis name")
        name_row.addWidget(self._name_edit, 1)
        left_vbox.addLayout(name_row)

        self._tabs = QTabWidget()
        self._tabs.tabBar().setUsesScrollButtons(False)
        self._tabs.tabBar().setExpanding(True)
        left_vbox.addWidget(self._tabs)

        self._build_tab_stackup()
        self._build_tab_via_geometry()
        self._build_tab_stitching()
        self._build_tab_feed()
        self._build_tab_simulation()
        self._build_tab_mesh()
        self._build_tab_script()

        # ---- Right panel ----
        right_container = QWidget()
        right_vbox = QVBoxLayout(right_container)
        right_vbox.setContentsMargins(2, 2, 2, 2)

        if _GL_AVAILABLE:
            self._gl_view = gl.GLViewWidget()
            self._gl_view.setCameraParams(distance=11, elevation=25, azimuth=45)
            right_vbox.addWidget(self._gl_view, 1)
            self._scene_tree = QTreeWidget()
            self._scene_tree.setHeaderLabel("3D Scene")
            self._scene_tree.setMinimumHeight(180)
            self._scene_tree.setContextMenuPolicy(Qt.CustomContextMenu)
            self._scene_tree.customContextMenuRequested.connect(self._on_scene_tree_context_menu)
            right_vbox.addWidget(self._scene_tree, 0)
        else:
            no_gl = QLabel("OpenGL not available.\nInstall PyOpenGL to enable 3D view.")
            no_gl.setAlignment(Qt.AlignCenter)
            right_vbox.addWidget(no_gl, 1)
            self._gl_view = None
            self._scene_tree = None

        # toolbar row
        toolbar_row = QHBoxLayout()
        self._btn_reset  = QPushButton("Reset View")
        self._btn_top    = QPushButton("Top ↑")
        self._btn_side   = QPushButton("Side →")
        self._btn_iso    = QPushButton("ISO")
        self._chk_autorefresh = QCheckBox("Auto-refresh")
        self._chk_autorefresh.setChecked(True)
        for btn in (self._btn_reset, self._btn_top, self._btn_side, self._btn_iso):
            toolbar_row.addWidget(btn)
        toolbar_row.addStretch(1)
        toolbar_row.addWidget(self._chk_autorefresh)
        right_vbox.addLayout(toolbar_row)

        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([480, 1020])

        # ── connect camera buttons ────────────────────────────────────────
        self._btn_reset.clicked.connect(self._cam_reset)
        self._btn_top.clicked.connect(self._cam_top)
        self._btn_side.clicked.connect(self._cam_side)
        self._btn_iso.clicked.connect(self._cam_iso)

        # ── name change ───────────────────────────────────────────────────
        self._name_edit.textChanged.connect(self._on_name_changed)

        # ── initial 3-D scene ─────────────────────────────────────────────
        if _GL_AVAILABLE:
            self._rebuild_3d()

    def _on_name_changed(self, text: str) -> None:
        title = text.strip() or f"Via Analysis #{self.window_number}"
        self.setWindowTitle(title)
        if not self._suppress_signals:
            self.project_modified.emit()

    # ── Tab 0 – Stackup ───────────────────────────────────────────────────

    def _build_tab_stackup(self):
        w = QWidget()
        vbox = QVBoxLayout(w)

        self._stackup_table = QTableWidget()
        self._stackup_table.setColumnCount(7)
        self._stackup_table.setHorizontalHeaderLabels(
            ["Layer Name", "Thick. (µm)", "Type", "Role", "Net", "εr", "tan δ"]
        )
        hdr = self._stackup_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setStretchLastSection(False)
        hdr.resizeSection(0, 165)
        hdr.resizeSection(1, 85)
        hdr.resizeSection(2, 95)
        hdr.resizeSection(3, 90)
        hdr.resizeSection(4, 60)
        hdr.resizeSection(5, 55)
        hdr.resizeSection(6, 55)
        self._stackup_table.setMinimumHeight(200)
        vbox.addWidget(self._stackup_table)

        btn_row = QHBoxLayout()
        self._btn_add_layer    = QPushButton("Add")
        self._btn_remove_layer = QPushButton("Remove")
        self._btn_move_up      = QPushButton("Move Up")
        self._btn_move_down    = QPushButton("Move Down")
        for b in (self._btn_add_layer, self._btn_remove_layer,
                  self._btn_move_up, self._btn_move_down):
            btn_row.addWidget(b)
        vbox.addLayout(btn_row)
        vbox.addStretch(1)

        self._btn_add_layer.clicked.connect(self._stackup_add)
        self._btn_remove_layer.clicked.connect(self._stackup_remove)
        self._btn_move_up.clicked.connect(self._stackup_move_up)
        self._btn_move_down.clicked.connect(self._stackup_move_down)

        self._tabs.addTab(w, "Stackup")

        # Populate default stackup
        self._load_stackup_rows(_DEFAULT_STACKUP)

    def _load_stackup_rows(self, rows: list[dict]):
        """Rebuild the stackup table from a list of dicts."""
        self._suppress_signals = True
        self._stackup_table.setRowCount(0)
        for row in rows:
            is_copper = row.get("is_copper", True)
            role = row.get("role")
            if not role:
                if not is_copper:
                    role = "Dielectric"
                elif any(tag in row.get("name", "").lower() for tag in ("gnd", "ground", "pwr", "power", "plane")):
                    role = "Plane"
                else:
                    role = "Signal"
            net = row.get("net", "")
            if not net and is_copper and role == "Plane":
                name_lower = row.get("name", "").lower()
                if any(tag in name_lower for tag in ("gnd", "ground")):
                    net = "GND"
                elif any(tag in name_lower for tag in ("pwr", "power", "vcc", "vdd")):
                    net = "PWR"
            self._stackup_insert_row(
                row["name"], row["thickness_um"], is_copper,
                role, net, row["er"], row["tand"]
            )
        self._suppress_signals = False
        self._stackup_changed()

    def _stackup_insert_row(self, name: str, thick: float,
                             is_copper: bool, role: str, net: str, er: float, tand: float,
                             at_row: Optional[int] = None):
        tbl = self._stackup_table
        row = tbl.rowCount() if at_row is None else at_row
        tbl.insertRow(row)

        tbl.setItem(row, 0, QTableWidgetItem(name))

        thick_item = QTableWidgetItem(str(thick))
        thick_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tbl.setItem(row, 1, thick_item)

        combo = QComboBox()
        combo.addItems(["Copper", "Dielectric"])
        combo.setCurrentIndex(0 if is_copper else 1)
        combo.currentIndexChanged.connect(self._stackup_changed)
        tbl.setCellWidget(row, 2, combo)

        role_combo = QComboBox()
        role_combo.addItems(["Signal", "Plane", "Dielectric"])
        role_index = role_combo.findText(role if role else ("Signal" if is_copper else "Dielectric"))
        role_combo.setCurrentIndex(role_index if role_index >= 0 else (0 if is_copper else 2))
        role_combo.currentIndexChanged.connect(self._stackup_changed)
        tbl.setCellWidget(row, 3, role_combo)

        net_item = QTableWidgetItem(net or "")
        tbl.setItem(row, 4, net_item)

        er_item = QTableWidgetItem(str(er))
        er_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tbl.setItem(row, 5, er_item)

        tand_item = QTableWidgetItem(str(tand))
        tand_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tbl.setItem(row, 6, tand_item)

        # Connect item changes
        tbl.itemChanged.connect(self._stackup_changed)

    def _read_stackup(self) -> list[dict]:
        """Read current stackup from the table widget."""
        tbl = self._stackup_table
        rows = []
        for r in range(tbl.rowCount()):
            name_item    = tbl.item(r, 0)
            thick_item   = tbl.item(r, 1)
            combo        = tbl.cellWidget(r, 2)
            role_combo   = tbl.cellWidget(r, 3)
            net_item     = tbl.item(r, 4)
            er_item      = tbl.item(r, 5)
            tand_item    = tbl.item(r, 6)

            name  = name_item.text()  if name_item  else ""
            try:
                thick = float(thick_item.text()) if thick_item else 35.0
            except ValueError:
                thick = 35.0
            is_copper = (combo.currentIndex() == 0) if combo else True
            role_text = role_combo.currentText().strip() if role_combo else ("Signal" if is_copper else "Dielectric")
            if not is_copper:
                role_text = "Dielectric"
            net_text = net_item.text().strip() if net_item else ""
            try:
                er    = float(er_item.text())   if er_item   else 1.0
            except ValueError:
                er    = 1.0
            try:
                tand  = float(tand_item.text()) if tand_item else 0.0
            except ValueError:
                tand  = 0.0

            rows.append({"name": name, "thickness_um": thick,
                         "is_copper": is_copper, "role": role_text, "net": net_text,
                         "er": er, "tand": tand})
        return rows

    def _copper_layer_names(self) -> list[str]:
        return [r["name"] for r in self._read_stackup() if r["is_copper"]]

    def _stackup_changed(self):
        if self._suppress_signals:
            return
        if hasattr(self, '_via_from_combo'):
            self._update_via_layer_combos()
        self.project_modified.emit()
        if hasattr(self, '_chk_autorefresh') and self._chk_autorefresh.isChecked() and _GL_AVAILABLE:
            self._rebuild_3d()

    def _stackup_add(self):
        self._stackup_insert_row("New Layer", 100.0, False, "Dielectric", "", 4.2, 0.02)
        self._stackup_changed()

    def _stackup_remove(self):
        row = self._stackup_table.currentRow()
        if row >= 0:
            self._stackup_table.removeRow(row)
            self._stackup_changed()

    def _stackup_move_up(self):
        tbl = self._stackup_table
        row = tbl.currentRow()
        if row <= 0:
            return
        self._swap_stackup_rows(row, row - 1)
        tbl.setCurrentCell(row - 1, tbl.currentColumn())
        self._stackup_changed()

    def _stackup_move_down(self):
        tbl = self._stackup_table
        row = tbl.currentRow()
        if row < 0 or row >= tbl.rowCount() - 1:
            return
        self._swap_stackup_rows(row, row + 1)
        tbl.setCurrentCell(row + 1, tbl.currentColumn())
        self._stackup_changed()

    def _swap_stackup_rows(self, a: int, b: int):
        """Swap two rows in the stackup table by reloading the normalized row data."""
        rows = self._read_stackup()
        rows[a], rows[b] = rows[b], rows[a]
        self._load_stackup_rows(rows)

    # ── Tab 1 – Via Geometry ──────────────────────────────────────────────

    def _build_tab_via_geometry(self):
        w = QWidget()
        form_widget = QGroupBox("Via Geometry")
        from PySide6.QtWidgets import QFormLayout
        form = QFormLayout(form_widget)

        self._drill_um   = self._mk_dspin(50,   5000,  250.0, " µm")
        self._pad_um     = self._mk_dspin(100, 10000,  500.0, " µm")
        self._antipad_um = self._mk_dspin(150, 20000,  800.0, " µm")
        form.addRow("Drill diameter (µm):", self._drill_um)
        form.addRow("Pad diameter (µm):",   self._pad_um)
        form.addRow("Antipad diameter (µm):", self._antipad_um)

        self._via_from_combo = QComboBox()
        self._via_to_combo   = QComboBox()
        form.addRow("Via From Layer:", self._via_from_combo)
        form.addRow("Via To Layer:",   self._via_to_combo)

        self._stub_combo = QComboBox()
        form.addRow("Stub extends to:", self._stub_combo)

        # Radio buttons
        radio_widget = QWidget()
        radio_hbox   = QHBoxLayout(radio_widget)
        radio_hbox.setContentsMargins(0, 0, 0, 0)
        self._radio_se   = QRadioButton("Single-ended")
        self._radio_diff = QRadioButton("Differential")
        self._radio_se.setChecked(True)
        radio_hbox.addWidget(self._radio_se)
        radio_hbox.addWidget(self._radio_diff)
        form.addRow("Mode:", radio_widget)

        self._diff_spacing = self._mk_dspin(100, 5000, 400.0, " µm")
        self._diff_spacing.setEnabled(False)
        form.addRow("Diff. spacing (µm):", self._diff_spacing)

        vbox = QVBoxLayout(w)
        vbox.addWidget(form_widget)
        vbox.addStretch(1)
        self._tabs.addTab(w, "Via Geometry")

        # Signals
        self._drill_um.valueChanged.connect(self._param_changed)
        self._pad_um.valueChanged.connect(self._param_changed)
        self._antipad_um.valueChanged.connect(self._param_changed)
        self._via_from_combo.currentIndexChanged.connect(self._via_from_changed)
        self._via_to_combo.currentIndexChanged.connect(self._via_to_changed)
        self._stub_combo.currentIndexChanged.connect(self._param_changed)
        self._radio_se.toggled.connect(self._diff_toggled)
        self._diff_spacing.valueChanged.connect(self._param_changed)

        # Initial populate
        self._update_via_layer_combos()

    def _update_via_layer_combos(self):
        self._suppress_signals = True
        copper_names = self._copper_layer_names()

        old_from = self._via_from_combo.currentText()
        old_to   = self._via_to_combo.currentText()
        old_stitch_from = self._stitch_from_combo.currentText() if hasattr(self, "_stitch_from_combo") else ""
        old_stitch_to = self._stitch_to_combo.currentText() if hasattr(self, "_stitch_to_combo") else ""

        self._via_from_combo.clear()
        self._via_from_combo.addItems(copper_names)
        self._via_to_combo.clear()
        self._via_to_combo.addItems(copper_names)

        # restore previous selections if still valid
        from_idx = self._via_from_combo.findText(old_from)
        if from_idx >= 0:
            self._via_from_combo.setCurrentIndex(from_idx)
        elif len(copper_names) > 0:
            self._via_from_combo.setCurrentIndex(0)

        to_idx = self._via_to_combo.findText(old_to)
        if to_idx >= 0:
            self._via_to_combo.setCurrentIndex(to_idx)
        elif len(copper_names) > 1:
            self._via_to_combo.setCurrentIndex(len(copper_names) - 1)

        if hasattr(self, "_stitch_from_combo") and hasattr(self, "_stitch_to_combo"):
            self._stitch_from_combo.clear()
            self._stitch_from_combo.addItems(copper_names)
            self._stitch_to_combo.clear()
            self._stitch_to_combo.addItems(copper_names)

            stitch_from_idx = self._stitch_from_combo.findText(old_stitch_from)
            if stitch_from_idx >= 0:
                self._stitch_from_combo.setCurrentIndex(stitch_from_idx)
            elif len(copper_names) > 0:
                self._stitch_from_combo.setCurrentIndex(0)

            stitch_to_idx = self._stitch_to_combo.findText(old_stitch_to)
            if stitch_to_idx >= 0:
                self._stitch_to_combo.setCurrentIndex(stitch_to_idx)
            elif len(copper_names) > 1:
                self._stitch_to_combo.setCurrentIndex(len(copper_names) - 1)

        self._suppress_signals = False
        self._update_stub_combo()

    def _update_stub_combo(self):
        self._suppress_signals = True
        old_stub = self._stub_combo.currentText()
        self._stub_combo.clear()
        self._stub_combo.addItem("None")

        copper_names = self._copper_layer_names()
        to_idx = self._via_to_combo.currentIndex()
        for name in copper_names[to_idx + 1:]:
            self._stub_combo.addItem(name)

        stub_idx = self._stub_combo.findText(old_stub)
        if stub_idx >= 0:
            self._stub_combo.setCurrentIndex(stub_idx)
        else:
            self._stub_combo.setCurrentIndex(0)
        self._suppress_signals = False

    def _via_from_changed(self):
        if not self._suppress_signals:
            self._update_stub_combo()
            self._param_changed()

    def _via_to_changed(self):
        if not self._suppress_signals:
            self._update_stub_combo()
            self._param_changed()

    def _diff_toggled(self, checked):
        self._diff_spacing.setEnabled(not checked)  # checked == SE → disable
        if hasattr(self, "_feed_port_controls"):
            self._feed_controls_changed()
        self._param_changed()

    # ── Tab 2 – Stitching ─────────────────────────────────────────────────

    def _build_tab_stitching(self):
        w = QWidget()
        grp = QGroupBox("Stitching Vias")
        from PySide6.QtWidgets import QFormLayout
        form = QFormLayout(grp)

        self._stitch_enable  = QCheckBox("Enable stitching")
        self._stitch_enable.setChecked(False)
        form.addRow(self._stitch_enable)

        self._stitch_pattern = QComboBox()
        self._stitch_pattern.addItems(["Ring", "Grid"])
        form.addRow("Pattern:", self._stitch_pattern)

        self._stitch_from_combo = QComboBox()
        self._stitch_to_combo = QComboBox()
        form.addRow("From copper layer:", self._stitch_from_combo)
        form.addRow("To copper layer:", self._stitch_to_combo)

        self._stitch_n  = QSpinBox()
        self._stitch_n.setRange(1, 64)
        self._stitch_n.setValue(8)
        form.addRow("N vias:", self._stitch_n)

        self._stitch_ring_r = self._mk_dspin(100, 20000, 2000.0, " µm")
        form.addRow("Ring radius (µm):", self._stitch_ring_r)

        self._stitch_drill = self._mk_dspin(50, 5000, 250.0, " µm")
        self._stitch_pad   = self._mk_dspin(100, 10000, 500.0, " µm")
        form.addRow("Stitching drill (µm):", self._stitch_drill)
        form.addRow("Stitching pad (µm):",   self._stitch_pad)

        vbox = QVBoxLayout(w)
        vbox.addWidget(grp)

        vbox.addWidget(QLabel("Stitching via coordinates (uncheck to exclude a via):"))
        self._stitch_coord_table = QTableWidget()
        self._stitch_coord_table.setColumnCount(4)
        self._stitch_coord_table.setHorizontalHeaderLabels(["\u2714", "#", "X (mm)", "Y (mm)"])
        self._stitch_coord_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._stitch_coord_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._stitch_coord_table.verticalHeader().setVisible(False)
        sh = self._stitch_coord_table.horizontalHeader()
        sh.setSectionResizeMode(QHeaderView.Interactive)
        sh.setStretchLastSection(False)
        sh.resizeSection(0, 32)
        sh.resizeSection(1, 38)
        sh.resizeSection(2, 115)
        sh.resizeSection(3, 115)
        self._stitch_coord_table.setAlternatingRowColors(True)
        self._stitch_coord_table.setMinimumHeight(170)
        vbox.addWidget(self._stitch_coord_table)

        vbox.addStretch(1)
        self._tabs.addTab(w, "Stitching")

        self._stitch_enable.stateChanged.connect(self._stitching_controls_changed)
        self._stitch_pattern.currentIndexChanged.connect(self._stitching_controls_changed)
        self._stitch_from_combo.currentIndexChanged.connect(self._stitching_controls_changed)
        self._stitch_to_combo.currentIndexChanged.connect(self._stitching_controls_changed)
        self._stitch_n.valueChanged.connect(self._stitching_controls_changed)
        self._stitch_ring_r.valueChanged.connect(self._stitching_controls_changed)
        self._stitch_drill.valueChanged.connect(self._stitching_controls_changed)
        self._stitch_pad.valueChanged.connect(self._stitching_controls_changed)
        self._stitch_coord_table.itemChanged.connect(self._on_stitch_checkbox_changed)

        self._update_via_layer_combos()
        self._refresh_stitching_coords_table()

    # ── Tab 3 – Simulation ────────────────────────────────────────────────

    def _build_tab_feed(self):
        w = QWidget()
        from PySide6.QtWidgets import QFormLayout

        self._feed_port_groups: dict[int, QGroupBox] = {}
        self._feed_port_controls: dict[int, dict[str, QDoubleSpinBox | QComboBox]] = {}

        port_specs = [
            (1, "Port 1 (Input)", 180.0),
            (2, "Port 2 (Output)", 0.0),
            (3, "Port 3 (Output)", 0.0),
            (4, "Port 4 (Output)", 0.0),
        ]

        vbox = QVBoxLayout(w)
        for port_idx, title, angle_default in port_specs:
            grp = QGroupBox(f"{title} Feed")
            form = QFormLayout(grp)

            feed_type = QComboBox()
            feed_type.addItems(["Coaxial", "Trace"])
            feed_type.setCurrentIndex(1)
            trace_w_um = self._mk_dspin(25.0, 10000.0, 250.0, " µm")
            trace_l_um = self._mk_dspin(50.0, 50000.0, 2000.0, " µm")
            trace_ang_deg = self._mk_dspin(-180.0, 180.0, angle_default, "°", decimals=1)

            form.addRow("Feed type:", feed_type)
            form.addRow("Trace width (µm):", trace_w_um)
            form.addRow("Trace length (µm):", trace_l_um)
            form.addRow("Trace angle (deg):", trace_ang_deg)

            self._feed_port_groups[port_idx] = grp
            self._feed_port_controls[port_idx] = {
                "type": feed_type,
                "trace_width_um": trace_w_um,
                "trace_length_um": trace_l_um,
                "trace_angle_deg": trace_ang_deg,
            }

            feed_type.currentIndexChanged.connect(self._feed_controls_changed)
            trace_w_um.valueChanged.connect(self._param_changed)
            trace_l_um.valueChanged.connect(self._param_changed)
            trace_ang_deg.valueChanged.connect(self._param_changed)

            vbox.addWidget(grp)

        # Legacy aliases used by existing preview/script/state code paths.
        self._feed_start_type = self._feed_port_controls[1]["type"]
        self._feed_start_trace_width_um = self._feed_port_controls[1]["trace_width_um"]
        self._feed_start_trace_length_um = self._feed_port_controls[1]["trace_length_um"]
        self._feed_start_trace_angle_deg = self._feed_port_controls[1]["trace_angle_deg"]
        self._feed_end_type = self._feed_port_controls[2]["type"]
        self._feed_end_trace_width_um = self._feed_port_controls[2]["trace_width_um"]
        self._feed_end_trace_length_um = self._feed_port_controls[2]["trace_length_um"]
        self._feed_end_trace_angle_deg = self._feed_port_controls[2]["trace_angle_deg"]

        hint = QLabel(
            "Set angle to steer the trace away from stitching vias. "
            "Differential mode uses port numbering: inputs 1,2 and outputs 3,4."
        )
        hint.setWordWrap(True)

        vbox.addWidget(hint)
        vbox.addStretch(1)
        self._tabs.addTab(w, "Feed")

        self._feed_controls_changed()

    def _active_feed_control_ports(self) -> list[int]:
        if self._radio_diff.isChecked():
            return [1, 2, 3, 4]
        return [1, 2]

    def _refresh_feed_group_titles(self) -> None:
        diff_enabled = self._radio_diff.isChecked()
        title_by_port = {
            1: "Port 1 (Input) Feed",
            2: "Port 2 (Input) Feed" if diff_enabled else "Port 2 (Output) Feed",
            3: "Port 3 (Output) Feed",
            4: "Port 4 (Output) Feed",
        }
        for port_idx, grp in self._feed_port_groups.items():
            grp.setTitle(title_by_port.get(port_idx, f"Port {port_idx} Feed"))

    def _feed_port_center_y_mm(self, port_idx: int) -> float:
        if port_idx in (2, 4) and self._radio_diff.isChecked():
            return self._diff_spacing.value() / 1000.0
        return 0.0

    def _get_feed_port_config(self, port_idx: int) -> dict[str, float | str]:
        controls = self._feed_port_controls.get(port_idx, self._feed_port_controls[1])
        return {
            "type": controls["type"].currentText(),
            "trace_width_um": controls["trace_width_um"].value(),
            "trace_length_um": controls["trace_length_um"].value(),
            "trace_angle_deg": controls["trace_angle_deg"].value(),
        }

    def _set_feed_port_config(self, port_idx: int, cfg: dict) -> None:
        controls = self._feed_port_controls.get(port_idx)
        if controls is None:
            return
        feed_type_idx = controls["type"].findText(str(cfg.get("type", "Trace")))
        if feed_type_idx >= 0:
            controls["type"].setCurrentIndex(feed_type_idx)
        controls["trace_width_um"].setValue(float(cfg.get("trace_width_um", 250.0)))
        controls["trace_length_um"].setValue(float(cfg.get("trace_length_um", 2000.0)))
        controls["trace_angle_deg"].setValue(float(cfg.get("trace_angle_deg", 0.0)))

    def _feed_controls_changed(self, *args):
        diff_enabled = self._radio_diff.isChecked()
        self._refresh_feed_group_titles()
        for port_idx, controls in self._feed_port_controls.items():
            port_enabled = diff_enabled or (port_idx in (1, 2))
            group = self._feed_port_groups.get(port_idx)
            if group is not None:
                group.setVisible(diff_enabled or (port_idx in (1, 2)))
                group.setEnabled(port_enabled)

            is_trace = controls["type"].currentText().strip().lower() == "trace"
            controls["trace_width_um"].setEnabled(port_enabled and is_trace)
            controls["trace_length_um"].setEnabled(port_enabled and is_trace)
            controls["trace_angle_deg"].setEnabled(port_enabled and is_trace)
        self._param_changed()

    # ── Tab 4 – Simulation ────────────────────────────────────────────────

    def _build_tab_simulation(self):
        w = QWidget()
        grp = QGroupBox("Simulation Parameters")
        fit_grp = QGroupBox("S-Parameter Fitting")
        from PySide6.QtWidgets import QFormLayout
        form = QFormLayout(grp)
        fit_form = QFormLayout(fit_grp)

        self._f_start  = self._mk_dspin(0.001, 100.0,  0.01, " GHz", decimals=3)
        self._f_stop   = self._mk_dspin(0.01,  200.0, 10.0,  " GHz", decimals=2)
        self._n_pts    = QSpinBox(); self._n_pts.setRange(3, 10001); self._n_pts.setValue(11)
        self._n_workers = QSpinBox(); self._n_workers.setRange(1, 32); self._n_workers.setValue(4)
        self._sparam_fit_enable = QCheckBox("Enable fitting to arbitrary points")
        self._sparam_fit_enable.setChecked(False)
        self._sparam_fit_n_pts = QSpinBox(); self._sparam_fit_n_pts.setRange(3, 50001); self._sparam_fit_n_pts.setValue(401)
        self._sparam_fit_n_pts.setEnabled(False)

        form.addRow("F start (GHz):", self._f_start)
        form.addRow("F stop (GHz):",  self._f_stop)
        form.addRow("N points:",      self._n_pts)
        form.addRow("N workers:",     self._n_workers)

        fit_form.addRow(self._sparam_fit_enable)
        fit_form.addRow("Fitted points:", self._sparam_fit_n_pts)

        vbox = QVBoxLayout(w)
        vbox.addWidget(grp)
        vbox.addWidget(fit_grp)
        vbox.addStretch(1)
        self._tabs.addTab(w, "Simulation")

        self._f_start.valueChanged.connect(self._param_changed)
        self._f_stop.valueChanged.connect(self._param_changed)
        self._n_pts.valueChanged.connect(self._param_changed)
        self._n_workers.valueChanged.connect(self._param_changed)
        self._sparam_fit_enable.toggled.connect(self._simulation_controls_changed)
        self._sparam_fit_n_pts.valueChanged.connect(self._param_changed)

    # ── Tab 5 – Mesh ─────────────────────────────────────────────────────

    def _build_tab_mesh(self):
        w = QWidget()
        from PySide6.QtWidgets import QFormLayout

        global_grp = QGroupBox("Global Mesh")
        global_form = QFormLayout(global_grp)
        self._res_mm = self._mk_dspin(0.01, 2.0, 0.25, "", decimals=2)
        global_form.addRow("Mesh resolution (fraction of lambda max):", self._res_mm)

        local_grp = QGroupBox("Local Mesh Subdivision")
        local_form = QFormLayout(local_grp)
        self._mesh_local_enable = QCheckBox("Enable local subdivision factors")
        self._mesh_local_enable.setChecked(False)
        local_form.addRow(self._mesh_local_enable)

        self._mesh_factor_sliders: dict[str, QSlider] = {}
        self._mesh_factor_labels: dict[str, QLabel] = {}
        mesh_specs = [
            ("via", "Via"),
            ("ports", "Ports"),
            ("feed", "Feed"),
            ("planes", "Planes"),
            ("stitching", "Stitching"),
        ]
        for mesh_key, mesh_label in mesh_specs:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(1, 16)
            slider.setSingleStep(1)
            slider.setPageStep(1)
            slider.setValue(1)
            slider.setTickInterval(1)
            slider.setTickPosition(QSlider.TicksBelow)
            value_lbl = QLabel("x1")
            value_lbl.setMinimumWidth(32)
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(value_lbl)
            local_form.addRow(f"{mesh_label} factor:", row_widget)
            self._mesh_factor_sliders[mesh_key] = slider
            self._mesh_factor_labels[mesh_key] = value_lbl
            slider.valueChanged.connect(self._mesh_controls_changed)

        hint = QLabel(
            "Higher factor means finer local mesh. Effective local target is Mesh Resolution / Factor."
        )
        hint.setWordWrap(True)

        vbox = QVBoxLayout(w)
        vbox.addWidget(global_grp)
        vbox.addWidget(local_grp)
        vbox.addWidget(hint)
        vbox.addStretch(1)
        self._tabs.addTab(w, "Mesh")

        self._res_mm.valueChanged.connect(self._param_changed)
        self._mesh_local_enable.toggled.connect(self._mesh_controls_changed)
        self._mesh_controls_changed()

    def _simulation_controls_changed(self, *args):
        self._sparam_fit_n_pts.setEnabled(self._sparam_fit_enable.isChecked())
        self._param_changed()

    def _mesh_controls_changed(self, *args):
        enabled = self._mesh_local_enable.isChecked()
        for mesh_key, slider in self._mesh_factor_sliders.items():
            slider.setEnabled(enabled)
            value = slider.value()
            if mesh_key in self._mesh_factor_labels:
                self._mesh_factor_labels[mesh_key].setText(f"x{value}")
                self._mesh_factor_labels[mesh_key].setEnabled(enabled)
        self._param_changed()

    # ── Tab 6 – Script ────────────────────────────────────────────────────

    def _build_tab_script(self):
        w = QWidget()
        vbox = QVBoxLayout(w)

        options_row = QHBoxLayout()
        self._show_structure_in_emerge = QCheckBox("Show structure in EMerge")
        self._show_structure_in_emerge.setChecked(True)
        self._show_labels_in_emerge = QCheckBox("Show labels")
        self._show_labels_in_emerge.setChecked(False)
        self._show_labels_in_emerge.setToolTip("Add plot_labels=True to m.view() (valid only when Show structure is enabled)")
        self._show_mesh_in_emerge = QCheckBox("Show mesh in EMerge")
        self._show_mesh_in_emerge.setChecked(True)
        options_row.addWidget(self._show_structure_in_emerge)
        options_row.addWidget(self._show_labels_in_emerge)
        options_row.addWidget(self._show_mesh_in_emerge)
        options_row.addStretch(1)
        vbox.addLayout(options_row)

        self._script_edit = QPlainTextEdit()
        self._script_edit.setReadOnly(True)
        mono = QFont("Courier New", 9)
        mono.setStyleHint(QFont.Monospace)
        self._script_edit.setFont(mono)
        vbox.addWidget(self._script_edit, 2)

        btn_row = QHBoxLayout()
        self._btn_gen_script  = QPushButton("Generate Script")
        self._btn_save_script = QPushButton("Save Script...")
        self._btn_copy_script = QPushButton("Copy")
        self._btn_run_emerge  = QPushButton("▶  Run EMerge")
        self._btn_run_emerge.setToolTip(
            "Salva lo script nella cartella <Progetto>_ViaAnalyzer ed esegue EMerge"
        )
        for b in (self._btn_gen_script, self._btn_save_script,
                  self._btn_copy_script, self._btn_run_emerge):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        vbox.addLayout(btn_row)

        # Output console (stdout/stderr of the EMerge process)
        vbox.addWidget(QLabel("Output EMerge:"))
        self._output_edit = QPlainTextEdit()
        self._output_edit.setReadOnly(True)
        out_mono = QFont("Courier New", 9)
        out_mono.setStyleHint(QFont.Monospace)
        self._output_edit.setFont(out_mono)
        self._output_edit.setMaximumBlockCount(5000)
        vbox.addWidget(self._output_edit, 1)

        # Stop button (initially hidden)
        self._btn_stop_emerge = QPushButton("■  Stop")
        self._btn_stop_emerge.setVisible(False)
        vbox.addWidget(self._btn_stop_emerge)

        self._tabs.addTab(w, "Script")

        self._btn_gen_script.clicked.connect(self._on_gen_script)
        self._btn_save_script.clicked.connect(self._on_save_script)
        self._btn_copy_script.clicked.connect(self._on_copy_script)
        self._btn_run_emerge.clicked.connect(self._on_run_emerge)
        self._btn_stop_emerge.clicked.connect(self._on_stop_emerge)
        self._show_structure_in_emerge.stateChanged.connect(self._param_changed)
        self._show_labels_in_emerge.stateChanged.connect(self._param_changed)
        self._show_mesh_in_emerge.stateChanged.connect(self._param_changed)

        self._emerge_process: QProcess | None = None

    # ── Parameter-change plumbing ─────────────────────────────────────────

    def _param_changed(self):
        if self._suppress_signals:
            return
        self.project_modified.emit()
        if hasattr(self, "_chk_autorefresh") and self._chk_autorefresh.isChecked() and _GL_AVAILABLE:
            self._rebuild_3d()

    def _stitching_controls_changed(self, *args):
        self._stitch_selected_row = -1
        self._refresh_stitching_coords_table()
        self._param_changed()

    def _compute_raw_stitch_coords(self) -> list[tuple[float, float]]:
        """Return ALL raw stitching-via XY coordinates (no enable/disable filtering)."""
        n_s = max(0, self._stitch_n.value())
        if n_s == 0:
            return []

        pattern = self._stitch_pattern.currentText().strip().lower()
        ring_r = max(0.001, self._stitch_ring_r.value() / 1000.0)

        if pattern == "grid":
            if n_s == 1:
                return [(ring_r, 0.0)]

            side = max(2, int(math.ceil(math.sqrt(n_s))))
            axis = np.linspace(-ring_r, ring_r, side)

            center_excl = max(
                self._pad_um.value() / 2000.0 + self._stitch_pad.value() / 2000.0,
                self._drill_um.value() / 1000.0,
            )

            candidates: list[tuple[float, float, float]] = []
            for y in axis:
                for x in axis:
                    xf = float(x)
                    yf = float(y)
                    candidates.append((xf, yf, math.hypot(xf, yf)))

            filtered = [p for p in candidates if p[2] >= center_excl]
            if len(filtered) < n_s:
                filtered = candidates

            filtered.sort(key=lambda p: (-p[2], p[1], p[0]))
            raw = [(x, y) for x, y, _ in filtered[:n_s]]
            raw.sort(key=lambda p: (p[1], p[0]))
            return raw

        raw = []
        for k in range(n_s):
            angle = 2.0 * math.pi * k / n_s
            raw.append((ring_r * math.cos(angle), ring_r * math.sin(angle)))
        return raw

    def _stitching_coordinates_mm(self) -> list[tuple[float, float]]:
        """Return active (enabled) stitching-via XY coordinates in mm."""
        raw = self._compute_raw_stitch_coords()
        # Clamp disabled-set to valid range
        self._stitch_deleted_indices = {i for i in self._stitch_deleted_indices if i < len(raw)}
        self._stitch_visible_base_indices = list(range(len(raw)))
        return [raw[i] for i in range(len(raw)) if i not in self._stitch_deleted_indices]

    def _on_stitch_checkbox_changed(self, item: "QTableWidgetItem"):
        """Toggle a via on/off when the user clicks the checkbox in col 0."""
        if item.column() != 0:
            return
        if not hasattr(self, "_stitch_coord_table"):
            return
        row = item.row()
        enabled = item.checkState() == Qt.Checked
        if enabled:
            self._stitch_deleted_indices.discard(row)
        else:
            self._stitch_deleted_indices.add(row)
        self._param_changed()
        if _GL_AVAILABLE and self._gl_view is not None:
            self._rebuild_3d()

    def _refresh_stitching_coords_table(self):
        if not hasattr(self, "_stitch_coord_table"):
            return

        raw = self._compute_raw_stitch_coords()
        # Clamp disabled set
        self._stitch_deleted_indices = {i for i in self._stitch_deleted_indices if i < len(raw)}
        self._stitch_visible_base_indices = list(range(len(raw)))

        tbl = self._stitch_coord_table
        tbl.blockSignals(True)
        tbl.setRowCount(len(raw))

        for i, (x_mm, y_mm) in enumerate(raw):
            chk_item = QTableWidgetItem()
            chk_item.setCheckState(Qt.Unchecked if i in self._stitch_deleted_indices else Qt.Checked)
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setTextAlignment(Qt.AlignCenter)

            idx_item = QTableWidgetItem(str(i + 1))
            x_item   = QTableWidgetItem(f"{x_mm:+.4f}")
            y_item   = QTableWidgetItem(f"{y_mm:+.4f}")

            idx_item.setTextAlignment(Qt.AlignCenter)
            x_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            y_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            for it in (idx_item, x_item, y_item):
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)

            tbl.setItem(i, 0, chk_item)
            tbl.setItem(i, 1, idx_item)
            tbl.setItem(i, 2, x_item)
            tbl.setItem(i, 3, y_item)

        tbl.blockSignals(False)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(value, high))

    def _simulation_domain_geometry(self) -> tuple[float, float, float, float]:
        """Return domain sizing in mm.

        Returns:
            half_extent_mm, boundary_pad_mm, outer_metal_radius_mm, dref_mm
        """
        drill_d_mm = self._drill_um.value() / 1000.0
        pad_d_mm = self._pad_um.value() / 1000.0
        pad_r_mm = pad_d_mm / 2.0

        dref_mm = max(drill_d_mm, 0.5 * pad_d_mm)

        # Outermost metal radius from center (signal via at origin)
        outer_r = pad_r_mm

        if self._radio_diff.isChecked():
            diff_mm = self._diff_spacing.value() / 1000.0
            outer_r = max(outer_r, abs(diff_mm) + pad_r_mm)

        if hasattr(self, "_feed_port_controls"):
            for port_idx in self._active_feed_control_ports():
                cfg = self._get_feed_port_config(port_idx)
                if str(cfg.get("type", "trace")).strip().lower() != "trace":
                    continue
                feed_len_mm = float(cfg.get("trace_length_um", 2000.0)) / 1000.0
                feed_w_mm = float(cfg.get("trace_width_um", 250.0)) / 1000.0
                y_off_mm = abs(self._feed_port_center_y_mm(port_idx))
                outer_r = max(outer_r, y_off_mm + feed_len_mm + 0.5 * feed_w_mm + pad_r_mm)

        stitching_enabled = self._stitch_enable.isChecked()
        if stitching_enabled:
            stitch_pad_r_mm = (self._stitch_pad.value() / 1000.0) / 2.0
            coords = self._stitching_coordinates_mm()
            if coords:
                outer_stitch_r = max(math.hypot(x, y) for x, y in coords) + stitch_pad_r_mm
                outer_r = max(outer_r, outer_stitch_r)

            c_def = max(2.5 * dref_mm, 0.5)
            c_min = max(2.0 * dref_mm, 0.4)
            c_max = max(4.0 * dref_mm, 1.5)
            boundary_pad_mm = self._clamp(c_def, c_min, c_max)
        else:
            c_def = max(4.5 * dref_mm, 0.8)
            c_min = max(3.5 * dref_mm, 0.6)
            c_max = max(6.0 * dref_mm, 2.0)
            boundary_pad_mm = self._clamp(c_def, c_min, c_max)

        half_extent_mm = max(outer_r + boundary_pad_mm, 1.5)
        return half_extent_mm, boundary_pad_mm, outer_r, dref_mm

    # ── Camera helpers ────────────────────────────────────────────────────

    def _cam_reset(self):
        if self._gl_view:
            self._gl_view.setCameraParams(distance=11, elevation=25, azimuth=45)

    def _cam_top(self):
        if self._gl_view:
            self._gl_view.setCameraParams(elevation=90, azimuth=0)

    def _cam_side(self):
        if self._gl_view:
            self._gl_view.setCameraParams(elevation=0, azimuth=0)

    def _cam_iso(self):
        if self._gl_view:
            self._gl_view.setCameraParams(distance=11, elevation=25, azimuth=45)

    def _begin_scene_tree_build(self):
        self._scene_group_items = {}
        self._scene_object_items = {}
        self._scene_gl_items = {}
        if self._scene_tree is not None:
            self._scene_tree.clear()

    def _ensure_scene_group_item(self, group_name: str) -> QTreeWidgetItem | None:
        if self._scene_tree is None:
            return None
        if group_name in self._scene_group_items:
            return self._scene_group_items[group_name]
        item = QTreeWidgetItem([group_name])
        item.setData(0, Qt.UserRole, "group")
        item.setData(0, Qt.UserRole + 1, group_name)
        self._scene_tree.addTopLevelItem(item)
        self._scene_group_items[group_name] = item
        return item

    def _set_scene_key_visibility(self, key: str, visible: bool):
        for gi in self._scene_gl_items.get(key, []):
            try:
                gi.setVisible(visible)
            except Exception:
                pass
        if visible:
            self._hidden_3d_keys.discard(key)
        else:
            self._hidden_3d_keys.add(key)
        t_item = self._scene_object_items.get(key)
        if t_item is not None:
            base = t_item.data(0, Qt.UserRole + 2) or t_item.text(0)
            t_item.setText(0, base if visible else f"{base} [hidden]")

    def _register_scene_mesh(self, group_name: str, key: str, label: str, mesh_item):
        if self._gl_view is None:
            return
        self._gl_view.addItem(mesh_item)
        self._scene_gl_items.setdefault(key, []).append(mesh_item)

        if self._scene_tree is not None:
            if key not in self._scene_object_items:
                parent = self._ensure_scene_group_item(group_name)
                if parent is not None:
                    node = QTreeWidgetItem([label])
                    node.setData(0, Qt.UserRole, "object")
                    node.setData(0, Qt.UserRole + 1, key)
                    node.setData(0, Qt.UserRole + 2, label)
                    parent.addChild(node)
                    self._scene_object_items[key] = node

        self._set_scene_key_visibility(key, key not in self._hidden_3d_keys)

    def _on_scene_tree_context_menu(self, pos):
        if self._scene_tree is None:
            return
        item = self._scene_tree.itemAt(pos)
        if item is None:
            return

        node_type = item.data(0, Qt.UserRole)
        menu = QMenu(self)

        if node_type == "object":
            key = item.data(0, Qt.UserRole + 1)
            if not key:
                return
            act_hide = menu.addAction("Hide")
            act_show = menu.addAction("Show")
            act_toggle = menu.addAction("Toggle")
            chosen = menu.exec(self._scene_tree.viewport().mapToGlobal(pos))
            if chosen == act_hide:
                self._set_scene_key_visibility(key, False)
            elif chosen == act_show:
                self._set_scene_key_visibility(key, True)
            elif chosen == act_toggle:
                self._set_scene_key_visibility(key, key in self._hidden_3d_keys)
            return

        if node_type == "group":
            group_name = item.data(0, Qt.UserRole + 1)
            if not group_name:
                return
            act_hide = menu.addAction("Hide Group")
            act_show = menu.addAction("Show Group")
            act_show_all = menu.addAction("Show All")
            chosen = menu.exec(self._scene_tree.viewport().mapToGlobal(pos))
            if chosen == act_show_all:
                for k in list(self._scene_object_items.keys()):
                    self._set_scene_key_visibility(k, True)
                return
            keys = [
                child.data(0, Qt.UserRole + 1)
                for child_idx in range(item.childCount())
                for child in [item.child(child_idx)]
                if child is not None and child.data(0, Qt.UserRole) == "object"
            ]
            if chosen == act_hide:
                for key in keys:
                    self._set_scene_key_visibility(key, False)
            elif chosen == act_show:
                for key in keys:
                    self._set_scene_key_visibility(key, True)

    def _layer_color(self, layer_index: int, is_copper: bool):
        """Return a visually distinct RGBA color for each layer."""
        copper_palette = [
            (0.95, 0.72, 0.18, 0.88),
            (0.91, 0.58, 0.14, 0.88),
            (0.98, 0.80, 0.26, 0.88),
            (0.84, 0.52, 0.12, 0.88),
            (0.93, 0.66, 0.22, 0.88),
        ]
        dielectric_palette = [
            (0.18, 0.50, 0.66, 0.38),
            (0.24, 0.58, 0.34, 0.38),
            (0.42, 0.42, 0.72, 0.38),
            (0.60, 0.38, 0.22, 0.36),
            (0.30, 0.46, 0.56, 0.36),
        ]
        palette = copper_palette if is_copper else dielectric_palette
        return palette[layer_index % len(palette)]

    # ── 3-D scene builder ─────────────────────────────────────────────────

    def _rebuild_3d(self):
        if not _GL_AVAILABLE or self._gl_view is None:
            return

        # Clear all items
        for item in list(self._gl_view.items):
            self._gl_view.removeItem(item)
        self._begin_scene_tree_build()

        # Grid + axis
        grid = gl.GLGridItem()
        grid.scale(0.5, 0.5, 0.5)
        self._register_scene_mesh("Helpers", "helpers/grid", "Grid", grid)

        stackup = self._read_stackup()
        if not stackup:
            return

        # Compute cumulative Z (mm, top = 0, increasing downward = negative Z in 3D)
        # We'll put layer tops at increasing Z (bottom of board at z=0, top at total_h_mm)
        hw, _, _, _ = self._simulation_domain_geometry()
        hd = hw

        z_cur = 0.0
        layer_z0 = []   # bottom Z of each layer
        layer_z1 = []   # top Z of each layer
        for r in stackup:
            h = r["thickness_um"] / 1000.0
            layer_z0.append(z_cur)
            layer_z1.append(z_cur + h)
            z_cur += h

        copper_layers = [(i, r) for i, r in enumerate(stackup) if r["is_copper"]]
        if not copper_layers:
            return

        via_from_idx_c = self._via_from_combo.currentIndex()
        via_to_idx_c   = self._via_to_combo.currentIndex()
        if via_from_idx_c < 0: via_from_idx_c = 0
        if via_to_idx_c   < 0: via_to_idx_c   = len(copper_layers) - 1

        via_from_idx_c = min(via_from_idx_c, len(copper_layers) - 1)
        via_to_idx_c   = min(via_to_idx_c,   len(copper_layers) - 1)

        from_stack_idx = copper_layers[via_from_idx_c][0]
        to_stack_idx   = copper_layers[via_to_idx_c][0]

        if from_stack_idx > to_stack_idx:
            from_stack_idx, to_stack_idx = to_stack_idx, from_stack_idx

        stitch_from_idx_c = self._stitch_from_combo.currentIndex() if hasattr(self, "_stitch_from_combo") else 0
        stitch_to_idx_c = self._stitch_to_combo.currentIndex() if hasattr(self, "_stitch_to_combo") else len(copper_layers) - 1
        if stitch_from_idx_c < 0:
            stitch_from_idx_c = 0
        if stitch_to_idx_c < 0:
            stitch_to_idx_c = len(copper_layers) - 1
        stitch_from_idx_c = min(stitch_from_idx_c, len(copper_layers) - 1)
        stitch_to_idx_c = min(stitch_to_idx_c, len(copper_layers) - 1)
        if stitch_from_idx_c > stitch_to_idx_c:
            stitch_from_idx_c, stitch_to_idx_c = stitch_to_idx_c, stitch_from_idx_c

        stitch_from_stack_idx = copper_layers[stitch_from_idx_c][0]
        stitch_to_stack_idx = copper_layers[stitch_to_idx_c][0]

        signal_landing_layers = {from_stack_idx, to_stack_idx}

        via_z_top = layer_z0[from_stack_idx]
        via_z_bot = layer_z1[to_stack_idx]

        drill_r  = self._drill_um.value()  / 2.0 / 1000.0   # mm
        pad_r    = self._pad_um.value()    / 2.0 / 1000.0
        apad_r   = self._antipad_um.value()/ 2.0 / 1000.0

        diff_offset_mm = self._diff_spacing.value() / 1000.0 if self._radio_diff.isChecked() else 0.0
        signal_via_centers = [(0.0, 0.0)]
        if self._radio_diff.isChecked():
            signal_via_centers.append((0.0, diff_offset_mm))

        stub_text = self._stub_combo.currentText()
        stub_stack_idx = None
        if stub_text and stub_text != "None":
            stub_stack_idx = next((i for i, r in enumerate(stackup) if r["name"] == stub_text), None)
            if stub_stack_idx is not None and stub_stack_idx <= to_stack_idx:
                stub_stack_idx = None

        def _layer_antipad_radius_mm(layer_index: int, min_radius_mm: float) -> float:
            # Via Geometry antipad is the only source of truth for 3D antipad holes.
            return max(apad_r, min_radius_mm)

        plane_holes_by_layer: dict[int, list[tuple[float, float, float]]] = {}
        for i, r in enumerate(stackup):
            if not (r["is_copper"] and str(r.get("role", "Signal")) == "Plane"):
                continue
            layer_holes: list[tuple[float, float, float]] = []
            in_signal_span = from_stack_idx < i < to_stack_idx
            in_stub_span = stub_stack_idx is not None and to_stack_idx < i <= stub_stack_idx
            if in_signal_span or in_stub_span:
                hole_r = _layer_antipad_radius_mm(i, drill_r)
                for cx, cy in signal_via_centers:
                    layer_holes.append((cx, cy, hole_r))

            if self._stitch_enable.isChecked() and stitch_from_stack_idx < i < stitch_to_stack_idx:
                s_drill = self._stitch_drill.value() / 2.0 / 1000.0
                # Plane layers: drill-size hole only so barrel sits flush (connected, no gap)
                for sx, sy in self._stitching_coordinates_mm():
                    layer_holes.append((sx, sy, s_drill))

            if layer_holes:
                plane_holes_by_layer[i] = layer_holes

        # Draw all layers as rectangular boxes. Antipad cavities on plane layers
        # are shown separately and centered on the via coordinates.
        for i, r in enumerate(stackup):
            if r["is_copper"] and i in signal_landing_layers:
                # Feed layers must show only landing pads + traces.
                continue
            z0 = layer_z0[i]
            z1 = layer_z1[i]
            color = self._layer_color(i, r["is_copper"])
            if r["is_copper"] and str(r.get("role", "Signal")) == "Plane" and i in plane_holes_by_layer:
                verts, faces = _perforated_plane_mesh(hw, hd, z0, z1, plane_holes_by_layer[i])
                if verts is None or faces is None:
                    verts, faces = _box_mesh(hw, hd, z0, z1)
            else:
                verts, faces = _box_mesh(hw, hd, z0, z1)
            mesh = gl.GLMeshItem(vertexes=verts, faces=faces,
                                  color=color, smooth=False, drawEdges=True)
            mesh.setGLOptions("additive" if not r["is_copper"] else "opaque")
            self._register_scene_mesh("Layers", f"layer/{i}", r["name"], mesh)

        def _add_via(
            cx: float,
            cy: float,
            z_top: float,
            z_bot: float,
            barrel_r: float,
            pad_radius: float,
            antipad_radius: float,
            landing_layers: set[int],
            color=(0.95, 0.80, 0.10, 1.0),
            scene_key: str = "via/main",
            scene_label: str = "Main Via",
            scene_group: str = "Vias",
        ):
            """Draw one via barrel + landing pads on landing copper layers."""
            h = z_bot - z_top
            if h <= 0:
                h = 0.001
            verts, faces = _cyl_mesh(barrel_r, h, z_top)
            verts[:, 0] += cx
            verts[:, 1] += cy
            mesh = gl.GLMeshItem(vertexes=verts, faces=faces,
                                  color=color, smooth=True, drawEdges=True)
            mesh.setGLOptions("opaque")
            self._register_scene_mesh(scene_group, scene_key, scene_label, mesh)

            # On landing layers draw pad as annular cylinder (barrel_r → pad_r, layer thickness).
            for si, r in enumerate(stackup):
                lz0 = layer_z0[si]
                lz1 = layer_z1[si]
                # only layers within via extent
                if lz1 < z_top - 0.001 or lz0 > z_bot + 0.001:
                    continue
                if r["is_copper"]:
                    layer_col = self._layer_color(si, True)
                    pad_color = (layer_col[0], min(1.0, layer_col[1] + 0.08), 0.05, 1.0)
                    if si in landing_layers:
                        # Annular pad with full layer thickness: inner=barrel, outer=pad → no barrel overlap
                        lh = lz1 - lz0
                        pv, pf = _annular_cyl_mesh(barrel_r, pad_radius, lh, lz0)
                        if pv is not None:
                            pv[:, 0] += cx
                            pv[:, 1] += cy
                            pmesh = gl.GLMeshItem(vertexes=pv, faces=pf,
                                                  color=pad_color, smooth=True, drawEdges=False)
                            pmesh.setGLOptions("opaque")
                            self._register_scene_mesh(scene_group, scene_key, scene_label, pmesh)

        # Main via
        _add_via(0.0, 0.0, via_z_top, via_z_bot, drill_r, pad_r, apad_r, signal_landing_layers,
             scene_key="via/main", scene_label="Main Via")

        # Stub
        if stub_stack_idx is not None:
                stub_z_top = layer_z1[to_stack_idx]
                stub_z_bot = layer_z1[stub_stack_idx]
                stub_color = (0.75, 0.75, 0.75, 0.7)
                _add_via(0.0, 0.0, stub_z_top, stub_z_bot, drill_r, pad_r, apad_r, {to_stack_idx}, color=stub_color,
                         scene_key="via/stub_main", scene_label="Stub Via")
                # Differential stub: mirror the stub barrel for the second via
                if self._radio_diff.isChecked():
                    _add_via(0.0, diff_offset_mm, stub_z_top, stub_z_bot, drill_r, pad_r, apad_r, {to_stack_idx}, color=stub_color,
                             scene_key="via/stub_diff", scene_label="Stub Via (Diff)")

        # Differential second via
        if self._radio_diff.isChecked():
            _add_via(0.0, diff_offset_mm, via_z_top, via_z_bot, drill_r, pad_r, apad_r, signal_landing_layers,
                     scene_key="via/diff", scene_label="Differential Via")

        # Feed geometry preview on via landing layers.
        # Differential mode shows all 4 feeds (1,2 inputs and 3,4 outputs).
        coax_h = max(0.4, 0.35 * (layer_z1[-1] - layer_z0[0]))
        is_diff_mode = self._radio_diff.isChecked()
        input_ports = [1, 2] if is_diff_mode else [1]
        output_ports = [3, 4] if is_diff_mode else [2]

        def _add_feed_preview(port_idx: int, layer_idx: int, prefix: str, color_seed: tuple[float, float, float, float]):
            if not hasattr(self, "_feed_port_controls"):
                return
            cfg = self._get_feed_port_config(port_idx)
            kind = str(cfg.get("type", "Trace")).strip().lower()
            y_off = self._feed_port_center_y_mm(port_idx)
            z0 = layer_z0[layer_idx]
            z1 = layer_z1[layer_idx]

            if kind == "trace":
                feed_len = float(cfg.get("trace_length_um", 2000.0)) / 1000.0
                feed_w = float(cfg.get("trace_width_um", 250.0)) / 1000.0
                feed_ang = float(cfg.get("trace_angle_deg", 0.0))
                corner_contact = math.sqrt(max(pad_r * pad_r - (0.5 * feed_w) * (0.5 * feed_w), 0.0))
                start_offset = max(0.0, corner_contact - 0.01 * feed_w)
                tv, tf = _trace_box_mesh(feed_len, feed_w, z0, z1, feed_ang, start_offset=start_offset)
                tv[:, 1] += y_off
                feed_color = (color_seed[0], min(1.0, color_seed[1] + 0.08), 0.05, 0.95)
                tmesh = gl.GLMeshItem(vertexes=tv, faces=tf,
                                      color=feed_color, smooth=False, drawEdges=False)
                tmesh.setGLOptions("opaque")
                self._register_scene_mesh("Feeds", f"feed/{prefix}_{port_idx}", f"Port {port_idx} Feed", tmesh)
            else:
                z_coax = layer_z1[layer_idx]
                cv, cf = _cyl_mesh(max(drill_r * 0.65, 0.03), coax_h, z_coax)
                cv[:, 1] += y_off
                cmesh = gl.GLMeshItem(vertexes=cv, faces=cf,
                                      color=(0.90, 0.90, 0.92, 0.95), smooth=True, drawEdges=True)
                cmesh.setGLOptions("opaque")
                self._register_scene_mesh("Feeds", f"feed/{prefix}_{port_idx}", f"Port {port_idx} Feed", cmesh)

        input_layer_col = self._layer_color(from_stack_idx, True)
        output_layer_col = self._layer_color(to_stack_idx, True)
        for pidx in input_ports:
            _add_feed_preview(pidx, from_stack_idx, "in", input_layer_col)
        for pidx in output_ports:
            _add_feed_preview(pidx, to_stack_idx, "out", output_layer_col)

        # Stitching vias
        if self._stitch_enable.isChecked():
            s_drill = self._stitch_drill.value() / 2.0 / 1000.0
            s_pad = self._stitch_pad.value() / 2.0 / 1000.0
            s_antipad = max(apad_r, s_pad * 1.35)
            s_color = (0.30, 0.60, 0.90, 0.85)
            s_highlight = (0.98, 0.20, 0.20, 0.95)
            stitch_z_top = layer_z0[stitch_from_stack_idx]
            stitch_z_bot = layer_z1[stitch_to_stack_idx]
            # Landing layers = top + bottom signal layers + all Plane layers in between
            # Only the top/bottom signal layers get annular pads drawn.
            # Intermediate Plane layers connect via the drill-size hole in the plane mesh — no extra pad ring.
            stitch_landing_layers = {stitch_from_stack_idx, stitch_to_stack_idx}
            for i, (sx, sy) in enumerate(self._stitching_coordinates_mm()):
                is_sel = i == self._stitch_selected_row
                color = s_highlight if is_sel else s_color
                r_mul = 1.25 if is_sel else 1.0
                _add_via(
                    sx,
                    sy,
                    stitch_z_top,
                    stitch_z_bot,
                    s_drill * r_mul,
                    s_pad * r_mul,
                    s_antipad * r_mul,
                    stitch_landing_layers,
                    color=color,
                    scene_key=f"stitch/{i}",
                    scene_label=f"Stitch {i + 1}",
                    scene_group="Stitching",
                )

        if self._scene_tree is not None:
            self._scene_tree.expandAll()

    # ── EMerge script generation ──────────────────────────────────────────

    # ── EMerge run helpers ────────────────────────────────────────────────

    def _get_host_main_window(self):
        host = getattr(self, "_host_main_window", None)
        if host is not None:
            return host
        return self.parent()

    def _ensure_saved_project_for_emerge(self) -> Path | None:
        host = self._get_host_main_window()
        if host is None:
            QMessageBox.warning(
                self,
                "Run EMerge",
                "Main project window not found. Save the project from the main window first.",
            )
            return None

        proj_path = getattr(host, "_project_path", None)
        if proj_path:
            return Path(str(proj_path))

        QMessageBox.information(
            self,
            "Run EMerge",
            "The SP Studio project is not saved yet. Save it now to continue.",
        )
        save_project = getattr(host, "_save_project", None)
        if callable(save_project):
            if not save_project():
                return None
            proj_path = getattr(host, "_project_path", None)
            if proj_path:
                return Path(str(proj_path))

        QMessageBox.warning(
            self,
            "Run EMerge",
            "Project save failed or was cancelled.",
        )
        return None

    def _get_emerge_folder(self):
        base = self._ensure_saved_project_for_emerge()
        if base is None:
            return None
        folder = base.parent / (base.stem + "_ViaAnalyzer")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _on_run_emerge(self):
        script = self._generate_emerge_script()
        self._script_edit.setPlainText(script)
        folder = self._get_emerge_folder()
        if folder is None:
            return
        script_path = folder / "via_simulation.py"
        script_path.write_text(script, encoding="utf-8")
        self._emerge_folder = folder
        self._output_edit.clear()
        self._output_edit.appendPlainText(
            "[S-Params Studio] Script salvato in:\n  " + str(script_path) + "\n"
            + "[S-Params Studio] Avvio EMerge dalla cartella:\n  " + str(folder) + "\n"
            + "-" * 60
        )
        import sys
        self._emerge_process = QProcess(self)
        self._emerge_process.setWorkingDirectory(str(folder))
        self._emerge_process.readyReadStandardOutput.connect(self._on_emerge_stdout)
        self._emerge_process.readyReadStandardError.connect(self._on_emerge_stderr)
        self._emerge_process.finished.connect(self._on_emerge_finished)
        self._emerge_process.errorOccurred.connect(self._on_emerge_error)
        self._btn_run_emerge.setEnabled(False)
        self._btn_stop_emerge.setVisible(True)
        self._emerge_process.start(sys.executable, [str(script_path)])

    def _on_stop_emerge(self):
        if self._emerge_process and self._emerge_process.state() != QProcess.NotRunning:
            self._emerge_process.kill()
            self._output_edit.appendPlainText(
                "\n[S-Params Studio] Processo interrotto dall'utente."
            )

    def _on_emerge_stdout(self):
        if self._emerge_process:
            data = bytes(self._emerge_process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self._output_edit.appendPlainText(data.rstrip())

    def _on_emerge_stderr(self):
        if self._emerge_process:
            data = bytes(self._emerge_process.readAllStandardError()).decode("utf-8", errors="replace")
            self._output_edit.appendPlainText(data.rstrip())

    def _on_emerge_finished(self, exit_code, exit_status):
        self._btn_run_emerge.setEnabled(True)
        self._btn_stop_emerge.setVisible(False)
        self._output_edit.appendPlainText(
            "\n" + "=" * 60
            + "\n[S-Params Studio] Processo terminato — exit code " + str(exit_code)
        )
        if exit_code == 0:
            folder = getattr(self, "_emerge_folder", None)
            if folder is not None:
                import glob as _glob
                candidates = _glob.glob(str(folder / "*.s[0-9]p"))
                if not candidates:
                    # fallback: any snp-like extension
                    candidates = [str(p) for p in folder.iterdir()
                                  if p.suffix.lower() in (".s2p", ".s4p", ".s1p", ".s3p")]
                if candidates:
                    result_file = max(candidates, key=lambda p: __import__('os').path.getmtime(p))
                    self._output_edit.appendPlainText(
                        "[S-Params Studio] Risultato: " + result_file
                    )
                    self.simulation_completed.emit(result_file)

    def _on_emerge_error(self, error):
        self._btn_run_emerge.setEnabled(True)
        self._btn_stop_emerge.setVisible(False)
        self._output_edit.appendPlainText(
            "\n[S-Params Studio] Errore avvio processo: " + str(error)
        )

    def _generate_emerge_script(self) -> str:  # noqa: C901
        import math as _math
        stackup   = self._read_stackup()
        drill_um  = self._drill_um.value()
        pad_um    = self._pad_um.value()
        antipad_um = self._antipad_um.value()
        f_start   = self._f_start.value()
        f_stop    = self._f_stop.value()
        n_pts     = self._n_pts.value()
        mesh_resolution = self._res_mm.value()
        n_workers = self._n_workers.value()
        sparam_fit_enabled = self._sparam_fit_enable.isChecked() if hasattr(self, "_sparam_fit_enable") else False
        sparam_fit_n_pts = self._sparam_fit_n_pts.value() if hasattr(self, "_sparam_fit_n_pts") else 401
        mesh_local_enabled = self._mesh_local_enable.isChecked() if hasattr(self, "_mesh_local_enable") else False
        mesh_div_via = self._mesh_factor_sliders["via"].value() if hasattr(self, "_mesh_factor_sliders") else 1
        mesh_div_ports = self._mesh_factor_sliders["ports"].value() if hasattr(self, "_mesh_factor_sliders") else 1
        mesh_div_feed = self._mesh_factor_sliders["feed"].value() if hasattr(self, "_mesh_factor_sliders") else 1
        mesh_div_planes = self._mesh_factor_sliders["planes"].value() if hasattr(self, "_mesh_factor_sliders") else 1
        mesh_div_stitching = self._mesh_factor_sliders["stitching"].value() if hasattr(self, "_mesh_factor_sliders") else 1
        show_structure_in_emerge = self._show_structure_in_emerge.isChecked() if hasattr(self, "_show_structure_in_emerge") else True
        show_labels_in_emerge = self._show_labels_in_emerge.isChecked() if hasattr(self, "_show_labels_in_emerge") else False
        show_mesh_in_emerge = self._show_mesh_in_emerge.isChecked() if hasattr(self, "_show_mesh_in_emerge") else True
        is_diff_mode = self._radio_diff.isChecked()
        active_script_ports = [1, 2, 3, 4] if is_diff_mode else [1, 2]
        input_control_ports = [1, 2] if is_diff_mode else [1]
        output_control_ports = [3, 4] if is_diff_mode else [2]
        control_to_script_port = {1: 1, 2: 2, 3: 3, 4: 4} if is_diff_mode else {1: 1, 2: 2}

        default_feed_cfg = {
            "type": "Trace",
            "trace_width_um": 250.0,
            "trace_length_um": 2000.0,
            "trace_angle_deg": 0.0,
        }

        feed_cfg_by_control: dict[int, dict[str, float | str]] = {}
        for ctrl_port in (1, 2, 3, 4):
            if hasattr(self, "_feed_port_controls"):
                feed_cfg_by_control[ctrl_port] = self._get_feed_port_config(ctrl_port)
            elif ctrl_port in (1, 2):
                feed_cfg_by_control[ctrl_port] = {
                    "type": self._feed_start_type.currentText() if hasattr(self, "_feed_start_type") else "Trace",
                    "trace_width_um": self._feed_start_trace_width_um.value() if hasattr(self, "_feed_start_trace_width_um") else 250.0,
                    "trace_length_um": self._feed_start_trace_length_um.value() if hasattr(self, "_feed_start_trace_length_um") else 2000.0,
                    "trace_angle_deg": self._feed_start_trace_angle_deg.value() if hasattr(self, "_feed_start_trace_angle_deg") else 180.0,
                }
            else:
                feed_cfg_by_control[ctrl_port] = {
                    "type": self._feed_end_type.currentText() if hasattr(self, "_feed_end_type") else "Trace",
                    "trace_width_um": self._feed_end_trace_width_um.value() if hasattr(self, "_feed_end_trace_width_um") else 250.0,
                    "trace_length_um": self._feed_end_trace_length_um.value() if hasattr(self, "_feed_end_trace_length_um") else 2000.0,
                    "trace_angle_deg": self._feed_end_trace_angle_deg.value() if hasattr(self, "_feed_end_trace_angle_deg") else 0.0,
                }

        for ctrl_port in (1, 2, 3, 4):
            cfg = dict(default_feed_cfg)
            cfg.update(feed_cfg_by_control.get(ctrl_port, {}))
            if not is_diff_mode and ctrl_port in (3, 4):
                cfg = dict(feed_cfg_by_control[2])
            feed_cfg_by_control[ctrl_port] = cfg

        def _resolve_port_feed_geometry(cfg: dict[str, float | str], default_angle_deg: float) -> dict[str, float | str]:
            kind = str(cfg.get("type", "Trace")).strip().lower()
            if kind == "trace":
                w_mm = max(float(cfg.get("trace_width_um", 250.0)) / 1000.0, 0.001)
                l_mm = max(float(cfg.get("trace_length_um", 2000.0)) / 1000.0, 0.001)
            else:
                w_mm = 0.0  # filled later once drill radius is known
                l_mm = 0.0
            angle_deg = float(cfg.get("trace_angle_deg", default_angle_deg))
            return {
                "kind": kind,
                "width_mm": w_mm,
                "length_mm": l_mm,
                "angle_deg": angle_deg,
            }

        entry_feed_by_control = {
            ctrl_port: _resolve_port_feed_geometry(feed_cfg_by_control[ctrl_port], 180.0)
            for ctrl_port in input_control_ports
        }
        exit_feed_by_control = {
            ctrl_port: _resolve_port_feed_geometry(feed_cfg_by_control[ctrl_port], 0.0)
            for ctrl_port in output_control_ports
        }

        def _make_port_sheet_geometry(feed_geom: dict[str, float | str], center_y_mm: float, contact_offset_mm: float) -> dict[str, float]:
            w_mm = float(feed_geom["width_mm"])
            l_mm = float(feed_geom["length_mm"])
            angle_deg = float(feed_geom["angle_deg"])
            kind = str(feed_geom["kind"])
            a = _math.radians(angle_deg)
            c, s = _math.cos(a), _math.sin(a)
            wx, wy = -s, c
            if kind == "trace":
                ox = (contact_offset_mm + l_mm) * c - (w_mm / 2.0) * wx
                oy = center_y_mm + (contact_offset_mm + l_mm) * s - (w_mm / 2.0) * wy
                ux, uy = w_mm * wx, w_mm * wy
            else:
                ox = -0.5 * w_mm
                oy = center_y_mm
                ux, uy = w_mm, 0.0
            return {
                "ox": ox,
                "oy": oy,
                "ux": ux,
                "uy": uy,
            }

        _, boundary_pad_mm, _, _ = self._simulation_domain_geometry()

        # ── Copper layer index mapping ────────────────────────────────────────
        copper_layers = [(i, r) for i, r in enumerate(stackup) if r["is_copper"]]
        via_from_idx_c = self._via_from_combo.currentIndex()
        via_to_idx_c   = self._via_to_combo.currentIndex()
        if via_from_idx_c < 0: via_from_idx_c = 0
        if via_to_idx_c   < 0: via_to_idx_c   = max(0, len(copper_layers) - 1)
        via_from_idx_c = min(via_from_idx_c, max(0, len(copper_layers) - 1))
        via_to_idx_c   = min(via_to_idx_c,   max(0, len(copper_layers) - 1))
        if via_from_idx_c > via_to_idx_c:
            via_from_idx_c, via_to_idx_c = via_to_idx_c, via_from_idx_c

        stitch_from_idx_c = self._stitch_from_combo.currentIndex() if hasattr(self, "_stitch_from_combo") else 0
        stitch_to_idx_c = self._stitch_to_combo.currentIndex() if hasattr(self, "_stitch_to_combo") else max(0, len(copper_layers) - 1)
        if stitch_from_idx_c < 0: stitch_from_idx_c = 0
        if stitch_to_idx_c   < 0: stitch_to_idx_c   = max(0, len(copper_layers) - 1)
        stitch_from_idx_c = min(stitch_from_idx_c, max(0, len(copper_layers) - 1))
        stitch_to_idx_c   = min(stitch_to_idx_c,   max(0, len(copper_layers) - 1))
        if stitch_from_idx_c > stitch_to_idx_c:
            stitch_from_idx_c, stitch_to_idx_c = stitch_to_idx_c, stitch_from_idx_c

        # ── Unique dielectric materials ───────────────────────────────────────
        diel_map: dict[tuple, str] = {}
        diel_vars: list[tuple] = []

        def _get_diel_var(er: float, tand: float) -> str:
            key = (round(er, 6), round(tand, 6))
            if key not in diel_map:
                vname = f"diel_{len(diel_map)}"
                diel_map[key] = vname
                diel_vars.append((vname, er, tand))
            return diel_map[key]

        for r in stackup:
            if not r["is_copper"]:
                _get_diel_var(r["er"], r["tand"])

        # ── Z coordinate computation (top=0, going negative) ─────────────────
        layer_z_mm: list[tuple[float, float]] = []   # (z_top, z_bot) per row
        _z = 0.0
        for r in stackup:
            _thick = r["thickness_um"] / 1000.0
            layer_z_mm.append((_z, _z - _thick))
            _z -= _thick

        # copper_info: (copper_idx, global_idx, row_dict, z_top_mm, z_bot_mm)
        copper_info = [
            (ci, gi, r, layer_z_mm[gi][0], layer_z_mm[gi][1])
            for ci, (gi, r) in enumerate(copper_layers)
        ]

        via_from = copper_info[via_from_idx_c]   # entry (upper) copper layer
        via_to   = copper_info[via_to_idx_c]     # exit  (lower) copper layer
        via_from_gi = via_from[1]
        via_to_gi   = via_to[1]
        z_via_top_mm = via_from[3]   # top face of upper barrel end
        z_via_bot_mm = via_to[4]     # bot face of lower barrel end
        via_barrel_h_mm = z_via_top_mm - z_via_bot_mm

        stub_text = self._stub_combo.currentText() if hasattr(self, "_stub_combo") else "None"
        stub_stack_idx = -1
        if stub_text and stub_text != "None":
            stub_stack_idx = next((i for i, row in enumerate(stackup) if row["name"] == stub_text), -1)
        stub_enabled = stub_stack_idx > via_to_gi
        z_stub_top_mm = layer_z_mm[via_to_gi][1] if stub_enabled else 0.0
        z_stub_bot_mm = layer_z_mm[stub_stack_idx][1] if stub_enabled else 0.0
        stub_h_mm = (z_stub_top_mm - z_stub_bot_mm) if stub_enabled else 0.0

        # Signal layer global indices (carry the feed trace)
        signal_gi_set = {via_from_gi, via_to_gi}

        # ── Physical geometry parameters (mm) ─────────────────────────────────
        drill_r_mm   = drill_um   / 2.0 / 1000.0
        pad_r_mm     = pad_um     / 2.0 / 1000.0
        antipad_r_mm = antipad_um / 2.0 / 1000.0
        diff_offset_mm = self._diff_spacing.value() / 1000.0 if self._radio_diff.isChecked() else 0.0
        signal_via_centers_mm = [(0.0, 0.0)]
        if self._radio_diff.isChecked():
            signal_via_centers_mm.append((0.0, diff_offset_mm))
        plane_antipad_r_mm_by_gi: dict[int, float] = {}
        for gi, row in enumerate(stackup):
            if row.get("is_copper") and str(row.get("role", "Signal")) == "Plane":
                plane_antipad_r_mm_by_gi[gi] = max(antipad_r_mm, drill_r_mm)

        z_entry_top_mm, z_entry_bot_mm = via_from[3], via_from[4]
        z_exit_top_mm,  z_exit_bot_mm  = via_to[3],   via_to[4]
        thick_entry_mm = z_entry_top_mm - z_entry_bot_mm
        thick_exit_mm  = z_exit_top_mm  - z_exit_bot_mm

        # Feed trace geometry (mm) and per-port sheet definitions
        for feed_geom in entry_feed_by_control.values():
            if str(feed_geom["kind"]) != "trace":
                feed_geom["width_mm"] = drill_r_mm * 2.0
                feed_geom["length_mm"] = max(drill_r_mm * 2.0, 0.05)
            w_mm = float(feed_geom["width_mm"])
            feed_geom["contact_offset_mm"] = max(
                0.0,
                _math.sqrt(max(pad_r_mm * pad_r_mm - (0.5 * w_mm) * (0.5 * w_mm), 0.0)) - 0.01 * w_mm,
            )
        for feed_geom in exit_feed_by_control.values():
            if str(feed_geom["kind"]) != "trace":
                feed_geom["width_mm"] = drill_r_mm * 2.0
                feed_geom["length_mm"] = max(drill_r_mm * 2.0, 0.05)
            w_mm = float(feed_geom["width_mm"])
            feed_geom["contact_offset_mm"] = max(
                0.0,
                _math.sqrt(max(pad_r_mm * pad_r_mm - (0.5 * w_mm) * (0.5 * w_mm), 0.0)) - 0.01 * w_mm,
            )

        max_trace_reach_mm = 0.0
        for ctrl_port, feed_geom in {**entry_feed_by_control, **exit_feed_by_control}.items():
            y_off_mm = abs(self._feed_port_center_y_mm(ctrl_port))
            max_trace_reach_mm = max(
                max_trace_reach_mm,
                y_off_mm + float(feed_geom["length_mm"]) + 0.5 * float(feed_geom["width_mm"]),
            )

        # Domain half-width (mm) — must contain traces + margin
        domain_hw_mm = max(boundary_pad_mm, max_trace_reach_mm + pad_r_mm + 0.5)

        # ── Port geometry ─────────────────────────────────────────────────────
        # Input ports (1,2): entry layer, nearest reference below entry
        entry_ci = via_from[0]
        if entry_ci + 1 < len(copper_info):
            entry_port_z_ref_mm = copper_info[entry_ci + 1][3]  # z_top of layer below
        else:
            entry_port_z_ref_mm = z_entry_bot_mm
        entry_port_h_mm = max(z_entry_top_mm - entry_port_z_ref_mm, thick_entry_mm)
        entry_port_oz_mm = entry_port_z_ref_mm if entry_port_z_ref_mm < z_entry_top_mm else z_entry_bot_mm

        # Output ports (3,4): exit layer, nearest reference above exit
        exit_ci = via_to[0]
        if exit_ci > 0:
            exit_port_z_ref_mm = copper_info[exit_ci - 1][4]   # z_bot of layer above
        else:
            exit_port_z_ref_mm = z_exit_top_mm
        exit_port_h_mm = max(exit_port_z_ref_mm - z_exit_bot_mm, thick_exit_mm)
        exit_port_oz_mm = z_exit_bot_mm

        port_plate_by_script: dict[int, dict[str, float]] = {}
        port_width_by_script: dict[int, float] = {}
        port_height_by_script: dict[int, float] = {}
        for ctrl_port in input_control_ports:
            script_port = control_to_script_port[ctrl_port]
            feed_geom = entry_feed_by_control[ctrl_port]
            port_plate = _make_port_sheet_geometry(
                feed_geom,
                self._feed_port_center_y_mm(ctrl_port),
                float(feed_geom["contact_offset_mm"]),
            )
            port_plate["oz"] = entry_port_oz_mm
            port_plate["vz"] = entry_port_h_mm
            port_plate_by_script[script_port] = port_plate
            port_width_by_script[script_port] = float(feed_geom["width_mm"])
            port_height_by_script[script_port] = entry_port_h_mm

        for ctrl_port in output_control_ports:
            script_port = control_to_script_port[ctrl_port]
            feed_geom = exit_feed_by_control[ctrl_port]
            port_plate = _make_port_sheet_geometry(
                feed_geom,
                self._feed_port_center_y_mm(ctrl_port),
                float(feed_geom["contact_offset_mm"]),
            )
            port_plate["oz"] = exit_port_oz_mm
            port_plate["vz"] = exit_port_h_mm
            port_plate_by_script[script_port] = port_plate
            port_width_by_script[script_port] = float(feed_geom["width_mm"])
            port_height_by_script[script_port] = exit_port_h_mm

        # ── Emit generated script ─────────────────────────────────────────────
        lines: list[str] = []
        lines.append('"""EMerge via simulation script — auto-generated by S-Params Studio.')
        lines.append('   Uses 3D primitives (Box, Cylinder) — no PCB layouter."""')
        lines.append("")
        lines.append("import datetime")
        lines.append("import emerge_iron")
        lines.append("")
        lines.append("from emerge_config import config")
        lines.append("config.set_pardiso_threads(8)")
        lines.append("config.set_acc_threads(10)")
        lines.append("")
        lines.append("import math")
        lines.append("import numpy as np")
        lines.append("import emerge as em")
        lines.append("import os, tempfile, shutil")
        lines.append("")
        # Derive a clean project name from the Via Analysis window name
        _raw_name = self._name_edit.text().strip() or f"Via Analysis #{self.window_number}"
        import re as _re
        _prj_name = _raw_name  # used verbatim as ProjectName in the generated script

        lines.append("#######################################################################################################################################")
        lines.append("# DEFINE PROJECT NAME")
        lines.append("#######################################################################################################################################")
        lines.append("#from Plot_Results import ProjectName")
        lines.append(f'ProjectName = "{_prj_name}"')
        lines.append("")
        lines.append("# Change current path to script file folder")
        lines.append("abspath = os.path.abspath(__file__)")
        lines.append("dname = os.path.dirname(abspath)")
        lines.append("os.chdir(dname)")
        lines.append("")
        lines.append("currDir = os.getcwd()")
        lines.append("print(currDir)")
        lines.append("## prepare simulation folder, if dir exits remove and create new one to be empty")
        lines.append("Sim_Path = os.path.join(currDir, ProjectName)")
        lines.append("if os.path.exists(Sim_Path):")
        lines.append("    shutil.rmtree(Sim_Path)   # clear previous directory")
        lines.append("    os.mkdir(Sim_Path)    # create empty simulation folder")
        lines.append("")
        lines.append("")
        lines.append("mm = 0.001  # metres per mm unit")
        lines.append("")
        lines.append('m = em.Simulation(ProjectName)')
        lines.append('m.check_version("2.5.5")')
        lines.append("")
        lines.append("# ── Materials ──────────────────────────────────────────────────────────────")

        for vname, er, tand in diel_vars:
            lines.append(f'{vname} = em.Material(er={er}, tand={tand}, color="#2ca02c", opacity=0.2)')
        if not diel_vars:
            lines.append('diel_0 = em.Material(er=4.2, tand=0.02, color="#2ca02c", opacity=0.2)')

        lines.append("")
        lines.append("# ── Geometry parameters ──────────────────────────────────────────────────")
        lines.append(f"drill_r   = {drill_r_mm:.6f} * mm  # via drill radius")
        lines.append(f"pad_r     = {pad_r_mm:.6f} * mm  # landing pad radius")
        lines.append(f"antipad_r_default = {antipad_r_mm:.6f} * mm  # default clearance (antipad) radius")
        lines.append(f"domain_hw = {domain_hw_mm:.4f} * mm  # domain half-width")
        lines.append(f"signal_via_centers_mm = {[(round(x, 6), round(y, 6)) for x, y in signal_via_centers_mm]}")
        lines.append(f"plane_antipad_r_mm_by_layer = {{ {', '.join(f'{gi}: {val:.6f}' for gi, val in plane_antipad_r_mm_by_gi.items())} }}")
        lines.append("")
        lines.append("# ── Stackup layer volumes (Z=0 at top surface, going negative) ──────────")
        for i, (r, (z_top, z_bot)) in enumerate(zip(stackup, layer_z_mm)):
            gi = i
            thick = r["thickness_um"] / 1000.0
            vname = f"layer_{i}"
            if r["is_copper"]:
                if gi in signal_gi_set:
                    # Signal layer: pads are created and united with trace in the feed section below.
                    lines.append(f"# Signal copper layer: {r['name']}  z=[{z_bot:.4f}, {z_top:.4f}] mm")
                    lines.append(f"# (pads created and united with trace in the feed section below)")
                    lines.append("")
                elif str(r.get("role", "Signal")) == "Plane":
                    # Plane copper layer: rectangular box with optional antipad subtraction.
                    lines.append(f"# Reference copper plane: {r['name']}  z=[{z_bot:.4f}, {z_top:.4f}] mm")
                    lines.append(f"{vname} = em.geo.Box(2*domain_hw, 2*domain_hw, {thick:.6f}*mm,")
                    lines.append(f"    position=(-domain_hw, -domain_hw, {z_bot:.6f}*mm),")
                    lines.append(f"    alignment=em.geo.Alignment.CORNER,")
                    lines.append(f"    name=\"{vname}\")")  
                    lines.append(f"{vname}.material = em.lib.PEC")
                    lines.append(f"_do_sig_clear = ({via_from_gi} < {gi} < {via_to_gi})")
                    lines.append(f"_do_stub_clear = ({stub_enabled} and {via_to_gi} < {gi} <= {stub_stack_idx})")
                    lines.append(f"if {gi} in plane_antipad_r_mm_by_layer and (_do_sig_clear or _do_stub_clear):")
                    lines.append(f"    _ap_r = plane_antipad_r_mm_by_layer[{gi}] * mm")
                    lines.append(f"    for _vi, (_vx, _vy) in enumerate(signal_via_centers_mm):")
                    lines.append(f"        _ap = em.geo.Cylinder(_ap_r, {thick:.6f}*mm,")
                    lines.append(f"            cs=em.GCS.displace(_vx*mm, _vy*mm, {z_bot:.6f}*mm),")
                    lines.append(f'            name=f"{vname}_antipad_{{_vi}}")' )
                    lines.append(f"        {vname} = em.geo.subtract({vname}, _ap)")
                    lines.append("")
                else:
                    # Other copper layer: full box, no via antipad subtraction.
                    lines.append(f"# Copper layer: {r['name']}  z=[{z_bot:.4f}, {z_top:.4f}] mm")
                    lines.append(f"{vname} = em.geo.Box(2*domain_hw, 2*domain_hw, {thick:.6f}*mm,")
                    lines.append(f"    position=(-domain_hw, -domain_hw, {z_bot:.6f}*mm),")
                    lines.append(f"    alignment=em.geo.Alignment.CORNER,")
                    lines.append(f"    name=\"{vname}\")")  
                    lines.append(f"{vname}.material = em.lib.PEC")
                    lines.append("")
            else:
                dv = _get_diel_var(r["er"], r["tand"])
                lines.append(f"# Dielectric layer: {r['name']}  z=[{z_bot:.4f}, {z_top:.4f}] mm")
                lines.append(f"{vname} = em.geo.Box(2*domain_hw, 2*domain_hw, {thick:.6f}*mm,")
                lines.append(f"    position=(-domain_hw, -domain_hw, {z_bot:.6f}*mm),")
                lines.append(f"    alignment=em.geo.Alignment.CORNER,")
                lines.append(f"    name=\"{vname}\")")  
                lines.append(f"{vname}.material = {dv}")
                lines.append("")

        lines.append("# ── Signal via barrel ────────────────────────────────────────────────────")
        lines.append(f"# From {via_from[2]['name']} z_top={z_via_top_mm:.4f}mm  "
                     f"to {via_to[2]['name']} z_bot={z_via_bot_mm:.4f}mm")
        lines.append(f"via_barrel = em.geo.Cylinder(drill_r, {via_barrel_h_mm:.6f}*mm,")
        lines.append(f"    cs=em.GCS.displace(0, 0, {z_via_bot_mm:.6f}*mm),")
        lines.append(f"    name=\"via_barrel\")")
        lines.append(f"via_barrel.material = em.lib.PEC")
        if self._radio_diff.isChecked():
            lines.append(f"via_barrel_diff = em.geo.Cylinder(drill_r, {via_barrel_h_mm:.6f}*mm,")
            lines.append(f"    cs=em.GCS.displace(0, {diff_offset_mm:.6f}*mm, {z_via_bot_mm:.6f}*mm),")
            lines.append(f"    name=\"via_barrel_diff\")")
            lines.append("via_barrel_diff.material = em.lib.PEC")
        if stub_enabled:
            lines.append(f"stub_barrel = em.geo.Cylinder(drill_r, {stub_h_mm:.6f}*mm,")
            lines.append(f"    cs=em.GCS.displace(0, 0, {z_stub_bot_mm:.6f}*mm),")
            lines.append(f"    name=\"stub_barrel\")")
            lines.append("stub_barrel.material = em.lib.PEC")
            if self._radio_diff.isChecked():
                lines.append(f"stub_barrel_diff = em.geo.Cylinder(drill_r, {stub_h_mm:.6f}*mm,")
                lines.append(f"    cs=em.GCS.displace(0, {diff_offset_mm:.6f}*mm, {z_stub_bot_mm:.6f}*mm),")
                lines.append(f"    name=\"stub_barrel_diff\")")
                lines.append("stub_barrel_diff.material = em.lib.PEC")
        lines.append("")
        lines.append("# ── Via holes used to clear intersected dielectric layers ───────────────")
        lines.append("_diel_via_holes = []  # (x[m], y[m], r[m], z_bot[m], z_top[m])")
        lines.append("for _vx, _vy in signal_via_centers_mm:")
        lines.append(f"    _diel_via_holes.append((_vx*mm, _vy*mm, drill_r, {z_via_bot_mm:.6f}*mm, {z_via_top_mm:.6f}*mm))")
        if self._radio_diff.isChecked():
            lines.append("for _vx, _vy in signal_via_centers_mm:")
            lines.append(f"    _diel_via_holes.append((_vx*mm, (_vy + {diff_offset_mm:.6f})*mm, drill_r, {z_via_bot_mm:.6f}*mm, {z_via_top_mm:.6f}*mm))")
        if stub_enabled:
            lines.append("for _vx, _vy in signal_via_centers_mm:")
            lines.append(f"    _diel_via_holes.append((_vx*mm, _vy*mm, drill_r, {z_stub_bot_mm:.6f}*mm, {z_stub_top_mm:.6f}*mm))")
            if self._radio_diff.isChecked():
                lines.append("for _vx, _vy in signal_via_centers_mm:")
                lines.append(f"    _diel_via_holes.append((_vx*mm, (_vy + {diff_offset_mm:.6f})*mm, drill_r, {z_stub_bot_mm:.6f}*mm, {z_stub_top_mm:.6f}*mm))")
        lines.append("")
        lines.append("# ── Entry feed geometry (inputs) ─────────────────────────────────────────")
        lines.append(f"# Layer: {via_from[2]['name']} (ports 1,2 in differential mode)")
        entry_script_ports = [control_to_script_port[p] for p in input_control_ports]
        for ctrl_port in input_control_ports:
            script_port = control_to_script_port[ctrl_port]
            geom = entry_feed_by_control[ctrl_port]
            y_off = self._feed_port_center_y_mm(ctrl_port)
            feed_name = f"trace_port{script_port}"
            lines.append(
                f"# Port {script_port} input feed: type={str(geom['kind'])}, angle={float(geom['angle_deg']):.1f}°, y={y_off:.6f}mm"
            )
            if str(geom["kind"]) == "trace":
                lines.append(
                    f"{feed_name} = em.geo.Box({float(geom['length_mm']):.6f}*mm, {float(geom['width_mm']):.6f}*mm, {thick_entry_mm:.6f}*mm,"
                )
                lines.append(
                    f"    position=({float(geom['contact_offset_mm']):.6f}*mm, {y_off - float(geom['width_mm']) / 2.0:.6f}*mm, {z_entry_bot_mm:.6f}*mm),"
                )
                lines.append("    alignment=em.geo.Alignment.CORNER,")
                lines.append(f"    name=\"{feed_name}\")")
                lines.append(f"{feed_name}.material = em.lib.PEC")
                if abs(float(geom["angle_deg"]) % 360) > 0.1:
                    lines.append(f"{feed_name} = em.geo.rotate({feed_name}, c0=(0, {y_off:.6f}*mm, 0),")
                    lines.append(f"    ax=(0, 0, 1), angle={float(geom['angle_deg']):.4f})")
            else:
                coax_r_start_mm = max(drill_r_mm * 0.65, 0.03)
                coax_h_start_mm = max(0.4, 0.35 * (layer_z_mm[-1][0] - layer_z_mm[0][1]))
                lines.append(f"{feed_name} = em.geo.Cylinder({coax_r_start_mm:.6f}*mm, {coax_h_start_mm:.6f}*mm,")
                lines.append(f"    cs=em.GCS.displace(0, {y_off:.6f}*mm, {z_entry_top_mm:.6f}*mm),")
                lines.append(f"    name=\"{feed_name}\")")
                lines.append(f"{feed_name}.material = em.lib.PEC")
            lines.append(f"_pad_p{script_port} = em.geo.Cylinder(pad_r, {thick_entry_mm:.6f}*mm,")
            lines.append(f"    cs=em.GCS.displace(0, {y_off:.6f}*mm, {z_entry_bot_mm:.6f}*mm),")
            lines.append(f"    name=\"entry_pad_p{script_port}\")")
            lines.append(f"_pad_p{script_port}.material = em.lib.PEC")
            lines.append(f"{feed_name} = em.geo.add({feed_name}, _pad_p{script_port})")

        lines.append(f"trace_start = trace_port{entry_script_ports[0]}")
        for script_port in entry_script_ports[1:]:
            lines.append(f"trace_start = em.geo.add(trace_start, trace_port{script_port})")

        lines.append("")
        lines.append("# ── Exit feed geometry (outputs) ─────────────────────────────────────────")
        lines.append(f"# Layer: {via_to[2]['name']} (ports 3,4 in differential mode)")
        exit_script_ports = [control_to_script_port[p] for p in output_control_ports]
        for ctrl_port in output_control_ports:
            script_port = control_to_script_port[ctrl_port]
            geom = exit_feed_by_control[ctrl_port]
            y_off = self._feed_port_center_y_mm(ctrl_port)
            feed_name = f"trace_port{script_port}"
            lines.append(
                f"# Port {script_port} output feed: type={str(geom['kind'])}, angle={float(geom['angle_deg']):.1f}°, y={y_off:.6f}mm"
            )
            if str(geom["kind"]) == "trace":
                lines.append(
                    f"{feed_name} = em.geo.Box({float(geom['length_mm']):.6f}*mm, {float(geom['width_mm']):.6f}*mm, {thick_exit_mm:.6f}*mm,"
                )
                lines.append(
                    f"    position=({float(geom['contact_offset_mm']):.6f}*mm, {y_off - float(geom['width_mm']) / 2.0:.6f}*mm, {z_exit_bot_mm:.6f}*mm),"
                )
                lines.append("    alignment=em.geo.Alignment.CORNER,")
                lines.append(f"    name=\"{feed_name}\")")
                lines.append(f"{feed_name}.material = em.lib.PEC")
                if abs(float(geom["angle_deg"]) % 360) > 0.1:
                    lines.append(f"{feed_name} = em.geo.rotate({feed_name}, c0=(0, {y_off:.6f}*mm, 0),")
                    lines.append(f"    ax=(0, 0, 1), angle={float(geom['angle_deg']):.4f})")
            else:
                coax_r_end_mm = max(drill_r_mm * 0.65, 0.03)
                coax_h_end_mm = max(0.4, 0.35 * (layer_z_mm[-1][0] - layer_z_mm[0][1]))
                lines.append(f"{feed_name} = em.geo.Cylinder({coax_r_end_mm:.6f}*mm, {coax_h_end_mm:.6f}*mm,")
                lines.append(f"    cs=em.GCS.displace(0, {y_off:.6f}*mm, {z_exit_top_mm:.6f}*mm),")
                lines.append(f"    name=\"{feed_name}\")")
                lines.append(f"{feed_name}.material = em.lib.PEC")
            lines.append(f"_pad_p{script_port} = em.geo.Cylinder(pad_r, {thick_exit_mm:.6f}*mm,")
            lines.append(f"    cs=em.GCS.displace(0, {y_off:.6f}*mm, {z_exit_bot_mm:.6f}*mm),")
            lines.append(f"    name=\"exit_pad_p{script_port}\")")
            lines.append(f"_pad_p{script_port}.material = em.lib.PEC")
            lines.append(f"{feed_name} = em.geo.add({feed_name}, _pad_p{script_port})")

        lines.append(f"trace_end = trace_port{exit_script_ports[0]}")
        for script_port in exit_script_ports[1:]:
            lines.append(f"trace_end = em.geo.add(trace_end, trace_port{script_port})")
        lines.append("")
        # Stitching vias
        if self._stitch_enable.isChecked():
            stitch_coords = self._stitching_coordinates_mm()
            s_drill_r_mm = self._stitch_drill.value() / 2.0 / 1000.0
            s_pad_r_mm = self._stitch_pad.value() / 2.0 / 1000.0
            s_from_gi = copper_info[stitch_from_idx_c][1] if stitch_from_idx_c < len(copper_info) else via_from_gi
            s_to_gi   = copper_info[stitch_to_idx_c][1]   if stitch_to_idx_c   < len(copper_info) else via_to_gi
            z_s_top_mm = layer_z_mm[s_from_gi][0]
            z_s_bot_mm = layer_z_mm[s_to_gi][1]
            s_h_mm = z_s_top_mm - z_s_bot_mm
            s_from_th_mm = stackup[s_from_gi]["thickness_um"] / 1000.0
            s_to_th_mm = stackup[s_to_gi]["thickness_um"] / 1000.0
            lines.append("# ── Stitching vias ───────────────────────────────────────────────────────")
            lines.append(f"# Pattern: {self._stitch_pattern.currentText()}, "
                         f"N={len(stitch_coords)} vias")
            lines.append(f"_s_drill_r = {s_drill_r_mm:.6f} * mm")
            lines.append(f"_s_pad_r   = {s_pad_r_mm:.6f} * mm")
            lines.append(f"_s_h       = {s_h_mm:.6f} * mm")
            lines.append(f"_s_z_bot   = {z_s_bot_mm:.6f} * mm")
            lines.append("_stitch_coords = [")
            for sx, sy in stitch_coords:
                lines.append(f"    ({sx:.6f}, {sy:.6f}),  # mm")
            lines.append("]")
            lines.append("for _si, (_sx, _sy) in enumerate(_stitch_coords):")
            lines.append("    _sv = em.geo.Cylinder(_s_drill_r, _s_h,")
            lines.append("        cs=em.GCS.displace(_sx*mm, _sy*mm, _s_z_bot),")
            lines.append('        name=f"stitch_{_si}_barrel")')
            lines.append("    _sv.material = em.lib.PEC")
            lines.append("    _diel_via_holes.append((_sx*mm, _sy*mm, _s_drill_r, _s_z_bot, _s_z_bot + _s_h))")
            lines.append(f"    # Annular pad on start layer (pad_r ring minus drill hole — no overlap with barrel)")
            lines.append(f"    _spad_from = em.geo.Cylinder(_s_pad_r, {s_from_th_mm:.6f}*mm,")
            lines.append(f"        cs=em.GCS.displace(_sx*mm, _sy*mm, {layer_z_mm[s_from_gi][1]:.6f}*mm),")
            lines.append('        name=f"stitch_{_si}_pad_from")')
            lines.append("    _spad_from.material = em.lib.PEC")
            lines.append(f"    _spad_from_hole = em.geo.Cylinder(_s_drill_r, {s_from_th_mm:.6f}*mm,")
            lines.append(f"        cs=em.GCS.displace(_sx*mm, _sy*mm, {layer_z_mm[s_from_gi][1]:.6f}*mm),")
            lines.append('        name=f"stitch_{_si}_pad_from_hole")')
            lines.append("    _spad_from = em.geo.subtract(_spad_from, _spad_from_hole)")
            lines.append(f"    # Annular pad on end layer (pad_r ring minus drill hole — no overlap with barrel)")
            lines.append(f"    _spad_to = em.geo.Cylinder(_s_pad_r, {s_to_th_mm:.6f}*mm,")
            lines.append(f"        cs=em.GCS.displace(_sx*mm, _sy*mm, {layer_z_mm[s_to_gi][1]:.6f}*mm),")
            lines.append('        name=f"stitch_{_si}_pad_to")')
            lines.append("    _spad_to.material = em.lib.PEC")
            lines.append(f"    _spad_to_hole = em.geo.Cylinder(_s_drill_r, {s_to_th_mm:.6f}*mm,")
            lines.append(f"        cs=em.GCS.displace(_sx*mm, _sy*mm, {layer_z_mm[s_to_gi][1]:.6f}*mm),")
            lines.append('        name=f"stitch_{_si}_pad_to_hole")')
            lines.append("    _spad_to = em.geo.subtract(_spad_to, _spad_to_hole)")
            lines.append("")
            lines.append("# Stitching barrel clearances on all crossed copper layers")
            lines.append("# All layers (Plane and Signal): subtract drill-size cylinder so barrel fits flush.")
            lines.append("# Plane layers → barrel surface touches plane copper = electrically connected.")
            lines.append("# Signal layers → drill-hole only, no antipad, no intersection.")
            for gi, row in enumerate(stackup):
                if not row.get("is_copper"):
                    continue
                if not (s_from_gi < gi < s_to_gi):
                    continue
                th_mm = row["thickness_um"] / 1000.0
                z_bot_mm = layer_z_mm[gi][1]
                role_label = str(row.get("role", "Signal")).strip()
                lines.append(f"# Stitch barrel clearance on {role_label} layer {gi} ({row['name']})")
                lines.append("for _si, (_sx, _sy) in enumerate(_stitch_coords):")
                lines.append(f"    _sdh = em.geo.Cylinder(_s_drill_r, {th_mm:.6f}*mm,")
                lines.append(f"        cs=em.GCS.displace(_sx*mm, _sy*mm, {z_bot_mm:.6f}*mm),")
                lines.append(f'        name=f"layer_{gi}_stitch_{{_si}}_drill_clear")')
                lines.append(f"    layer_{gi} = em.geo.subtract(layer_{gi}, _sdh)")
            lines.append("")

        lines.append("# ── Dielectric clearances for all crossing vias ─────────────────────────")
        for gi, row in enumerate(stackup):
            if row.get("is_copper"):
                continue
            z_top_mm, z_bot_mm = layer_z_mm[gi]
            th_mm = row["thickness_um"] / 1000.0
            lines.append(f"# Dielectric via clearances on layer {gi} ({row['name']})")
            lines.append("for _hi, (_hx, _hy, _hr, _hz_bot, _hz_top) in enumerate(_diel_via_holes):")
            lines.append(f"    if not (_hz_top <= {z_bot_mm:.6f}*mm or _hz_bot >= {z_top_mm:.6f}*mm):")
            lines.append(f"        _dh = em.geo.Cylinder(_hr, {th_mm:.6f}*mm,")
            lines.append(f"            cs=em.GCS.displace(_hx, _hy, {z_bot_mm:.6f}*mm),")
            lines.append(f"            name=f\"layer_{gi}_diel_clear_{{_hi}}\")")
            lines.append(f"        layer_{gi} = em.geo.subtract(layer_{gi}, _dh)")
            lines.append("")

        lines.append("# ── Non-plane copper clearances for crossing vias ───────────────────────")
        for gi, row in enumerate(stackup):
            if not row.get("is_copper"):
                continue
            if gi in signal_gi_set:
                continue
            if str(row.get("role", "Signal")) == "Plane":
                continue
            z_top_mm, z_bot_mm = layer_z_mm[gi]
            th_mm = row["thickness_um"] / 1000.0
            lines.append(f"# Copper via clearances on layer {gi} ({row['name']})")
            lines.append("for _hi, (_hx, _hy, _hr, _hz_bot, _hz_top) in enumerate(_diel_via_holes):")
            lines.append(f"    if not (_hz_top <= {z_bot_mm:.6f}*mm or _hz_bot >= {z_top_mm:.6f}*mm):")
            lines.append(f"        _ch = em.geo.Cylinder(_hr, {th_mm:.6f}*mm,")
            lines.append(f"            cs=em.GCS.displace(_hx, _hy, {z_bot_mm:.6f}*mm),")
            lines.append(f"            name=f\"layer_{gi}_cu_clear_{{_hi}}\")")
            lines.append(f"        layer_{gi} = em.geo.subtract(layer_{gi}, _ch)")
            lines.append("")

        lines.append("# ── Lumped port sheets ───────────────────────────────────────────────────")
        lines.append("# Port sheet = 2D Plate at the far end of each trace.")
        lines.append("# Width = trace width, Height = distance to nearest reference plane.")
        lines.append("# direction=(0,0,1) = E-field vertical (GND→signal).")
        for script_port in active_script_ports:
            plate = port_plate_by_script[script_port]
            role_txt = "input" if script_port in (1, 2) and is_diff_mode else ("output" if script_port in (3, 4) else ("input" if script_port == 1 else "output"))
            lines.append(
                f"# Port {script_port} ({role_txt}): far end of feed trace — height={port_height_by_script[script_port]:.4f}mm"
            )
            lines.append(f"port{script_port}_sheet = em.geo.Plate(")
            lines.append(f"    ({plate['ox']:.6f}*mm, {plate['oy']:.6f}*mm, {plate['oz']:.6f}*mm),")
            lines.append(f"    ({plate['ux']:.6f}*mm, {plate['uy']:.6f}*mm, 0),")
            lines.append(f"    (0, 0, {plate['vz']:.6f}*mm),")
            lines.append(f"    name=\"port{script_port}_sheet\")")

        lines.append("# ── Geometry finalisation ────────────────────────────────────────────────")
        lines.append("# Open region (1mm padding around structure)")
        lines.append("air = em.geo.open_region(1*mm, 1*mm, 1*mm)")
        lines.append("")
        lines.append("m.commit_geometry()")
        if show_structure_in_emerge:
            if show_labels_in_emerge:
                lines.append("m.view(labels=True)")
            else:
                lines.append("m.view()")
        lines.append("")
        lines.append("# ── Simulation setup ─────────────────────────────────────────────────────")
        lines.append(f"m.mw.set_frequency_range({f_start:.4f}e9, {f_stop:.4f}e9, {n_pts})")
        lines.append("")
        lines.append("# ── Mesh ─────────────────────────────────────────────────────────────────")
        lines.append(f"m.mw.set_resolution({mesh_resolution:.3f})  # fraction of max wavelength")
        lines.append("m.settings.safe_mode = True")
        lines.append(f"mesh_local_enabled = {mesh_local_enabled}")
        lines.append(f"mesh_div_via = {mesh_div_via}")
        lines.append(f"mesh_div_ports = {mesh_div_ports}")
        lines.append(f"mesh_div_feed = {mesh_div_feed}")
        lines.append(f"mesh_div_planes = {mesh_div_planes}")
        lines.append(f"mesh_div_stitching = {mesh_div_stitching}")
        lines.append("if mesh_local_enabled:")
        lines.append("    try:")
        lines.append(f"        _local_base = max(1e-6, {mesh_resolution:.6f}) * mm")
        lines.append("        _via_size = _local_base / max(mesh_div_via, 1)")
        lines.append("        _ports_size = _local_base / max(mesh_div_ports, 1)")
        lines.append("        _feed_size = _local_base / max(mesh_div_feed, 1)")
        lines.append("        _plane_size = _local_base / max(mesh_div_planes, 1)")
        lines.append("        _stitch_size = _local_base / max(mesh_div_stitching, 1)")
        lines.append("        m.mesher.set_boundary_size(via_barrel, _via_size)")
        if is_diff_mode:
            lines.append("        m.mesher.set_boundary_size(via_barrel_diff, _via_size)")
        if stub_enabled:
            lines.append("        m.mesher.set_boundary_size(stub_barrel, _via_size)")
            if is_diff_mode:
                lines.append("        m.mesher.set_boundary_size(stub_barrel_diff, _via_size)")
        lines.append("        m.mesher.set_boundary_size(trace_start, _feed_size)")
        lines.append("        m.mesher.set_boundary_size(trace_end, _feed_size)")
        for script_port in active_script_ports:
            lines.append(f"        m.mesher.set_boundary_size(port{script_port}_sheet, _ports_size)")
        for li, row in enumerate(stackup):
            if row.get("is_copper") and str(row.get("role", "Signal")) == "Plane":
                lines.append(f"        m.mesher.set_boundary_size(layer_{li}, _plane_size)")
        if self._stitch_enable.isChecked():
            lines.append("        if 'stitch_group' in locals():")
            lines.append("            m.mesher.set_boundary_size(stitch_group, _stitch_size)")
        lines.append("    except Exception as _mesh_err:")
        lines.append("        print(f'Local mesh refinement skipped: {_mesh_err}')")
        lines.append("")
        lines.append("")
        lines.append("")
        lines.append("")
        lines.append("# ── Lumped ports ─────────────────────────────────────────────────────────")
        if is_diff_mode:
            lines.append("# Differential numbering: ports 1,2 = inputs on entry layer; ports 3,4 = outputs on exit layer.")
        else:
            lines.append("# Single-ended numbering: port 1 = input on entry layer; port 2 = output on exit layer.")
        for script_port in active_script_ports:
            lines.append(f"p{script_port} = m.mw.bc.LumpedPort(port{script_port}_sheet, {script_port},")
            lines.append(
                f"    width={port_width_by_script[script_port]:.6f}*mm, height={port_height_by_script[script_port]:.6f}*mm,"
            )
            lines.append("    direction=(0, 0, 1))")
            lines.append("")
        lines.append("m.generate_mesh()")
        if show_mesh_in_emerge:
            lines.append("m.view(plot_mesh=True)")
        lines.append("")
        lines.append("# ── Simulation ───────────────────────────────────────────────────────────")
        lines.append(f"data = m.mw.run_sweep(False, n_workers={n_workers})")
        lines.append("m.save()")
        lines.append("")
        lines.append("ResultTimeCode = datetime.datetime.now().strftime(\"%Y%m%d-%H%M%S\")")
        lines.append(f"sparam_fit_enabled = {sparam_fit_enabled}")
        lines.append(f"sparam_fit_points = {sparam_fit_n_pts}")
        lines.append("if sparam_fit_enabled:")
        lines.append("    try:")
        lines.append("        _fit_freq = data.scalar.grid.dense_f(int(max(3, sparam_fit_points)))")
        lines.append("        _fit_export = {'freq_hz': np.asarray(_fit_freq)}")
        lines.append(f"        for _i in range(1, {len(active_script_ports) + 1}):")
        lines.append(f"            for _j in range(1, {len(active_script_ports) + 1}):")
        lines.append("                _fit_export[f'S{_i}{_j}'] = data.scalar.grid.model_S(_i, _j, _fit_freq)")
        lines.append("        _fit_path = os.path.join(currDir, f\"{ProjectName}_{ResultTimeCode}_fit.npz\")")
        lines.append("        np.savez(_fit_path, **_fit_export)")
        lines.append("        print(f'Fitted S-parameters saved to: {_fit_path}')")
        lines.append("    except Exception as _fit_err:")
        lines.append("        print(f'S-parameter fitting skipped: {_fit_err}')")
        lines.append("")
        _n_ports = len(active_script_ports)
        _ts_ext = f"s{_n_ports}p"
        lines.append("data.scalar.grid.export_touchstone(")
        lines.append(f"    os.path.join(currDir, f\"{{ProjectName}}_{{ResultTimeCode}}.{_ts_ext}\"),")
        lines.append("    Z0ref=50, format=\"RI\",")
        lines.append("    custom_comments=[ProjectName, \"SP Studio\"],")
        lines.append("    funit=\"HZ\")")
        lines.append("print(f\"Done. Results saved to: {currDir}\")")
        lines.append("")





        return "\n".join(lines)

    def _on_gen_script(self):
        script = self._generate_emerge_script()
        self._script_edit.setPlainText(script)

    def _on_save_script(self):
        script = self._generate_emerge_script()
        self._script_edit.setPlainText(script)
        folder = self._get_emerge_folder()
        if folder is None:
            return
        script_path = folder / "via_simulation.py"
        script_path.write_text(script, encoding="utf-8")
        QMessageBox.information(self, "Save Script", f"Script saved to:\n{script_path}")

    def _on_copy_script(self):
        clip = QApplication.clipboard()
        if clip:
            clip.setText(self._script_edit.toPlainText())

    # ── State serialisation ───────────────────────────────────────────────

    def export_project_state(self) -> dict:
        stackup = self._read_stackup()
        copper_layers = [(i, r) for i, r in enumerate(stackup) if r["is_copper"]]

        via_from_c = self._via_from_combo.currentIndex()
        via_to_c   = self._via_to_combo.currentIndex()
        stub_text  = self._stub_combo.currentText()

        # stub_idx: index in full stackup list, -1 = None
        stub_idx = -1
        if stub_text and stub_text != "None":
            for i, r in enumerate(stackup):
                if r["name"] == stub_text:
                    stub_idx = i
                    break

        return {
            "window_number": self.window_number,
            "name": self._name_edit.text().strip(),
            "stackup": stackup,
            "via": {
                "drill_um":       self._drill_um.value(),
                "pad_um":         self._pad_um.value(),
                "antipad_um":     self._antipad_um.value(),
                "via_from_idx":   via_from_c,
                "via_to_idx":     via_to_c,
                "stub_idx":       stub_idx,
                "is_differential": self._radio_diff.isChecked(),
                "diff_spacing_um": self._diff_spacing.value(),
            },
            "stitching": {
                "enabled":       self._stitch_enable.isChecked(),
                "pattern":       self._stitch_pattern.currentText(),
                "from_idx":      self._stitch_from_combo.currentIndex(),
                "to_idx":        self._stitch_to_combo.currentIndex(),
                "deleted_indices": sorted(self._stitch_deleted_indices),
                "n_vias":        self._stitch_n.value(),
                "ring_radius_um": self._stitch_ring_r.value(),
                "drill_um":      self._stitch_drill.value(),
                "pad_um":        self._stitch_pad.value(),
            },
            "feed": {
                "ports": {
                    "1": self._get_feed_port_config(1),
                    "2": self._get_feed_port_config(2),
                    "3": self._get_feed_port_config(3),
                    "4": self._get_feed_port_config(4),
                },
                "start": {
                    "type": self._feed_start_type.currentText(),
                    "trace_width_um": self._feed_start_trace_width_um.value(),
                    "trace_length_um": self._feed_start_trace_length_um.value(),
                    "trace_angle_deg": self._feed_start_trace_angle_deg.value(),
                },
                "end": {
                    "type": self._feed_end_type.currentText(),
                    "trace_width_um": self._feed_end_trace_width_um.value(),
                    "trace_length_um": self._feed_end_trace_length_um.value(),
                    "trace_angle_deg": self._feed_end_trace_angle_deg.value(),
                },
            },
            "simulation": {
                "f_start_ghz": self._f_start.value(),
                "f_stop_ghz":  self._f_stop.value(),
                "n_pts":       self._n_pts.value(),
                "resolution_mm": self._res_mm.value(),
                "mesh_local_enabled": self._mesh_local_enable.isChecked(),
                "mesh_div_via": self._mesh_factor_sliders["via"].value(),
                "mesh_div_ports": self._mesh_factor_sliders["ports"].value(),
                "mesh_div_feed": self._mesh_factor_sliders["feed"].value(),
                "mesh_div_planes": self._mesh_factor_sliders["planes"].value(),
                "mesh_div_stitching": self._mesh_factor_sliders["stitching"].value(),
                "sparam_fit_enabled": self._sparam_fit_enable.isChecked(),
                "sparam_fit_n_pts": self._sparam_fit_n_pts.value(),
                "n_workers":   self._n_workers.value(),
                "show_structure_in_emerge": self._show_structure_in_emerge.isChecked(),
                "show_labels_in_emerge": self._show_labels_in_emerge.isChecked(),
                "show_mesh_in_emerge": self._show_mesh_in_emerge.isChecked(),
            },
            "view3d": {
                "hidden_keys": sorted(self._hidden_3d_keys),
            },
        }

    def import_project_state(self, state: dict) -> None:
        self._suppress_signals = True

        self.window_number = state.get("window_number", self.window_number)

        saved_name = state.get("name", "").strip()
        default_name = f"Via Analysis #{self.window_number}"
        self._name_edit.setText(saved_name if saved_name else default_name)
        self.setWindowTitle(self._name_edit.text())

        via = state.get("via", {})
        stackup = state.get("stackup", _DEFAULT_STACKUP)
        normalized_stackup: list[dict] = []
        for row in stackup:
            row_copy = dict(row)
            is_copper = row_copy.get("is_copper", True)
            if not row_copy.get("role"):
                if not is_copper:
                    row_copy["role"] = "Dielectric"
                elif any(tag in row_copy.get("name", "").lower() for tag in ("gnd", "ground", "pwr", "power", "plane")):
                    row_copy["role"] = "Plane"
                else:
                    row_copy["role"] = "Signal"
            if "net" not in row_copy:
                name_lower = row_copy.get("name", "").lower()
                if row_copy.get("is_copper") and row_copy.get("role") == "Plane":
                    if any(tag in name_lower for tag in ("gnd", "ground")):
                        row_copy["net"] = "GND"
                    elif any(tag in name_lower for tag in ("pwr", "power", "vcc", "vdd")):
                        row_copy["net"] = "PWR"
                    else:
                        row_copy["net"] = ""
                else:
                    row_copy["net"] = ""
            normalized_stackup.append(row_copy)
        self._load_stackup_rows(normalized_stackup)

        self._drill_um.setValue(  via.get("drill_um",   250.0))
        self._pad_um.setValue(    via.get("pad_um",     500.0))
        self._antipad_um.setValue(via.get("antipad_um", 800.0))

        self._update_via_layer_combos()

        vf = via.get("via_from_idx", 0)
        vt = via.get("via_to_idx",   max(0, self._via_from_combo.count() - 1))
        if vf < self._via_from_combo.count():
            self._via_from_combo.setCurrentIndex(vf)
        if vt < self._via_to_combo.count():
            self._via_to_combo.setCurrentIndex(vt)

        self._update_stub_combo()
        stub_idx = via.get("stub_idx", -1)
        if stub_idx >= 0:
            stub_name = stackup[stub_idx]["name"] if stub_idx < len(stackup) else ""
            si = self._stub_combo.findText(stub_name)
            if si >= 0:
                self._stub_combo.setCurrentIndex(si)

        is_diff = via.get("is_differential", False)
        self._radio_diff.setChecked(is_diff)
        self._radio_se.setChecked(not is_diff)
        self._diff_spacing.setEnabled(is_diff)
        self._diff_spacing.setValue(via.get("diff_spacing_um", 400.0))

        stitch = state.get("stitching", {})
        self._stitch_enable.setChecked(stitch.get("enabled", False))
        pat_idx = self._stitch_pattern.findText(stitch.get("pattern", "Ring"))
        if pat_idx >= 0:
            self._stitch_pattern.setCurrentIndex(pat_idx)
        sf = stitch.get("from_idx", 0)
        st = stitch.get("to_idx", max(0, self._stitch_to_combo.count() - 1))
        if 0 <= sf < self._stitch_from_combo.count():
            self._stitch_from_combo.setCurrentIndex(sf)
        if 0 <= st < self._stitch_to_combo.count():
            self._stitch_to_combo.setCurrentIndex(st)
        deleted = stitch.get("deleted_indices", [])
        if isinstance(deleted, list):
            self._stitch_deleted_indices = {int(i) for i in deleted if isinstance(i, (int, float)) and int(i) >= 0}
        else:
            self._stitch_deleted_indices = set()
        self._stitch_n.setValue(       stitch.get("n_vias",        8))
        self._stitch_ring_r.setValue(  stitch.get("ring_radius_um", 2000.0))
        self._stitch_drill.setValue(   stitch.get("drill_um",       250.0))
        self._stitch_pad.setValue(     stitch.get("pad_um",         500.0))

        feed = state.get("feed", {})
        legacy_feed = {
            "type": feed.get("type", "Trace"),
            "trace_width_um": feed.get("trace_width_um", 250.0),
            "trace_length_um": feed.get("trace_length_um", 2000.0),
            "trace_angle_deg": feed.get("trace_angle_deg", 0.0),
        }
        start_feed = feed.get("start", legacy_feed)
        end_feed = feed.get("end", legacy_feed)
        feed_ports = feed.get("ports", {}) if isinstance(feed.get("ports", {}), dict) else {}

        port1_feed = feed_ports.get("1", start_feed)
        port2_feed = feed_ports.get("2", start_feed if is_diff else end_feed)
        port3_feed = feed_ports.get("3", end_feed)
        port4_feed = feed_ports.get("4", end_feed)

        # Backward compatibility: legacy files only had start/end groups.
        # In that case, mirror start→port2 and end→port4 when missing.
        if "2" not in feed_ports:
            port2_feed = dict(start_feed if is_diff else end_feed)
        if "4" not in feed_ports:
            port4_feed = dict(end_feed)

        self._set_feed_port_config(1, {
            "type": port1_feed.get("type", "Trace"),
            "trace_width_um": port1_feed.get("trace_width_um", 250.0),
            "trace_length_um": port1_feed.get("trace_length_um", 2000.0),
            "trace_angle_deg": port1_feed.get("trace_angle_deg", 180.0),
        })
        self._set_feed_port_config(2, {
            "type": port2_feed.get("type", "Trace"),
            "trace_width_um": port2_feed.get("trace_width_um", 250.0),
            "trace_length_um": port2_feed.get("trace_length_um", 2000.0),
            "trace_angle_deg": port2_feed.get("trace_angle_deg", 180.0 if is_diff else 0.0),
        })
        self._set_feed_port_config(3, {
            "type": port3_feed.get("type", "Trace"),
            "trace_width_um": port3_feed.get("trace_width_um", 250.0),
            "trace_length_um": port3_feed.get("trace_length_um", 2000.0),
            "trace_angle_deg": port3_feed.get("trace_angle_deg", 0.0),
        })
        self._set_feed_port_config(4, {
            "type": port4_feed.get("type", "Trace"),
            "trace_width_um": port4_feed.get("trace_width_um", 250.0),
            "trace_length_um": port4_feed.get("trace_length_um", 2000.0),
            "trace_angle_deg": port4_feed.get("trace_angle_deg", 0.0),
        })
        self._feed_controls_changed()

        sim = state.get("simulation", {})
        self._f_start.setValue(   sim.get("f_start_ghz",  0.01))
        self._f_stop.setValue(    sim.get("f_stop_ghz",   10.0))
        self._n_pts.setValue(     sim.get("n_pts",        11))
        self._res_mm.setValue(    sim.get("resolution_mm", 0.25))
        self._mesh_local_enable.setChecked(sim.get("mesh_local_enabled", False))
        self._mesh_factor_sliders["via"].setValue(int(sim.get("mesh_div_via", 1)))
        self._mesh_factor_sliders["ports"].setValue(int(sim.get("mesh_div_ports", 1)))
        self._mesh_factor_sliders["feed"].setValue(int(sim.get("mesh_div_feed", 1)))
        self._mesh_factor_sliders["planes"].setValue(int(sim.get("mesh_div_planes", 1)))
        self._mesh_factor_sliders["stitching"].setValue(int(sim.get("mesh_div_stitching", 1)))
        self._sparam_fit_enable.setChecked(sim.get("sparam_fit_enabled", False))
        self._sparam_fit_n_pts.setValue(sim.get("sparam_fit_n_pts", 401))
        self._n_workers.setValue( sim.get("n_workers",    4))
        self._show_structure_in_emerge.setChecked(sim.get("show_structure_in_emerge", True))
        self._show_labels_in_emerge.setChecked(sim.get("show_labels_in_emerge", False))
        self._show_mesh_in_emerge.setChecked(sim.get("show_mesh_in_emerge", True))
        self._simulation_controls_changed()
        self._mesh_controls_changed()

        view3d = state.get("view3d", {})
        hidden_keys = view3d.get("hidden_keys", [])
        if isinstance(hidden_keys, list):
            self._hidden_3d_keys = {str(k) for k in hidden_keys}
        else:
            self._hidden_3d_keys = set()

        self._suppress_signals = False
        self._refresh_stitching_coords_table()
        if _GL_AVAILABLE:
            self._rebuild_3d()

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _mk_dspin(lo: float, hi: float, val: float,
                  suffix: str = "", decimals: int = 1) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setDecimals(decimals)
        if suffix:
            sb.setSuffix(suffix)
        return sb
