"""Interactive transmission-line calculator dialog.

Opened from the right-click context menu of a substrate block. Every
parameter — substrate stack-up (εr, h, t, tan δ, σ, h_top/h_bot) and
trace geometry (W, S) — is editable. The user picks a single variable to
solve for, sets a target Z_se or Z_diff, and the dialog runs a bisection
that holds all the other parameters fixed.

An "Apply to substrate" button invokes a caller-supplied callback with
the freshly computed `SubstrateSpec` so the schematic block is updated
in place. The dialog itself never mutates the document.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sparams_utility.models.circuit import SubstrateSpec, TransmissionLineSpec
from sparams_utility.transmission_lines import (
    coupled_microstrip_modes,
    coupled_stripline_modes,
    cpw_coupled_modes,
    cpw_z0_eeff,
    microstrip_z0_eeff,
    stripline_z0,
)


_LINE_KIND_LABELS = {
    "microstrip": "Microstrip (single-ended)",
    "microstrip_coupled": "Coupled Microstrip (edge-coupled)",
    "stripline": "Stripline (single-ended)",
    "stripline_coupled": "Coupled Stripline (edge-coupled)",
    "cpw": "CPW (single-ended)",
    "cpw_coupled": "Coupled CPW (edge-coupled)",
}


def _variables_for(kind: str, sub_kind: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = [
        ("W", "W (trace width)"),
        ("epsilon_r", "εr (relative permittivity)"),
        ("t", "t (conductor thickness)"),
    ]
    if sub_kind == "substrate":
        items.append(("h", "h (dielectric thickness)"))
    else:
        items.append(("h_top", "h_top (gap above conductor)"))
        items.append(("h_bot", "h_bot (gap below conductor)"))
    if kind.startswith("cpw"):
        items.append(("S_slot", "S_slot (CPW slot to ground)"))
    if kind.endswith("_coupled"):
        items.append(("S", "S (edge-to-edge spacing)"))
    return items


def _bisect(fn, target: float, lo: float, hi: float, *,
            tol: float = 1e-3, max_iter: int = 100) -> Optional[float]:
    """Bisect for fn(x) == target in [lo, hi]; expand bracket if needed."""
    f_lo = fn(lo) - target
    f_hi = fn(hi) - target
    for _ in range(10):
        if f_lo * f_hi <= 0.0:
            break
        lo = max(lo * 0.5, 1e-12)
        hi *= 2.0
        f_lo = fn(lo) - target
        f_hi = fn(hi) - target
    if f_lo * f_hi > 0.0:
        return None
    a, b = lo, hi
    fa = f_lo
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = fn(m) - target
        if abs(fm) < max(tol * abs(target), 1e-6):
            return m
        if fa * fm <= 0.0:
            b = m
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)


class TLineCalculatorDialog(QDialog):
    """Modal dialog: trace + substrate calculator with a single-unknown solver."""

    def __init__(
        self,
        substrate: SubstrateSpec,
        substrate_name: str,
        substrate_kind: str,
        parent: Optional[QWidget] = None,
        *,
        apply_callback: Optional[Callable[[SubstrateSpec], None]] = None,
        tline_spec: Optional[TransmissionLineSpec] = None,
        tline_label: str = "",
        tline_callback: Optional[Callable[[TransmissionLineSpec], None]] = None,
    ) -> None:
        super().__init__(parent)
        title_subject = tline_label if tline_spec is not None else substrate_name
        self.setWindowTitle(f"T Line Calculator \u2014 {title_subject}")
        self._initial_substrate = substrate
        self._substrate_kind = substrate_kind  # "substrate" | "substrate_stripline"
        self._apply_callback = apply_callback
        self._tline_spec = tline_spec
        self._tline_callback = tline_callback
        self._block_signals = False

        outer = QVBoxLayout(self)

        # Line kind.
        kind_box = QGroupBox("Line kind")
        kind_form = QFormLayout(kind_box)
        self._kind_combo = QComboBox()
        for key, label in _LINE_KIND_LABELS.items():
            if substrate_kind == "substrate" and not (
                key.startswith("microstrip") or key.startswith("cpw")
            ):
                continue
            if substrate_kind == "substrate_stripline" and not key.startswith("stripline"):
                continue
            # In tline mode, only allow kinds matching the coupling state
            # of the schematic block (port count is fixed).
            if tline_spec is not None:
                spec_is_coupled = tline_spec.line_kind.endswith("_coupled")
                if key.endswith("_coupled") != spec_is_coupled:
                    continue
            self._kind_combo.addItem(label, key)
        # Pre-select the line kind coming from the tline spec, if any.
        if tline_spec is not None:
            preset_idx = self._kind_combo.findData(tline_spec.line_kind)
            if preset_idx >= 0:
                self._kind_combo.setCurrentIndex(preset_idx)
        kind_form.addRow("Kind", self._kind_combo)
        outer.addWidget(kind_box)

        # Substrate parameters (editable).
        sub_box = QGroupBox(f"Substrate — {substrate_name}")
        sub_form = QFormLayout(sub_box)
        self._eps_spin = self._mk_spin(1.0, 50.0, 0.01, substrate.epsilon_r, decimals=3)
        sub_form.addRow("εr", self._eps_spin)
        self._tand_spin = self._mk_spin(0.0, 1.0, 0.0001, substrate.loss_tangent, decimals=4)
        sub_form.addRow("tan δ", self._tand_spin)
        self._sigma_spin = self._mk_spin(1e4, 1e9, 1e5, substrate.conductivity_s_per_m,
                                         decimals=0, suffix=" S/m")
        sub_form.addRow("σ", self._sigma_spin)
        self._t_spin = self._mk_dim(default_um=substrate.conductor_thickness_m * 1e6,
                                    max_um=20000.0)
        sub_form.addRow("t (conductor)", self._t_spin)
        if substrate_kind == "substrate":
            self._h_spin = self._mk_dim(default_um=substrate.height_m * 1e6, max_um=50000.0)
            sub_form.addRow("h (dielectric)", self._h_spin)
            self._h_top_spin = None
            self._h_bot_spin = None
        else:
            self._h_spin = None
            self._h_top_spin = self._mk_dim(
                default_um=substrate.stripline_h_top_m * 1e6, max_um=50000.0
            )
            sub_form.addRow("h_top (above)", self._h_top_spin)
            self._h_bot_spin = self._mk_dim(
                default_um=substrate.stripline_h_bottom_m * 1e6, max_um=50000.0
            )
            sub_form.addRow("h_bot (below)", self._h_bot_spin)
        outer.addWidget(sub_box)

        # Geometry.
        geom_box = QGroupBox("Trace geometry")
        geom_form = QFormLayout(geom_box)
        w_default_um = (tline_spec.width_m * 1e6) if tline_spec is not None else 200.0
        s_default_um = (tline_spec.spacing_m * 1e6) if tline_spec is not None else 200.0
        self._w_spin = self._mk_dim(default_um=w_default_um, max_um=20000.0)
        geom_form.addRow("W (width)", self._w_spin)
        self._s_label = QLabel("S (spacing)")
        self._s_spin = self._mk_dim(default_um=s_default_um, max_um=20000.0)
        geom_form.addRow(self._s_label, self._s_spin)
        # CPW slot (gap to outer ground). Only shown for CPW kinds.
        slot_default_um = (
            (tline_spec.cpw_slot_m * 1e6) if tline_spec is not None else 150.0
        )
        self._slot_label = QLabel("S_slot (CPW)")
        self._slot_spin = self._mk_dim(default_um=slot_default_um, max_um=20000.0)
        geom_form.addRow(self._slot_label, self._slot_spin)
        # Length spin (mm) — only meaningful when applying to a tline.
        self._l_label = QLabel("L (length)")
        self._l_spin = QDoubleSpinBox()
        self._l_spin.setDecimals(3)
        self._l_spin.setRange(0.001, 10000.0)
        self._l_spin.setSingleStep(0.5)
        self._l_spin.setSuffix(" mm")
        self._l_spin.setValue((tline_spec.length_m * 1e3) if tline_spec is not None else 10.0)
        self._l_spin.setVisible(tline_spec is not None)
        self._l_label.setVisible(tline_spec is not None)
        geom_form.addRow(self._l_label, self._l_spin)
        outer.addWidget(geom_box)

        # Solver.
        solver_box = QGroupBox("Solver")
        solver_form = QFormLayout(solver_box)
        self._var_combo = QComboBox()
        solver_form.addRow("Solve for", self._var_combo)
        self._target_quantity = QComboBox()
        self._target_quantity.addItem("Z_se (single-ended Z₀)", "z_se")
        self._target_quantity.addItem("Z_diff (differential)", "z_diff")
        solver_form.addRow("Target quantity", self._target_quantity)
        self._target_value = QDoubleSpinBox()
        self._target_value.setDecimals(2)
        self._target_value.setRange(1.0, 1000.0)
        self._target_value.setSingleStep(1.0)
        self._target_value.setValue(50.0)
        self._target_value.setSuffix(" Ω")
        solver_form.addRow("Target value", self._target_value)
        self._solve_btn = QPushButton("Solve")
        solver_form.addRow("", self._solve_btn)
        outer.addWidget(solver_box)

        # Results.
        res_box = QGroupBox("Results")
        res_form = QFormLayout(res_box)
        self._res_z0 = QLabel("—")
        self._res_eeff = QLabel("—")
        self._res_z0e = QLabel("—")
        self._res_z0o = QLabel("—")
        self._res_zdiff = QLabel("—")
        self._res_zcommon = QLabel("—")
        res_form.addRow("Z_se (Ω)", self._res_z0)
        res_form.addRow("ε_eff", self._res_eeff)
        res_form.addRow("Z₀,e (Ω)", self._res_z0e)
        res_form.addRow("Z₀,o (Ω)", self._res_z0o)
        res_form.addRow("Z_diff (Ω)", self._res_zdiff)
        res_form.addRow("Z_common (Ω)", self._res_zcommon)
        outer.addWidget(res_box)

        # Buttons.
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        if self._tline_callback is not None:
            self._apply_btn = QPushButton("Apply to T Line")
        else:
            self._apply_btn = QPushButton("Apply to substrate")
        bb.addButton(self._apply_btn, QDialogButtonBox.ActionRole)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)
        self._apply_btn.setEnabled(
            self._apply_callback is not None or self._tline_callback is not None
        )

        # Wiring.
        self._kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        for sp in self._all_param_spins():
            sp.valueChanged.connect(self._recompute)
        self._target_quantity.currentIndexChanged.connect(self._on_target_quantity_changed)
        self._solve_btn.clicked.connect(self._solve_for_unknown)
        self._apply_btn.clicked.connect(self._on_apply_clicked)

        self._on_kind_changed()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mk_spin(lo: float, hi: float, step: float, value: float,
                 decimals: int = 3, suffix: str = "") -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(step)
        sp.setValue(value)
        if suffix:
            sp.setSuffix(suffix)
        return sp

    @staticmethod
    def _mk_dim(*, default_um: float, max_um: float) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setDecimals(2)
        sp.setRange(0.1, max_um)
        sp.setSingleStep(5.0)
        sp.setValue(default_um)
        sp.setSuffix(" µm")
        return sp

    def _all_param_spins(self) -> list[QDoubleSpinBox]:
        spins: list[QDoubleSpinBox] = [
            self._eps_spin, self._tand_spin, self._sigma_spin, self._t_spin,
            self._w_spin, self._s_spin, self._slot_spin,
        ]
        if self._h_spin is not None:
            spins.append(self._h_spin)
        if self._h_top_spin is not None:
            spins.append(self._h_top_spin)
        if self._h_bot_spin is not None:
            spins.append(self._h_bot_spin)
        return spins

    # ------------------------------------------------------------------ #
    def _current_kind(self) -> str:
        return str(self._kind_combo.currentData())

    def _is_coupled(self) -> bool:
        return self._current_kind().endswith("_coupled")

    def _on_kind_changed(self) -> None:
        coupled = self._is_coupled()
        is_cpw = self._current_kind().startswith("cpw")
        self._s_spin.setVisible(coupled)
        self._s_label.setVisible(coupled)
        self._slot_spin.setVisible(is_cpw)
        self._slot_label.setVisible(is_cpw)
        self._res_z0e.setEnabled(coupled)
        self._res_z0o.setEnabled(coupled)
        self._res_zdiff.setEnabled(coupled)
        self._res_zcommon.setEnabled(coupled)
        self._var_combo.blockSignals(True)
        self._var_combo.clear()
        for key, label in _variables_for(self._current_kind(), self._substrate_kind):
            self._var_combo.addItem(label, key)
        self._var_combo.blockSignals(False)
        # Enable/disable Z_diff target choice.
        idx = self._target_quantity.findData("z_diff")
        if idx >= 0:
            self._target_quantity.model().item(idx).setEnabled(coupled)
            if not coupled and self._target_quantity.currentData() == "z_diff":
                self._target_quantity.setCurrentIndex(self._target_quantity.findData("z_se"))
        self._on_target_quantity_changed()
        self._recompute()

    def _on_target_quantity_changed(self) -> None:
        is_diff = self._target_quantity.currentData() == "z_diff"
        self._target_value.setValue(100.0 if is_diff else 50.0)

    # ------------------------------------------------------------------ #
    def _build_substrate(self) -> SubstrateSpec:
        eps = float(self._eps_spin.value())
        tand = float(self._tand_spin.value())
        sigma = float(self._sigma_spin.value())
        t_m = float(self._t_spin.value()) * 1e-6
        if self._substrate_kind == "substrate":
            h_m = float(self._h_spin.value()) * 1e-6 if self._h_spin else self._initial_substrate.height_m
            h_top_m = self._initial_substrate.stripline_h_top_m
            h_bot_m = self._initial_substrate.stripline_h_bottom_m
        else:
            h_top_m = float(self._h_top_spin.value()) * 1e-6
            h_bot_m = float(self._h_bot_spin.value()) * 1e-6
            h_m = h_top_m + t_m + h_bot_m
        return replace(
            self._initial_substrate,
            epsilon_r=eps,
            loss_tangent=tand,
            conductivity_s_per_m=sigma,
            conductor_thickness_m=t_m,
            height_m=h_m,
            stripline_h_top_m=h_top_m,
            stripline_h_bottom_m=h_bot_m,
        )

    def current_substrate(self) -> SubstrateSpec:
        return self._build_substrate()

    def _stripline_b(self, sub: SubstrateSpec) -> float:
        return float(sub.stripline_h_top_m + sub.conductor_thickness_m + sub.stripline_h_bottom_m)

    def _z_se(self, sub: SubstrateSpec, w_m: float, slot_m: float = 0.0) -> float:
        kind = self._current_kind()
        if kind.startswith("microstrip"):
            z0, _ = microstrip_z0_eeff(w_m, sub.height_m, sub.epsilon_r, sub.conductor_thickness_m)
            return z0
        if kind.startswith("cpw"):
            z0, _ = cpw_z0_eeff(w_m, slot_m, sub.height_m, sub.epsilon_r, sub.conductor_thickness_m)
            return z0
        b = self._stripline_b(sub)
        return stripline_z0(w_m, b, sub.epsilon_r, sub.conductor_thickness_m)

    def _z_diff(self, sub: SubstrateSpec, w_m: float, s_m: float, slot_m: float = 0.0) -> float:
        kind = self._current_kind()
        if kind == "microstrip_coupled":
            z0e, z0o, _, _ = coupled_microstrip_modes(
                w_m, s_m, sub.height_m, sub.epsilon_r, sub.conductor_thickness_m
            )
        elif kind == "cpw_coupled":
            z0e, z0o, _, _ = cpw_coupled_modes(
                w_m, slot_m, s_m, sub.height_m, sub.epsilon_r, sub.conductor_thickness_m
            )
        else:
            b = self._stripline_b(sub)
            z0e, z0o = coupled_stripline_modes(
                w_m, s_m, b, sub.epsilon_r, sub.conductor_thickness_m
            )
        return 2.0 * z0o

    # ------------------------------------------------------------------ #
    def _recompute(self) -> None:
        if self._block_signals:
            return
        sub = self._build_substrate()
        w_m = float(self._w_spin.value()) * 1e-6
        slot_m = float(self._slot_spin.value()) * 1e-6
        kind = self._current_kind()
        if kind.startswith("microstrip"):
            z0, eeff = microstrip_z0_eeff(w_m, sub.height_m, sub.epsilon_r, sub.conductor_thickness_m)
            self._res_z0.setText(f"{z0:.2f}")
            self._res_eeff.setText(f"{eeff:.3f}")
        elif kind.startswith("cpw"):
            z0, eeff = cpw_z0_eeff(w_m, slot_m, sub.height_m, sub.epsilon_r, sub.conductor_thickness_m)
            self._res_z0.setText(f"{z0:.2f}")
            self._res_eeff.setText(f"{eeff:.3f}")
        else:
            b = self._stripline_b(sub)
            z0 = stripline_z0(w_m, b, sub.epsilon_r, sub.conductor_thickness_m)
            self._res_z0.setText(f"{z0:.2f}")
            self._res_eeff.setText(f"{sub.epsilon_r:.3f}")
        if self._is_coupled():
            s_m = float(self._s_spin.value()) * 1e-6
            if kind == "microstrip_coupled":
                z0e, z0o, eeff_e, eeff_o = coupled_microstrip_modes(
                    w_m, s_m, sub.height_m, sub.epsilon_r, sub.conductor_thickness_m
                )
                self._res_eeff.setText(f"e={eeff_e:.3f} / o={eeff_o:.3f}")
            elif kind == "cpw_coupled":
                z0e, z0o, eeff_e, eeff_o = cpw_coupled_modes(
                    w_m, slot_m, s_m, sub.height_m, sub.epsilon_r, sub.conductor_thickness_m
                )
                self._res_eeff.setText(f"e={eeff_e:.3f} / o={eeff_o:.3f}")
            else:
                b = self._stripline_b(sub)
                z0e, z0o = coupled_stripline_modes(
                    w_m, s_m, b, sub.epsilon_r, sub.conductor_thickness_m
                )
            self._res_z0e.setText(f"{z0e:.2f}")
            self._res_z0o.setText(f"{z0o:.2f}")
            self._res_zdiff.setText(f"{2.0 * z0o:.2f}")
            self._res_zcommon.setText(f"{0.5 * z0e:.2f}")
        else:
            self._res_z0e.setText("—")
            self._res_z0o.setText("—")
            self._res_zdiff.setText("—")
            self._res_zcommon.setText("—")

    # ------------------------------------------------------------------ #
    def _solve_for_unknown(self) -> None:
        var = str(self._var_combo.currentData())
        target_kind = self._target_quantity.currentData()
        target = float(self._target_value.value())
        is_diff = target_kind == "z_diff"
        if is_diff and not self._is_coupled():
            QMessageBox.information(self, "T Line Calculator",
                                    "Z_diff is only defined for coupled lines.")
            return

        sub_now = self._build_substrate()
        w_now = float(self._w_spin.value()) * 1e-6
        s_now = float(self._s_spin.value()) * 1e-6 if self._is_coupled() else 0.0
        slot_now = float(self._slot_spin.value()) * 1e-6

        def evaluate(x: float) -> float:
            sub = sub_now
            w_m, s_m, slot_m = w_now, s_now, slot_now
            if var == "W":
                w_m = x * 1e-6
            elif var == "S":
                s_m = x * 1e-6
            elif var == "S_slot":
                slot_m = x * 1e-6
            elif var == "epsilon_r":
                sub = replace(sub, epsilon_r=x)
            elif var == "t":
                t_m = x * 1e-6
                if self._substrate_kind == "substrate_stripline":
                    sub = replace(
                        sub,
                        conductor_thickness_m=t_m,
                        height_m=sub.stripline_h_top_m + t_m + sub.stripline_h_bottom_m,
                    )
                else:
                    sub = replace(sub, conductor_thickness_m=t_m)
            elif var == "h":
                sub = replace(sub, height_m=x * 1e-6)
            elif var == "h_top":
                h_top = x * 1e-6
                sub = replace(
                    sub,
                    stripline_h_top_m=h_top,
                    height_m=h_top + sub.conductor_thickness_m + sub.stripline_h_bottom_m,
                )
            elif var == "h_bot":
                h_bot = x * 1e-6
                sub = replace(
                    sub,
                    stripline_h_bottom_m=h_bot,
                    height_m=sub.stripline_h_top_m + sub.conductor_thickness_m + h_bot,
                )
            if is_diff:
                return self._z_diff(sub, w_m, s_m, slot_m)
            return self._z_se(sub, w_m, slot_m)

        lo, hi = self._search_range_for(var)
        x = _bisect(evaluate, target, lo, hi)
        if x is None:
            QMessageBox.information(
                self, "T Line Calculator",
                "Could not find a value of the chosen variable that meets the target.\n"
                "Try adjusting the other parameters.",
            )
            return

        self._block_signals = True
        try:
            if var == "W":
                self._w_spin.setValue(max(0.1, x))
            elif var == "S":
                self._s_spin.setValue(max(0.1, x))
            elif var == "S_slot":
                self._slot_spin.setValue(max(0.1, x))
            elif var == "epsilon_r":
                self._eps_spin.setValue(max(1.0, x))
            elif var == "t":
                self._t_spin.setValue(max(0.1, x))
            elif var == "h":
                self._h_spin.setValue(max(0.1, x))
            elif var == "h_top":
                self._h_top_spin.setValue(max(0.1, x))
            elif var == "h_bot":
                self._h_bot_spin.setValue(max(0.1, x))
        finally:
            self._block_signals = False
        self._recompute()

    @staticmethod
    def _search_range_for(var: str) -> tuple[float, float]:
        if var == "epsilon_r":
            return 1.01, 30.0
        return 1.0, 10000.0

    # ------------------------------------------------------------------ #
    def _on_apply_clicked(self) -> None:
        if self._tline_callback is not None and self._tline_spec is not None:
            new_tspec = TransmissionLineSpec(
                line_kind=self._current_kind(),
                substrate_name=self._tline_spec.substrate_name,
                width_m=float(self._w_spin.value()) * 1e-6,
                length_m=float(self._l_spin.value()) * 1e-3,
                spacing_m=float(self._s_spin.value()) * 1e-6,
                z0_ref_ohm=self._tline_spec.z0_ref_ohm,
                cpw_slot_m=float(self._slot_spin.value()) * 1e-6,
            )
            self._tline_callback(new_tspec)
            QMessageBox.information(self, "T Line Calculator",
                                    "Transmission line updated with the calculator parameters.")
            return
        if self._apply_callback is None:
            return
        new_spec = self._build_substrate()
        self._apply_callback(new_spec)
        QMessageBox.information(self, "T Line Calculator",
                                "Substrate updated with the calculator parameters.")
