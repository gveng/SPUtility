"""Via analysis model for PCB signal integrity.

Models a PCB via as a lumped pi-network (series barrel inductance with
skin-effect resistance + shunt antipad/pad capacitances) with an optional
open-circuit stub for non-through or un-back-drilled vias.

Physical model
--------------
The via barrel is treated as a short coaxial section of length *h*
(board thickness):

  * Series inductance  L = (μ₀ h)/(2π) · ln(r_antipad / r_drill)  [H]
  * Antipad capacitance C_ap = 2π ε₀ εr h / ln(r_antipad / r_pad)  [F]
  * Pad capacitance    C_pad = ε₀ εr · π(r_pad²−r_drill²) / g_pad  [F/pad]
    where g_pad = (r_antipad − r_pad) is the annular gap to the plane.

Skin-effect barrel resistance (Wheeler):

  R = h / (σ · 2π r_drill · δ_s),  δ_s = √(2 / (ω μ₀ σ))

Open-circuit stub admittance (if l_stub > 0):

  Y_stub = j · tan(2π f l_stub / v_p) / Z_via
  Z_via  = (η₀ / (2π √εr)) · ln(r_antipad / r_drill)

The 2-port pi-network S-parameters are computed via the ABCD matrix.

References
----------
* S. Hall, G. Hall, J. McCall, "High-Speed Signal Propagation", 2003.
* E. Bogatin, "Signal and Power Integrity – Simplified", 2nd ed. 2009.
* D. Pozar, "Microwave Engineering", 4th ed., §7.1 (coaxial line).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
_MU0 = 4.0e-7 * np.pi         # H/m
_EPS0 = 8.854187817e-12        # F/m
_C0 = 2.997924580e8            # m/s
_ETA0 = _MU0 * _C0             # ≈ 376.730 Ω  (free-space impedance)


# ── Parameter dataclass ───────────────────────────────────────────────────────

@dataclass
class ViaParams:
    """Geometric and material parameters of a single PCB via.

    All dimensional quantities are in SI units (metres).
    """
    drill_diameter_m: float = 250e-6     # finished drill hole diameter
    pad_diameter_m: float = 500e-6       # copper pad outer diameter
    antipad_diameter_m: float = 800e-6   # clearance hole diameter in reference planes
    board_thickness_m: float = 1.6e-3    # total via barrel height
    stub_length_m: float = 0.0           # length of unused stub (0 if fully back-drilled)
    n_signal_pads: int = 1               # number of signal-layer pads adding to C_pad
    epsilon_r: float = 4.2               # PCB dielectric permittivity
    loss_tangent: float = 0.020          # dielectric loss tangent tan δ
    conductivity_s_per_m: float = 5.8e7  # conductor (copper) conductivity σ [S/m]
    reference_z_ohm: float = 50.0        # port reference impedance Z₀ [Ω]


# ── Derived model dataclass ───────────────────────────────────────────────────

@dataclass
class ViaModel:
    """Lumped-element parameters derived from :class:`ViaParams`."""
    L_barrel_H: float        # via barrel inductance  [H]
    C_antipad_F: float       # barrel ↔ antipad capacitance  [F]
    C_pad_F: float           # signal-pad ↔ plane capacitance (all pads total)  [F]
    Z_via_ohm: float         # via characteristic impedance  [Ω]
    stub_resonance_hz: float # quarter-wave stub frequency  [Hz]  (0 if no stub)


# ── Model computation ─────────────────────────────────────────────────────────

def compute_via_model(p: ViaParams) -> ViaModel:
    """Compute lumped electrical parameters from *p*.

    Raises
    ------
    ValueError
        If any dimensional constraint is violated (antipad ≤ drill, etc.).
    """
    r_d = p.drill_diameter_m / 2.0    # drill radius
    r_p = p.pad_diameter_m / 2.0      # pad radius
    r_a = p.antipad_diameter_m / 2.0  # antipad radius
    h = p.board_thickness_m

    if r_a <= r_d:
        raise ValueError(
            f"Antipad diameter ({p.antipad_diameter_m*1e6:.1f} µm) must be "
            f"larger than drill diameter ({p.drill_diameter_m*1e6:.1f} µm)."
        )
    if r_p <= r_d:
        raise ValueError(
            f"Pad diameter ({p.pad_diameter_m*1e6:.1f} µm) must be "
            f"larger than drill diameter ({p.drill_diameter_m*1e6:.1f} µm)."
        )
    if r_a <= r_p:
        raise ValueError(
            f"Antipad diameter ({p.antipad_diameter_m*1e6:.1f} µm) must be "
            f"larger than pad diameter ({p.pad_diameter_m*1e6:.1f} µm)."
        )

    # Barrel inductance (coaxial, outer conductor = antipad)
    L_barrel = (_MU0 * h) / (2.0 * np.pi) * np.log(r_a / r_d)

    # Antipad (barrel↔plane) capacitance (coaxial, outer = antipad, inner = pad)
    C_antipad = (2.0 * np.pi * _EPS0 * p.epsilon_r * h) / np.log(r_a / r_p)

    # Pad capacitance: annular pad area / annular gap to plane edge
    gap_pad = r_a - r_p
    A_annular = np.pi * (r_p**2 - r_d**2)
    C_pad_one = (_EPS0 * p.epsilon_r * A_annular / gap_pad) if gap_pad > 0.0 else 0.0
    C_pad = C_pad_one * max(1, p.n_signal_pads)

    # Via characteristic impedance (coaxial)
    Z_via = (_ETA0 / (2.0 * np.pi * np.sqrt(max(p.epsilon_r, 1.0)))) * np.log(r_a / r_d)

    # Quarter-wave stub resonance
    if p.stub_length_m > 1e-9:
        v_p = _C0 / np.sqrt(max(p.epsilon_r, 1.0))
        f_stub_res = v_p / (4.0 * p.stub_length_m)
    else:
        f_stub_res = 0.0

    return ViaModel(
        L_barrel_H=L_barrel,
        C_antipad_F=C_antipad,
        C_pad_F=C_pad,
        Z_via_ohm=Z_via,
        stub_resonance_hz=f_stub_res,
    )


# ── S-parameter synthesis ─────────────────────────────────────────────────────

def via_sparameters(
    p: ViaParams,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    """Compute 2-port S-parameter matrix for a through via.

    The pi-network model is::

             Z_series (R + jωL)
      p1 ──┬──────────────────────┬── p2
           │                      │
         Y_sh1                  Y_sh2
           │                      │
          GND                    GND

    Each shunt admittance is::

      Y_sh = jω · (C_antipad/2 + C_pad/2) + G_dielectric + Y_stub

    where the stub term ``Y_stub = j·tan(βl)/Z_via`` is non-zero only when
    ``p.stub_length_m > 0``.

    Parameters
    ----------
    p:
        Via parameters.
    freqs_hz:
        1-D array of frequency points [Hz].

    Returns
    -------
    S : np.ndarray, shape (N, 2, 2), complex
        S-parameter matrices, referenced to *p.reference_z_ohm*.
    """
    model = compute_via_model(p)

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    n = len(freqs_hz)

    omega = 2.0 * np.pi * freqs_hz  # (N,)
    Z0 = float(p.reference_z_ohm)
    sigma = float(p.conductivity_s_per_m)
    r_d = p.drill_diameter_m / 2.0

    # ── Series impedance (barrel inductance + skin resistance) ────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_s = np.where(
            omega > 0.0,
            np.sqrt(2.0 / (omega * _MU0 * sigma)),
            np.inf,
        )
    R_barrel = np.where(
        omega > 0.0,
        p.board_thickness_m / (sigma * 2.0 * np.pi * r_d * delta_s),
        0.0,
    )
    Z_series = R_barrel + 1j * omega * model.L_barrel_H

    # ── Shunt admittance at each port ─────────────────────────────────────
    C_shunt = model.C_antipad_F / 2.0 + model.C_pad_F / 2.0

    # Dielectric loss conductance G = ω C tan_δ
    G_diel = omega * C_shunt * float(p.loss_tangent)
    Y_shunt = 1j * omega * C_shunt + G_diel

    # Open-circuit stub admittance Y = j tan(βl) / Z_via
    if p.stub_length_m > 1e-9 and model.Z_via_ohm > 0.0:
        v_p = _C0 / np.sqrt(max(p.epsilon_r, 1.0))
        beta_l = omega / v_p * p.stub_length_m
        Y_stub = 1j * np.tan(beta_l) / model.Z_via_ohm
        Y_shunt = Y_shunt + Y_stub

    # ── Pi-network ABCD → S-parameters ───────────────────────────────────
    # Symmetric pi: Y1 = Y2 = Y_shunt, Z = Z_series
    Z = Z_series
    Y = Y_shunt

    A = 1.0 + Z * Y
    B = Z
    C_mat = 2.0 * Y + Z * Y**2
    D = A  # symmetric network

    denom = A + B / Z0 + C_mat * Z0 + D

    S = np.empty((n, 2, 2), dtype=complex)
    S[:, 0, 0] = (A + B / Z0 - C_mat * Z0 - D) / denom
    S[:, 1, 0] = 2.0 / denom
    S[:, 0, 1] = S[:, 1, 0]       # reciprocal
    S[:, 1, 1] = (-A + B / Z0 - C_mat * Z0 + D) / denom

    return S
