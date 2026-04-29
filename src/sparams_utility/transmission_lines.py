"""Closed-form transmission-line S-parameter synthesis.

Given a `SubstrateSpec` (PCB stack-up) and a `TransmissionLineSpec`
(geometry: width, length, optional spacing) this module returns an
N-port complex S-matrix at one frequency, with N=2 for single-ended
microstrip / stripline and N=4 for the corresponding edge-coupled
variants.

Reference impedance for the returned S-matrix is `Z0_ref` (defaults to
50 Ω single-ended, 50 Ω per port for coupled — i.e. 100 Ω differential).
The synthesis uses standard textbook formulas:

  * Microstrip Z0 / εeff: Hammerstad & Jensen, IEEE T-MTT 1980.
  * Stripline Z0       : Wheeler / Cohn closed-form (Pozar §3.7).
  * Coupled lines      : even/odd-mode Z0e/Z0o decomposition with the
    standard ABCD model for two transmission lines of equal length.

Losses are modelled as:

  * Conductor: αc = Rs / (2·Z0·W)         (Wheeler approximation, Np/m)
  * Dielectric: αd = (π·f·√εeff·tanδ)/c   (Pozar eq. 3.30, Np/m)

These give first-order accurate insertion loss for typical PCB lines.
The model is intentionally lightweight — its purpose is to drive the
schematic-level circuit solver, not to replace a full 2.5D field
solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public dataclasses (lightweight container — the persistent specs live in
# `models.circuit` to keep the model layer independent of numpy).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SubstrateGeom:
    epsilon_r: float
    loss_tangent: float
    height_m: float           # microstrip h, or stripline total b = h_top+t+h_bot
    conductor_thickness_m: float
    conductivity_s_per_m: float
    stripline_h_top_m: float = 0.0
    stripline_h_bottom_m: float = 0.0


# ---------------------------------------------------------------------------
# Microstrip — Hammerstad & Jensen
# ---------------------------------------------------------------------------


def microstrip_z0_eeff(W: float, h: float, eps_r: float, t: float = 0.0) -> Tuple[float, float]:
    """Return (Z0 [Ω], ε_eff) for a single microstrip line.

    Hammerstad–Jensen formulas with conductor-thickness correction.
    """
    W = max(W, 1e-9)
    h = max(h, 1e-9)
    if t > 0.0 and t < h:
        # Effective width correction (Hammerstad).
        dW = (t / np.pi) * (1.0 + np.log(2.0 * h / t))
        We = W + dW
    else:
        We = W

    u = We / h
    a = 1.0 + (1.0 / 49.0) * np.log((u**4 + (u / 52.0) ** 2) / (u**4 + 0.432))
    a += (1.0 / 18.7) * np.log(1.0 + (u / 18.1) ** 3)
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0)) ** 0.053
    eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 10.0 / u) ** (-a * b)

    eta0 = 376.730313668
    f_u = 6.0 + (2.0 * np.pi - 6.0) * np.exp(-((30.666 / u) ** 0.7528))
    z0_air = (eta0 / (2.0 * np.pi)) * np.log(f_u / u + np.sqrt(1.0 + (2.0 / u) ** 2))
    z0 = z0_air / np.sqrt(eps_eff)
    return float(z0), float(eps_eff)


# ---------------------------------------------------------------------------
# Stripline — Cohn / Wheeler
# ---------------------------------------------------------------------------


def stripline_z0(W: float, b: float, eps_r: float, t: float = 0.0) -> float:
    """Return Z0 [Ω] for a symmetric stripline (Cohn / Wheeler).

    `b` is the full ground-plane separation (dielectric thickness).
    Pozar §3.7 formulation valid for W/(b−t) up to ~0.35; an empirical
    correction is used for wider strips.
    """
    eps_r = max(eps_r, 1.0)
    b = max(b, 1e-9)
    W = max(W, 1e-9)
    t = max(0.0, min(t, 0.5 * b))

    eta0 = 376.730313668
    # Effective width with thickness correction (Wheeler).
    if t > 0.0:
        dW = (t / np.pi) * (1.0 + np.log(2.0 * b / t))
    else:
        dW = 0.0
    We = W + dW
    ratio = We / (b - t) if b > t else We / b
    if ratio <= 0.35:
        # Narrow strip (Cohn closed-form).
        z0 = (eta0 / (4.0 * np.pi * np.sqrt(eps_r))) * np.log(
            1.0 + (4.0 * (b - t) / (np.pi * We))
            * ((8.0 * (b - t) / (np.pi * We))
               + np.sqrt((8.0 * (b - t) / (np.pi * We)) ** 2 + 6.27))
        )
    else:
        # Wide strip — Wheeler/IPC empirical form.
        cf = (2.0 / np.pi) * np.log((b + t) / (b - t)) - (t / b) * np.log(
            ((b - t) ** 2) / (b * t + 1e-30) + 1.0
        )
        denom = (We / (b - t)) + (cf / np.pi)
        z0 = (eta0 / (2.0 * np.sqrt(eps_r))) / max(denom, 1e-9)
    return float(z0)


def asymmetric_stripline_z0(W: float, h_top: float, h_bot: float, eps_r: float, t: float = 0.0) -> float:
    """Approximate Z0 [Ω] for an offset (asymmetric) stripline.

    Uses the Wheeler / IPC parallel-combination model: treat the upper
    and lower half-lines as independent symmetric striplines and combine
    their characteristic impedances in parallel.
    """
    h_top = max(h_top, 1e-9)
    h_bot = max(h_bot, 1e-9)
    z0_top = stripline_z0(W, 2.0 * h_top + t, eps_r, t)
    z0_bot = stripline_z0(W, 2.0 * h_bot + t, eps_r, t)
    if z0_top + z0_bot <= 0.0:
        return float(z0_top)
    return float(2.0 * z0_top * z0_bot / (z0_top + z0_bot))


# ---------------------------------------------------------------------------
# Coupled microstrip — Hammerstad–Jensen even/odd modes
# ---------------------------------------------------------------------------


def coupled_microstrip_modes(W: float, S: float, h: float, eps_r: float,
                              t: float = 0.0) -> Tuple[float, float, float, float]:
    """Return (Z0e, Z0o, εeff_e, εeff_o) for edge-coupled microstrip.

    Compact Hammerstad–Jensen-style model (Garg & Bahl).
    """
    W = max(W, 1e-9)
    S = max(S, 1e-9)
    h = max(h, 1e-9)
    z0_se, eeff_se = microstrip_z0_eeff(W, h, eps_r, t)

    g = S / h
    u = W / h
    # Even-mode εeff (Garg–Bahl 1979).
    v = u * (20.0 + g * g) / (10.0 + g * g) + g * np.exp(-g)
    a_e = 1.0 + (1.0 / 49.0) * np.log((v**4 + (v / 52.0) ** 2) / (v**4 + 0.432))
    a_e += (1.0 / 18.7) * np.log(1.0 + (v / 18.1) ** 3)
    b_e = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0)) ** 0.053
    eeff_e = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 10.0 / max(v, 1e-9)) ** (-a_e * b_e)
    # Odd-mode εeff.
    a_o = 0.7287 * (eeff_se - (eps_r + 1.0) / 2.0) * (1.0 - np.exp(-0.179 * u))
    b_o = 0.747 * eps_r / (0.15 + eps_r)
    c_o = b_o - (b_o - 0.207) * np.exp(-0.414 * u)
    d_o = 0.593 + 0.694 * np.exp(-0.562 * u)
    eeff_o = ((eps_r + 1.0) / 2.0 + a_o - eeff_se) * np.exp(-c_o * (g ** d_o)) + eeff_se

    # Mode characteristic impedances (Kirschning–Jansen approximations).
    eta0 = 376.730313668
    # Single-line air impedance (no dielectric) for normalization.
    z0_air, _ = microstrip_z0_eeff(W, h, 1.0, t)
    # Even-mode air Z (Hammerstad–Jensen approximation).
    q1 = 0.8695 * u ** 0.194
    q2 = 1.0 + 0.7519 * g + 0.189 * (g ** 2.31)
    q3 = 0.1975 + (16.6 + (8.4 / g) ** 6) ** (-0.387) + (1.0 / 241.0) * np.log(
        (g ** 10) / (1.0 + (g / 3.4) ** 10)
    )
    q4 = (2.0 * q1 / q2) * (np.exp(-g) * (u ** q3) + (2.0 - np.exp(-g)) * (u ** -q3)) ** -1
    z0e_air = z0_air * np.sqrt(eeff_se / 1.0) / (1.0 - z0_air * np.sqrt(eeff_se) * q4 / eta0)
    z0e = z0e_air / np.sqrt(eeff_e)
    # Odd-mode air Z.
    q5 = 1.794 + 1.14 * np.log(1.0 + 0.638 / (g + 0.517 * g ** 2.43))
    q6 = 0.2305 + (1.0 / 281.3) * np.log((g ** 10) / (1.0 + (g / 5.8) ** 10)) + (1.0 / 5.1) * np.log(
        1.0 + 0.598 * g ** 1.154
    )
    q7 = (10.0 + 190.0 * g ** 2) / (1.0 + 82.3 * g ** 3)
    q8 = np.exp(-6.5 - 0.95 * np.log(g) - (g / 0.15) ** 5)
    q9 = np.log(q7) * (q8 + 1.0 / 16.5)
    q10 = (q2 * q4 - q5 * np.exp(np.log(u) * q6 * (u ** -q9))) / q2
    z0o_air = z0_air * np.sqrt(eeff_se) / (1.0 - z0_air * np.sqrt(eeff_se) * q10 / eta0)
    z0o = z0o_air / np.sqrt(eeff_o)

    # Sanity clamps — keep the modes physical.
    z0e = float(max(min(z0e, 4.0 * z0_se), 0.5 * z0_se))
    z0o = float(max(min(z0o, z0_se), 0.05 * z0_se))
    return float(z0e), float(z0o), float(eeff_e), float(eeff_o)


def coupled_stripline_modes(W: float, S: float, b: float, eps_r: float,
                              t: float = 0.0) -> Tuple[float, float]:
    """Return (Z0e, Z0o) for edge-coupled symmetric stripline (Cohn)."""
    eps_r = max(eps_r, 1.0)
    b = max(b, 1e-9)
    W = max(W, 1e-9)
    S = max(S, 1e-9)
    eta0 = 376.730313668

    # Cohn even/odd-mode capacitive expressions (Garg & Bahl).
    k_e = np.tanh(np.pi * W / (2.0 * b)) * np.tanh(np.pi * (W + S) / (2.0 * b))
    k_o = np.tanh(np.pi * W / (2.0 * b)) / np.tanh(np.pi * (W + S) / (2.0 * b))
    k_e = float(np.clip(k_e, 1e-12, 1.0 - 1e-12))
    k_o = float(np.clip(k_o, 1e-12, 1.0 - 1e-12))

    def _kr(k: float) -> float:
        kp = float(np.sqrt(1.0 - k * k))
        if k <= np.sqrt(0.5):
            return float(np.pi / np.log(2.0 * (1.0 + np.sqrt(kp)) / (1.0 - np.sqrt(kp))))
        return float(np.log(2.0 * (1.0 + np.sqrt(k)) / (1.0 - np.sqrt(k))) / np.pi)

    z0e = (eta0 / (4.0 * np.sqrt(eps_r))) * _kr(k_e)
    z0o = (eta0 / (4.0 * np.sqrt(eps_r))) * _kr(k_o)
    return float(z0e), float(z0o)


# ---------------------------------------------------------------------------
# Per-unit-length losses
# ---------------------------------------------------------------------------


def _alpha_dielectric(f: float, eps_eff: float, tan_delta: float) -> float:
    """Dielectric loss [Np/m] (TEM approximation)."""
    if f <= 0.0:
        return 0.0
    c0 = 299_792_458.0
    return float(np.pi * f * np.sqrt(max(eps_eff, 1.0)) * tan_delta / c0)


def _alpha_conductor(f: float, z0: float, W: float, sigma: float) -> float:
    """Conductor loss [Np/m] (Wheeler approximation, single conductor over GND)."""
    if f <= 0.0 or z0 <= 0.0 or W <= 0.0 or sigma <= 0.0:
        return 0.0
    mu0 = 4.0e-7 * np.pi
    rs = float(np.sqrt(np.pi * f * mu0 / sigma))  # surface resistance Ω/sq
    return float(rs / (2.0 * z0 * W))


# ---------------------------------------------------------------------------
# 2-port single-ended line: ABCD → S
# ---------------------------------------------------------------------------


def _abcd_lossy_line(z0: complex, gamma: complex, length: float) -> np.ndarray:
    """ABCD matrix of a lossy transmission line."""
    gl = gamma * length
    cosh_gl = np.cosh(gl)
    sinh_gl = np.sinh(gl)
    return np.array([[cosh_gl, z0 * sinh_gl],
                     [sinh_gl / z0, cosh_gl]], dtype=np.complex128)


def _abcd_to_s(abcd: np.ndarray, z_ref: float) -> np.ndarray:
    a, b, c, d = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
    z0 = complex(z_ref, 0.0)
    denom = a + b / z0 + c * z0 + d
    s11 = (a + b / z0 - c * z0 - d) / denom
    s12 = 2.0 * (a * d - b * c) / denom
    s21 = 2.0 / denom
    s22 = (-a + b / z0 - c * z0 + d) / denom
    return np.array([[s11, s12], [s21, s22]], dtype=np.complex128)


def _single_line_s_matrix(z0_line: float, eps_eff: float, tan_d: float,
                            W: float, sigma: float, length: float, z_ref: float,
                            f: float) -> np.ndarray:
    c0 = 299_792_458.0
    beta = 2.0 * np.pi * f * np.sqrt(max(eps_eff, 1.0)) / c0 if f > 0.0 else 0.0
    alpha = _alpha_conductor(f, z0_line, W, sigma) + _alpha_dielectric(f, eps_eff, tan_d)
    gamma = complex(alpha, beta)
    abcd = _abcd_lossy_line(complex(z0_line, 0.0), gamma, length)
    return _abcd_to_s(abcd, z_ref)


# ---------------------------------------------------------------------------
# 4-port coupled line: even/odd-mode → S
# ---------------------------------------------------------------------------


def _coupled_line_s_matrix(z0e: float, z0o: float, eeff_e: float, eeff_o: float,
                             tan_d: float, W: float, sigma: float, length: float,
                             z_ref: float, f: float) -> np.ndarray:
    """Edge-coupled symmetric pair → 4-port S.

    Port numbering convention used here:
      P1 — line A near end
      P2 — line A far end
      P3 — line B near end
      P4 — line B far end

    Derived from the standard Pozar §7.6 closed-form for a coupled
    section, augmented with per-mode propagation constants and the
    same conductor / dielectric loss model used for single lines.
    """
    c0 = 299_792_458.0
    if f <= 0.0:
        # DC: behave as two independent shorts of length L (resistive).
        rdc = max(0.0, _alpha_conductor(1.0, (z0e + z0o) / 2.0, W, sigma)) * length
        s = np.zeros((4, 4), dtype=np.complex128)
        # Identity-like trivial pass-through at DC (R≈0 short).
        s[0, 1] = s[1, 0] = s[2, 3] = s[3, 2] = 1.0 / (1.0 + rdc)
        return s

    beta_e = 2.0 * np.pi * f * np.sqrt(max(eeff_e, 1.0)) / c0
    beta_o = 2.0 * np.pi * f * np.sqrt(max(eeff_o, 1.0)) / c0
    alpha_e = _alpha_conductor(f, z0e, W, sigma) + _alpha_dielectric(f, eeff_e, tan_d)
    alpha_o = _alpha_conductor(f, z0o, W, sigma) + _alpha_dielectric(f, eeff_o, tan_d)
    gamma_e = complex(alpha_e, beta_e)
    gamma_o = complex(alpha_o, beta_o)

    # Build single-line ABCDs for each mode, convert to single-mode S(2x2)
    # at reference 2*Z_ref (even) and Z_ref/2... actually the simplest
    # exact route is the "two-port modal" decomposition:
    #   S_even = S(z0e, gamma_e, L)  with reference Z_ref
    #   S_odd  = S(z0o, gamma_o, L)  with reference Z_ref
    # Then the 4-port matrix is built from the modal S-parameters.
    se = _abcd_to_s(_abcd_lossy_line(complex(z0e, 0.0), gamma_e, length), z_ref)
    so = _abcd_to_s(_abcd_lossy_line(complex(z0o, 0.0), gamma_o, length), z_ref)

    s11e, s12e = se[0, 0], se[0, 1]
    s21e, s22e = se[1, 0], se[1, 1]
    s11o, s12o = so[0, 0], so[0, 1]
    s21o, s22o = so[1, 0], so[1, 1]

    # Pozar §7.6: standard combination for a symmetric coupled pair.
    # Port map: 1=A near, 2=A far, 3=B near, 4=B far.
    s = np.zeros((4, 4), dtype=np.complex128)
    s[0, 0] = 0.5 * (s11e + s11o); s[1, 1] = 0.5 * (s22e + s22o)
    s[2, 2] = s[0, 0];             s[3, 3] = s[1, 1]
    s[0, 1] = 0.5 * (s12e + s12o); s[1, 0] = 0.5 * (s21e + s21o)
    s[2, 3] = s[0, 1];             s[3, 2] = s[1, 0]
    s[0, 2] = 0.5 * (s11e - s11o); s[2, 0] = s[0, 2]
    s[1, 3] = 0.5 * (s22e - s22o); s[3, 1] = s[1, 3]
    s[0, 3] = 0.5 * (s12e - s12o); s[3, 0] = s[0, 3]
    s[1, 2] = 0.5 * (s21e - s21o); s[2, 1] = s[1, 2]
    return s


# ---------------------------------------------------------------------------
# Public synthesis entry point
# ---------------------------------------------------------------------------


def synthesize_tline_s_matrix(
    *,
    line_kind: str,
    width_m: float,
    length_m: float,
    spacing_m: float,
    z0_ref: float,
    substrate: _SubstrateGeom,
    frequency_hz: float,
) -> np.ndarray:
    """Return the S-matrix at one frequency for the requested t-line kind.

    `line_kind` ∈ {"microstrip", "stripline", "microstrip_coupled",
    "stripline_coupled"}. The returned matrix is 2×2 for the
    single-ended kinds and 4×4 for the coupled kinds.
    """
    eps_r = substrate.epsilon_r
    tan_d = substrate.loss_tangent
    h = substrate.height_m
    t = substrate.conductor_thickness_m
    sigma = substrate.conductivity_s_per_m
    h_top = substrate.stripline_h_top_m
    h_bot = substrate.stripline_h_bottom_m

    if line_kind == "microstrip":
        z0_line, eps_eff = microstrip_z0_eeff(width_m, h, eps_r, t)
        return _single_line_s_matrix(
            z0_line, eps_eff, tan_d, width_m, sigma, length_m, z0_ref, frequency_hz
        )
    if line_kind == "stripline":
        if h_top > 0.0 and h_bot > 0.0 and abs(h_top - h_bot) > 1e-9:
            z0_line = asymmetric_stripline_z0(width_m, h_top, h_bot, eps_r, t)
        else:
            z0_line = stripline_z0(width_m, h, eps_r, t)
        # Stripline is pure TEM — εeff equals εr.
        return _single_line_s_matrix(
            z0_line, eps_r, tan_d, width_m, sigma, length_m, z0_ref, frequency_hz
        )
    if line_kind == "microstrip_coupled":
        z0e, z0o, eeff_e, eeff_o = coupled_microstrip_modes(
            width_m, spacing_m, h, eps_r, t
        )
        return _coupled_line_s_matrix(
            z0e, z0o, eeff_e, eeff_o, tan_d, width_m, sigma, length_m, z0_ref, frequency_hz
        )
    if line_kind == "stripline_coupled":
        z0e, z0o = coupled_stripline_modes(width_m, spacing_m, h, eps_r, t)
        return _coupled_line_s_matrix(
            z0e, z0o, eps_r, eps_r, tan_d, width_m, sigma, length_m, z0_ref, frequency_hz
        )
    raise ValueError(f"Unknown transmission-line kind: {line_kind!r}")


def substrate_geom_from_spec(substrate_spec) -> _SubstrateGeom:
    """Build the math-only geometry container from a `SubstrateSpec`."""
    return _SubstrateGeom(
        epsilon_r=float(substrate_spec.epsilon_r),
        loss_tangent=float(substrate_spec.loss_tangent),
        height_m=float(substrate_spec.height_m),
        conductor_thickness_m=float(substrate_spec.conductor_thickness_m),
        conductivity_s_per_m=float(substrate_spec.conductivity_s_per_m),
        stripline_h_top_m=float(getattr(substrate_spec, "stripline_h_top_m", 0.0)),
        stripline_h_bottom_m=float(getattr(substrate_spec, "stripline_h_bottom_m", 0.0)),
    )


# ===========================================================================
# Coplanar waveguide (CPW) — Wen 1969 / Simons (Coplanar Waveguide Circuits,
# Components, and Systems, 2001).  Conductor-backed (grounded) variant.
# ===========================================================================

try:                                                       # pragma: no cover
    from scipy.special import ellipk as _scipy_ellipk      # type: ignore
    _HAVE_SCIPY_ELLIPK = True
except Exception:                                          # pragma: no cover
    _scipy_ellipk = None
    _HAVE_SCIPY_ELLIPK = False


def _ellipk_ratio(k: float) -> float:
    """Return K(k)/K(k') with k' = sqrt(1-k^2).

    Uses scipy.special.ellipk when available, otherwise Hilberg's
    closed-form approximation (accuracy < 8 ppm), which is the standard
    workaround for CPW design without scipy.
    """
    k = float(np.clip(abs(k), 1e-12, 1.0 - 1e-12))
    if _HAVE_SCIPY_ELLIPK:
        # scipy.special.ellipk takes the parameter m = k**2.
        kp = float(np.sqrt(1.0 - k * k))
        return float(_scipy_ellipk(k * k) / _scipy_ellipk(kp * kp))
    # Hilberg approximation.
    if k <= np.sqrt(0.5):
        kp = float(np.sqrt(1.0 - k * k))
        return float(np.pi / np.log(2.0 * (1.0 + np.sqrt(kp)) / (1.0 - np.sqrt(kp))))
    return float(np.log(2.0 * (1.0 + np.sqrt(k)) / (1.0 - np.sqrt(k))) / np.pi)


def cpw_z0_eeff(W: float, S: float, h: float, eps_r: float,
                t: float = 0.0) -> Tuple[float, float]:
    """Conductor-backed CPW Z0 [Ω] and ε_eff (Wen 1969 / Simons §2.2).

    W = center-conductor width, S = ground-to-conductor slot width
    (each side), h = dielectric height to the bottom ground plane,
    t = conductor thickness (Wheeler thickness correction applied to
    effective W and S).
    """
    W = max(float(W), 1e-9)
    S = max(float(S), 1e-9)
    h = max(float(h), 1e-9)
    eps_r = max(float(eps_r), 1.0)

    # Wheeler-style thickness correction (Simons eq. 2.41).
    if t > 0.0:
        delta = (1.25 * t / np.pi) * (1.0 + np.log(4.0 * np.pi * W / max(t, 1e-12)))
        We = W + delta
        Se = max(S - delta, 1e-9)
    else:
        We = W
        Se = S

    a = 0.5 * We
    b = a + Se
    k1 = a / b
    # Conductor-backed lower-half-plane modulus.
    k3 = float(np.tanh(np.pi * a / (2.0 * h)) / np.tanh(np.pi * b / (2.0 * h)))

    r1 = _ellipk_ratio(k1)        # K(k1)/K(k1')
    r3 = _ellipk_ratio(k3)        # K(k3)/K(k3')

    # Effective permittivity for grounded CPW (Simons eq. 2.45).
    eps_eff = (1.0 + eps_r * (r3 / r1)) / (1.0 + (r3 / r1))
    eta0 = 376.730313668
    # Z0 for conductor-backed CPW (Simons eq. 2.46-2.47):
    #   Z0 = 60π / (sqrt(εeff) · [K(k1)/K(k1') + K(k3)/K(k3')])
    z0 = (eta0 / 2.0) / (np.sqrt(eps_eff) * (r1 + r3))
    return float(z0), float(eps_eff)


def cpw_coupled_modes(W: float, S_slot: float, S_coupled: float, h: float,
                       eps_r: float, t: float = 0.0
                       ) -> Tuple[float, float, float, float]:
    """Edge-coupled grounded-CPW even/odd modes (Simons §5, approximate).

    Two CPW lines of width `W` with outer slot `S_slot` to their outer
    ground, separated edge-to-edge by `S_coupled`.  Implementation uses
    the single-line CPW result and superimposes a mutual-capacitance
    correction whose strength decays as exp(-π·S_coupled/W) — this is
    the Simons 2001 approximation suitable for moderate coupling.
    """
    z0_se, eeff_se = cpw_z0_eeff(W, S_slot, h, eps_r, t)
    # Mutual coupling coefficient (heuristic exponential decay vs gap).
    arg = np.pi * max(S_coupled, 1e-9) / max(W, 1e-9)
    k_c = 0.45 * float(np.exp(-arg))
    k_c = float(np.clip(k_c, 0.0, 0.45))
    z0e = z0_se * np.sqrt((1.0 + k_c) / (1.0 - k_c))
    z0o = z0_se * np.sqrt((1.0 - k_c) / (1.0 + k_c))
    # Even mode concentrates field in dielectric → slightly higher εeff.
    eeff_e = eeff_se * (1.0 + 0.05 * k_c)
    eeff_o = eeff_se * (1.0 - 0.05 * k_c)
    return float(z0e), float(z0o), float(eeff_e), float(eeff_o)


def cpw_line_s_matrix(z0_line: float, eps_eff: float, tan_d: float,
                       W: float, sigma: float, length: float, z_ref: float,
                       f: float) -> np.ndarray:
    """2-port S-matrix of a single CPW line (cf. _single_line_s_matrix)."""
    return _single_line_s_matrix(z0_line, eps_eff, tan_d, W, sigma, length,
                                  z_ref, f)


def cpw_coupled_s_matrix(z0e: float, z0o: float, eeff_e: float, eeff_o: float,
                          tan_d: float, W: float, sigma: float, length: float,
                          z_ref: float, f: float) -> np.ndarray:
    """4-port S-matrix of a coupled CPW pair (re-uses the modal model)."""
    return _coupled_line_s_matrix(z0e, z0o, eeff_e, eeff_o, tan_d, W, sigma,
                                   length, z_ref, f)


# ===========================================================================
# Impedance taper — linear / exponential / Klopfenstein (1956)
# ===========================================================================


def _kind_z0_eeff(line_kind: str, W: float, substrate: _SubstrateGeom,
                   S_cpw: float = 0.0) -> Tuple[float, float]:
    """Return (Z0, eps_eff) for the requested line kind at width W."""
    eps_r = substrate.epsilon_r
    h = substrate.height_m
    t = substrate.conductor_thickness_m
    if line_kind == "microstrip":
        return microstrip_z0_eeff(W, h, eps_r, t)
    if line_kind == "stripline":
        return stripline_z0(W, h, eps_r, t), eps_r
    if line_kind == "cpw":
        S = S_cpw if S_cpw > 0.0 else 0.5 * W
        return cpw_z0_eeff(W, S, h, eps_r, t)
    raise ValueError(f"Unknown line_kind for taper: {line_kind!r}")


def _invert_w_for_z0(line_kind: str, z0_target: float,
                      substrate: _SubstrateGeom, W_lo: float, W_hi: float,
                      S_cpw: float = 0.0) -> float:
    """Bisection: find W such that Z0(W) ≈ z0_target on [W_lo, W_hi]."""
    W_lo = max(W_lo, 1e-7)
    W_hi = max(W_hi, W_lo * 1.0001)
    z_lo = _kind_z0_eeff(line_kind, W_lo, substrate, S_cpw)[0]
    z_hi = _kind_z0_eeff(line_kind, W_hi, substrate, S_cpw)[0]
    # Z0 is monotonically decreasing in W for all three kinds; flip if not.
    decreasing = z_lo > z_hi
    for _ in range(60):
        Wm = 0.5 * (W_lo + W_hi)
        zm = _kind_z0_eeff(line_kind, Wm, substrate, S_cpw)[0]
        if abs(zm - z0_target) < 1e-4:
            return Wm
        if (zm > z0_target) == decreasing:
            W_lo = Wm
        else:
            W_hi = Wm
    return 0.5 * (W_lo + W_hi)


def taper_s_matrix(line_kind: str, substrate: _SubstrateGeom,
                    W_start: float, W_end: float, length: float,
                    z_ref: float, f: float, profile: str = "linear",
                    n_segments: int = 32, S_cpw: float = 0.0) -> np.ndarray:
    """2-port S-matrix of an impedance taper (Klopfenstein 1956 et al.).

    Discretizes the taper into `n_segments` uniform sub-sections and
    cascades their ABCD matrices.  Profiles:
      * "linear"       — width varies linearly with x.
      * "exponential"  — ln(W) varies linearly with x (≈ exponential Z0).
      * "klopfenstein" — Z0(x) follows a tanh-shaped Klopfenstein-like
        profile; the per-segment width is found by bisection on the
        kind-specific Z0(W) closed form.
    """
    n = max(int(n_segments), 1)
    W_start = max(float(W_start), 1e-9)
    W_end = max(float(W_end), 1e-9)
    L = max(float(length), 0.0)
    dL = L / n if n > 0 else L

    # Pre-compute endpoint impedances for the Klopfenstein profile.
    z1, _ = _kind_z0_eeff(line_kind, W_start, substrate, S_cpw)
    z2, _ = _kind_z0_eeff(line_kind, W_end, substrate, S_cpw)
    ln_z1 = np.log(max(z1, 1e-9))
    ln_z2 = np.log(max(z2, 1e-9))
    A_klop = 3.0  # Klopfenstein shape parameter (Γm small → A≈3 typical).
    tanhA = np.tanh(A_klop)

    W_lo = min(W_start, W_end) * 0.25
    W_hi = max(W_start, W_end) * 4.0

    abcd = np.eye(2, dtype=np.complex128)
    for i in range(n):
        x_center = (i + 0.5) * dL
        u = x_center / L if L > 0.0 else 0.5            # u in (0,1)
        if profile == "linear":
            W_i = W_start + (W_end - W_start) * u
        elif profile == "exponential":
            W_i = W_start * (W_end / W_start) ** u
        elif profile == "klopfenstein":
            uu = 2.0 * u - 1.0                          # uu in (-1, 1)
            ln_z = 0.5 * (ln_z1 + ln_z2) + 0.5 * (ln_z2 - ln_z1) * \
                   float(np.tanh(A_klop * uu) / tanhA)
            z_target = float(np.exp(ln_z))
            W_i = _invert_w_for_z0(line_kind, z_target, substrate,
                                    W_lo, W_hi, S_cpw)
        else:
            raise ValueError(f"Unknown taper profile: {profile!r}")

        z0_i, eeff_i = _kind_z0_eeff(line_kind, W_i, substrate, S_cpw)
        c0 = 299_792_458.0
        beta = 2.0 * np.pi * f * np.sqrt(max(eeff_i, 1.0)) / c0 if f > 0.0 else 0.0
        alpha = (_alpha_conductor(f, z0_i, W_i, substrate.conductivity_s_per_m)
                 + _alpha_dielectric(f, eeff_i, substrate.loss_tangent))
        gamma = complex(alpha, beta)
        abcd = abcd @ _abcd_lossy_line(complex(z0_i, 0.0), gamma, dL)

    return _abcd_to_s(abcd, z_ref)


# ===========================================================================
# Lumped / ideal n-port building blocks
# ===========================================================================


def attenuator_s_matrix(attenuation_db: float, z_ref: float = 50.0) -> np.ndarray:
    """Ideal symmetric matched 2-port attenuator (S11=S22=0)."""
    a = 10.0 ** (-float(attenuation_db) / 20.0)
    s = np.zeros((2, 2), dtype=np.complex128)
    s[0, 1] = s[1, 0] = a
    return s


def circulator_s_matrix(insertion_loss_db: float = 0.0,
                         isolation_db: float = 30.0,
                         return_loss_db: float = 25.0,
                         direction: str = "cw",
                         z_ref: float = 50.0) -> np.ndarray:
    """Ideal matched 3-port circulator (Pozar §9.4).

    CW direction: 1→2, 2→3, 3→1 are the low-loss paths.
    """
    il = 10.0 ** (-float(insertion_loss_db) / 20.0)
    iso = 10.0 ** (-float(isolation_db) / 20.0)
    rl = 10.0 ** (-float(return_loss_db) / 20.0)
    s = np.full((3, 3), 0.0 + 0.0j)
    for i in range(3):
        s[i, i] = rl
    if str(direction).lower() == "ccw":
        fwd = [(0, 1), (1, 2), (2, 0)]   # 2→1, 3→2, 1→3
    else:
        fwd = [(1, 0), (2, 1), (0, 2)]   # 1→2, 2→3, 3→1  (S[j,i] = signal i→j)
    for j, i in fwd:
        s[j, i] = il
        s[i, j] = iso
    return s


def coupler_s_matrix(kind: str, coupling_db: float,
                      insertion_loss_db: float = 0.0,
                      isolation_db: float = 30.0,
                      return_loss_db: float = 25.0,
                      z_ref: float = 50.0) -> np.ndarray:
    """Ideal 4-port directional coupler (Pozar §5.6 / §7.5).

    `kind` ∈ {"branch_line_90", "rat_race_180", "directional"}.
    Port convention: 1=input, 2=through, 3=coupled, 4=isolated.
    Returned matrix is reciprocal (S[i,j]=S[j,i]).
    """
    c = 10.0 ** (-float(coupling_db) / 20.0)
    il = 10.0 ** (-float(insertion_loss_db) / 20.0)
    iso = 10.0 ** (-float(isolation_db) / 20.0)
    rl = 10.0 ** (-float(return_loss_db) / 20.0)
    c = float(np.clip(c, 0.0, 1.0))
    thru_mag = float(np.sqrt(max(1.0 - c * c, 0.0))) * il

    s = np.zeros((4, 4), dtype=np.complex128)
    for i in range(4):
        s[i, i] = rl

    k = str(kind).lower()
    if k in ("branch_line_90", "branch_line", "90", "hybrid_90"):
        thru = -1j * thru_mag
        coup = -c * il
    elif k in ("rat_race_180", "rat_race", "180", "hybrid_180"):
        # 180° hybrid: through and coupled both 1/sqrt(2)*il in magnitude;
        # phase difference handled by the sign of `coup` between port pairs.
        thru = (1.0 / np.sqrt(2.0)) * il
        coup = -(1.0 / np.sqrt(2.0)) * il   # 180° relative phase
    elif k in ("directional",):
        thru = thru_mag + 0.0j
        coup = -1j * c * il
    else:
        raise ValueError(f"Unknown coupler kind: {kind!r}")

    iso_c = iso + 0.0j
    # Port pairs: (input, through, coupled, isolated) = (1,2,3,4) → indices 0..3
    # Through path:  1↔2, 3↔4
    s[0, 1] = s[1, 0] = thru
    s[2, 3] = s[3, 2] = thru
    # Coupled path: 1↔3, 2↔4
    s[0, 2] = s[2, 0] = coup
    s[1, 3] = s[3, 1] = coup
    # Isolated path: 1↔4, 2↔3
    s[0, 3] = s[3, 0] = iso_c
    s[1, 2] = s[2, 1] = iso_c
    return s
