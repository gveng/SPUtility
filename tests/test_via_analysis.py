"""Unit tests for the via_analysis physics model."""
from __future__ import annotations

import math
import pytest
import numpy as np

from sparams_utility.via_analysis import ViaParams, ViaModel, compute_via_model, via_sparameters


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_params(**kwargs) -> ViaParams:
    """Return a physically valid set of via parameters, optionally overriding fields."""
    defaults = dict(
        drill_diameter_m=250e-6,
        pad_diameter_m=500e-6,
        antipad_diameter_m=800e-6,
        board_thickness_m=1.6e-3,
        stub_length_m=0.0,
        n_signal_pads=1,
        epsilon_r=4.2,
        loss_tangent=0.020,
        conductivity_s_per_m=5.8e7,
        reference_z_ohm=50.0,
    )
    defaults.update(kwargs)
    return ViaParams(**defaults)


# ── compute_via_model ─────────────────────────────────────────────────────────

class TestComputeViaModel:
    def test_returns_via_model(self):
        m = compute_via_model(_default_params())
        assert isinstance(m, ViaModel)

    def test_inductance_positive(self):
        m = compute_via_model(_default_params())
        assert m.L_barrel_H > 0.0

    def test_inductance_increases_with_board_thickness(self):
        m_thin = compute_via_model(_default_params(board_thickness_m=0.8e-3))
        m_thick = compute_via_model(_default_params(board_thickness_m=3.2e-3))
        assert m_thick.L_barrel_H > m_thin.L_barrel_H

    def test_inductance_increases_with_antipad_ratio(self):
        """Larger antipad → higher ln(r_a/r_d) → more inductance."""
        m_small = compute_via_model(_default_params(antipad_diameter_m=700e-6))
        m_large = compute_via_model(_default_params(antipad_diameter_m=1200e-6))
        assert m_large.L_barrel_H > m_small.L_barrel_H

    def test_capacitances_positive(self):
        m = compute_via_model(_default_params())
        assert m.C_antipad_F > 0.0
        assert m.C_pad_F > 0.0

    def test_c_pad_scales_with_n_pads(self):
        m1 = compute_via_model(_default_params(n_signal_pads=1))
        m2 = compute_via_model(_default_params(n_signal_pads=4))
        assert math.isclose(m2.C_pad_F, 4.0 * m1.C_pad_F, rel_tol=1e-9)

    def test_z_via_positive(self):
        m = compute_via_model(_default_params())
        assert m.Z_via_ohm > 0.0

    def test_no_stub_resonance_when_stub_zero(self):
        m = compute_via_model(_default_params(stub_length_m=0.0))
        assert m.stub_resonance_hz == 0.0

    def test_stub_resonance_present(self):
        m = compute_via_model(_default_params(stub_length_m=0.6e-3))
        assert m.stub_resonance_hz > 0.0

    def test_stub_resonance_decreases_with_longer_stub(self):
        """Longer stub → lower quarter-wave frequency."""
        m_short = compute_via_model(_default_params(stub_length_m=0.3e-3))
        m_long = compute_via_model(_default_params(stub_length_m=0.8e-3))
        assert m_long.stub_resonance_hz < m_short.stub_resonance_hz

    def test_invalid_antipad_raises(self):
        with pytest.raises(ValueError, match="Antipad"):
            compute_via_model(_default_params(antipad_diameter_m=200e-6))  # antipad < drill

    def test_invalid_pad_raises(self):
        with pytest.raises(ValueError, match="Pad"):
            compute_via_model(_default_params(pad_diameter_m=100e-6))  # pad < drill

    def test_antipad_less_than_pad_raises(self):
        with pytest.raises(ValueError, match="Antipad"):
            compute_via_model(_default_params(
                pad_diameter_m=600e-6,
                antipad_diameter_m=550e-6,  # antipad < pad
            ))


# ── via_sparameters ───────────────────────────────────────────────────────────

class TestViaSparameters:
    def _compute(self, **kwargs):
        p = _default_params(**kwargs)
        freqs = np.linspace(0.1e9, 20e9, 200)
        return via_sparameters(p, freqs), freqs

    def test_shape(self):
        S, freqs = self._compute()
        assert S.shape == (200, 2, 2)

    def test_reciprocal(self):
        S, _ = self._compute()
        np.testing.assert_allclose(S[:, 0, 1], S[:, 1, 0], atol=1e-12)

    def test_symmetric_s11_s22(self):
        S, _ = self._compute()
        np.testing.assert_allclose(S[:, 0, 0], S[:, 1, 1], atol=1e-12)

    def test_s21_near_zero_db_at_low_freq(self):
        """A small lossless via should be nearly transparent at low frequency."""
        S, _ = self._compute(
            stub_length_m=0.0,
            loss_tangent=0.0,
            conductivity_s_per_m=5.8e7,
        )
        s21_db_low = 20.0 * np.log10(np.abs(S[0, 1, 0]))  # first frequency point
        assert s21_db_low > -0.5, f"S21 at low freq should be near 0 dB, got {s21_db_low:.2f} dB"

    def test_insertion_loss_increases_with_frequency(self):
        """Inductance + capacitance cause increasing insertion loss with frequency."""
        S, _ = self._compute(stub_length_m=0.0, loss_tangent=0.0)
        s21_mag = np.abs(S[:, 1, 0])
        # Average first 10 vs last 10 points
        assert np.mean(s21_mag[:10]) > np.mean(s21_mag[-10:])

    def test_stub_dip_visible(self):
        """Stub resonance should cause a significant S21 dip."""
        stub = 0.6e-3
        import sparams_utility.via_analysis as va
        _C0 = va._C0
        eps_r = 4.2
        f_res = _C0 / (4.0 * stub * math.sqrt(eps_r))

        freqs = np.linspace(0.1e9, 3.0 * f_res, 1000)
        p = _default_params(stub_length_m=stub)
        S = via_sparameters(p, freqs)

        s21_db = 20.0 * np.log10(np.abs(S[:, 1, 0]))
        dip = np.min(s21_db)
        assert dip < -3.0, f"Expected stub dip below -3 dB, got {dip:.1f} dB"

    def test_passivity(self):
        """For a low-loss via |S21|² + |S11|² ≤ 1 (lossless/passive)."""
        S, _ = self._compute(loss_tangent=0.0)
        power = np.abs(S[:, 0, 0])**2 + np.abs(S[:, 1, 0])**2
        assert np.all(power <= 1.0 + 1e-9), f"Passivity violated: max={np.max(power):.6f}"
