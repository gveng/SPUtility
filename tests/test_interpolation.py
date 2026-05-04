from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import skrf as rf  # noqa: E402

from sparams_utility.interpolation import (  # noqa: E402
    cascade_networks,
    common_frequency_grid,
    interpolate_networks,
    tdr_from_network,
)


def _make_thru(freqs_hz: np.ndarray, z0: float = 50.0) -> rf.Network:
    """Lossless ideal thru: S11=S22=0, S12=S21=1."""
    n = freqs_hz.size
    s = np.zeros((n, 2, 2), dtype=complex)
    s[:, 0, 1] = 1.0
    s[:, 1, 0] = 1.0
    freq = rf.Frequency.from_f(freqs_hz, unit="Hz")
    return rf.Network(frequency=freq, s=s, z0=z0)


def _make_attenuator(freqs_hz: np.ndarray, atten_db: float, z0: float = 50.0) -> rf.Network:
    """Matched attenuator: S11=S22=0, S21=S12=10^(-A/20)."""
    n = freqs_hz.size
    a = 10.0 ** (-atten_db / 20.0)
    s = np.zeros((n, 2, 2), dtype=complex)
    s[:, 0, 1] = a
    s[:, 1, 0] = a
    freq = rf.Frequency.from_f(freqs_hz, unit="Hz")
    return rf.Network(frequency=freq, s=s, z0=z0)


class InterpolationTests(unittest.TestCase):
    def test_common_grid_union(self) -> None:
        a = _make_thru(np.array([1e9, 2e9, 3e9]))
        b = _make_thru(np.array([2e9, 4e9]))
        grid = common_frequency_grid([a, b], mode="union")
        np.testing.assert_array_equal(grid, [1e9, 2e9, 3e9, 4e9])

    def test_common_grid_intersection(self) -> None:
        a = _make_thru(np.array([1e9, 2e9, 3e9]))
        b = _make_thru(np.array([2e9, 3e9, 4e9]))
        grid = common_frequency_grid([a, b], mode="intersection")
        np.testing.assert_array_equal(grid, [2e9, 3e9])

    def test_interpolate_networks_aligns_frequencies(self) -> None:
        a = _make_thru(np.array([1e9, 2e9, 3e9]))
        b = _make_attenuator(np.array([1.5e9, 2.5e9]), atten_db=6.0)
        out = interpolate_networks([a, b], mode="union")
        self.assertEqual(out[0].f.size, out[1].f.size)
        np.testing.assert_array_equal(out[0].f, out[1].f)

    def test_cascade_two_attenuators(self) -> None:
        freqs = np.linspace(1e9, 10e9, 11)
        a = _make_attenuator(freqs, atten_db=3.0)
        b = _make_attenuator(freqs, atten_db=3.0)
        casc = cascade_networks([a, b])
        # Total insertion loss should be 6 dB across the band
        s21_db = 20.0 * np.log10(np.abs(casc.s[:, 1, 0]))
        np.testing.assert_allclose(s21_db, -6.0, atol=1e-9)

    def test_cascade_handles_different_grids(self) -> None:
        a = _make_attenuator(np.array([1e9, 2e9, 3e9, 4e9]), atten_db=3.0)
        b = _make_attenuator(np.array([1e9, 2.5e9, 4e9]), atten_db=3.0)
        casc = cascade_networks([a, b], freq_mode="intersection")
        np.testing.assert_array_equal(casc.f, [1e9, 4e9])
        s21_db = 20.0 * np.log10(np.abs(casc.s[:, 1, 0]))
        np.testing.assert_allclose(s21_db, -6.0, atol=1e-9)

    def test_tdr_from_matched_network_returns_z0(self) -> None:
        # A perfectly matched 2-port has S11=0 -> Z(t) ~= Z0 everywhere.
        freqs = np.linspace(0.0, 20e9, 201)
        nw = _make_thru(freqs, z0=50.0)
        t, z = tdr_from_network(nw, port=0, window=None)
        self.assertEqual(t.shape, z.shape)
        # Discard the first sample (DC pile-up) and check Z stays close to 50 Ohm
        np.testing.assert_allclose(z[10:100], 50.0, atol=1.0)


if __name__ == "__main__":
    unittest.main()
