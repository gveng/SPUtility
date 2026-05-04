"""
Touchstone interpolation, cascading and TDR helpers based on scikit-rf.

This module provides ADS-compatible utilities to:
  * load multiple Touchstone files,
  * resample them on a common frequency grid (interpolation),
  * cascade them through T-matrix multiplication,
  * compute Time-Domain Reflectometry (TDR) traces from a Network.

scikit-rf is used as the reference numerical engine because it implements
the same S<->T conversion and complex interpolation conventions used by
commercial tools such as Keysight ADS.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import skrf as rf


def load_networks(filepaths: Sequence[str]) -> List[rf.Network]:
    """Load a sequence of Touchstone files as scikit-rf Network objects."""
    return [rf.Network(str(fp)) for fp in filepaths]


def common_frequency_grid(
    networks: Sequence[rf.Network], mode: str = "union"
) -> np.ndarray:
    """Build a common frequency grid (Hz) from a list of networks.

    mode = 'union'        -> sorted union of all sample frequencies
    mode = 'intersection' -> sorted intersection of all sample frequencies
    mode = 'finest'       -> uniform grid using the finest spacing found
    """
    if not networks:
        return np.array([], dtype=float)

    freq_sets = [set(np.asarray(nw.f, dtype=float).tolist()) for nw in networks]
    if mode == "intersection":
        common = sorted(set.intersection(*freq_sets))
        return np.array(common, dtype=float)
    if mode == "finest":
        f_min = max(nw.f.min() for nw in networks)
        f_max = min(nw.f.max() for nw in networks)
        finest = min(np.diff(nw.f).min() for nw in networks if nw.f.size > 1)
        if f_max <= f_min or finest <= 0:
            return np.array([], dtype=float)
        n = int(np.floor((f_max - f_min) / finest)) + 1
        return f_min + np.arange(n, dtype=float) * finest
    common = sorted(set.union(*freq_sets))
    return np.array(common, dtype=float)


def interpolate_networks(
    networks: Sequence[rf.Network],
    freqs_hz: np.ndarray | None = None,
    mode: str = "union",
) -> List[rf.Network]:
    """Resample each network on a common frequency grid using scikit-rf.

    If freqs_hz is None, a grid is built from the inputs using `mode`.
    """
    if not networks:
        return []
    grid = freqs_hz if freqs_hz is not None else common_frequency_grid(networks, mode=mode)
    if grid.size == 0:
        return [nw.copy() for nw in networks]
    target = rf.Frequency.from_f(grid, unit="Hz")
    return [
        nw.interpolate(target, kind="linear", fill_value="extrapolate", bounds_error=False)
        for nw in networks
    ]


def cascade_networks(
    networks: Sequence[rf.Network],
    freq_mode: str = "intersection",
) -> rf.Network:
    """Cascade a sequence of 2N-port networks (T-Cascade).

    All inputs are first resampled on a common frequency grid, then
    cascaded via T-parameter multiplication using scikit-rf's built-in
    ``Network.__pow__`` operator (left-to-right). The resulting network
    has the same number of ports as the inputs.
    """
    if not networks:
        raise ValueError("cascade_networks requires at least one network.")
    if any(nw.nports != networks[0].nports for nw in networks):
        raise ValueError("All networks must have the same number of ports to cascade.")

    resampled = interpolate_networks(networks, mode=freq_mode)
    result = resampled[0]
    for nw in resampled[1:]:
        result = result ** nw
    return result


def tdr_from_network(
    network: rf.Network,
    port: int = 0,
    window: str | None = "hamming",
    n_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a TDR impedance trace from a Network's Sii.

    Returns (time_seconds, Z_ohm) computed from the step response of the
    selected port reflection coefficient. A Hamming window is applied by
    default to limit Gibbs ringing, mirroring ADS defaults.
    """
    if network.nports < 1:
        raise ValueError("Network has no ports.")

    nw = network.copy()
    if window:
        try:
            nw = nw.windowed(window=window, normalize=True)
        except Exception:
            # Older scikit-rf returns None / mutates in place; ignore.
            pass

    s_port = nw.s[:, port, port]
    freq = np.asarray(nw.f, dtype=float)
    if freq[0] != 0.0:
        s0 = np.interp(0.0, freq, s_port.real) + 1j * np.interp(0.0, freq, s_port.imag)
        freq = np.concatenate(([0.0], freq))
        s_port = np.concatenate(([s0], s_port))

    n_fft = n_points or (1 << int(np.ceil(np.log2(2 * len(freq)))))
    impulse = np.fft.irfft(s_port, n=n_fft)
    step = np.cumsum(impulse)

    df = freq[1] - freq[0] if len(freq) > 1 else 1.0
    t = np.arange(n_fft) / (n_fft * df)

    z0 = float(np.real(nw.z0[0, port]))
    rho = np.clip(step, -0.999999, 0.999999)
    z = z0 * (1.0 + rho) / (1.0 - rho)
    return t, z


__all__ = [
    "load_networks",
    "common_frequency_grid",
    "interpolate_networks",
    "cascade_networks",
    "tdr_from_network",
]
