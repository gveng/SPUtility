from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import linalg
from scipy.interpolate import PchipInterpolator
from scipy.signal import fftconvolve
from scipy.signal.windows import tukey
from scipy.special import erf as _scipy_erf

from sparams_utility.models.circuit import CircuitDocument, CircuitPortRef, DriverSpec, TransientSourceSpec
from sparams_utility.models.state import AppState
from sparams_utility.transmission_lines import (
    substrate_geom_from_spec as _tline_substrate_geom_from_spec,
    synthesize_tline_s_matrix as _tline_synthesize_s_matrix,
)


_TLINE_KINDS = {
    "tline_microstrip",
    "tline_stripline",
    "tline_microstrip_coupled",
    "tline_stripline_coupled",
    "tline_cpw",
    "tline_cpw_coupled",
    "taper",
}

_TLINE_KIND_TO_LINE_KIND = {
    "tline_microstrip": "microstrip",
    "tline_stripline": "stripline",
    "tline_microstrip_coupled": "microstrip_coupled",
    "tline_stripline_coupled": "stripline_coupled",
    "tline_cpw": "tline_cpw",
    "tline_cpw_coupled": "tline_cpw_coupled",
    "taper": "taper",
}

# Required substrate kind per tline block-kind. "taper" accepts either,
# so it is intentionally absent from this mapping (handled in _stamp_tline).
_TLINE_REQUIRED_SUBSTRATE_KIND = {
    "tline_microstrip": "substrate",
    "tline_stripline": "substrate_stripline",
    "tline_microstrip_coupled": "substrate",
    "tline_stripline_coupled": "substrate_stripline",
    "tline_cpw": "substrate",
    "tline_cpw_coupled": "substrate",
}


@dataclass(frozen=True)
class CircuitSolveResult:
    nports: int
    frequencies_hz: np.ndarray
    s_matrices: np.ndarray  # shape (nfreq, nports, nports)
    z0_ohm: np.ndarray  # shape (nports,)
    passivity: "PassivityDiagnostic | None" = None


@dataclass(frozen=True)
class PassivityThresholds:
    sigma_noise_tol: float = 1e-6
    sigma_warn_tol: float = 1e-4
    sigma_hard_tol: float = 1e-3
    persistent_run_points: int = 3
    persistent_fraction: float = 0.005


@dataclass(frozen=True)
class PassivitySummary:
    severity: str
    worst_frequency_hz: float | None
    worst_sigma_max: float
    worst_sigma_excess: float
    worst_min_margin: float
    points_over_noise: int
    points_over_warn: int
    points_over_hard: int
    longest_warn_run: int
    longest_hard_run: int


@dataclass(frozen=True)
class PassivityDiagnostic:
    thresholds: PassivityThresholds
    frequencies_hz: np.ndarray
    sigma_max: np.ndarray
    sigma_excess: np.ndarray
    min_margin: np.ndarray
    summary: PassivitySummary


@dataclass(frozen=True)
class _TouchstoneInterpolationCache:
    frequencies_hz: np.ndarray
    interpolation_axis: np.ndarray
    y_cube: np.ndarray
    z0_ohm: float
    use_log_axis: bool
    interpolation_mode: str
    real_interpolator: Any | None = None
    imag_interpolator: Any | None = None


def solve_circuit_network(document: CircuitDocument, state: AppState) -> CircuitSolveResult:
    frequencies = _build_frequency_grid(document)
    if frequencies.size == 0:
        raise ValueError("Sweep has no valid frequency points.")

    se_assignments = sorted(document.external_ports, key=lambda item: item.external_port_number)
    diff_assignments = sorted(document.differential_ports, key=lambda item: item.external_port_number)
    if not se_assignments and not diff_assignments:
        raise ValueError("At least one external port block is required.")

    uf = _UnionFind()
    for instance in document.instances:
        for port in range(1, instance.nports + 1):
            uf.add((instance.instance_id, port))

    for connection in document.connections:
        uf.union(connection.port_a.key(), connection.port_b.key())

    grounded_roots: set[Tuple[str, int]] = set()
    for instance in document.instances:
        if instance.block_kind != "gnd":
            continue
        root = uf.find((instance.instance_id, 1))
        grounded_roots.add(root)

    root_to_node: Dict[Tuple[str, int], int] = {}
    next_node = 0
    for instance in document.instances:
        for port in range(1, instance.nports + 1):
            root = uf.find((instance.instance_id, port))
            if root in grounded_roots:
                continue
            if root not in root_to_node:
                root_to_node[root] = next_node
                next_node += 1

    node_count = next_node
    if node_count == 0:
        raise ValueError("No active electrical nodes found in circuit.")

    # Build internal SE node list: first all single-ended ports, then + and − of each diff pair.
    internal_se_nodes: List[int] = []
    internal_se_z0: List[float] = []
    # Track which internal indices correspond to differential pairs for later MM conversion.
    diff_pair_indices: List[Tuple[int, int]] = []  # (plus_idx, minus_idx) into internal_se_nodes

    for assignment in se_assignments:
        key = assignment.port_ref.key()
        if key not in uf.parent:
            raise ValueError("An external port references a missing block port.")
        root = uf.find(key)
        if root in grounded_roots:
            raise ValueError("An external port cannot be tied directly to GND.")
        internal_se_nodes.append(root_to_node[root])
        instance = document.get_instance(assignment.port_ref.instance_id)
        internal_se_z0.append(float(instance.impedance_ohm) if instance is not None else 50.0)

    for diff_assign in diff_assignments:
        plus_key = diff_assign.port_ref_plus.key()
        minus_key = diff_assign.port_ref_minus.key()
        if plus_key not in uf.parent or minus_key not in uf.parent:
            raise ValueError("A differential port references a missing block port.")
        root_p = uf.find(plus_key)
        root_m = uf.find(minus_key)
        if root_p in grounded_roots or root_m in grounded_roots:
            raise ValueError("A differential port terminal cannot be tied directly to GND.")
        plus_idx = len(internal_se_nodes)
        internal_se_nodes.append(root_to_node[root_p])
        minus_idx = len(internal_se_nodes)
        internal_se_nodes.append(root_to_node[root_m])
        diff_instance = document.get_instance(diff_assign.port_ref_plus.instance_id)
        z0_se = float(diff_instance.impedance_ohm) / 2.0 if diff_instance is not None else 50.0
        internal_se_z0.append(z0_se)
        internal_se_z0.append(z0_se)
        diff_pair_indices.append((plus_idx, minus_idx))

    if len(set(internal_se_nodes)) != len(internal_se_nodes):
        raise ValueError("Two or more external ports are shorted together. Use distinct nodes for exported ports.")

    z0_se_array = np.array(internal_se_z0, dtype=float)
    if np.any(z0_se_array <= 0.0):
        raise ValueError("External port impedance must be > 0.")

    touchstone_cache = _build_touchstone_cache(document, state)

    n_se = len(internal_se_nodes)
    s_se_all = np.zeros((frequencies.size, n_se, n_se), dtype=np.complex128)
    for idx, frequency in enumerate(frequencies):
        y_global = np.zeros((node_count, node_count), dtype=np.complex128)

        for instance in document.instances:
            if instance.block_kind in {"port_ground", "port_diff", "gnd", "driver_se", "driver_diff", "eyescope_se", "eyescope_diff", "scope_se", "scope_diff", "net_node", "substrate", "substrate_stripline"}:
                continue
            if instance.block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
                _stamp_lumped(instance, frequency, uf, grounded_roots, root_to_node, y_global)
                continue
            if instance.block_kind in _TLINE_KINDS:
                _stamp_tline(instance, frequency, uf, grounded_roots, root_to_node, y_global, document)
                continue
            if instance.block_kind == "attenuator":
                _stamp_attenuator(instance, uf, grounded_roots, root_to_node, y_global)
                continue
            if instance.block_kind == "circulator":
                _stamp_circulator(instance, uf, grounded_roots, root_to_node, y_global)
                continue
            if instance.block_kind == "coupler":
                _stamp_coupler(instance, uf, grounded_roots, root_to_node, y_global)
                continue
            if instance.block_kind == "touchstone":
                _stamp_touchstone(instance, frequency, uf, grounded_roots, root_to_node, y_global, touchstone_cache)

        y_ports = _reduce_to_external_ports(y_global, internal_se_nodes)
        s_se_all[idx] = _y_to_s(y_ports, z0_se_array)

    # Apply SE → mixed-mode (differential) conversion if there are differential ports.
    if diff_pair_indices:
        n_se_only = len(se_assignments)
        n_diff = len(diff_assignments)
        n_out = n_se_only + n_diff
        z0_out_list: List[float] = []
        for assignment in se_assignments:
            inst = document.get_instance(assignment.port_ref.instance_id)
            z0_out_list.append(float(inst.impedance_ohm) if inst is not None else 50.0)
        for diff_assign in diff_assignments:
            inst = document.get_instance(diff_assign.port_ref_plus.instance_id)
            z0_out_list.append(float(inst.impedance_ohm) if inst is not None else 100.0)
        z0_out = np.array(z0_out_list, dtype=float)

        s_out = np.zeros((frequencies.size, n_out, n_out), dtype=np.complex128)
        for idx in range(frequencies.size):
            s_out[idx] = _se_to_mixed_mode(
                s_se_all[idx], n_se_only, diff_pair_indices,
            )
    else:
        n_out = len(se_assignments)
        z0_out = z0_se_array
        s_out = s_se_all

    return CircuitSolveResult(
        nports=n_out,
        frequencies_hz=frequencies,
        s_matrices=s_out,
        z0_ohm=z0_out,
        passivity=_analyze_passivity(frequencies, s_out),
    )


def to_touchstone_string(result: CircuitSolveResult) -> str:
    return to_touchstone_string_with_format(result, data_format="RI", frequency_unit="GHz")


def _analyze_passivity(
    frequencies_hz: np.ndarray,
    s_matrices: np.ndarray,
    thresholds: PassivityThresholds | None = None,
) -> PassivityDiagnostic:
    active_thresholds = thresholds or PassivityThresholds()
    sigma_max = np.zeros(frequencies_hz.size, dtype=float)
    sigma_excess = np.zeros(frequencies_hz.size, dtype=float)
    min_margin = np.zeros(frequencies_hz.size, dtype=float)

    for idx, s_matrix in enumerate(s_matrices):
        singular_values = np.linalg.svd(s_matrix, compute_uv=False)
        sigma = float(singular_values[0]) if singular_values.size else 0.0
        sigma_max[idx] = sigma
        sigma_excess[idx] = max(0.0, sigma - 1.0)
        passive_margin = np.eye(s_matrix.shape[0], dtype=np.complex128) - s_matrix.conj().T @ s_matrix
        hermitian_margin = 0.5 * (passive_margin + passive_margin.conj().T)
        eigenvalues = np.linalg.eigvalsh(hermitian_margin)
        min_margin[idx] = float(eigenvalues[0]) if eigenvalues.size else 0.0

    points_over_noise = int(np.count_nonzero(sigma_excess > active_thresholds.sigma_noise_tol))
    points_over_warn = int(np.count_nonzero(sigma_excess > active_thresholds.sigma_warn_tol))
    points_over_hard = int(np.count_nonzero(sigma_excess > active_thresholds.sigma_hard_tol))
    longest_warn_run = _longest_true_run(sigma_excess > active_thresholds.sigma_warn_tol)
    longest_hard_run = _longest_true_run(sigma_excess > active_thresholds.sigma_hard_tol)
    persistent_points = max(
        active_thresholds.persistent_run_points,
        int(math.ceil(frequencies_hz.size * active_thresholds.persistent_fraction)),
    )
    worst_idx = int(np.argmax(sigma_excess)) if sigma_excess.size else 0
    worst_excess = float(sigma_excess[worst_idx]) if sigma_excess.size else 0.0
    worst_sigma = float(sigma_max[worst_idx]) if sigma_max.size else 0.0
    worst_margin = float(min_margin[np.argmin(min_margin)]) if min_margin.size else 0.0
    worst_frequency = float(frequencies_hz[worst_idx]) if sigma_excess.size else None

    severity = "pass"
    if points_over_hard > 0 or longest_warn_run >= persistent_points:
        severity = "hard"
    elif worst_excess > active_thresholds.sigma_warn_tol or points_over_warn > 0:
        severity = "borderline"
    elif worst_excess > active_thresholds.sigma_noise_tol:
        severity = "noise"

    return PassivityDiagnostic(
        thresholds=active_thresholds,
        frequencies_hz=frequencies_hz.copy(),
        sigma_max=sigma_max,
        sigma_excess=sigma_excess,
        min_margin=min_margin,
        summary=PassivitySummary(
            severity=severity,
            worst_frequency_hz=worst_frequency,
            worst_sigma_max=worst_sigma,
            worst_sigma_excess=worst_excess,
            worst_min_margin=worst_margin,
            points_over_noise=points_over_noise,
            points_over_warn=points_over_warn,
            points_over_hard=points_over_hard,
            longest_warn_run=longest_warn_run,
            longest_hard_run=longest_hard_run,
        ),
    )


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def to_touchstone_string_with_format(
    result: CircuitSolveResult,
    *,
    data_format: str = "RI",
    frequency_unit: str = "GHz",
) -> str:
    fmt = data_format.upper()
    if fmt not in {"RI", "MA", "DB"}:
        raise ValueError(f"Unsupported Touchstone format '{data_format}'.")

    unit = frequency_unit.upper()
    scale = {
        "HZ": 1.0,
        "KHZ": 1e3,
        "MHZ": 1e6,
        "GHZ": 1e9,
    }.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported frequency unit '{frequency_unit}'.")

    lines = [
        "! Generated by SPUtility circuit solver",
        f"# {unit} S {fmt} R 50",
    ]

    for f_idx, freq_hz in enumerate(result.frequencies_hz):
        freq_scaled = freq_hz / scale
        parts: List[str] = [f"{freq_scaled:.12g}"]
        for row in range(result.nports):
            for col in range(result.nports):
                value = result.s_matrices[f_idx, row, col]
                if fmt == "RI":
                    parts.append(f"{value.real:.12g}")
                    parts.append(f"{value.imag:.12g}")
                else:
                    magnitude = abs(value)
                    angle_deg = math.degrees(math.atan2(value.imag, value.real))
                    if fmt == "MA":
                        parts.append(f"{magnitude:.12g}")
                    else:
                        mag_db = -300.0 if magnitude <= 0.0 else 20.0 * math.log10(magnitude)
                        parts.append(f"{mag_db:.12g}")
                    parts.append(f"{angle_deg:.12g}")
        lines.append(" ".join(parts))

    return "\n".join(lines) + "\n"


def _build_frequency_grid(document: CircuitDocument) -> np.ndarray:
    fmin = float(document.sweep.fmin_hz)
    fmax = float(document.sweep.fmax_hz)
    fstep = float(document.sweep.fstep_hz)
    if fmin <= 0.0 or fmax < fmin or fstep <= 0.0:
        return np.array([], dtype=float)

    count = int(np.floor((fmax - fmin) / fstep)) + 1
    if count < 1:
        return np.array([], dtype=float)
    freqs = fmin + np.arange(count, dtype=float) * fstep
    if freqs[-1] < fmax:
        freqs = np.append(freqs, fmax)
    return freqs


def _build_touchstone_cache(document: CircuitDocument, state: AppState) -> Dict[str, _TouchstoneInterpolationCache]:
    cache: Dict[str, _TouchstoneInterpolationCache] = {}
    for instance in document.instances:
        if instance.block_kind != "touchstone":
            continue
        if instance.source_file_id in cache:
            continue
        loaded = state.get_file(instance.source_file_id)
        if loaded is None:
            raise ValueError(f"Missing Touchstone file for block '{instance.display_label}'.")

        points = loaded.data.points
        freqs = np.array([p.frequency_hz for p in points], dtype=float)
        s_cube = np.zeros((len(points), loaded.data.nports, loaded.data.nports), dtype=np.complex128)
        for f_idx, point in enumerate(points):
            for row in range(loaded.data.nports):
                for col in range(loaded.data.nports):
                    s_cube[f_idx, row, col] = point.s_matrix[row][col].complex_value
        freqs, s_cube = _fill_missing_frequency_points(freqs, s_cube)
        z0 = float(loaded.data.options.reference_resistance)
        y_cube = np.zeros_like(s_cube)
        for f_idx in range(freqs.size):
            y_cube[f_idx] = _s_to_y(s_cube[f_idx], z0)

        interpolation_axis, use_log_axis = _choose_interpolation_axis(freqs)
        mode, real_interpolator, imag_interpolator = _build_matrix_interpolator(interpolation_axis, y_cube)
        cache[instance.source_file_id] = _TouchstoneInterpolationCache(
            frequencies_hz=freqs,
            interpolation_axis=interpolation_axis,
            y_cube=y_cube,
            z0_ohm=z0,
            use_log_axis=use_log_axis,
            interpolation_mode=mode,
            real_interpolator=real_interpolator,
            imag_interpolator=imag_interpolator,
        )
    return cache


def _fill_missing_frequency_points(source_freqs: np.ndarray, s_cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(source_freqs)
    freqs_sorted = source_freqs[order]
    s_sorted = s_cube[order]

    unique_freqs, unique_idx = np.unique(freqs_sorted, return_index=True)
    return unique_freqs, s_sorted[unique_idx]


def _choose_interpolation_axis(freqs_hz: np.ndarray) -> Tuple[np.ndarray, bool]:
    if np.all(freqs_hz > 0.0):
        return np.log(freqs_hz), True
    return freqs_hz.astype(float), False


def _build_matrix_interpolator(x_axis: np.ndarray, matrix_cube: np.ndarray) -> Tuple[str, Any | None, Any | None]:
    if x_axis.size <= 1:
        return "constant", None, None
    if x_axis.size == 2:
        return "linear", None, None
    return (
        "pchip",
        PchipInterpolator(x_axis, matrix_cube.real, axis=0, extrapolate=False),
        PchipInterpolator(x_axis, matrix_cube.imag, axis=0, extrapolate=False),
    )


def _stamp_lumped(
    instance,
    frequency: float,
    uf: "_UnionFind",
    grounded_roots: set[Tuple[str, int]],
    root_to_node: Dict[Tuple[str, int], int],
    y_global: np.ndarray,
) -> None:
    port_a = (instance.instance_id, 1)
    port_b = (instance.instance_id, 2)
    root_a = uf.find(port_a)
    root_b = uf.find(port_b)

    if instance.block_kind == "lumped_r":
        impedance = complex(instance.impedance_ohm, 0.0)
    elif instance.block_kind == "lumped_l":
        impedance = complex(0.0, 2.0 * np.pi * frequency * instance.impedance_ohm)
    elif instance.block_kind == "lumped_c":
        if frequency <= 0.0:
            impedance = complex(1e18, 0.0)
        else:
            impedance = complex(0.0, -1.0 / (2.0 * np.pi * frequency * instance.impedance_ohm))
    else:
        return

    if abs(impedance) < 1e-18:
        admittance = complex(1e18, 0.0)
    else:
        admittance = 1.0 / impedance

    _stamp_admittance_between_nodes(root_a, root_b, admittance, grounded_roots, root_to_node, y_global)


def _stamp_touchstone(
    instance,
    frequency: float,
    uf: "_UnionFind",
    grounded_roots: set[Tuple[str, int]],
    root_to_node: Dict[Tuple[str, int], int],
    y_global: np.ndarray,
    cache: Dict[str, _TouchstoneInterpolationCache],
) -> None:
    if instance.source_file_id not in cache:
        return
    y_block = _interpolate_y_matrix(cache[instance.source_file_id], frequency)

    roots = [uf.find((instance.instance_id, idx + 1)) for idx in range(instance.nports)]
    for i in range(instance.nports):
        for j in range(instance.nports):
            _stamp_general_admittance(
                roots[i],
                roots[j],
                y_block[i, j],
                grounded_roots,
                root_to_node,
                y_global,
            )


def _stamp_tline(
    instance,
    frequency: float,
    uf: "_UnionFind",
    grounded_roots: set[Tuple[str, int]],
    root_to_node: Dict[Tuple[str, int], int],
    y_global: np.ndarray,
    document: CircuitDocument,
) -> None:
    """Synthesize the line S-matrix on the fly and stamp it as Y-block.

    The block references a substrate by display label; if the substrate
    cannot be resolved or has the wrong kind for the line type, the
    block is silently skipped at this frequency point (the schematic
    validator surfaces these conditions to the user separately).
    """
    spec = getattr(instance, "transmission_line_spec", None)
    if spec is None:
        return
    line_kind = _TLINE_KIND_TO_LINE_KIND.get(instance.block_kind)
    if line_kind is None:
        return

    # Resolve referenced substrate by display label.
    # "taper" accepts either substrate kind; everything else has a fixed kind.
    expected_kind = _TLINE_REQUIRED_SUBSTRATE_KIND.get(instance.block_kind)
    if expected_kind is None:
        # Taper: accept any substrate kind.
        substrate_inst = next(
            (
                i
                for i in document.instances
                if i.block_kind in ("substrate", "substrate_stripline")
                and i.display_label == spec.substrate_name
            ),
            None,
        )
    else:
        substrate_inst = next(
            (
                i
                for i in document.instances
                if i.block_kind == expected_kind and i.display_label == spec.substrate_name
            ),
            None,
        )
    if substrate_inst is None or substrate_inst.substrate_spec is None:
        return

    geom = _tline_substrate_geom_from_spec(substrate_inst.substrate_spec)

    if instance.block_kind == "tline_cpw":
        from sparams_utility.transmission_lines import (
            cpw_z0_eeff,
            cpw_line_s_matrix,
        )
        z0_line, eps_eff = cpw_z0_eeff(
            float(spec.width_m),
            float(spec.cpw_slot_m),
            float(geom.height_m),
            float(geom.epsilon_r),
            t=float(geom.conductor_thickness_m),
        )
        s_matrix = cpw_line_s_matrix(
            z0_line,
            eps_eff,
            float(geom.loss_tangent),
            float(spec.width_m),
            float(geom.conductivity_s_per_m),
            float(spec.length_m),
            float(spec.z0_ref_ohm),
            float(frequency),
        )
    elif instance.block_kind == "tline_cpw_coupled":
        from sparams_utility.transmission_lines import (
            cpw_coupled_modes,
            cpw_coupled_s_matrix,
        )
        z0e, z0o, eeff_e, eeff_o = cpw_coupled_modes(
            float(spec.width_m),
            float(spec.cpw_slot_m),
            float(spec.spacing_m),
            float(geom.height_m),
            float(geom.epsilon_r),
            t=float(geom.conductor_thickness_m),
        )
        s_matrix = cpw_coupled_s_matrix(
            z0e,
            z0o,
            eeff_e,
            eeff_o,
            float(geom.loss_tangent),
            float(spec.width_m),
            float(geom.conductivity_s_per_m),
            float(spec.length_m),
            float(spec.z0_ref_ohm),
            float(frequency),
        )
    elif instance.block_kind == "taper":
        from sparams_utility.transmission_lines import taper_s_matrix
        # Map substrate block-kind to base line kind for the taper.
        sub_line_kind = (
            "stripline" if substrate_inst.block_kind == "substrate_stripline" else "microstrip"
        )
        w_end = float(spec.width_end_m) if spec.width_end_m > 0.0 else float(spec.width_m)
        s_matrix = taper_s_matrix(
            line_kind=sub_line_kind,
            substrate=geom,
            W_start=float(spec.width_m),
            W_end=w_end,
            length=float(spec.length_m),
            z_ref=float(spec.z0_ref_ohm),
            f=float(frequency),
            profile=str(spec.taper_profile or "linear"),
            S_cpw=float(spec.cpw_slot_m),
        )
    else:
        s_matrix = _tline_synthesize_s_matrix(
            line_kind=line_kind,
            width_m=float(spec.width_m),
            length_m=float(spec.length_m),
            spacing_m=float(spec.spacing_m),
            z0_ref=float(spec.z0_ref_ohm),
            substrate=geom,
            frequency_hz=float(frequency),
        )
    y_block = _s_to_y(s_matrix, float(spec.z0_ref_ohm))

    nports = s_matrix.shape[0]
    roots = [uf.find((instance.instance_id, idx + 1)) for idx in range(nports)]
    for i in range(nports):
        for j in range(nports):
            _stamp_general_admittance(
                roots[i],
                roots[j],
                y_block[i, j],
                grounded_roots,
                root_to_node,
                y_global,
            )


def _stamp_lumped_block(
    instance,
    s_matrix: np.ndarray,
    z_ref: float,
    uf: "_UnionFind",
    grounded_roots: set[Tuple[str, int]],
    root_to_node: Dict[Tuple[str, int], int],
    y_global: np.ndarray,
) -> None:
    """Stamp a generic, frequency-flat S-matrix block as a Y-admittance block."""
    y_block = _s_to_y(s_matrix, float(z_ref))
    nports = s_matrix.shape[0]
    roots = [uf.find((instance.instance_id, idx + 1)) for idx in range(nports)]
    for i in range(nports):
        for j in range(nports):
            _stamp_general_admittance(
                roots[i], roots[j], y_block[i, j],
                grounded_roots, root_to_node, y_global,
            )


def _stamp_attenuator(
    instance,
    uf: "_UnionFind",
    grounded_roots: set[Tuple[str, int]],
    root_to_node: Dict[Tuple[str, int], int],
    y_global: np.ndarray,
) -> None:
    spec = getattr(instance, "attenuator_spec", None)
    if spec is None:
        return
    from sparams_utility.transmission_lines import attenuator_s_matrix
    s_matrix = attenuator_s_matrix(
        attenuation_db=float(spec.attenuation_db),
        z_ref=float(spec.z0_ref_ohm),
    )
    _stamp_lumped_block(
        instance, s_matrix, float(spec.z0_ref_ohm),
        uf, grounded_roots, root_to_node, y_global,
    )


def _stamp_circulator(
    instance,
    uf: "_UnionFind",
    grounded_roots: set[Tuple[str, int]],
    root_to_node: Dict[Tuple[str, int], int],
    y_global: np.ndarray,
) -> None:
    spec = getattr(instance, "circulator_spec", None)
    if spec is None:
        return
    from sparams_utility.transmission_lines import circulator_s_matrix
    s_matrix = circulator_s_matrix(
        insertion_loss_db=float(spec.insertion_loss_db),
        isolation_db=float(spec.isolation_db),
        return_loss_db=float(spec.return_loss_db),
        direction=str(spec.direction or "cw"),
        z_ref=float(spec.z0_ref_ohm),
    )
    _stamp_lumped_block(
        instance, s_matrix, float(spec.z0_ref_ohm),
        uf, grounded_roots, root_to_node, y_global,
    )


def _stamp_coupler(
    instance,
    uf: "_UnionFind",
    grounded_roots: set[Tuple[str, int]],
    root_to_node: Dict[Tuple[str, int], int],
    y_global: np.ndarray,
) -> None:
    spec = getattr(instance, "coupler_spec", None)
    if spec is None:
        return
    from sparams_utility.transmission_lines import coupler_s_matrix
    s_matrix = coupler_s_matrix(
        kind=str(spec.kind or "branch_line_90"),
        coupling_db=float(spec.coupling_db),
        insertion_loss_db=float(spec.insertion_loss_db),
        isolation_db=float(spec.isolation_db),
        return_loss_db=float(spec.return_loss_db),
        z_ref=float(spec.z0_ref_ohm),
    )
    _stamp_lumped_block(
        instance, s_matrix, float(spec.z0_ref_ohm),
        uf, grounded_roots, root_to_node, y_global,
    )


def _stamp_admittance_between_nodes(
    root_a: Tuple[str, int],
    root_b: Tuple[str, int],
    admittance: complex,
    grounded_roots: set[Tuple[str, int]],
    root_to_node: Dict[Tuple[str, int], int],
    y_global: np.ndarray,
) -> None:
    a_is_ground = root_a in grounded_roots
    b_is_ground = root_b in grounded_roots

    if not a_is_ground:
        ia = root_to_node[root_a]
        y_global[ia, ia] += admittance
    if not b_is_ground:
        ib = root_to_node[root_b]
        y_global[ib, ib] += admittance
    if not a_is_ground and not b_is_ground:
        ia = root_to_node[root_a]
        ib = root_to_node[root_b]
        y_global[ia, ib] -= admittance
        y_global[ib, ia] -= admittance


def _stamp_general_admittance(
    root_i: Tuple[str, int],
    root_j: Tuple[str, int],
    value: complex,
    grounded_roots: set[Tuple[str, int]],
    root_to_node: Dict[Tuple[str, int], int],
    y_global: np.ndarray,
) -> None:
    i_ground = root_i in grounded_roots
    j_ground = root_j in grounded_roots
    if i_ground:
        return
    ii = root_to_node[root_i]
    if j_ground:
        if root_i == root_j:
            y_global[ii, ii] += value
        return
    jj = root_to_node[root_j]
    y_global[ii, jj] += value


def _se_to_mixed_mode(
    s_se: np.ndarray,
    n_se_only: int,
    diff_pair_indices: List[Tuple[int, int]],
) -> np.ndarray:
    """Convert a single-ended S-matrix to mixed-mode, keeping only Sdd for differential pairs.

    The internal SE matrix has indices [0..n_se_only-1] for single-ended ports,
    then pairs of (+, −) indices for each differential port.
    The output has n_se_only + len(diff_pair_indices) ports: first the SE ports,
    then one differential (dd) port per pair.

    The modal transformation for each differential pair converts the SE pair (p, n)
    into differential and common modes using:
        a_d = (a_p − a_n) / √2
        a_c = (a_p + a_n) / √2

    We build a full transformation matrix M that is identity for SE ports and
    applies the modal conversion for each diff pair, then extract only the
    rows/columns corresponding to SE ports and differential modes (discarding
    common-mode rows/columns).
    """
    n_se_total = s_se.shape[0]
    n_diff = len(diff_pair_indices)
    n_out = n_se_only + n_diff

    # Build the full modal transformation matrix M (n_se_total × n_se_total).
    # For SE-only ports: identity rows.
    # For each diff pair: row for diff mode, row for common mode.
    m_full = np.zeros((n_se_total, n_se_total), dtype=np.complex128)
    for i in range(n_se_only):
        m_full[i, i] = 1.0

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for d_idx, (plus_idx, minus_idx) in enumerate(diff_pair_indices):
        row_d = n_se_only + 2 * d_idx       # differential mode row
        row_c = n_se_only + 2 * d_idx + 1   # common mode row
        # Differential: (plus − minus) / √2
        m_full[row_d, plus_idx] = inv_sqrt2
        m_full[row_d, minus_idx] = -inv_sqrt2
        # Common: (plus + minus) / √2
        m_full[row_c, plus_idx] = inv_sqrt2
        m_full[row_c, minus_idx] = inv_sqrt2

    # S_mm_full = M · S_se · M^{-1}  (M is unitary so M^{-1} = M^H)
    s_mm_full = m_full @ s_se @ m_full.conj().T

    # Extract only the rows/columns for SE ports and differential modes (drop common modes).
    keep_indices = list(range(n_se_only))
    for d_idx in range(n_diff):
        keep_indices.append(n_se_only + 2 * d_idx)  # differential mode only
    keep = np.array(keep_indices, dtype=int)
    return s_mm_full[np.ix_(keep, keep)]


def _reduce_to_external_ports(y_global: np.ndarray, external_nodes: List[int]) -> np.ndarray:
    n_total = y_global.shape[0]
    ext = np.array(external_nodes, dtype=int)
    all_nodes = np.arange(n_total, dtype=int)
    int_nodes = np.array([n for n in all_nodes if n not in set(external_nodes)], dtype=int)

    y_ee = y_global[np.ix_(ext, ext)]
    if int_nodes.size == 0:
        return y_ee

    y_ei = y_global[np.ix_(ext, int_nodes)]
    y_ie = y_global[np.ix_(int_nodes, ext)]
    y_ii = y_global[np.ix_(int_nodes, int_nodes)]

    correction = y_ei @ _solve_linear_system(y_ii, y_ie, context="Schur complement reduction")

    return y_ee - correction


def _s_to_y(s_matrix: np.ndarray, z0: float) -> np.ndarray:
    n = s_matrix.shape[0]
    ident = np.eye(n, dtype=np.complex128)
    a = ident + s_matrix
    b = ident - s_matrix
    return _solve_linear_system(a, b, context="S to Y conversion") / z0


def _y_to_s(y_matrix: np.ndarray, z0_ports: np.ndarray) -> np.ndarray:
    n = y_matrix.shape[0]
    ident = np.eye(n, dtype=np.complex128)
    sqrt_z0 = np.diag(np.sqrt(z0_ports.astype(np.complex128)))
    a = sqrt_z0 @ y_matrix @ sqrt_z0
    left = ident - a
    right = ident + a
    return _solve_linear_system(right.T, left.T, context="Y to S conversion").T


def _solve_linear_system(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    context: str,
    condition_limit: float = 1e12,
    regularization_factor: float = 1e-12,
) -> np.ndarray:
    if matrix_a.size == 0:
        return np.empty_like(matrix_b)

    condition = np.linalg.cond(matrix_a)
    if np.isfinite(condition) and condition <= condition_limit:
        try:
            return linalg.solve(matrix_a, matrix_b, assume_a="gen", check_finite=False)
        except linalg.LinAlgError:
            pass

    scale = max(float(np.linalg.norm(matrix_a, ord=np.inf)), 1.0)
    regularization = regularization_factor * scale
    regularized = matrix_a + np.eye(matrix_a.shape[0], dtype=np.complex128) * regularization
    try:
        return linalg.solve(regularized, matrix_b, assume_a="gen", check_finite=False)
    except linalg.LinAlgError:
        solution, *_ = linalg.lstsq(matrix_a, matrix_b, check_finite=False)
        if not np.all(np.isfinite(solution)):
            raise ValueError(f"{context} failed: matrix is singular or badly conditioned.")
        return solution


def _interpolate_y_matrix(cache_entry: _TouchstoneInterpolationCache, target_freq: float) -> np.ndarray:
    if target_freq < cache_entry.frequencies_hz[0] or target_freq > cache_entry.frequencies_hz[-1]:
        f_min = cache_entry.frequencies_hz[0]
        f_max = cache_entry.frequencies_hz[-1]
        raise ValueError(
            f"Frequency {target_freq:.12g} Hz is outside Touchstone data range [{f_min:.12g}, {f_max:.12g}] Hz."
        )

    if cache_entry.interpolation_mode == "constant":
        return cache_entry.y_cube[0]

    target_axis = math.log(target_freq) if cache_entry.use_log_axis else target_freq
    if cache_entry.interpolation_mode == "linear":
        idx = int(np.searchsorted(cache_entry.interpolation_axis, target_axis))
        if idx <= 0:
            return cache_entry.y_cube[0]
        if idx >= cache_entry.interpolation_axis.size:
            return cache_entry.y_cube[-1]
        x0 = cache_entry.interpolation_axis[idx - 1]
        x1 = cache_entry.interpolation_axis[idx]
        if x1 <= x0:
            return cache_entry.y_cube[idx]
        alpha = (target_axis - x0) / (x1 - x0)
        return (1.0 - alpha) * cache_entry.y_cube[idx - 1] + alpha * cache_entry.y_cube[idx]

    real_part = cache_entry.real_interpolator(target_axis)
    imag_part = cache_entry.imag_interpolator(target_axis)
    return np.asarray(real_part, dtype=np.float64) + 1j * np.asarray(imag_part, dtype=np.float64)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

    def add(self, item: Tuple[str, int]) -> None:
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item: Tuple[str, int]) -> Tuple[str, int]:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: Tuple[str, int], b: Tuple[str, int]) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# Channel Simulation (eye diagram)
# ---------------------------------------------------------------------------

_PRBS_TAPS: Dict[str, List[List[int]]] = {
    # Standard primitive polynomials (ITU-T O.150, IEEE 802.3 etc.).
    # Each entry is the canonical Fibonacci-LFSR tap list: the leading value
    # is the polynomial degree N and the remaining values are the non-leading
    # polynomial exponents. For example PRBS-7 corresponds to x^7 + x^6 + 1,
    # so the tap list is [7, 6]. The generator below shifts LEFT and feeds the
    # XOR of the tapped register bits into the LSB, which is the convention
    # used by every reference implementation (e.g. ADS, Keysight, NumPy).
    "PRBS-7":  [[7, 6]],
    "PRBS-8":  [[8, 6, 5, 4]],
    "PRBS-9":  [[9, 5]],
    "PRBS-10": [[10, 7]],
    "PRBS-11": [[11, 9]],
    "PRBS-12": [[12, 11, 10, 4]],
    "PRBS-13": [[13, 12, 2, 1]],
    "PRBS-15": [[15, 14]],
    "PRBS-20": [[20, 17]],
    "PRBS-23": [[23, 18]],
    "PRBS-31": [[31, 28]],
}


def _generate_prbs(pattern: str, num_bits: int) -> np.ndarray:
    taps_list = _PRBS_TAPS.get(pattern)
    if taps_list is None:
        raise ValueError(f"Unknown PRBS pattern: {pattern}")
    taps = taps_list[0]
    order = taps[0]
    register = (1 << order) - 1   # all-ones seed (nonzero, on the maximal cycle)
    mask = (1 << order) - 1
    bits = np.empty(num_bits, dtype=np.int8)
    for i in range(num_bits):
        bits[i] = register & 1
        feedback = 0
        for tap in taps:
            feedback ^= (register >> (tap - 1)) & 1
        # Shift LEFT (canonical Fibonacci LFSR) and inject feedback at LSB.
        register = ((register << 1) | feedback) & mask
    return bits


def _prbs_period_length(pattern: str) -> int:
    """Return one full LFSR period length: 2^order − 1."""
    taps_list = _PRBS_TAPS.get(pattern)
    if taps_list is None:
        raise ValueError(f"Unknown PRBS pattern: {pattern}")
    order = taps_list[0][0]
    return (1 << order) - 1


# --- 8b/10b (Widmer & Franaszek, IEEE 802.3 Cl.36) ----------------------------
# 5b/6b D-code table. Each entry is (RD-_code, RD+_code) where the 6-bit code
# is stored MSB-first in transmission order (a,b,c,d,e,i).
_8B10B_5B6B: tuple = (
    (0b100111, 0b011000),  # D.00
    (0b011101, 0b100010),  # D.01
    (0b101101, 0b010010),  # D.02
    (0b110001, 0b110001),  # D.03
    (0b110101, 0b001010),  # D.04
    (0b101001, 0b101001),  # D.05
    (0b011001, 0b011001),  # D.06
    (0b111000, 0b000111),  # D.07
    (0b111001, 0b000110),  # D.08
    (0b100101, 0b100101),  # D.09
    (0b010101, 0b010101),  # D.10
    (0b110100, 0b110100),  # D.11
    (0b001101, 0b001101),  # D.12
    (0b101100, 0b101100),  # D.13
    (0b011100, 0b011100),  # D.14
    (0b010111, 0b101000),  # D.15
    (0b011011, 0b100100),  # D.16
    (0b100011, 0b100011),  # D.17
    (0b010011, 0b010011),  # D.18
    (0b110010, 0b110010),  # D.19
    (0b001011, 0b001011),  # D.20
    (0b101010, 0b101010),  # D.21
    (0b011010, 0b011010),  # D.22
    (0b111010, 0b000101),  # D.23
    (0b110011, 0b001100),  # D.24
    (0b100110, 0b100110),  # D.25
    (0b010110, 0b010110),  # D.26
    (0b110110, 0b001001),  # D.27
    (0b001110, 0b001110),  # D.28
    (0b101110, 0b010001),  # D.29
    (0b011110, 0b100001),  # D.30
    (0b101011, 0b010100),  # D.31
)

# 3b/4b D-code table. Each entry is (RD-_code, RD+_code), MSB-first (f,g,h,j).
_8B10B_3B4B: tuple = (
    (0b1011, 0b0100),  # D.x.0
    (0b1001, 0b1001),  # D.x.1
    (0b0101, 0b0101),  # D.x.2
    (0b1100, 0b0011),  # D.x.3
    (0b1101, 0b0010),  # D.x.4
    (0b1010, 0b1010),  # D.x.5
    (0b0110, 0b0110),  # D.x.6
    (0b1110, 0b0001),  # D.x.7 (primary)
)

# Alternate D.x.A7 form (used to avoid 5-in-a-row across the 6b/4b boundary).
_8B10B_3B4B_A7: tuple = (0b0111, 0b1000)
_A7_X_RD_MINUS = frozenset({17, 18, 20})
_A7_X_RD_PLUS = frozenset({11, 13, 14})


def _encode_8b10b(bits: np.ndarray) -> np.ndarray:
    """Encode an arbitrary bit stream with running-disparity 8b/10b.

    Bits are grouped into bytes LSB-first (bit 0 of byte = first bit in stream =
    A bit in the 8b/10b standard). The stream is zero-padded to a multiple of 8
    bits. Output length is therefore ``ceil(N/8) * 10``.
    All bytes are encoded as D-characters (no K-characters injected).
    """
    n = len(bits)
    if n == 0:
        return bits.astype(np.int8, copy=False)
    pad = (-n) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    nbytes = len(bits) // 8
    packed = np.packbits(
        bits.astype(np.uint8).reshape(nbytes, 8), axis=1, bitorder="little"
    ).reshape(-1)

    out = np.empty(nbytes * 10, dtype=np.int8)
    rd = -1  # current running disparity, -1 or +1
    for i in range(nbytes):
        byte = int(packed[i])
        x = byte & 0x1F          # EDCBA (lower 5)
        y = (byte >> 5) & 0x07   # HGF   (upper 3)

        c6_m, c6_p = _8B10B_5B6B[x]
        c6 = c6_m if rd < 0 else c6_p
        d6 = 2 * bin(c6).count("1") - 6
        if d6:
            rd = -rd

        if y == 7:
            use_alt = (rd < 0 and x in _A7_X_RD_MINUS) or (rd > 0 and x in _A7_X_RD_PLUS)
            if use_alt:
                c4 = _8B10B_3B4B_A7[0] if rd < 0 else _8B10B_3B4B_A7[1]
            else:
                c4_m, c4_p = _8B10B_3B4B[7]
                c4 = c4_m if rd < 0 else c4_p
        else:
            c4_m, c4_p = _8B10B_3B4B[y]
            c4 = c4_m if rd < 0 else c4_p
        d4 = 2 * bin(c4).count("1") - 4
        if d4:
            rd = -rd

        base = i * 10
        out[base + 0] = (c6 >> 5) & 1
        out[base + 1] = (c6 >> 4) & 1
        out[base + 2] = (c6 >> 3) & 1
        out[base + 3] = (c6 >> 2) & 1
        out[base + 4] = (c6 >> 1) & 1
        out[base + 5] = c6 & 1
        out[base + 6] = (c4 >> 3) & 1
        out[base + 7] = (c4 >> 2) & 1
        out[base + 8] = (c4 >> 1) & 1
        out[base + 9] = c4 & 1

    return out


def _apply_encoding(bits: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "None" or encoding == "":
        return bits
    if encoding == "8b10b":
        return _encode_8b10b(bits)
    if encoding == "64b66b":
        # Scrambler/header overhead approximated as a no-op (2 bits per 64).
        # Real 64b/66b would scramble payload with x^58+x^39+1; not implemented.
        return bits
    if encoding == "128b130b":
        # Same approximation as 64b/66b; full PCIe Gen3+ scrambler not implemented.
        return bits
    if encoding == "PAM4":
        return bits
    return bits


@dataclass(frozen=True)
class ChannelSimResult:
    time_s: np.ndarray
    waveform_v: np.ndarray
    ui_s: float
    driver_spec: DriverSpec
    is_differential: bool


@dataclass(frozen=True)
class TransientTrace:
    output_instance_id: str
    output_port_number: int
    label: str
    waveform_v: np.ndarray


@dataclass(frozen=True)
class TransientSimResult:
    time_s: np.ndarray
    traces: Tuple[TransientTrace, ...]
    source_spec: TransientSourceSpec
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _SolvedTransferPath:
    solve_result: CircuitSolveResult
    source_port_idx: int
    output_port_indices: Tuple[int, ...]


def _emit_progress(
    progress_callback: Callable[[int, str], None] | None,
    percent: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(percent, message)


def _solve_transfer_path(
    document: CircuitDocument,
    state: AppState,
    source_instance_id: str,
    output_refs: Sequence[CircuitPortRef],
    progress_callback: Callable[[int, str], None] | None = None,
) -> _SolvedTransferPath:
    _emit_progress(progress_callback, 5, "Solving S-parameter network...")
    result = solve_circuit_network(document, state)

    source_port_idx = _find_port_index(document, source_instance_id, 1)
    if source_port_idx is None:
        raise ValueError("Could not map source port to a solved S-parameter index.")

    output_port_indices: list[int] = []
    for output_ref in output_refs:
        output_port_idx = _find_port_index(
            document,
            output_ref.instance_id,
            output_ref.port_number,
        )
        if output_port_idx is None:
            raise ValueError("Could not map one or more output ports to solved S-parameter indices.")
        output_port_indices.append(output_port_idx)

    return _SolvedTransferPath(
        solve_result=result,
        source_port_idx=source_port_idx,
        output_port_indices=tuple(output_port_indices),
    )


def _interpolate_transfer_function(
    frequencies_hz: np.ndarray,
    transfer_freq: np.ndarray,
    freq_fft: np.ndarray,
) -> np.ndarray:
    h_freq = np.zeros(len(freq_fft), dtype=np.complex128)
    for i, freq_hz in enumerate(freq_fft):
        if freq_hz <= 0:
            h_freq[i] = transfer_freq[0] if len(transfer_freq) > 0 else 0.0
        elif freq_hz < frequencies_hz[0]:
            h_freq[i] = transfer_freq[0]
        elif freq_hz > frequencies_hz[-1]:
            delayed = transfer_freq[-1] * np.exp(-1j * 2 * np.pi * (freq_hz - frequencies_hz[-1]) * 1e-12)
            mag_rolloff = abs(transfer_freq[-1]) * np.exp(-((freq_hz - frequencies_hz[-1]) / frequencies_hz[-1]) * 2.0)
            h_freq[i] = mag_rolloff * np.exp(1j * np.angle(delayed))
        else:
            idx = np.searchsorted(frequencies_hz, freq_hz)
            if idx <= 0:
                h_freq[i] = transfer_freq[0]
            elif idx >= len(frequencies_hz):
                h_freq[i] = transfer_freq[-1]
            else:
                alpha = (freq_hz - frequencies_hz[idx - 1]) / (frequencies_hz[idx] - frequencies_hz[idx - 1])
                h_freq[i] = (1 - alpha) * transfer_freq[idx - 1] + alpha * transfer_freq[idx]
    return h_freq


def _voltage_transfer_function(
    solve_result: CircuitSolveResult,
    source_port_idx: int,
    output_port_idx: int,
    source_impedance_ohm: float | None = None,
) -> np.ndarray:
    """Voltage transfer function from a Thevenin source to a high-impedance probe.

    Solves the multi-port network as a nodal-admittance problem:

        Y · V = I

    with the boundary conditions
        * Port ``source_port_idx`` driven by a Thevenin source
          (``V_src`` open-circuit, ``Z_src`` series),
        * Every other external port left open (``I_k = 0``), since any actual
          loading must be represented *explicitly* in the schematic.

    For ``Z_src > 0`` the source-row equation
        ``Y[s, :] · V = (V_src − V_s) / Z_src``
    is rewritten as
        ``(Y + (1/Z_src)·e_s·e_s^T) · V = (V_src/Z_src) · e_s``
    which is well-conditioned for *any* port reference impedance — including
    high-Z probe ports such as ``scope_diff`` (``Z0 = 2 MΩ``) where the
    classical ``(1 − S22)`` denominator becomes singular.

    For ``Z_src = 0`` the source node is forced to ``V_s = V_src`` directly
    (substitution row).
    """
    s_matrices = solve_result.s_matrices
    z0 = np.asarray(solve_result.z0_ohm, dtype=complex)
    n_freq = s_matrices.shape[0]
    n_port = s_matrices.shape[1]

    z_src = (
        float(z0[source_port_idx].real)
        if source_impedance_ohm is None
        else float(source_impedance_ohm)
    )

    identity = np.eye(n_port, dtype=complex)
    sqrt_z0 = np.sqrt(z0)
    inv_sqrt_z0 = 1.0 / sqrt_z0

    h = np.zeros(n_freq, dtype=complex)

    for k in range(n_freq):
        s = s_matrices[k]
        # Y = (1/√Z0) · (I − S) · (I + S)^{-1} · (1/√Z0)
        try:
            mat_right = np.linalg.solve(identity + s, identity - s)
        except np.linalg.LinAlgError:
            h[k] = 0.0
            continue
        y_mat = (inv_sqrt_z0[:, None] * mat_right) * inv_sqrt_z0[None, :]

        a = y_mat.copy()
        b = np.zeros(n_port, dtype=complex)
        if z_src > 0.0:
            a[source_port_idx, source_port_idx] += 1.0 / z_src
            b[source_port_idx] = 1.0 / z_src  # unit V_src
        else:
            # Ideal voltage source: V_s = V_src.
            a[source_port_idx, :] = 0.0
            a[source_port_idx, source_port_idx] = 1.0
            b[source_port_idx] = 1.0

        try:
            v = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            h[k] = 0.0
            continue
        h[k] = v[output_port_idx]

    return h


def _polarity_sign(polarity: str) -> float:
    return -1.0 if str(polarity).strip().lower().startswith("neg") else 1.0


def _build_step_waveform(spec: TransientSourceSpec, time_s: np.ndarray) -> np.ndarray:
    target_level = abs(spec.amplitude_v) * _polarity_sign(spec.polarity)
    edge_time_s = spec.rise_time_s if target_level >= 0.0 else spec.fall_time_s
    edge_time_s = max(edge_time_s, 0.0)
    delay_s = max(spec.delay_s, 0.0)
    waveform = np.zeros_like(time_s, dtype=float)
    if edge_time_s == 0.0:
        waveform[time_s >= delay_s] = target_level
        return waveform
    active = (time_s >= delay_s) & (time_s < delay_s + edge_time_s)
    waveform[active] = target_level * ((time_s[active] - delay_s) / edge_time_s)
    waveform[time_s >= delay_s + edge_time_s] = target_level
    return waveform


def _build_pulse_waveform(spec: TransientSourceSpec, time_s: np.ndarray) -> np.ndarray:
    target_level = abs(spec.amplitude_v) * _polarity_sign(spec.polarity)
    lead_time_s = spec.rise_time_s if target_level >= 0.0 else spec.fall_time_s
    trail_time_s = spec.fall_time_s if target_level >= 0.0 else spec.rise_time_s
    lead_time_s = max(lead_time_s, 0.0)
    trail_time_s = max(trail_time_s, 0.0)
    delay_s = max(spec.delay_s, 0.0)
    pulse_width_s = max(spec.pulse_width_s, 0.0)
    fall_start_s = delay_s + lead_time_s + pulse_width_s

    waveform = np.zeros_like(time_s, dtype=float)
    if lead_time_s == 0.0:
        waveform[time_s >= delay_s] = target_level
    else:
        active = (time_s >= delay_s) & (time_s < delay_s + lead_time_s)
        waveform[active] = target_level * ((time_s[active] - delay_s) / lead_time_s)
        waveform[time_s >= delay_s + lead_time_s] = target_level

    if trail_time_s == 0.0:
        waveform[time_s >= fall_start_s] = 0.0
        return waveform

    trail_active = (time_s >= fall_start_s) & (time_s < fall_start_s + trail_time_s)
    waveform[trail_active] = target_level * (1.0 - ((time_s[trail_active] - fall_start_s) / trail_time_s))
    waveform[time_s >= fall_start_s + trail_time_s] = 0.0
    return waveform


def _build_transient_source_waveform(
    block_kind: str,
    spec: TransientSourceSpec,
    time_s: np.ndarray,
) -> np.ndarray:
    if block_kind == "transient_step_se":
        return _build_step_waveform(spec, time_s)
    if block_kind == "transient_pulse_se":
        return _build_pulse_waveform(spec, time_s)
    raise ValueError(f"Unsupported transient source kind: {block_kind}")


def _build_driver_transient_waveform(
    spec: DriverSpec,
    time_s: np.ndarray,
) -> np.ndarray:
    """Sample a PRBS NRZ bitstream from a DriverSpec onto the given time grid."""
    if spec.bitrate_gbps <= 0.0:
        raise ValueError("Driver bitrate must be greater than zero.")
    ui_s = 1.0 / (spec.bitrate_gbps * 1e9)
    if getattr(spec, "maximal_length_lfsr", False):
        period = _prbs_period_length(spec.prbs_pattern)
        one_period = _generate_prbs(spec.prbs_pattern, period)
        n_total = max(int(spec.num_bits), period)
        reps = int(np.ceil(n_total / period))
        raw_bits = np.tile(one_period, reps)[:n_total]
    else:
        raw_bits = _generate_prbs(spec.prbs_pattern, spec.num_bits)
    encoded_bits = np.asarray(_apply_encoding(raw_bits, spec.encoding), dtype=float)
    n_bits = int(encoded_bits.size)
    if n_bits <= 0:
        return np.zeros_like(time_s, dtype=float)
    bit_idx = np.floor(np.clip(time_s, 0.0, None) / ui_s).astype(int)
    in_range = bit_idx < n_bits
    waveform = np.full(time_s.shape, spec.voltage_low_v, dtype=float)
    levels = np.where(encoded_bits > 0.5, spec.voltage_high_v, spec.voltage_low_v)
    waveform[in_range] = levels[bit_idx[in_range]]
    waveform[~in_range] = spec.voltage_low_v
    # Apply a simple moving-average filter to approximate edge rates, as done
    # in simulate_channel.
    dt = float(time_s[1] - time_s[0]) if time_s.size > 1 else ui_s
    rise_samples = max(1, int(spec.rise_time_s / dt)) if spec.rise_time_s > 0.0 else 1
    fall_samples = max(1, int(spec.fall_time_s / dt)) if spec.fall_time_s > 0.0 else 1
    kernel_len = max(rise_samples, fall_samples)
    if kernel_len > 1:
        kernel = np.ones(kernel_len, dtype=float) / kernel_len
        waveform = np.convolve(waveform, kernel, mode="same")
    return waveform


def _choose_transient_timebase(
    stop_time_s: float,
    spec: TransientSourceSpec,
    frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, float]:
    if stop_time_s <= 0.0:
        raise ValueError("Stop time must be greater than zero.")
    dt_target = stop_time_s / 4096.0
    edge_candidates = [edge for edge in (spec.rise_time_s, spec.fall_time_s) if edge > 0.0]
    if edge_candidates:
        dt_target = min(dt_target, min(edge_candidates) / 20.0)
    if frequencies_hz.size > 0 and frequencies_hz[-1] > 0.0:
        dt_target = min(dt_target, 1.0 / (8.0 * float(frequencies_hz[-1])))
    dt_target = max(dt_target, stop_time_s / 200000.0)
    sample_count = int(np.ceil(stop_time_s / max(dt_target, 1e-18))) + 1
    sample_count = max(512, min(200000, sample_count))
    time_s = np.linspace(0.0, stop_time_s, sample_count, dtype=float)
    dt = float(time_s[1] - time_s[0]) if sample_count > 1 else stop_time_s
    return time_s, dt


def _format_frequency_hz(value_hz: float) -> str:
    if value_hz >= 1e9:
        return f"{value_hz / 1e9:.4g} GHz"
    if value_hz >= 1e6:
        return f"{value_hz / 1e6:.4g} MHz"
    if value_hz >= 1e3:
        return f"{value_hz / 1e3:.4g} KHz"
    return f"{value_hz:.4g} Hz"


def _collect_transient_warnings(
    frequencies_hz: np.ndarray,
    spec: TransientSourceSpec,
    stop_time_s: float,
) -> Tuple[str, ...]:
    warnings: list[str] = []
    if frequencies_hz.size == 0:
        return ()
    low_frequency_limit_hz = 1.0 / max(stop_time_s, 1e-18)
    if frequencies_hz[0] > low_frequency_limit_hz:
        warnings.append(
            "Sweep Fmin may be too high for the requested stop time "
            f"({_format_frequency_hz(frequencies_hz[0])} > {_format_frequency_hz(low_frequency_limit_hz)}); "
            "low-frequency settling may be inaccurate."
        )
    edge_candidates = [edge for edge in (spec.rise_time_s, spec.fall_time_s) if edge > 0.0]
    if edge_candidates:
        recommended_fmax_hz = 0.35 / min(edge_candidates)
        if frequencies_hz[-1] < recommended_fmax_hz:
            warnings.append(
                "Sweep Fmax may be too low for the configured edge rate "
                f"({_format_frequency_hz(frequencies_hz[-1])} < {_format_frequency_hz(recommended_fmax_hz)}); "
                "edge fidelity may be limited."
            )
    return tuple(warnings)


def _channel_effective_bandwidth_hz(
    freq_hz: np.ndarray, h_freq: np.ndarray, threshold_db: float = -60.0
) -> float:
    """Return the highest frequency above which |H(f)| stays below ``threshold_db``
    relative to the in-band peak. Used to right-size the time-domain sample rate.
    """
    if freq_hz.size == 0 or h_freq.size == 0:
        return 0.0
    mag = np.abs(h_freq)
    peak = float(np.max(mag))
    if peak <= 0.0:
        return float(freq_hz[-1])
    threshold = peak * (10.0 ** (threshold_db / 20.0))
    above = np.where(mag >= threshold)[0]
    if above.size == 0:
        return float(freq_hz[0])
    return float(freq_hz[int(above[-1])])


def _build_causal_impulse_response(
    h_freq_grid: np.ndarray,
    nfft: int,
    dt: float,
    ir_length_s: float,
    tukey_alpha: float = 0.10,
) -> np.ndarray:
    """Build a causal, windowed impulse response from a one-sided spectrum.

    ``h_freq_grid`` is the channel transfer function sampled on
    ``np.fft.rfftfreq(nfft, d=dt)``. The IR is computed via inverse rFFT,
    truncated to ``ir_length_s`` (so the strictly causal main response is kept
    while late acausal/wrap-around tail is discarded), and tapered with a Tukey
    window to suppress spectral truncation ringing on the convolution output.
    """
    h_time = np.fft.irfft(h_freq_grid, n=nfft)
    n_keep = max(8, min(nfft, int(np.ceil(ir_length_s / dt))))
    ir = h_time[:n_keep].copy()
    # Apply a tail-only cosine taper to suppress truncation ringing without
    # touching the leading samples (which carry the main impulse for nearly
    # resistive / broadband-flat channels). A symmetric Tukey would zero out
    # ir[0] and silently kill the entire response in those cases.
    if n_keep > 1 and tukey_alpha > 0.0:
        taper_len = max(1, int(np.ceil(tukey_alpha * n_keep)))
        if taper_len > 1:
            ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(taper_len) / (taper_len - 1)))
            ir[-taper_len:] *= ramp
    return ir


def _interpolate_channel_transfer(
    freq_hz_src: np.ndarray,
    h_src: np.ndarray,
    freq_fft: np.ndarray,
) -> np.ndarray:
    """High-accuracy interpolation of the voltage transfer function on the FFT grid.

    Uses PCHIP on magnitude and unwrapped phase (numerically stable around DC).
    Below ``fmin`` the response is held at the lowest measured value. Above
    ``fmax`` we apply a slow cosine fade from ``mag[-1]`` down to zero spread
    over the entire ``[fmax .. Nyquist_FFT]`` band, with phase continued by
    linear extrapolation. This preserves the impulse-response energy when the
    sweep ``Fmax`` is much smaller than the FFT Nyquist (typical of low-bitrate
    simulations on broadband resistive networks). DC and Nyquist bins are also
    forced real to keep the corresponding time-domain signal strictly real.
    """
    fmax = float(freq_hz_src[-1])
    fmin = float(freq_hz_src[0])
    mag = np.abs(h_src)
    phase = np.unwrap(np.angle(h_src))

    pchip_mag = PchipInterpolator(freq_hz_src, mag, extrapolate=False)
    pchip_phase = PchipInterpolator(freq_hz_src, phase, extrapolate=False)

    m = np.zeros_like(freq_fft, dtype=float)
    p = np.zeros_like(freq_fft, dtype=float)

    inband = (freq_fft >= fmin) & (freq_fft <= fmax)
    if np.any(inband):
        m[inband] = pchip_mag(freq_fft[inband])
        p[inband] = pchip_phase(freq_fft[inband])

    below = freq_fft < fmin
    if np.any(below):
        m[below] = mag[0]
        p[below] = phase[0]

    if len(freq_hz_src) >= 2:
        slope = (phase[-1] - phase[-2]) / max(freq_hz_src[-1] - freq_hz_src[-2], 1.0)
    else:
        slope = 0.0

    nyquist_fft = float(freq_fft[-1]) if freq_fft.size > 0 else fmax
    above = freq_fft > fmax
    if np.any(above) and nyquist_fft > fmax:
        # Two regimes for the out-of-band extension:
        #   1) "Flat" channel (mag[-1] is still close to the in-band peak):
        #      hold the last value all the way to Nyquist. A flat lossless
        #      network really does extend beyond the user's sweep, and a
        #      cosine fade here would smear the impulse response over many
        #      samples and destroy the DC/transient amplitude after IR
        #      truncation.
        #   2) "Roll-off" channel (mag[-1] is already small): apply a cosine
        #      fade from mag[-1] down to 0 spread over the [fmax..Nyquist]
        #      range, so we don't inject a brick-wall cutoff but also don't
        #      hold a non-physical residual response.
        peak = float(np.max(mag)) if mag.size > 0 else 0.0
        ratio = (mag[-1] / peak) if peak > 0.0 else 0.0
        rel = np.clip((freq_fft[above] - fmax) / (nyquist_fft - fmax), 0.0, 1.0)
        if ratio >= 0.5:
            # Flat channel: hold to Nyquist with only a tiny cosine touch in
            # the last 5 % of the band, just to keep the Nyquist bin clean.
            taper = np.where(rel < 0.95, 1.0,
                             0.5 * (1.0 + np.cos(np.pi * (rel - 0.95) / 0.05)))
            m[above] = mag[-1] * taper
        else:
            # Rolling-off channel: smooth full-band cosine fade.
            m[above] = mag[-1] * 0.5 * (1.0 + np.cos(np.pi * rel))
        p[above] = phase[-1] + slope * (freq_fft[above] - fmax)

    h_freq = m * np.exp(1j * p)
    if h_freq.size > 0:
        h_freq[0] = complex(h_freq[0].real, 0.0)
        h_freq[-1] = complex(h_freq[-1].real, 0.0)
    return h_freq


# 10%-90% to gaussian sigma:  rise = 2*sqrt(2)*erfinv(0.8) * sigma  ≈ 2.5631 * sigma
_RISE_TO_SIGMA = 1.0 / 2.5631031311216

def _build_smooth_nrz_waveform(
    encoded_bits: Sequence[int],
    samples_per_bit: int,
    dt: float,
    v_high: float,
    v_low: float,
    rise_time_s: float,
    fall_time_s: float,
) -> np.ndarray:
    """Build an NRZ waveform with analytic erf (gaussian-edge) transitions.

    Each bit boundary k is rendered as the difference between the ideal
    rectangular step and a smooth erf step centered on the boundary, scaled by
    the symbol delta. The correction is non-zero only inside ±6*sigma around
    the boundary, so per-edge cost is O(samples_per_bit). This is the standard
    SerDes / IEEE 802.3 COM model for finite-rise NRZ stimuli and is much more
    accurate than convolving an ideal NRZ with a moving-average filter.
    """
    n_bits = len(encoded_bits)
    if n_bits == 0:
        return np.zeros(0, dtype=float)
    total = n_bits * samples_per_bit
    bits = np.asarray(encoded_bits, dtype=int)
    levels = np.where(bits > 0, v_high, v_low).astype(float)

    # Initial rectangular NRZ.
    waveform = np.repeat(levels, samples_per_bit)

    sigma_rise = max(rise_time_s, dt) * _RISE_TO_SIGMA
    sigma_fall = max(fall_time_s, dt) * _RISE_TO_SIGMA
    sample_indices = np.arange(total, dtype=float)
    # sample times at the centre of each sample (matches np.repeat alignment)
    t = (sample_indices + 0.5) * dt
    sqrt2 = math.sqrt(2.0)

    for k in range(1, n_bits):
        delta = float(levels[k] - levels[k - 1])
        if delta == 0.0:
            continue
        sigma = sigma_rise if delta > 0.0 else sigma_fall
        t_edge = k * samples_per_bit * dt
        half_window = 6.0 * sigma
        i0 = max(0, int(np.floor((t_edge - half_window) / dt)))
        i1 = min(total, int(np.ceil((t_edge + half_window) / dt)) + 1)
        if i0 >= i1:
            continue
        local_t = t[i0:i1]
        smooth_step = 0.5 * (1.0 + _scipy_erf((local_t - t_edge) / (sigma * sqrt2)))
        ideal_step = (local_t >= t_edge).astype(float)
        # Replace the rectangular boundary with the analytic smooth boundary.
        waveform[i0:i1] += delta * (smooth_step - ideal_step)
    return waveform


def simulate_channel(
    document: CircuitDocument,
    state: AppState,
    driver_instance_id: str,
    output_port_instance_id: str,
    output_port_number: int = 1,
    progress_callback: callable | None = None,
) -> ChannelSimResult:
    driver_inst = document.get_instance(driver_instance_id)
    if driver_inst is None or driver_inst.driver_spec is None:
        raise ValueError("Driver instance not found or has no driver specification.")

    spec = driver_inst.driver_spec
    is_diff = driver_inst.block_kind == "driver_diff"

    transfer_path = _solve_transfer_path(
        document,
        state,
        driver_instance_id,
        [CircuitPortRef(output_port_instance_id, output_port_number)],
        progress_callback,
    )
    out_port_idx = transfer_path.output_port_indices[0]
    freq_hz = transfer_path.solve_result.frequencies_hz

    _emit_progress(progress_callback, 25, "Generating PRBS pattern...")
    bitrate_hz = spec.bitrate_gbps * 1e9
    ui_s = 1.0 / bitrate_hz

    if getattr(spec, "maximal_length_lfsr", False):
        period = _prbs_period_length(spec.prbs_pattern)
        one_period = _generate_prbs(spec.prbs_pattern, period)
        n_total = max(int(spec.num_bits), period)
        reps = int(np.ceil(n_total / period))
        raw_bits = np.tile(one_period, reps)[:n_total]
    else:
        raw_bits = _generate_prbs(spec.prbs_pattern, spec.num_bits)
    encoded_bits = _apply_encoding(raw_bits, spec.encoding)
    num_bits = len(encoded_bits)

    # ── (D) Sample rate from effective channel bandwidth, not just Fmax ──
    # Use ~3x the bandwidth where |H(f)| is above -60 dB of its peak so the
    # frequency content that actually affects the time-domain output is
    # critically sampled, while we don't waste CPU on energy-less bins.
    h_sweep = _voltage_transfer_function(
        transfer_path.solve_result,
        transfer_path.source_port_idx,
        out_port_idx,
        source_impedance_ohm=getattr(spec, "source_impedance_ohm", None),
    )
    fmax_sweep_hz = float(freq_hz[-1]) if len(freq_hz) > 0 else bitrate_hz
    bw_hz = _channel_effective_bandwidth_hz(freq_hz, h_sweep, threshold_db=-60.0)
    bw_hz = min(max(bw_hz, 5.0 * bitrate_hz), fmax_sweep_hz)
    samples_per_bit = int(np.ceil(max(3.0 * bw_hz * ui_s, 32.0)))
    samples_per_bit = max(64, min(samples_per_bit, 512))
    dt = ui_s / samples_per_bit
    total_samples = num_bits * samples_per_bit

    # FFT grid sized for IR extraction. The IR is then *truncated* to a few
    # tens of UI and convolved with the stimulus by ``fftconvolve`` — this is
    # equivalent to (B) pulse-response superposition for any LTI channel and
    # avoids the circular wrap-around inherent to a single big FFT.
    nfft_ir = int(2 ** np.ceil(np.log2(max(samples_per_bit * 256, 4096))))
    freq_fft = np.fft.rfftfreq(nfft_ir, d=dt)

    _emit_progress(progress_callback, 40, "Interpolating channel transfer function...")
    h_freq = _interpolate_channel_transfer(freq_hz, h_sweep, freq_fft)

    # ── (A) Causal, windowed impulse response ─────────────────────────────
    # Keep up to 64 UI of impulse response (or the full IFFT, whichever is
    # smaller). 64 UI is enough for typical lossy channels; lossless channels
    # decay much faster and the Tukey taper hides the tail anyway.
    ir_length_s = min(64.0 * ui_s, nfft_ir * dt)
    impulse_response = _build_causal_impulse_response(
        h_freq, nfft_ir, dt, ir_length_s
    )

    v_high = spec.voltage_high_v
    v_low = spec.voltage_low_v

    _emit_progress(progress_callback, 55, "Building NRZ waveform...")
    nrz_waveform = _build_smooth_nrz_waveform(
        encoded_bits,
        samples_per_bit,
        dt,
        v_high,
        v_low,
        spec.rise_time_s,
        spec.fall_time_s,
    )

    _emit_progress(progress_callback, 70, "FFT convolution (pulse-response superposition)...")
    # ── (B) Linear convolution via fftconvolve ────────────────────────────
    # fftconvolve picks the optimal block size internally. Mathematically
    # identical to per-bit pulse-response superposition for LTI channels.
    convolved = fftconvolve(nrz_waveform, impulse_response, mode="full")
    output_waveform = convolved[:total_samples]

    _emit_progress(progress_callback, 90, "Building result...")
    time_s = np.arange(total_samples, dtype=float) * dt

    return ChannelSimResult(
        time_s=time_s,
        waveform_v=output_waveform,
        ui_s=ui_s,
        driver_spec=spec,
        is_differential=is_diff,
    )


def simulate_transient(
    document: CircuitDocument,
    state: AppState,
    source_instance_id: str,
    output_refs: Sequence[CircuitPortRef],
    stop_time_s: float,
    progress_callback: Callable[[int, str], None] | None = None,
) -> TransientSimResult:
    source_inst = document.get_instance(source_instance_id)
    if source_inst is None:
        raise ValueError("Transient source instance not found.")
    is_driver_source = source_inst.block_kind in {"driver_se", "driver_diff"}
    is_pulse_step_source = source_inst.block_kind in {"transient_step_se", "transient_pulse_se"}
    if not (is_driver_source or is_pulse_step_source):
        raise ValueError(
            "Selected source is not a transient step/pulse source or a Channel Sim driver."
        )
    if is_pulse_step_source and source_inst.transient_source_spec is None:
        raise ValueError("Transient source has no transient specification.")
    if is_driver_source and source_inst.driver_spec is None:
        raise ValueError("Driver block has no driver specification.")
    if not output_refs:
        raise ValueError("At least one transient output must be selected.")
    transfer_path = _solve_transfer_path(
        document,
        state,
        source_instance_id,
        output_refs,
        progress_callback,
    )
    if is_driver_source:
        driver_spec = source_inst.driver_spec
        edges_spec = TransientSourceSpec(
            rise_time_s=driver_spec.rise_time_s,
            fall_time_s=driver_spec.fall_time_s,
        )
        warnings = _collect_transient_warnings(
            transfer_path.solve_result.frequencies_hz,
            edges_spec,
            stop_time_s,
        )
    else:
        edges_spec = source_inst.transient_source_spec
        warnings = _collect_transient_warnings(
            transfer_path.solve_result.frequencies_hz,
            edges_spec,
            stop_time_s,
        )

    _emit_progress(progress_callback, 25, "Building transient stimulus...")
    time_s, dt = _choose_transient_timebase(
        stop_time_s,
        edges_spec,
        transfer_path.solve_result.frequencies_hz,
    )
    if is_driver_source:
        driver_spec = source_inst.driver_spec
        ui_s = 1.0 / (driver_spec.bitrate_gbps * 1e9) if driver_spec.bitrate_gbps > 0 else dt
        # Ensure at least ~16 samples per bit for a decent NRZ representation.
        if dt > ui_s / 16.0:
            sample_count = int(np.ceil(stop_time_s / max(ui_s / 16.0, 1e-18))) + 1
            sample_count = max(512, min(200000, sample_count))
            time_s = np.linspace(0.0, stop_time_s, sample_count, dtype=float)
            dt = float(time_s[1] - time_s[0]) if sample_count > 1 else stop_time_s
        source_waveform = _build_driver_transient_waveform(driver_spec, time_s)
        result_spec = edges_spec
    else:
        source_waveform = _build_transient_source_waveform(
            source_inst.block_kind, edges_spec, time_s
        )
        result_spec = edges_spec

    _emit_progress(progress_callback, 45, "Preparing FFT convolution...")
    total_samples = len(time_s)
    nfft = int(2 ** np.ceil(np.log2(max(total_samples, 1))))
    freq_fft = np.fft.rfftfreq(nfft, d=dt)
    padded = np.zeros(nfft, dtype=float)
    padded[:total_samples] = source_waveform
    source_fft = np.fft.rfft(padded)

    traces: list[TransientTrace] = []
    output_count = len(output_refs)
    for index, (output_ref, output_port_idx) in enumerate(
        zip(output_refs, transfer_path.output_port_indices, strict=False)
    ):
        progress = 55 + int(((index + 1) / max(output_count, 1)) * 30)
        instance = document.get_instance(output_ref.instance_id)
        label = instance.display_label if instance is not None else output_ref.instance_id
        _emit_progress(progress_callback, progress, f"Computing transient trace for {label}...")
        h_freq = _interpolate_transfer_function(
            transfer_path.solve_result.frequencies_hz,
            _voltage_transfer_function(
                transfer_path.solve_result,
                transfer_path.source_port_idx,
                output_port_idx,
                source_impedance_ohm=(
                    getattr(source_inst.driver_spec, "source_impedance_ohm", None)
                    if is_driver_source
                    else None
                ),
            ),
            freq_fft,
        )
        output_fft = source_fft * h_freq
        waveform_v = np.fft.irfft(output_fft, n=nfft)[:total_samples]
        traces.append(
            TransientTrace(
                output_instance_id=output_ref.instance_id,
                output_port_number=output_ref.port_number,
                label=label,
                waveform_v=waveform_v,
            )
        )

    _emit_progress(progress_callback, 90, "Building transient result...")
    return TransientSimResult(
        time_s=time_s,
        traces=tuple(traces),
        source_spec=result_spec,
        warnings=warnings,
    )


def _find_port_index(document: CircuitDocument, instance_id: str, port_number: int) -> int | None:
    se_assignments = sorted(document.external_ports, key=lambda item: item.external_port_number)
    diff_assignments = sorted(document.differential_ports, key=lambda item: item.external_port_number)

    idx = 0
    for assignment in se_assignments:
        if assignment.port_ref.instance_id == instance_id and assignment.port_ref.port_number == port_number:
            return idx
        idx += 1

    for diff_assign in diff_assignments:
        if diff_assign.port_ref_plus.instance_id == instance_id:
            return idx
        idx += 1

    return None
