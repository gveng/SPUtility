from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import linalg
from scipy.interpolate import PchipInterpolator

from sparams_utility.models.circuit import CircuitDocument, CircuitPortRef
from sparams_utility.models.state import AppState


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
            if instance.block_kind in {"port_ground", "port_diff", "gnd"}:
                continue
            if instance.block_kind in {"lumped_r", "lumped_l", "lumped_c"}:
                _stamp_lumped(instance, frequency, uf, grounded_roots, root_to_node, y_global)
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
