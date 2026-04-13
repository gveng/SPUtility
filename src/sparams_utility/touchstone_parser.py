from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class TouchstoneParseError(ValueError):
    """Raised when a Touchstone file cannot be parsed safely."""


class TouchstoneFormat(str, Enum):
    RI = "RI"
    MA = "MA"
    DB = "DB"


@dataclass(frozen=True)
class TouchstoneOptions:
    frequency_unit: str
    parameter: str
    data_format: TouchstoneFormat
    reference_resistance: float


@dataclass(frozen=True)
class SParameterCell:
    row: int
    col: int
    raw_primary: float
    raw_secondary: float
    complex_value: complex
    magnitude_db: float


@dataclass(frozen=True)
class TouchstonePoint:
    frequency_hz: float
    s_matrix: List[List[SParameterCell]]


@dataclass(frozen=True)
class MagnitudeTable:
    frequencies_hz: List[float]
    traces_db: Dict[str, List[float]]


@dataclass(frozen=True)
class TouchstoneFile:
    source_name: str
    nports: int
    options: TouchstoneOptions
    trace_names: List[str]
    points: List[TouchstonePoint]
    magnitude_table: MagnitudeTable
    comments: List[str]


_FREQ_SCALE_TO_HZ = {
    "HZ": 1.0,
    "KHZ": 1e3,
    "MHZ": 1e6,
    "GHZ": 1e9,
}


def parse_touchstone_file(file_path: str | Path) -> TouchstoneFile:
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    return parse_touchstone_string(content, source_name=str(path))


def parse_touchstone_string(content: str, source_name: str = "<string>") -> TouchstoneFile:
    lines = content.splitlines()
    option_line: Optional[str] = None
    data_tokens: List[str] = []
    comments: List[str] = []

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped.startswith("!"):
            comments.append(stripped[1:].strip())
            continue

        if stripped.startswith("#"):
            if option_line is not None:
                raise TouchstoneParseError(
                    f"{source_name}:{line_no}: multiple option lines are not supported"
                )
            option_line = stripped
            continue

        if "!" in raw_line:
            before_comment, comment = raw_line.split("!", 1)
            comments.append(comment.strip())
            raw_line = before_comment

        segment = raw_line.strip()
        if not segment:
            continue

        data_tokens.extend(segment.split())

    if option_line is None:
        raise TouchstoneParseError(
            f"{source_name}: missing option line (expected '# ...')"
        )

    options = _parse_option_line(option_line, source_name)
    if options.frequency_unit not in _FREQ_SCALE_TO_HZ:
        allowed = ", ".join(sorted(_FREQ_SCALE_TO_HZ))
        raise TouchstoneParseError(
            f"{source_name}: unsupported frequency unit '{options.frequency_unit}', expected one of {allowed}"
        )

    numeric_values = [_parse_float_token(tok, source_name) for tok in data_tokens]
    if not numeric_values:
        raise TouchstoneParseError(f"{source_name}: no numeric data found")

    ext_nports = _infer_nports_from_source_name(source_name)

    nports: Optional[int] = None
    if ext_nports is not None and _data_length_matches_nports(len(numeric_values), ext_nports):
        # Extension gives an unambiguous match — use it directly.
        nports = ext_nports
    else:
        port_matches = [n for n in range(1, 65) if _data_length_matches_nports(len(numeric_values), n)]
        if len(port_matches) == 1:
            nports = port_matches[0]
        elif len(port_matches) == 0:
            raise TouchstoneParseError(
                f"{source_name}: could not infer number of ports from extension or data width"
            )
        elif ext_nports is not None and ext_nports in port_matches:
            nports = ext_nports
        elif ext_nports is not None:
            raise TouchstoneParseError(
                f"{source_name}: extension implies {ext_nports} ports, but data length "
                f"{len(numeric_values)} does not match Touchstone row width"
            )
        else:
            raise TouchstoneParseError(
                f"{source_name}: ambiguous port count inferred from data width "
                f"(matches {port_matches}); provide a .sNp extension"
            )

    points = _build_points(
        numeric_values=numeric_values,
        nports=nports,
        data_format=options.data_format,
        freq_scale=_FREQ_SCALE_TO_HZ[options.frequency_unit],
        source_name=source_name,
    )

    trace_names = [f"S{r}{c}" for r in range(1, nports + 1) for c in range(1, nports + 1)]
    magnitude_table = _build_magnitude_table(points, trace_names)

    return TouchstoneFile(
        source_name=source_name,
        nports=nports,
        options=options,
        trace_names=trace_names,
        points=points,
        magnitude_table=magnitude_table,
        comments=comments,
    )


def _parse_option_line(line: str, source_name: str) -> TouchstoneOptions:
    tokens = line[1:].split()
    if not tokens:
        raise TouchstoneParseError(f"{source_name}: option line is empty")

    frequency_unit = "GHZ"
    parameter = "S"
    data_format = TouchstoneFormat.MA
    reference_resistance = 50.0

    seen_r = False
    idx = 0
    while idx < len(tokens):
        token = tokens[idx].upper()

        if token in _FREQ_SCALE_TO_HZ:
            frequency_unit = token
        elif token in {"S", "Y", "Z", "H", "G"}:
            parameter = token
        elif token in {"RI", "MA", "DB"}:
            data_format = TouchstoneFormat(token)
        elif token == "R":
            if idx + 1 >= len(tokens):
                raise TouchstoneParseError(
                    f"{source_name}: option line contains 'R' without resistance value"
                )
            reference_resistance = _parse_float_token(tokens[idx + 1], source_name)
            seen_r = True
            idx += 1
        else:
            raise TouchstoneParseError(
                f"{source_name}: unsupported option token '{tokens[idx]}'"
            )

        idx += 1

    if parameter != "S":
        raise TouchstoneParseError(
            f"{source_name}: only S-parameters are supported, got '{parameter}'"
        )

    if not seen_r:
        reference_resistance = 50.0

    return TouchstoneOptions(
        frequency_unit=frequency_unit,
        parameter=parameter,
        data_format=data_format,
        reference_resistance=reference_resistance,
    )


def _parse_float_token(token: str, source_name: str) -> float:
    candidate = token.replace("D", "E").replace("d", "e")
    try:
        return float(candidate)
    except ValueError as exc:
        raise TouchstoneParseError(
            f"{source_name}: invalid numeric token '{token}'"
        ) from exc


def _infer_nports_from_source_name(source_name: str) -> Optional[int]:
    suffix = Path(source_name).suffix
    if len(suffix) >= 4 and suffix[1].lower() == "s" and suffix[-1].lower() == "p":
        body = suffix[2:-1]
        if body.isdigit():
            nports = int(body)
            if nports > 0:
                return nports
    return None


def _data_length_matches_nports(value_count: int, nports: int) -> bool:
    block = 1 + (2 * nports * nports)
    return value_count > 0 and value_count % block == 0


def _infer_nports_from_data_length(value_count: int) -> Optional[int]:
    matches: List[int] = []
    for n in range(1, 65):
        if _data_length_matches_nports(value_count, n):
            matches.append(n)

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    # Ambiguous — caller must use extension hint to resolve.
    return None


def _build_points(
    numeric_values: Sequence[float],
    nports: int,
    data_format: TouchstoneFormat,
    freq_scale: float,
    source_name: str,
) -> List[TouchstonePoint]:
    block_size = 1 + (2 * nports * nports)
    if len(numeric_values) % block_size != 0:
        raise TouchstoneParseError(
            f"{source_name}: data token count {len(numeric_values)} is not a multiple of row width {block_size}"
        )

    points: List[TouchstonePoint] = []
    cursor = 0
    while cursor < len(numeric_values):
        freq_raw = numeric_values[cursor]
        freq_hz = freq_raw * freq_scale
        cursor += 1

        cells: List[SParameterCell] = []
        for linear_index in range(nports * nports):
            primary = numeric_values[cursor]
            secondary = numeric_values[cursor + 1]
            cursor += 2

            row = (linear_index // nports) + 1
            col = (linear_index % nports) + 1
            complex_value, mag_db = _convert_pair_to_complex_and_db(
                primary, secondary, data_format, source_name
            )
            cells.append(
                SParameterCell(
                    row=row,
                    col=col,
                    raw_primary=primary,
                    raw_secondary=secondary,
                    complex_value=complex_value,
                    magnitude_db=mag_db,
                )
            )

        matrix: List[List[SParameterCell]] = []
        for row_start in range(0, len(cells), nports):
            matrix.append(cells[row_start : row_start + nports])

        points.append(TouchstonePoint(frequency_hz=freq_hz, s_matrix=matrix))

    return points


def _convert_pair_to_complex_and_db(
    primary: float,
    secondary: float,
    data_format: TouchstoneFormat,
    source_name: str,
) -> Tuple[complex, float]:
    if data_format == TouchstoneFormat.RI:
        real = primary
        imag = secondary
        magnitude = math.hypot(real, imag)
        mag_db = _safe_mag_to_db(magnitude)
        return complex(real, imag), mag_db

    if data_format == TouchstoneFormat.MA:
        magnitude = primary
        if magnitude < 0:
            raise TouchstoneParseError(
                f"{source_name}: MA magnitude cannot be negative ({magnitude})"
            )
        angle_rad = math.radians(secondary)
        complex_value = complex(
            magnitude * math.cos(angle_rad),
            magnitude * math.sin(angle_rad),
        )
        mag_db = _safe_mag_to_db(magnitude)
        return complex_value, mag_db

    # DB format uses dB magnitude directly in primary.
    mag_db = primary
    magnitude = math.pow(10.0, mag_db / 20.0)
    angle_rad = math.radians(secondary)
    complex_value = complex(
        magnitude * math.cos(angle_rad),
        magnitude * math.sin(angle_rad),
    )
    return complex_value, mag_db


def _safe_mag_to_db(magnitude: float) -> float:
    if magnitude <= 0:
        return float("-inf")
    return 20.0 * math.log10(magnitude)


def _build_magnitude_table(
    points: Sequence[TouchstonePoint], trace_names: Sequence[str]
) -> MagnitudeTable:
    traces_db: Dict[str, List[float]] = {name: [] for name in trace_names}
    frequencies_hz: List[float] = []

    for point in points:
        frequencies_hz.append(point.frequency_hz)
        for row in point.s_matrix:
            for cell in row:
                trace = f"S{cell.row}{cell.col}"
                traces_db[trace].append(cell.magnitude_db)

    return MagnitudeTable(frequencies_hz=frequencies_hz, traces_db=traces_db)
