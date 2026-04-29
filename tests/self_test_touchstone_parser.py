from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sparams_utility.touchstone_parser import parse_touchstone_string


def run_self_test() -> None:
    ma_sample = """# MHz S MA R 50
! A 2-port MA sample with comments and wrapped rows
100   0.5 0    0.1 -90
! Port Impedance 50 50
      2.0 180  0.1 45
200   0.25 0   0.2 0  1.5 -180
! Port Impedance 50 50
      0.3 90
"""

    ri_sample = """# GHz S RI R 50
! A 2-port RI sample with wrapped rows
1.0   1.0 0.0   0.0 0.5
! Port Impedance 50 50
      0.0 -1.0  -0.5 0.0
2.0   0.0 1.0   0.3 0.4   0.1 0.2
      -0.2 -0.1
"""

    ma_data = parse_touchstone_string(ma_sample, source_name="ma_sample.s2p")
    assert ma_data.nports == 2
    assert ma_data.options.data_format.value == "MA"
    assert len(ma_data.points) == 2

    s11_db_f1 = ma_data.points[0].s_matrix[0][0].magnitude_db
    expected_s11_db_f1 = 20.0 * math.log10(0.5)
    assert abs(s11_db_f1 - expected_s11_db_f1) < 1e-12

    ri_data = parse_touchstone_string(ri_sample, source_name="ri_sample.s2p")
    assert ri_data.nports == 2
    assert ri_data.options.data_format.value == "RI"
    assert len(ri_data.points) == 2

    s21_complex_f1 = ri_data.points[0].s_matrix[1][0].complex_value
    assert abs(s21_complex_f1.real - 0.0) < 1e-12
    assert abs(s21_complex_f1.imag + 1.0) < 1e-12

    s12_db_f2 = ri_data.points[1].s_matrix[0][1].magnitude_db
    expected_s12_db_f2 = 20.0 * math.log10(math.hypot(0.3, 0.4))
    assert abs(s12_db_f2 - expected_s12_db_f2) < 1e-12

    print("Touchstone parser self-test passed for MA and RI samples.")


if __name__ == "__main__":
    run_self_test()
