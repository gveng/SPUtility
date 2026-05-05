__version__ = "2.6.6"
__app_name__ = "S-Params Studio"
__build_date__ = "2026-05-05"
__author__ = "Gabriele Vittori"

from .touchstone_parser import (
    MagnitudeTable,
    SParameterCell,
    TouchstoneFile,
    TouchstoneFormat,
    TouchstoneOptions,
    TouchstoneParseError,
    TouchstonePoint,
    parse_touchstone_file,
    parse_touchstone_string,
)

__all__ = [
    "MagnitudeTable",
    "SParameterCell",
    "TouchstoneFile",
    "TouchstoneFormat",
    "TouchstoneOptions",
    "TouchstoneParseError",
    "TouchstonePoint",
    "parse_touchstone_file",
    "parse_touchstone_string",
]
