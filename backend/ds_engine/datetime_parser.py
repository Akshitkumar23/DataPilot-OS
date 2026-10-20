"""
DataPilot OS — Canonical Explicit Datetime Normalizer & Robust Vectorized Parser.

Provides:
- normalize_datetime_format: Deterministically translates human date format tokens
  (e.g., 'DD/MM/YYYY', 'YYYY-MM-DD', 'MM/DD/YYYY', 'DD/MM/YYYY HH:mm') into standard
  Python strptime format strings (e.g., '%d/%m/%Y', '%Y-%m-%d', etc.), or validates
  pre-existing strptime directives.
- parse_datetime_column: Vectorized pandas datetime parsing using exact/normalized
  format specifications, returning a DatetimeParseResult with parsed Series, row counts,
  and unparseable record statistics without mutating input data.
"""

from dataclasses import dataclass
import re
from typing import Optional, Tuple
import pandas as pd


# Recognized valid strptime conversion directives
VALID_STRPTIME_DIRECTIVES = {
    "%Y", "%y", "%m", "%B", "%b", "%d", "%H", "%I", "%M", "%S",
    "%f", "%p", "%z", "%Z", "%j", "%U", "%W", "%w", "%a", "%A",
    "%c", "%x", "%X", "%%"
}

# Token map from human-friendly date tokens to Python strptime format tokens.
# Ordered by token string length descending to prevent partial match collisions (e.g., YYYY before YY).
HUMAN_DATETIME_TOKENS = [
    ("YYYY", "%Y"),
    ("yyyy", "%Y"),
    ("YY", "%y"),
    ("yy", "%y"),
    ("MM", "%m"),
    ("DD", "%d"),
    ("dd", "%d"),
    ("HH", "%H"),
    ("hh", "%H"),
    ("mm", "%M"),
    ("SS", "%S"),
    ("ss", "%S"),
]


@dataclass(frozen=True)
class DatetimeParseResult:
    """Outcome of parsing a Series with an explicit or inferred format."""
    series: pd.Series
    valid_count: int
    unparseable_count: int
    null_count: int
    total_count: int
    format_used: Optional[str]


def normalize_datetime_format(format_str: str) -> str:
    """
    Normalizes and strictly validates a user- or model-specified datetime format string.

    Converts tokens like 'DD/MM/YYYY' -> '%d/%m/%Y'.
    If the string is already a strptime string (contains '%'), validates that all '%'
    specifiers are recognized standard directives.
    Raises ValueError on unrecognized, empty, or invalid formats without silent guessing.
    """
    if not isinstance(format_str, str) or not format_str.strip():
        raise ValueError("Datetime format must be a non-empty string.")

    cleaned = format_str.strip()

    # If it already contains '%', validate standard strptime directives
    if "%" in cleaned:
        directives = re.findall(r"%[a-zA-Z%]", cleaned)
        if not directives:
            raise ValueError(f"Invalid format string '{format_str}': contains '%' but no valid directives.")
        for d in directives:
            if d not in VALID_STRPTIME_DIRECTIVES:
                raise ValueError(f"Unrecognized strptime directive '{d}' in format '{format_str}'.")
        return cleaned

    # Human-friendly tokens translation
    sorted_tokens = sorted(HUMAN_DATETIME_TOKENS, key=lambda x: len(x[0]), reverse=True)
    pattern = "|".join(re.escape(k) for k, _ in sorted_tokens)
    token_dict = dict(sorted_tokens)

    normalized = re.sub(pattern, lambda m: token_dict[m.group(0)], cleaned)

    # Must have produced at least one valid strptime directive
    directives = re.findall(r"%[a-zA-Z%]", normalized)
    if not directives:
        raise ValueError(
            f"Invalid datetime format '{format_str}': cannot map to recognized datetime tokens (e.g., YYYY, MM, DD, HH, mm, ss)."
        )

    for d in directives:
        if d not in VALID_STRPTIME_DIRECTIVES:
            raise ValueError(f"Normalized format '{normalized}' contains invalid directive '{d}'.")

    return normalized


def parse_datetime_column(
    series: pd.Series,
    format_str: Optional[str] = None
) -> DatetimeParseResult:
    """
    Vectorized parse of a pandas Series into datetime64.

    If format_str is provided, it is normalized and strictly enforced via format=...
    Coerces invalid records to NaT to measure unparseable counts precisely.
    Does NOT mutate input series.
    """
    total = len(series)
    null_count = int(series.isna().sum())

    if format_str is not None and str(format_str).strip():
        normalized_fmt = normalize_datetime_format(format_str)
        # Using exact=False to permit slight whitespace/separator flexibility if needed, or exact format
        parsed = pd.to_datetime(series, format=normalized_fmt, errors="coerce")
        format_used = normalized_fmt
    else:
        # Standard parsing without format specification
        parsed = pd.to_datetime(series, errors="coerce")
        format_used = None

    valid_mask = parsed.notna()
    valid_count = int(valid_mask.sum())
    unparseable_count = int((series.notna() & ~valid_mask).sum())

    return DatetimeParseResult(
        series=parsed,
        valid_count=valid_count,
        unparseable_count=unparseable_count,
        null_count=null_count,
        total_count=total,
        format_used=format_used,
    )
