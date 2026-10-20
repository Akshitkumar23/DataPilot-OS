"""Version-safe, typed normalization for pandas calendar frequencies.

The planner reasons about a *calendar interval*; pandas versions define the
string spelling accepted by their offset parser. This module is the sole
boundary between those two concerns. It never infers an interval from a
dataset or query: callers must still provide one explicitly.
"""

from __future__ import annotations

import re

import pandas as pd


# Pandas 2.2/3.x renamed these historical aliases. This is a library
# compatibility protocol, not a query-to-operation fallback: every mapping is
# one-to-one with the same documented interval in the newer pandas API.
_DEPRECATED_PANDAS_ALIASES = {
    "H": "h", "BH": "bh", "CBH": "cbh", "T": "min", "S": "s",
    "L": "ms", "U": "us", "N": "ns", "M": "ME", "BM": "BME",
    "CBM": "CBME", "Q": "QE", "BQ": "BQE", "Y": "YE", "A": "YE",
    "BY": "BYE", "BA": "BYE", "SM": "SME",
}


def canonicalize_pandas_frequency(value: str) -> str:
    """Return the runtime-supported canonical alias for an explicit interval.

    Values accepted by the installed pandas parser are used directly. If the
    parser rejects a historical alias that pandas renamed, only that known
    spelling is translated and then parsed again. Invalid values fail loudly;
    no cadence is inferred from the dataset or query text.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A non-empty pandas frequency is required.")

    raw = value.strip()
    try:
        return pd.tseries.frequencies.to_offset(raw).freqstr
    except (TypeError, ValueError) as original_error:
        match = re.fullmatch(r"(?P<count>\d*)(?P<unit>[A-Za-z]+)(?P<suffix>-.+)?", raw)
        if not match:
            raise ValueError(f"Unsupported pandas frequency '{value}'.") from original_error

        replacement = _DEPRECATED_PANDAS_ALIASES.get(match.group("unit"))
        if replacement is None:
            raise ValueError(f"Unsupported pandas frequency '{value}'.") from original_error

        candidate = f"{match.group('count')}{replacement}{match.group('suffix') or ''}"
        try:
            return pd.tseries.frequencies.to_offset(candidate).freqstr
        except (TypeError, ValueError) as normalized_error:
            raise ValueError(
                f"Unsupported pandas frequency '{value}' (normalized as '{candidate}')."
            ) from normalized_error
