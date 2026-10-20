"""
DataPilot OS — Dataset Profiler & Schema Role Engine.
Automatically profiles connected datasets, detects semantic column roles through
pure statistical and type-theoretic properties, calculates exact hashes across entire datasets,
and generates compact schema context for LLM planning.
100% Zero domain keyword hardcoding.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SemanticRoleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)


class RoleCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    role: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    null_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    cardinality: int = 0


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    semantic_role: str = Field(
        ...,
        description="Detected role: 'entity_id', 'temporal', 'binary_flag', 'continuous_numeric', 'discrete_numeric', 'categorical_nominal', 'text'"
    )
    null_count: int
    null_percentage: float
    cardinality: int
    is_unique: bool
    unique_ratio: float = Field(ge=0.0, le=1.0)
    is_constant: bool = False
    # Structural evidence only.  These fields deliberately do not depend on
    # column names, so a prospective sequence/order column can be presented
    # for confirmation without guessing its semantic meaning.
    is_monotonic_increasing: bool = False
    is_monotonic_decreasing: bool = False
    sample_values: List[Any] = Field(default_factory=list)
    stats: Optional[Dict[str, float]] = None
    datetime_parse_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    inferred_frequency: Optional[str] = None
    role_candidates: List[SemanticRoleEvidence] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_hash: str
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    temporal_column_candidates: List[str] = Field(default_factory=list)
    target_column_candidates: List[str] = Field(default_factory=list)
    entity_id_candidates: List[str] = Field(default_factory=list)
    binary_flag_candidates: List[str] = Field(default_factory=list)
    continuous_numeric_candidates: List[str] = Field(default_factory=list)
    categorical_candidates: List[str] = Field(default_factory=list)
    sequence_column_candidates: List[str] = Field(default_factory=list)
    datetime_parse_candidates: List[str] = Field(default_factory=list)
    role_ambiguities: Dict[str, List[str]] = Field(default_factory=dict)
    duplicate_row_count: int = 0
    total_missing_percentage: float = 0.0

    # Ranked statistical role candidates
    ranked_numeric_measures: List[RoleCandidate] = Field(default_factory=list)
    ranked_categorical_dimensions: List[RoleCandidate] = Field(default_factory=list)
    ranked_temporal_columns: List[RoleCandidate] = Field(default_factory=list)
    ranked_entity_identifiers: List[RoleCandidate] = Field(default_factory=list)
    ranked_target_candidates: List[RoleCandidate] = Field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return self.dataset_hash

    def get_ranked_candidates(self, role: str) -> List[RoleCandidate]:
        role_normalized = role.lower().strip()
        role_map = {
            "numeric_measure": self.ranked_numeric_measures,
            "numeric": self.ranked_numeric_measures,
            "measure": self.ranked_numeric_measures,
            "numeric_measures": self.ranked_numeric_measures,
            "categorical_dimension": self.ranked_categorical_dimensions,
            "categorical": self.ranked_categorical_dimensions,
            "dimension": self.ranked_categorical_dimensions,
            "categorical_dimensions": self.ranked_categorical_dimensions,
            "dimensions": self.ranked_categorical_dimensions,
            "temporal": self.ranked_temporal_columns,
            "time": self.ranked_temporal_columns,
            "temporal_columns": self.ranked_temporal_columns,
            "entity_id": self.ranked_entity_identifiers,
            "entity": self.ranked_entity_identifiers,
            "entity_identifiers": self.ranked_entity_identifiers,
            "target": self.ranked_target_candidates,
            "target_candidate": self.ranked_target_candidates,
            "target_candidates": self.ranked_target_candidates,
        }
        return role_map.get(role_normalized, [])

    def get_top_candidate(
        self,
        role: str,
        threshold: float = 0.85,
        ambiguity_margin: float = 0.15,
    ) -> Tuple[Optional[RoleCandidate], List[RoleCandidate]]:
        """
        Returns (top_candidate, all_candidates) if exactly one top candidate
        clears the high confidence threshold and ambiguity margin over any runner-up.
        If zero or multiple ambiguous candidates exist, returns (None, all_candidates).
        """
        candidates = self.get_ranked_candidates(role)
        if not candidates:
            return None, []
        top = candidates[0]
        if top.confidence < threshold:
            return None, candidates
        if len(candidates) > 1:
            second = candidates[1]
            if (top.confidence - second.confidence) < ambiguity_margin:
                # Ambiguous: runner-up is too close in confidence
                return None, candidates
        return top, candidates

    def to_compact_prompt_context(self, max_cols: int = 40) -> str:
        """
        Formats a compact, token-efficient summary (<300 tokens) 
        grounding the LLM planner with exact column names, roles, and stats.
        """
        lines = [
            f"Dataset: '{self.dataset_name}' ({self.row_count:,} rows, {self.column_count} columns | Hash: {self.dataset_hash[:8]})",
            "Columns & Semantic Roles:"
        ]
        for col_name, profile in list(self.columns.items())[:max_cols]:
            extra_info = []
            if profile.stats:
                s = profile.stats
                min_v = round(s.get("min", 0), 2)
                max_v = round(s.get("max", 0), 2)
                extra_info.append(f"range=[{min_v}..{max_v}]")
                if "iqr" in s:
                    extra_info.append(f"iqr={round(s['iqr'], 2)}")
            elif profile.sample_values:
                samples = ", ".join(repr(str(v)[:20]) for v in profile.sample_values[:3])
                extra_info.append(f"samples=[{samples}]")
            
            extra_str = f" ({'; '.join(extra_info)})" if extra_info else ""
            lines.append(f"- `{col_name}` [{profile.dtype} | {profile.semantic_role}]{extra_str}")

        if self.temporal_column_candidates:
            lines.append(f"Temporal Columns: {', '.join(self.temporal_column_candidates)}")
        if self.binary_flag_candidates:
            lines.append(f"Binary/Flag Columns: {', '.join(self.binary_flag_candidates)}")
        if self.sequence_column_candidates:
            lines.append(f"Ordered Sequence Candidates: {', '.join(self.sequence_column_candidates)}")
        if self.role_ambiguities:
            ambiguity_text = "; ".join(
                f"{name}=>{','.join(roles)}" for name, roles in self.role_ambiguities.items()
            )
            lines.append(f"Ambiguous Roles (confirm before use): {ambiguity_text}")

        return "\n".join(lines)


class DatasetProfiler:
    """Fast, vectorized, pure statistical dataset profiler for DataPilot OS."""

    _CACHE: Dict[str, DatasetProfile] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Clears the dataset profile cache."""
        cls._CACHE.clear()

    @classmethod
    def get_cached_profile(cls, dataset_hash: str) -> Optional[DatasetProfile]:
        """Retrieve a cached profile by dataset SHA-256 hash if present."""
        return cls._CACHE.get(dataset_hash)

    @classmethod
    def compute_hash(cls, df: pd.DataFrame, dataset_name: str = "dataset") -> str:
        """Computes a cryptographically strong SHA-256 hash over the entire dataset."""
        h = hashlib.sha256()
        h.update(dataset_name.encode("utf-8"))
        h.update(str(len(df)).encode("utf-8"))
        h.update(",".join(str(c) for c in df.columns).encode("utf-8"))
        h.update(",".join(str(dt) for dt in df.dtypes).encode("utf-8"))
        if not df.empty:
            row_hashes = pd.util.hash_pandas_object(df, index=False).values
            h.update(row_hashes.tobytes())
        return h.hexdigest()

    @classmethod
    def profile(
        cls,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        use_cache: bool = True,
    ) -> DatasetProfile:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(df)}")

        dataset_hash = cls.compute_hash(df, dataset_name)
        if use_cache and dataset_hash in cls._CACHE:
            return cls._CACHE[dataset_hash]

        row_count = len(df)
        col_count = len(df.columns)

        col_profiles: Dict[str, ColumnProfile] = {}
        temporal_candidates = []
        target_candidates = []
        entity_id_candidates = []
        binary_flag_candidates = []
        continuous_numeric_candidates = []
        categorical_candidates = []
        sequence_candidates = []
        datetime_parse_candidates = []
        role_ambiguities: Dict[str, List[str]] = {}

        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            null_count = int(series.isnull().sum())
            null_pct = round((null_count / max(row_count, 1)) * 100, 2)
            cardinality = int(series.nunique(dropna=True))
            is_unique = (cardinality == row_count) and (row_count > 0)
            non_null_count = int(series.notna().sum())
            unique_ratio = cardinality / max(non_null_count, 1)
            is_constant = cardinality <= 1
            is_monotonic_increasing = False
            is_monotonic_decreasing = False
            if pd.api.types.is_numeric_dtype(series) and not series.dropna().empty:
                try:
                    observed_values = series.dropna()
                    is_monotonic_increasing = bool(observed_values.is_monotonic_increasing)
                    is_monotonic_decreasing = bool(observed_values.is_monotonic_decreasing)
                except (TypeError, ValueError):
                    pass

            # Sample values
            valid_series = series.dropna()
            samples = valid_series.head(5).tolist() if not valid_series.empty else []

            # Infer semantic role & stats via mathematical & structural properties
            stats = None
            role = "categorical_nominal"
            role_evidence: List[SemanticRoleEvidence] = []
            datetime_parse_ratio = 0.0
            inferred_frequency = None

            if pd.api.types.is_numeric_dtype(series):
                if pd.api.types.is_bool_dtype(series) or cardinality == 2:
                    unique_vals = set(valid_series.unique().tolist())
                    if unique_vals.issubset({0, 1, True, False, 0.0, 1.0}):
                        role = "binary_flag"
                        binary_flag_candidates.append(str(col))
                        target_candidates.append(str(col))
                        role_evidence.append(SemanticRoleEvidence(
                            role="binary_flag", confidence=1.0,
                            evidence=["exactly two observed values in the boolean/0/1 value set"],
                        ))
                    else:
                        role = "discrete_numeric"
                        role_evidence.extend([
                            SemanticRoleEvidence(
                                role="discrete_numeric", confidence=0.8,
                                evidence=["numeric dtype with exactly two non-boolean values"],
                            ),
                            SemanticRoleEvidence(
                                role="categorical_nominal", confidence=0.7,
                                evidence=["two-level observed domain"],
                            ),
                        ])
                        binary_flag_candidates.append(str(col))
                        target_candidates.append(str(col))
                elif is_unique and pd.api.types.is_integer_dtype(series) and row_count >= 10:
                    role = "entity_id"
                    entity_id_candidates.append(str(col))
                    role_evidence.append(SemanticRoleEvidence(
                        role="entity_id", confidence=0.85,
                        evidence=["integer values are unique across all rows"],
                    ))
                    if is_monotonic_increasing or is_monotonic_decreasing:
                        sequence_candidates.append(str(col))
                        role_evidence.append(SemanticRoleEvidence(
                            role="ordered_sequence", confidence=0.95,
                            evidence=["values are unique and monotonic"],
                        ))
                elif pd.api.types.is_float_dtype(series) or cardinality > 20:
                    role = "continuous_numeric"
                    continuous_numeric_candidates.append(str(col))
                    role_evidence.append(SemanticRoleEvidence(
                        role="continuous_numeric", confidence=0.9,
                        evidence=["numeric dtype with sufficiently varied observed values"],
                    ))
                else:
                    role = "discrete_numeric"
                    continuous_numeric_candidates.append(str(col))
                    role_evidence.append(SemanticRoleEvidence(
                        role="discrete_numeric", confidence=0.85,
                        evidence=["numeric dtype with a bounded discrete observed domain"],
                    ))

                # Compute numeric stats
                if not valid_series.empty:
                    valid_num = valid_series.astype(float) if pd.api.types.is_bool_dtype(series) else valid_series
                    q25 = float(valid_num.quantile(0.25))
                    q50 = float(valid_num.quantile(0.50))
                    q75 = float(valid_num.quantile(0.75))
                    stats = {
                        "min": float(valid_num.min()),
                        "max": float(valid_num.max()),
                        "mean": float(valid_num.mean()),
                        "std": float(valid_num.std()) if len(valid_num) > 1 else 0.0,
                        "p25": q25,
                        "median": q50,
                        "p75": q75,
                        "iqr": float(q75 - q25),
                        "skew": float(valid_num.skew()) if len(valid_num) > 2 else 0.0,
                    }
            elif pd.api.types.is_datetime64_any_dtype(series):
                role = "temporal"
                temporal_candidates.append(str(col))
                datetime_parse_candidates.append(str(col))
                datetime_parse_ratio = 1.0
                role_evidence.append(SemanticRoleEvidence(
                    role="temporal", confidence=1.0,
                    evidence=["native pandas datetime dtype"],
                ))
                observed_times = pd.DatetimeIndex(valid_series.drop_duplicates().sort_values())
                if len(observed_times) >= 3:
                    try:
                        inferred_frequency = pd.infer_freq(observed_times)
                    except (TypeError, ValueError):
                        inferred_frequency = None
            else:
                # String / Object / Categorical
                is_dt = False
                if not valid_series.empty:
                    try:
                        sample_check = valid_series.iloc[: min(len(valid_series), 1000)]
                        parsed = pd.to_datetime(sample_check, errors="coerce", format="mixed")
                        datetime_parse_ratio = float(parsed.notnull().mean())
                        is_dt = datetime_parse_ratio >= 0.95
                    except (TypeError, ValueError, OverflowError):
                        is_dt = False

                if is_dt:
                    role = "temporal"
                    temporal_candidates.append(str(col))
                    datetime_parse_candidates.append(str(col))
                    role_evidence.append(SemanticRoleEvidence(
                        role="temporal", confidence=datetime_parse_ratio,
                        evidence=[f"{datetime_parse_ratio:.1%} of sampled non-null values parse as datetimes"],
                    ))
                    parsed_full = pd.to_datetime(valid_series, errors="coerce", format="mixed")
                    observed_times = pd.DatetimeIndex(parsed_full.dropna().drop_duplicates().sort_values())
                    if len(observed_times) >= 3:
                        try:
                            inferred_frequency = pd.infer_freq(observed_times)
                        except (TypeError, ValueError):
                            inferred_frequency = None
                elif is_unique and row_count >= 10:
                    role = "entity_id"
                    entity_id_candidates.append(str(col))
                    role_evidence.append(SemanticRoleEvidence(
                        role="entity_id", confidence=0.85,
                        evidence=["non-null values are unique across all rows"],
                    ))
                elif cardinality == 2:
                    unique_vals_lower = {str(v).lower() for v in valid_series.unique()}
                    if unique_vals_lower.issubset({"true", "false", "yes", "no", "0", "1", "y", "n"}):
                        role = "binary_flag"
                        binary_flag_candidates.append(str(col))
                        target_candidates.append(str(col))
                        role_evidence.append(SemanticRoleEvidence(
                            role="binary_flag", confidence=1.0,
                            evidence=["two observed values form a recognized boolean value set"],
                        ))
                    else:
                        role = "categorical_nominal"
                        categorical_candidates.append(str(col))
                        binary_flag_candidates.append(str(col))
                        target_candidates.append(str(col))
                        role_evidence.extend([
                            SemanticRoleEvidence(
                                role="categorical_nominal", confidence=0.95,
                                evidence=["object dtype with exactly two observed levels"],
                            ),
                            SemanticRoleEvidence(
                                role="binary_flag", confidence=0.7,
                                evidence=["exactly two observed levels; event meaning requires confirmation"],
                            ),
                        ])
                else:
                    if cardinality > 100 and (cardinality / max(row_count, 1)) > 0.6:
                        role = "text"
                        role_evidence.append(SemanticRoleEvidence(
                            role="text", confidence=0.8,
                            evidence=["high-cardinality non-numeric values"],
                        ))
                    else:
                        role = "categorical_nominal"
                        categorical_candidates.append(str(col))
                        role_evidence.append(SemanticRoleEvidence(
                            role="categorical_nominal", confidence=0.85,
                            evidence=["repeated non-numeric values with limited cardinality"],
                        ))

            if not role_evidence:
                role_evidence.append(SemanticRoleEvidence(
                    role=role, confidence=0.75, evidence=["physical dtype and observed cardinality"],
                ))
            plausible_roles = [item.role for item in role_evidence if item.confidence >= 0.65]
            if len(set(plausible_roles)) > 1:
                role_ambiguities[str(col)] = list(dict.fromkeys(plausible_roles))

            col_profiles[str(col)] = ColumnProfile(
                name=str(col),
                dtype=dtype_str,
                semantic_role=role,
                null_count=null_count,
                null_percentage=null_pct,
                cardinality=cardinality,
                is_unique=is_unique,
                unique_ratio=unique_ratio,
                is_constant=is_constant,
                is_monotonic_increasing=is_monotonic_increasing,
                is_monotonic_decreasing=is_monotonic_decreasing,
                sample_values=[v if isinstance(v, (int, float, bool, str)) else str(v) for v in samples],
                stats=stats,
                datetime_parse_ratio=datetime_parse_ratio,
                inferred_frequency=inferred_frequency,
                role_candidates=role_evidence,
            )

        # Build ranked role candidates purely from statistical and structural properties
        ranked_numeric: List[RoleCandidate] = []
        ranked_categorical: List[RoleCandidate] = []
        ranked_temporal: List[RoleCandidate] = []
        ranked_entity: List[RoleCandidate] = []
        ranked_target: List[RoleCandidate] = []

        for col_name, cp in col_profiles.items():
            col_null_rate = cp.null_percentage / 100.0

            # 1. Temporal candidate
            if cp.semantic_role == "temporal" or cp.datetime_parse_ratio >= 0.8:
                conf = max(cp.datetime_parse_ratio, 1.0 if cp.dtype.startswith("datetime") else 0.85)
                conf = max(0.0, min(1.0, conf - 0.2 * col_null_rate))
                ev = [f"dtype {cp.dtype}", f"datetime parse ratio {cp.datetime_parse_ratio:.2f}"]
                ranked_temporal.append(RoleCandidate(
                    column=col_name, role="temporal", confidence=round(conf, 3),
                    evidence=ev, null_rate=round(col_null_rate, 4), cardinality=cp.cardinality
                ))

            # 2. Entity Identifier
            if cp.is_unique and row_count >= 10:
                conf = 0.95 if ("int" in cp.dtype or "object" in cp.dtype) else 0.85
                conf = max(0.0, min(1.0, conf - 0.3 * col_null_rate))
                ev = [f"unique across all {row_count} rows", f"cardinality={cp.cardinality}"]
                ranked_entity.append(RoleCandidate(
                    column=col_name, role="entity_id", confidence=round(conf, 3),
                    evidence=ev, null_rate=round(col_null_rate, 4), cardinality=cp.cardinality
                ))

            # 3. Numeric Measure
            if cp.semantic_role in ("continuous_numeric", "discrete_numeric") and not cp.is_constant:
                is_sequential_id = (cp.is_unique and row_count >= 20 and (cp.is_monotonic_increasing or cp.is_monotonic_decreasing))
                if not is_sequential_id:
                    base_conf = 0.95 if cp.semantic_role == "continuous_numeric" else 0.80
                    if cp.cardinality > 50:
                        base_conf = min(0.98, base_conf + 0.03)
                    if cp.stats and cp.stats.get("std", 0.0) == 0.0:
                        base_conf = 0.0
                    conf = max(0.0, min(1.0, base_conf - 0.25 * col_null_rate))
                    ev = [f"numeric dtype {cp.dtype}", f"cardinality={cp.cardinality}", f"null_rate={col_null_rate:.1%}"]
                    if cp.stats:
                        ev.append(f"mean={cp.stats.get('mean', 0.0):.2f}, std={cp.stats.get('std', 0.0):.2f}")
                    ranked_numeric.append(RoleCandidate(
                        column=col_name, role="numeric_measure", confidence=round(conf, 3),
                        evidence=ev, null_rate=round(col_null_rate, 4), cardinality=cp.cardinality
                    ))

            # 4. Categorical Dimension
            if cp.semantic_role in ("categorical_nominal", "binary_flag") or (
                cp.semantic_role == "discrete_numeric" and 2 <= cp.cardinality <= 20 and not cp.is_unique
            ):
                if not cp.is_unique and not cp.is_constant:
                    base_conf = 0.95 if ("object" in cp.dtype or "category" in cp.dtype or "string" in cp.dtype) else 0.75
                    if cp.cardinality > 100:
                        base_conf = max(0.3, base_conf - 0.4)
                    conf = max(0.0, min(1.0, base_conf - 0.2 * col_null_rate))
                    ev = [f"dtype {cp.dtype}", f"cardinality={cp.cardinality} distinct classes", f"null_rate={col_null_rate:.1%}"]
                    ranked_categorical.append(RoleCandidate(
                        column=col_name, role="categorical_dimension", confidence=round(conf, 3),
                        evidence=ev, null_rate=round(col_null_rate, 4), cardinality=cp.cardinality
                    ))

            # 5. Target Candidate
            if cp.semantic_role == "binary_flag" or cp.cardinality == 2:
                conf = max(0.0, min(1.0, 0.95 - 0.2 * col_null_rate))
                ev = [f"binary flag with 2 distinct levels", f"null_rate={col_null_rate:.1%}"]
                ranked_target.append(RoleCandidate(
                    column=col_name, role="binary_target", confidence=round(conf, 3),
                    evidence=ev, null_rate=round(col_null_rate, 4), cardinality=cp.cardinality
                ))
            elif 3 <= cp.cardinality <= 10 and not cp.is_unique:
                conf = max(0.0, min(1.0, 0.85 - 0.2 * col_null_rate))
                ev = [f"multiclass discrete label with {cp.cardinality} classes", f"null_rate={col_null_rate:.1%}"]
                ranked_target.append(RoleCandidate(
                    column=col_name, role="multiclass_target", confidence=round(conf, 3),
                    evidence=ev, null_rate=round(col_null_rate, 4), cardinality=cp.cardinality
                ))
            elif cp.semantic_role == "continuous_numeric" and not cp.is_constant and col_null_rate < 0.2:
                conf = max(0.0, min(1.0, 0.70 - 0.3 * col_null_rate))
                ev = [f"continuous numeric metric for regression", f"null_rate={col_null_rate:.1%}"]
                ranked_target.append(RoleCandidate(
                    column=col_name, role="continuous_target", confidence=round(conf, 3),
                    evidence=ev, null_rate=round(col_null_rate, 4), cardinality=cp.cardinality
                ))

        ranked_numeric.sort(key=lambda c: (-c.confidence, c.null_rate, c.column))
        ranked_categorical.sort(key=lambda c: (-c.confidence, c.null_rate, c.column))
        ranked_temporal.sort(key=lambda c: (-c.confidence, c.null_rate, c.column))
        ranked_entity.sort(key=lambda c: (-c.confidence, c.null_rate, c.column))
        ranked_target.sort(key=lambda c: (-c.confidence, c.null_rate, c.column))

        profile_obj = DatasetProfile(
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            row_count=row_count,
            column_count=col_count,
            columns=col_profiles,
            temporal_column_candidates=list(dict.fromkeys(temporal_candidates)),
            target_column_candidates=list(dict.fromkeys(target_candidates)),
            entity_id_candidates=list(dict.fromkeys(entity_id_candidates)),
            binary_flag_candidates=list(dict.fromkeys(binary_flag_candidates)),
            continuous_numeric_candidates=list(dict.fromkeys(continuous_numeric_candidates)),
            categorical_candidates=list(dict.fromkeys(categorical_candidates)),
            sequence_column_candidates=list(dict.fromkeys(sequence_candidates)),
            datetime_parse_candidates=list(dict.fromkeys(datetime_parse_candidates)),
            role_ambiguities=role_ambiguities,
            duplicate_row_count=int(df.duplicated().sum()),
            total_missing_percentage=round(
                float(df.isna().sum().sum()) / max(row_count * col_count, 1) * 100,
                4,
            ),
            ranked_numeric_measures=ranked_numeric,
            ranked_categorical_dimensions=ranked_categorical,
            ranked_temporal_columns=ranked_temporal,
            ranked_entity_identifiers=ranked_entity,
            ranked_target_candidates=ranked_target,
        )

        if use_cache:
            cls._CACHE[dataset_hash] = profile_obj

        return profile_obj
