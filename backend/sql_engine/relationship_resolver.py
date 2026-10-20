"""
DataPilot OS — Phase 7 / FW-04 Relationship Resolution Engine.
Implements the Three-Tier Relationship Evidence Model:
  Tier 1: Explicit User Declaration (orders.customer_id = customers.customer_id)
  Tier 2: Database-Declared Metadata
  Tier 3: Inferred Candidate (Evidence Contract: Uniqueness >= 0.999, Inclusion >= 0.70)
Computes strict relational cardinality: 1:1, 1:m, m:1, m:m.
Detects fan-out and multiplicative row explosion risk before execution.
Zero domain or dataset hardcoding.
"""

from typing import Dict, List, Optional, Tuple, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np


class RelationshipEvidence(BaseModel):
    """Structured report on relational integrity between two candidate key sets."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    parent_table: str
    parent_key: str
    child_table: str
    child_key: str
    evidence_tier: Literal["explicit", "database_declared", "inferred_candidate"]
    
    parent_rows: int
    child_rows: int
    parent_null_keys: int
    child_null_keys: int
    parent_unique_keys: int
    child_unique_keys: int
    matched_child_keys: int
    orphan_child_keys: int

    parent_uniqueness_ratio: float
    child_uniqueness_ratio: float
    inclusion_ratio: float  # matched_child_keys / child_unique_keys

    cardinality: Literal["1:1", "1:m", "m:1", "m:m"]
    is_valid_relationship: bool
    requires_user_confirmation: bool
    status: str
    warnings: List[str] = Field(default_factory=list)


class RelationshipResolver:
    """
    Authoritative deterministic engine to evaluate relational integrity,
    candidate foreign keys, and join cardinality across tables.
    """

    # Generic column names that must never be inferred as foreign keys on their own
    GENERIC_NAMES = {
        "status", "code", "type", "date", "created_at", "updated_at",
        "name", "value", "description", "category", "id", "year", "month",
        "day", "comment", "flag", "time", "count"
    }

    @classmethod
    def are_types_compatible(cls, s1: pd.Series, s2: pd.Series) -> bool:
        """Determines whether two Series can be joined without lossy type coercions."""
        d1 = s1.dtype
        d2 = s2.dtype
        if d1 == d2:
            return True
        if pd.api.types.is_numeric_dtype(d1) and pd.api.types.is_numeric_dtype(d2):
            return True
        if pd.api.types.is_string_dtype(d1) and pd.api.types.is_string_dtype(d2):
            return True
        return False

    @classmethod
    def evaluate_relationship(
        cls,
        parent_df: pd.DataFrame,
        parent_table: str,
        parent_key: str,
        child_df: pd.DataFrame,
        child_table: str,
        child_key: str,
        evidence_tier: Literal["explicit", "database_declared", "inferred_candidate"] = "explicit"
    ) -> RelationshipEvidence:
        """
        Evaluates relational evidence and cardinality between candidate key columns.
        """
        if parent_key not in parent_df.columns:
            raise ValueError(f"Parent key '{parent_key}' not found in table '{parent_table}'")
        if child_key not in child_df.columns:
            raise ValueError(f"Child key '{child_key}' not found in table '{child_table}'")

        p_s = parent_df[parent_key]
        c_s = child_df[child_key]

        # Verify type compatibility
        if not cls.are_types_compatible(p_s, c_s):
            return RelationshipEvidence(
                parent_table=parent_table,
                parent_key=parent_key,
                child_table=child_table,
                child_key=child_key,
                evidence_tier=evidence_tier,
                parent_rows=len(parent_df),
                child_rows=len(child_df),
                parent_null_keys=int(p_s.isna().sum()),
                child_null_keys=int(c_s.isna().sum()),
                parent_unique_keys=int(p_s.dropna().nunique()),
                child_unique_keys=int(c_s.dropna().nunique()),
                matched_child_keys=0,
                orphan_child_keys=int(c_s.dropna().nunique()),
                parent_uniqueness_ratio=0.0,
                child_uniqueness_ratio=0.0,
                inclusion_ratio=0.0,
                cardinality="m:m",
                is_valid_relationship=False,
                requires_user_confirmation=True,
                status="TYPE_MISMATCH",
                warnings=[f"Incompatible types: {p_s.dtype} in '{parent_table}' and {c_s.dtype} in '{child_table}'"]
            )

        p_clean = p_s.dropna()
        c_clean = c_s.dropna()

        p_rows = len(parent_df)
        c_rows = len(child_df)
        p_nulls = int(p_s.isna().sum())
        c_nulls = int(c_s.isna().sum())

        p_uniq_count = int(p_clean.nunique())
        c_uniq_count = int(c_clean.nunique())

        p_uniqueness = (p_uniq_count / len(p_clean)) if len(p_clean) > 0 else 0.0
        c_uniqueness = (c_uniq_count / len(c_clean)) if len(c_clean) > 0 else 0.0

        # Compute inclusion / overlap
        p_set = set(p_clean)
        c_set = set(c_clean)
        matched_keys = len(c_set.intersection(p_set))
        orphan_keys = len(c_set - p_set)
        inclusion_ratio = (matched_keys / len(c_set)) if len(c_set) > 0 else 0.0

        # Determine Cardinality from key uniqueness
        # 1:1 -> unique in left & unique in right
        # 1:m -> unique in left & duplicate in right
        # m:1 -> duplicate in left & unique in right
        # m:m -> duplicate in left & duplicate in right
        p_is_unique = p_uniqueness >= 0.999 and p_nulls == 0
        c_is_unique = c_uniqueness >= 0.999 and c_nulls == 0

        if p_is_unique and c_is_unique:
            cardinality = "1:1"
        elif p_is_unique and not c_is_unique:
            cardinality = "1:m"
        elif not p_is_unique and c_is_unique:
            cardinality = "m:1"
        else:
            cardinality = "m:m"

        warnings = []
        if p_nulls > 0:
            warnings.append(f"Parent key '{parent_table}.{parent_key}' has {p_nulls} NULL values; primary keys should be NOT NULL.")
        if orphan_keys > 0:
            warnings.append(f"{orphan_keys} distinct keys in '{child_table}.{child_key}' ({1.0 - inclusion_ratio:.1%}) do not exist in '{parent_table}'.")
        if cardinality == "m:m":
            warnings.append("Many-to-Many cardinality detected. Risk of multiplicative fan-out row explosion.")

        # Validation rules based on tier
        if evidence_tier == "explicit":
            # Explicit declaration: user decided, but we warn
            is_valid = True
            req_confirm = False
            status = "CONFIRMED"
        elif evidence_tier == "database_declared":
            is_valid = True
            req_confirm = False
            status = "CONFIRMED"
        else:
            # Inferred Candidate: Must pass statistical evidence contract
            # 1. Parent side must be unique (PK candidate)
            # 2. Inclusion ratio must be >= 0.70
            # 3. Not a disallowed generic column
            if parent_key.lower() in cls.GENERIC_NAMES and child_key.lower() in cls.GENERIC_NAMES:
                is_valid = False
                req_confirm = True
                status = "REJECTED_GENERIC_NAME"
                warnings.append(f"Column '{parent_key}' is a generic column name; automatic foreign key inference is blocked.")
            elif not p_is_unique:
                is_valid = False
                req_confirm = True
                status = "REJECTED_NON_UNIQUE_PARENT"
                warnings.append(f"Parent key '{parent_table}.{parent_key}' is not unique ({p_uniqueness:.1%}); cannot serve as candidate primary key.")
            elif inclusion_ratio < 0.70:
                is_valid = False
                req_confirm = True
                status = "REJECTED_LOW_INCLUSION"
                warnings.append(f"Child key inclusion ratio ({inclusion_ratio:.1%}) is below candidate threshold (70%).")
            else:
                is_valid = True
                req_confirm = True  # Always requires analyst confirmation before auto-join
                status = "CANDIDATE_RELATIONSHIP"

        return RelationshipEvidence(
            parent_table=parent_table,
            parent_key=parent_key,
            child_table=child_table,
            child_key=child_key,
            evidence_tier=evidence_tier,
            parent_rows=p_rows,
            child_rows=c_rows,
            parent_null_keys=p_nulls,
            child_null_keys=c_nulls,
            parent_unique_keys=p_uniq_count,
            child_unique_keys=c_uniq_count,
            matched_child_keys=matched_keys,
            orphan_child_keys=orphan_keys,
            parent_uniqueness_ratio=p_uniqueness,
            child_uniqueness_ratio=c_uniqueness,
            inclusion_ratio=inclusion_ratio,
            cardinality=cardinality,
            is_valid_relationship=is_valid,
            requires_user_confirmation=req_confirm,
            status=status,
            warnings=warnings
        )

    @classmethod
    def discover_candidate_relationships(
        cls,
        tables_map: Dict[str, pd.DataFrame]
    ) -> List[RelationshipEvidence]:
        """
        Inspects all pairs of tables in the catalog and discovers statistically sound candidate FKs.
        Never infers across generic columns or without high uniqueness and overlap.
        """
        candidates: List[RelationshipEvidence] = []
        table_names = list(tables_map.keys())

        for i in range(len(table_names)):
            for j in range(len(table_names)):
                if i == j:
                    continue
                t1 = table_names[i]
                t2 = table_names[j]
                df1 = tables_map[t1]
                df2 = tables_map[t2]

                # Find candidate key pairs:
                # Case 1: Exact column name match
                common_cols = set(df1.columns).intersection(set(df2.columns))
                for col in common_cols:
                    if col.lower() in cls.GENERIC_NAMES:
                        continue
                    # Check if df1 can be parent and df2 child
                    ev = cls.evaluate_relationship(
                        parent_df=df1, parent_table=t1, parent_key=col,
                        child_df=df2, child_table=t2, child_key=col,
                        evidence_tier="inferred_candidate"
                    )
                    if ev.is_valid_relationship:
                        candidates.append(ev)

                # Case 2: Canonical _id suffix match (e.g. df1 has 'id', df2 has 'user_id' where t1 is 'user')
                for c1 in df1.columns:
                    for c2 in df2.columns:
                        if c1 == c2:
                            continue
                        if c1.lower() == "id" and c2.lower() == f"{t1.rstrip('s').lower()}_id":
                            ev = cls.evaluate_relationship(
                                parent_df=df1, parent_table=t1, parent_key=c1,
                                child_df=df2, child_table=t2, child_key=c2,
                                evidence_tier="inferred_candidate"
                            )
                            if ev.is_valid_relationship:
                                candidates.append(ev)

        return candidates
