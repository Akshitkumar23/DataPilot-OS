import hashlib
import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SchemaAnalyzer:
    """
    Extracts deep dataset schema metadata for LLM Semantic Planning:
    - Detects dtypes & semantic types (currency, department, age, years, id, category).
    - Extracts row/col counts, cardinality, min, max, and sample values.
    - Caches schema snapshot per dataset hash to avoid redundant computation.
    """
    _schema_cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def compute_df_hash(cls, df: pd.DataFrame) -> str:
        sample_str = ""
        if not df.empty:
            sample_head = df.head(3).to_string()
            sample_tail = df.tail(3).to_string()
            sample_str = f"_{sample_head}_{sample_tail}"
        shape_str = f"{df.shape}_{list(df.columns)}{sample_str}"
        return hashlib.md5(shape_str.encode('utf-8')).hexdigest()

    @classmethod
    def analyze(cls, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"rows": 0, "columns": {}}

        df_hash = cls.compute_df_hash(df)
        if df_hash in cls._schema_cache:
            logger.info("SCHEMA_CACHE_HIT | Returned cached schema snapshot.")
            return cls._schema_cache[df_hash]

        logger.info(f"SCHEMA_ANALYZING | Computing schema for DataFrame shape {df.shape}")
        col_meta = {}
        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            n_uniq = int(s.nunique())
            
            # Infer semantic type
            col_lower = str(col).lower()
            if any(id_kw in col_lower for id_kw in ['_id', 'id', 'index', 'number', 'code']):
                sem_type = "id"
            elif any(curr_kw in col_lower for curr_kw in ['salary', 'price', 'cost', 'revenue', 'profit', 'sales', 'amount', 'fee', 'charge']):
                sem_type = "currency"
            elif any(age_kw in col_lower for age_kw in ['age', 'year_of_birth']):
                sem_type = "age"
            elif any(yr_kw in col_lower for yr_kw in ['experience', 'tenure', 'years']):
                sem_type = "years"
            elif any(dt_kw in col_lower for dt_kw in ['date', 'time', 'timestamp']):
                sem_type = "datetime"
            elif pd.api.types.is_numeric_dtype(s):
                sem_type = "numeric"
            elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                sem_type = "category"
            else:
                sem_type = "general"

            info: Dict[str, Any] = {
                "dtype": dtype,
                "semantic_type": sem_type,
                "cardinality": n_uniq,
                "null_count": int(s.isna().sum())
            }

            if pd.api.types.is_numeric_dtype(s) and not s.dropna().empty:
                info["min"] = float(s.min())
                info["max"] = float(s.max())
                info["mean"] = round(float(s.mean()), 2)
            
            samples = s.dropna().unique()[:3].tolist()
            info["sample_values"] = [str(x) for x in samples]
            col_meta[str(col)] = info

        schema_snapshot = {
            "rows": len(df),
            "col_count": len(df.columns),
            "columns": col_meta
        }
        cls._schema_cache[df_hash] = schema_snapshot
        return schema_snapshot
