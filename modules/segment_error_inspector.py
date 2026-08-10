import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SegmentErrorInspectorEngine:
    def evaluate_baseline_vs_ml(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        y = df[target_col].dropna()
        b_mae = float(np.mean(np.abs(y - y.mean())))
        return {'target': target_col, 'baseline_mae': b_mae}

    def inspect_segment_errors(self, df: pd.DataFrame, target_col: str, pred_col: str, cat_col: str) -> Dict[str, Any]:
        temp_df = df[[target_col, pred_col, cat_col]].dropna().copy()
        temp_df['residual'] = np.abs(temp_df[target_col] - temp_df[pred_col])
        res = temp_df.groupby(cat_col)['residual'].mean().round(4).to_dict()
        return {'segment_errors': res}

segment_error_inspector = SegmentErrorInspectorEngine()
