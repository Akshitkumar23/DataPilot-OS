import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModelMLEngine:
    """
    Engine to generate Python code for training and comparing multiple machine learning models,
    as well as handling class imbalance.
    """

    def generate_model_comparison_code(self, df: pd.DataFrame, target_col: str) -> str:
        """
        Generates Python code to train, compare, and benchmark 5+ models and output a comparative Leaderboard Matrix dataframe.
        
        Args:
            df (pd.DataFrame): The dataframe containing the dataset.
            target_col (str): The name of the target column.
            
        Returns:
            str: The generated Python code.
        """
        try:
            logger.info(f"Generating model comparison code for target column: {target_col}")
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataframe.")
                
            code = f"""
import logging
import time
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_model_comparison(df: pd.DataFrame, target_col: str = '{target_col}') -> pd.DataFrame:
    \"\"\"
    Train and compare multiple models, returning a leaderboard DataFrame.
    \"\"\"
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {{
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }}
        
        results = []
        
        for name, model in models.items():
            logger.info(f"Training {{name}}...")
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            
            results.append({{
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'Train Time (s)': train_time
            }})
            
        leaderboard_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        return leaderboard_df
    except Exception as e:
        logger.error(f"Error in model comparison: {{str(e)}}")
        raise
"""
            return code
        except Exception as e:
            logger.error(f"Error generating model comparison code: {str(e)}")
            raise

    def generate_imbalance_handling_code(self, df: pd.DataFrame, target_col: str, strategy: str) -> str:
        """
        Generates Python code for handling dataset imbalance using SMOTE, Random Oversampling, or Undersampling.
        
        Args:
            df (pd.DataFrame): The dataframe containing the dataset.
            target_col (str): The name of the target column.
            strategy (str): The strategy to use ('smote', 'oversample', 'undersample').
            
        Returns:
            str: The generated Python code.
        """
        try:
            logger.info(f"Generating imbalance handling code with strategy: {strategy} for target column: {target_col}")
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataframe.")
            
            valid_strategies = ['smote', 'oversample', 'undersample']
            if strategy.lower() not in valid_strategies:
                 raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
                 
            code = f"""
import logging
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_imbalance(df: pd.DataFrame, target_col: str = '{target_col}', strategy: str = '{strategy.lower()}') -> pd.DataFrame:
    \"\"\"
    Handle class imbalance using the specified strategy.
    \"\"\"
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Applying {{strategy}} to handle class imbalance...")
        if strategy == 'smote':
            sampler = SMOTE(random_state=42)
        elif strategy == 'oversample':
            sampler = RandomOverSampler(random_state=42)
        elif strategy == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            raise ValueError("Invalid strategy. Must be 'smote', 'oversample', or 'undersample'.")
            
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_col)], axis=1)
        return df_resampled
    except Exception as e:
        logger.error(f"Error handling imbalance: {{str(e)}}")
        raise
"""
            return code
        except Exception as e:
            logger.error(f"Error generating imbalance handling code: {str(e)}")
            raise

    def detect_data_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> dict:
        """
        Detects Data Drift between reference baseline dataset and current incoming dataset using Kolmogorov-Smirnov test.
        """
        from scipy.stats import ks_2samp
        from typing import Dict, Any
        drift_results = {}
        drifted_count = 0
        common_cols = [c for c in reference_df.columns if c in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[c])]

        for col in common_cols:
            stat, p_value = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
            is_drifted = bool(p_value < threshold)
            if is_drifted:
                drifted_count += 1
            drift_results[str(col)] = {
                "ks_stat": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": is_drifted
            }

        return {
            "total_numeric_features": len(common_cols),
            "features_drifted_count": drifted_count,
            "drift_status": "DATA DRIFT DETECTED" if drifted_count > 0 else "STABLE NO DRIFT",
            "feature_drift_details": drift_results
        }


# Global Singleton Instance
ml_engine = MultiModelMLEngine()
