import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
import textwrap

logger = logging.getLogger(__name__)

class AutoEDAEngine:
    """
    AutoEDAEngine provides automated exploratory data analysis features,
    including statistical summaries, multicollinearity checks, missing value
    computations, and executable code generation for dashboards.
    """
    
    @staticmethod
    def generate_eda_report(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates automated EDA metrics: correlation matrix, distribution skewness/kurtosis,
        null percentages, outlier counts, cardinality analysis, high collinearity flags (VIF > 5.0),
        and top feature importance previews (via simple variance or correlation).
        Also generates executable Python code string for full 1-click visual EDA dashboard.
        """
        try:
            report: Dict[str, Any] = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Null percentages
            null_pct = (df.isnull().sum() / len(df) * 100).to_dict()
            report['null_percentages'] = null_pct
            
            # Cardinality
            cardinality = {col: df[col].nunique() for col in categorical_cols}
            report['cardinality'] = cardinality
            
            # Skewness and Kurtosis
            if numeric_cols:
                skewness = df[numeric_cols].skew().to_dict()
                kurtosis = df[numeric_cols].kurtosis().to_dict()
                report['skewness'] = skewness
                report['kurtosis'] = kurtosis
                
                # Outliers count using IQR
                outliers = {}
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
                    outliers[col] = outlier_count
                report['outlier_counts'] = outliers
                
                # Correlation
                corr_matrix = df[numeric_cols].corr()
                report['correlation_matrix'] = corr_matrix.to_dict()
                
                # High collinearity flags (VIF > 5.0)
                high_collinearity = {}
                try:
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    # Drop rows with NaN for VIF calculation
                    df_vif = df[numeric_cols].dropna()
                    if not df_vif.empty and df_vif.shape[1] > 1:
                        # Adding a constant is typical for VIF, but we just use the data
                        vif_data = pd.DataFrame()
                        vif_data["feature"] = df_vif.columns
                        vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) 
                                           for i in range(len(df_vif.columns))]
                        
                        high_vif = vif_data[vif_data["VIF"] > 5.0]
                        high_collinearity = dict(zip(high_vif["feature"], high_vif["VIF"]))
                except ImportError:
                    logger.warning("statsmodels is not installed. Skipping VIF calculation.")
                    high_collinearity = {"error": "statsmodels not installed"}
                except Exception as e:
                    logger.warning(f"Could not calculate VIF: {e}")
                    high_collinearity = {"error": str(e)}
                
                report['high_collinearity'] = high_collinearity
                
            else:
                report['skewness'] = {}
                report['kurtosis'] = {}
                report['outlier_counts'] = {}
                report['correlation_matrix'] = {}
                report['high_collinearity'] = {}

            # Dashboard code generation
            dashboard_code = textwrap.dedent(f'''\
                import matplotlib.pyplot as plt
                import seaborn as sns
                import pandas as pd
                
                def run_eda_dashboard(df: pd.DataFrame):
                    """
                    1-Click Visual EDA Dashboard
                    """
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    
                    if not numeric_cols.empty:
                        # Correlation Matrix
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
                        plt.title("Correlation Matrix")
                        plt.tight_layout()
                        plt.show()
                        
                        # Distributions
                        for col in numeric_cols[:5]:  # Plot up to 5 numeric columns to avoid freezing
                            plt.figure(figsize=(8, 4))
                            sns.histplot(df[col].dropna(), kde=True)
                            plt.title(f"Distribution of {{col}}")
                            plt.tight_layout()
                            plt.show()
                            
                    if not categorical_cols.empty:
                        # Categorical counts
                        for col in categorical_cols[:3]: # Plot up to 3 categorical columns
                            plt.figure(figsize=(8, 4))
                            order = df[col].value_counts().index[:10]
                            sns.countplot(y=col, data=df, order=order)
                            plt.title(f"Top 10 Categories in {{col}}")
                            plt.tight_layout()
                            plt.show()
            ''')
            report['dashboard_code'] = dashboard_code
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating EDA report: {e}")
            raise


# Global Singleton Instance
auto_eda = AutoEDAEngine()
