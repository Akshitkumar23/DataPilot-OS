import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoInsightsEngine:
    """
    Engine for generating automated insights, root cause analysis,
    and what-if simulations on datasets.
    """

    def __init__(self) -> None:
        """Initializes the AutoInsightsEngine."""
        pass

    def run_root_cause_analysis(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Scans a numerical target variable against categorical dimensions
        to identify key drivers of drops or spikes.
        
        Args:
            df (pd.DataFrame): The input dataset.
            target_col (str): The numerical column to analyze.
            
        Returns:
            Dict[str, Any]: A dictionary containing insight text summary and visual driver plot code.
            
        Raises:
            ValueError: If the target column is missing or not numerical, or dataframe is empty.
        """
        if df.empty:
            logger.error("Empty dataframe provided for root cause analysis.")
            raise ValueError("Input dataframe cannot be empty.")
            
        if target_col not in df.columns:
            logger.error("Target column '%s' not found in dataframe.", target_col)
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")
            
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            logger.error("Target column '%s' must be numerical.", target_col)
            raise ValueError(f"Target column '{target_col}' must be numerical.")

        logger.info("Running root cause analysis for target column '%s'", target_col)
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        insights = []
        plot_data: Dict[str, Any] = {}
        
        # Analyze categorical dimensions to find high variance drivers
        for col in categorical_cols:
            if df[col].nunique() < 20:  # Limit to columns with manageable categories
                grouped = df.groupby(col, observed=True)[target_col].mean()
                if not grouped.empty:
                    overall_mean = float(df[target_col].mean())
                    max_cat = grouped.idxmax()
                    min_cat = grouped.idxmin()
                    max_val = float(grouped.max())
                    min_val = float(grouped.min())
                    
                    if pd.notna(max_cat) and pd.notna(min_cat):
                        insights.append(
                            f"Dimension '{col}': '{max_cat}' drives the highest {target_col} ({max_val:.2f}), "
                            f"while '{min_cat}' drives the lowest ({min_val:.2f})."
                        )
                        
                        plot_data[col] = {
                            "categories": grouped.index.tolist(),
                            "values": grouped.values.tolist(),
                            "overall_mean": overall_mean
                        }
                        
        summary = f"Root cause analysis on '{target_col}' completed. "
        if insights:
            summary += "Key findings: " + " | ".join(insights[:3])
        else:
            summary += "No significant categorical drivers found."
            
        plot_code = """
import plotly.express as px
import pandas as pd

def generate_driver_plot(plot_data, dimension):
    data = plot_data.get(dimension)
    if not data:
        return None
    df_plot = pd.DataFrame({'Category': data['categories'], 'Value': data['values']})
    fig = px.bar(df_plot, x='Category', y='Value', title=f"Driver Analysis: {dimension}")
    fig.add_hline(y=data['overall_mean'], line_dash="dash", annotation_text="Overall Mean")
    return fig
"""

        logger.info("Root cause analysis successfully completed.")
        return {
            "summary": summary,
            "plot_data": plot_data,
            "plot_code_snippet": plot_code.strip()
        }

    def run_whatif_simulation(self, df: pd.DataFrame, target_col: str, feature_changes: Dict[str, float]) -> Dict[str, float]:
        """
        Simulates the impact of changing feature values on the projected target metric.
        
        Args:
            df (pd.DataFrame): The input dataset.
            target_col (str): The target metric to project.
            feature_changes (Dict[str, float]): A dictionary mapping feature column names 
                                                to their percentage change (e.g., {'discount': -0.05}).
                                                
        Returns:
            Dict[str, float]: A dictionary containing the original and simulated target values.
            
        Raises:
            ValueError: If target_col or feature columns are missing or invalid, or dataframe is empty.
        """
        if df.empty:
            logger.error("Empty dataframe provided for what-if simulation.")
            raise ValueError("Input dataframe cannot be empty.")
            
        if target_col not in df.columns:
            logger.error("Target column '%s' not found in dataframe.", target_col)
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")
            
        for feature in feature_changes.keys():
            if feature not in df.columns:
                logger.error("Feature column '%s' not found in dataframe.", feature)
                raise ValueError(f"Feature column '{feature}' not found in dataframe.")
            if not pd.api.types.is_numeric_dtype(df[feature]):
                logger.error("Feature column '%s' must be numerical for simulation.", feature)
                raise ValueError(f"Feature column '{feature}' must be numerical for simulation.")
                
        logger.info("Running what-if simulation for '%s' with changes: %s", target_col, feature_changes)
        
        original_mean = float(df[target_col].mean())
        
        # Calculate impact based on linear correlation with the target column
        # to provide a data-driven simulation of feature changes.
        total_impact = 1.0
        for feature, pct_change in feature_changes.items():
            correlation = df[feature].corr(df[target_col])
            
            if pd.isna(correlation):
                correlation = 0.0
                
            total_impact += pct_change * correlation
            
        simulated_mean = original_mean * total_impact
        impact_percentage = (simulated_mean - original_mean) / original_mean if original_mean != 0 else 0.0
        
        logger.info("What-if simulation successfully completed.")
        return {
            "original_target_mean": original_mean,
            "simulated_target_mean": simulated_mean,
            "impact_percentage": float(impact_percentage)
        }

# Global Singleton Instance
auto_insights = AutoInsightsEngine()
