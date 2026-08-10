import logging
from typing import List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnterpriseVisualizationEngine:
    """
    Enterprise Visualization Engine for DataPilot OS.
    Generates high-resolution Seaborn/Matplotlib/Plotly Python code for various enterprise chart types.
    """

    def __init__(self, theme: str = 'light'):
        """
        Initialize the EnterpriseVisualizationEngine.

        Args:
            theme (str): Theme for the charts ('light' or 'dark').
        """
        self.theme = theme
        logger.info(f"Initialized EnterpriseVisualizationEngine with theme: {self.theme}")

    def _get_base_setup(self) -> str:
        """Helper to get base imports and setup code for matplotlib/seaborn."""
        style = "dark_background" if self.theme == 'dark' else "seaborn-v0_8-whitegrid"
        return f"""import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic parameters
plt.style.use('{style}')
sns.set_context("talk")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
"""

    def _get_plotly_setup(self) -> str:
        """Helper to get base imports and setup code for plotly."""
        template = "plotly_dark" if self.theme == 'dark' else "plotly_white"
        return f"""import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "{template}"
"""

    def recommend_chart_type(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        date_cols: List[str]
    ) -> str:
        """
        Smart geometry recommender to suggest the best chart type based on available data columns.

        Args:
            df (pd.DataFrame): The dataset.
            num_cols (List[str]): List of numerical columns.
            cat_cols (List[str]): List of categorical columns.
            date_cols (List[str]): List of datetime columns.

        Returns:
            str: Recommended chart type.
        """
        try:
            if date_cols and num_cols:
                if len(num_cols) >= 2:
                    return "line_chart"
                return "line_chart"
            elif cat_cols and num_cols:
                if len(cat_cols) >= 2:
                    return "grouped_bar_chart"
                return "bar_chart"
            elif len(num_cols) >= 2:
                if len(num_cols) > 2:
                    return "pairplot"
                return "scatterplot"
            elif len(num_cols) == 1:
                return "histogram"
            elif len(cat_cols) == 1:
                return "bar_chart"
            else:
                return "table"
        except Exception as e:
            logger.error(f"Error recommending chart type: {e}")
            return "bar_chart"

    # ================= Comparison Charts =================

    def generate_bar_chart(self, df_name: str, x_col: str, y_col: str, title: str = "Bar Chart") -> str:
        """Generates code for a vertical bar chart."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data={df_name}, x='{x_col}', y='{y_col}', ax=ax, palette='viridis')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating bar chart code: {e}")
            raise

    def generate_horizontal_bar(
        self, df_name: str, y_col: str, x_col: str, title: str = "Horizontal Bar Chart"
    ) -> str:
        """Generates code for a horizontal bar chart."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data={df_name}, x='{x_col}', y='{y_col}', ax=ax, palette='mako')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating horizontal bar chart code: {e}")
            raise

    def generate_grouped_bar(
        self, df_name: str, x_col: str, y_col: str, hue_col: str, title: str = "Grouped Bar Chart"
    ) -> str:
        """Generates code for a grouped bar chart."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data={df_name}, x='{x_col}', y='{y_col}', hue='{hue_col}', ax=ax, palette='Set2')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.legend(title='{hue_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating grouped bar chart code: {e}")
            raise

    def generate_stacked_bar(
        self, df_name: str, x_col: str, y_cols: List[str], title: str = "Stacked Bar Chart"
    ) -> str:
        """Generates code for a stacked bar chart using pandas plotting."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(10, 6))
{df_name}.set_index('{x_col}')[{y_cols}].plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('Values')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating stacked bar chart code: {e}")
            raise

    def generate_100pct_stacked_bar(
        self, df_name: str, x_col: str, y_cols: List[str], title: str = "100% Stacked Bar Chart"
    ) -> str:
        """Generates code for a 100% stacked bar chart."""
        try:
            code = self._get_base_setup()
            code += f"""
df_pct = {df_name}.set_index('{x_col}')[{y_cols}]
df_pct = df_pct.div(df_pct.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 6))
df_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Spectral')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating 100% stacked bar chart code: {e}")
            raise

    # ================= Trends Charts =================

    def generate_line_chart(self, df_name: str, x_col: str, y_col: str, title: str = "Line Chart") -> str:
        """Generates code for a line chart."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data={df_name}, x='{x_col}', y='{y_col}', ax=ax, marker='o', linewidth=2)
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating line chart code: {e}")
            raise

    def generate_area_chart(self, df_name: str, x_col: str, y_col: str, title: str = "Area Chart") -> str:
        """Generates code for an area chart."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(12, 6))
plt.fill_between({df_name}['{x_col}'], {df_name}['{y_col}'], color="skyblue", alpha=0.4)
plt.plot({df_name}['{x_col}'], {df_name}['{y_col}'], color="Slateblue", alpha=0.6, linewidth=2)
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating area chart code: {e}")
            raise

    def generate_step_chart(self, df_name: str, x_col: str, y_col: str, title: str = "Step Chart") -> str:
        """Generates code for a step chart."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(12, 6))
ax.step({df_name}['{x_col}'], {df_name}['{y_col}'], where='mid', linewidth=2, color='teal')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating step chart code: {e}")
            raise

    def generate_moving_average_chart(
        self, df_name: str, x_col: str, y_col: str, window: int = 7, title: str = "Moving Average Chart"
    ) -> str:
        """Generates code for a moving average line chart."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data={df_name}, x='{x_col}', y='{y_col}', ax=ax, alpha=0.3, label='Original')
# Calculate moving average
rolling_mean = {df_name}['{y_col}'].rolling(window={window}).mean()
ax.plot({df_name}['{x_col}'], rolling_mean, color='red', linewidth=2, label=f'{window}-period MA')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating moving average chart code: {e}")
            raise

    # ================= Distribution Charts =================

    def generate_histogram(self, df_name: str, col: str, bins: int = 30, title: str = "Histogram") -> str:
        """Generates code for a histogram."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data={df_name}, x='{col}', bins={bins}, kde=True, ax=ax, color='purple')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{col}')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating histogram code: {e}")
            raise

    def generate_kde_plot(self, df_name: str, col: str, title: str = "KDE Plot") -> str:
        """Generates code for a Kernel Density Estimation (KDE) plot."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data={df_name}, x='{col}', fill=True, ax=ax, color='crimson')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{col}')
ax.set_ylabel('Density')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating kde plot code: {e}")
            raise

    def generate_boxplot(self, df_name: str, x_col: str, y_col: str, title: str = "Boxplot") -> str:
        """Generates code for a boxplot."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data={df_name}, x='{x_col}', y='{y_col}', ax=ax, palette='Set3')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating boxplot code: {e}")
            raise

    def generate_violin_plot(self, df_name: str, x_col: str, y_col: str, title: str = "Violin Plot") -> str:
        """Generates code for a violin plot."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data={df_name}, x='{x_col}', y='{y_col}', ax=ax, palette='muted')
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating violin plot code: {e}")
            raise

    def generate_ecdf_plot(self, df_name: str, col: str, title: str = "ECDF Plot") -> str:
        """Generates code for an Empirical Cumulative Distribution Function plot."""
        try:
            code = self._get_base_setup()
            code += f"""
fig, ax = plt.subplots(figsize=(10, 6))
sns.ecdfplot(data={df_name}, x='{col}', ax=ax, linewidth=2)
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{col}')
ax.set_ylabel('Proportion')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating ecdf plot code: {e}")
            raise

    # ================= Relationships Charts =================

    def generate_scatterplot(
        self, df_name: str, x_col: str, y_col: str, hue_col: Optional[str] = None, title: str = "Scatterplot"
    ) -> str:
        """Generates code for a scatterplot."""
        try:
            code = self._get_base_setup()
            hue_arg = f", hue='{hue_col}'" if hue_col else ""
            code += f"""
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data={df_name}, x='{x_col}', y='{y_col}'{hue_arg}, ax=ax, alpha=0.7)
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating scatterplot code: {e}")
            raise

    def generate_bubble_chart(
        self, df_name: str, x_col: str, y_col: str, size_col: str, hue_col: Optional[str] = None, title: str = "Bubble Chart"
    ) -> str:
        """Generates code for a bubble chart."""
        try:
            code = self._get_base_setup()
            hue_arg = f", hue='{hue_col}'" if hue_col else ""
            code += f"""
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data={df_name}, x='{x_col}', y='{y_col}', size='{size_col}', sizes=(20, 500){hue_arg}, ax=ax, alpha=0.6)
ax.set_title('{title}', fontsize=16, pad=15)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating bubble chart code: {e}")
            raise

    def generate_pairplot(self, df_name: str, hue_col: Optional[str] = None, title: str = "Pairplot") -> str:
        """Generates code for a pairplot."""
        try:
            code = self._get_base_setup()
            hue_arg = f", hue='{hue_col}'" if hue_col else ""
            code += f"""
pair_fig = sns.pairplot(data={df_name}{hue_arg}, corner=True)
pair_fig.fig.suptitle('{title}', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating pairplot code: {e}")
            raise

    def generate_heatmap(self, df_name: str, title: str = "Correlation Heatmap") -> str:
        """Generates code for a correlation heatmap."""
        try:
            code = self._get_base_setup()
            code += f"""
# Select numerical columns for correlation
num_df = {df_name}.select_dtypes(include=[np.number])
corr = num_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, vmin=-1, vmax=1)
ax.set_title('{title}', fontsize=16, pad=15)
plt.tight_layout()
plt.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating heatmap code: {e}")
            raise

    # ================= Financial & Hierarchical Charts =================

    def generate_waterfall_chart(
        self, df_name: str, cat_col: str, val_col: str, title: str = "Waterfall Chart"
    ) -> str:
        """Generates code for a waterfall chart using Plotly."""
        try:
            code = self._get_plotly_setup()
            code += f"""
fig = go.Figure(go.Waterfall(
    name="20", orientation="v",
    measure=["relative"] * len({df_name}) + ["total"],
    x={df_name}['{cat_col}'].tolist() + ["Total"],
    textposition="outside",
    y={df_name}['{val_col}'].tolist() + [{df_name}['{val_col}'].sum()],
    connector={{"line": {{"color": "rgb(63, 63, 63)"}}}}
))
fig.update_layout(
    title='{title}', showlegend=False, width=800, height=500,
    margin=dict(t=50, l=25, r=25, b=25)
)
fig.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating waterfall chart code: {e}")
            raise

    def generate_candlestick_chart(
        self, df_name: str, date_col: str, open_col: str, high_col: str, low_col: str, close_col: str, title: str = "Candlestick Chart"
    ) -> str:
        """Generates code for a financial candlestick chart using Plotly."""
        try:
            code = self._get_plotly_setup()
            code += f"""
fig = go.Figure(data=[go.Candlestick(
    x={df_name}['{date_col}'],
    open={df_name}['{open_col}'],
    high={df_name}['{high_col}'],
    low={df_name}['{low_col}'],
    close={df_name}['{close_col}']
)])
fig.update_layout(
    title='{title}', xaxis_rangeslider_visible=False,
    margin=dict(t=50, l=25, r=25, b=25)
)
fig.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating candlestick chart code: {e}")
            raise

    def generate_treemap(self, df_name: str, path_cols: List[str], val_col: str, title: str = "Treemap") -> str:
        """Generates code for a treemap using Plotly."""
        try:
            code = self._get_plotly_setup()
            code += f"""
fig = px.treemap(
    {df_name}, path={path_cols}, values='{val_col}',
    title='{title}', color='{val_col}',
    color_continuous_scale='RdBu'
)
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
fig.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating treemap code: {e}")
            raise

    def generate_sunburst_chart(
        self, df_name: str, path_cols: List[str], val_col: str, title: str = "Sunburst Chart"
    ) -> str:
        """Generates code for a sunburst chart using Plotly."""
        try:
            code = self._get_plotly_setup()
            code += f"""
fig = px.sunburst(
    {df_name}, path={path_cols}, values='{val_col}',
    title='{title}', color='{val_col}',
    color_continuous_scale='Viridis'
)
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
fig.show()
"""
            return code
        except Exception as e:
            logger.error(f"Error generating sunburst chart code: {e}")
            raise

if __name__ == "__main__":
    logger.info("Visualization engine module loaded successfully.")

# Global Singleton Instance
visualization_engine = EnterpriseVisualizationEngine()
