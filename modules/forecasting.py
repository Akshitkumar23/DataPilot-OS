import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

# Optional imports for diagnostics
try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    import statsmodels.api as sm
except ImportError:
    adfuller = None
    seasonal_decompose = None
    sm = None

logger = logging.getLogger(__name__)

class ForecastingEngine:
    """
    Forecasting Engine for DataPilot OS.
    Provides time-series diagnostic methods and forecasting code generators.
    """

    def __init__(self):
        """Initialize the ForecastingEngine."""
        pass

    def adf_stationarity_test(self, series: pd.Series) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Args:
            series (pd.Series): Time series data.

        Returns:
            Dict[str, Any]: ADF test results.
            
        Raises:
            ImportError: If statsmodels is not installed.
            ValueError: If series is empty or invalid.
        """
        if adfuller is None:
            raise ImportError("statsmodels is required for adf_stationarity_test.")
            
        if series is None or len(series) == 0:
            raise ValueError("Input series cannot be empty.")
            
        # Drop NAs
        series = series.dropna()
        if len(series) < 3:
            raise ValueError("Series is too short for ADF test.")
            
        try:
            result = adfuller(series)
            return {
                "test_statistic": result[0],
                "p_value": result[1],
                "used_lag": result[2],
                "n_observations": result[3],
                "critical_values": result[4],
                "is_stationary": result[1] < 0.05
            }
        except Exception as e:
            logger.error(f"Error in adf_stationarity_test: {e}")
            raise

    def calculate_lag_features(self, df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
        """
        Calculate lag features for a time series.

        Args:
            df (pd.DataFrame): DataFrame containing the time series.
            target_col (str): The name of the target column.
            lags (List[int]): List of lag intervals to calculate.

        Returns:
            pd.DataFrame: DataFrame with new lag feature columns.
            
        Raises:
            ValueError: If target_col is not in df or df is empty.
        """
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame.")
            
        if df.empty:
            raise ValueError("DataFrame cannot be empty.")
            
        df_out = df.copy()
        try:
            for lag in lags:
                col_name = f"{target_col}_lag_{lag}"
                df_out[col_name] = df_out[target_col].shift(lag)
            return df_out
        except Exception as e:
            logger.error(f"Error in calculate_lag_features: {e}")
            raise

    def calculate_rolling_windows(self, df: pd.DataFrame, target_col: str, windows: List[int], agg_funcs: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Calculate rolling window statistics.

        Args:
            df (pd.DataFrame): DataFrame containing the time series.
            target_col (str): The name of the target column.
            windows (List[int]): List of window sizes.
            agg_funcs (List[str]): List of aggregation functions to apply (e.g., 'mean', 'std').

        Returns:
            pd.DataFrame: DataFrame with new rolling window feature columns.
            
        Raises:
            ValueError: If target_col is not in df or df is empty.
        """
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame.")
            
        if df.empty:
            raise ValueError("DataFrame cannot be empty.")
            
        df_out = df.copy()
        try:
            for window in windows:
                for func in agg_funcs:
                    col_name = f"{target_col}_rolling_{window}_{func}"
                    if hasattr(df_out[target_col].rolling(window=window), func):
                        df_out[col_name] = getattr(df_out[target_col].rolling(window=window), func)()
                    else:
                        logger.warning(f"Aggregation function '{func}' not supported for rolling window.")
            return df_out
        except Exception as e:
            logger.error(f"Error in calculate_rolling_windows: {e}")
            raise

    def seasonal_decompose_analysis(self, series: pd.Series, model: str = 'additive', period: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Perform seasonal decomposition.

        Args:
            series (pd.Series): Time series data.
            model (str): Type of seasonal component ('additive' or 'multiplicative').
            period (Optional[int]): Period of the series.

        Returns:
            Dict[str, pd.Series]: Dictionary containing 'trend', 'seasonal', and 'resid' components.
            
        Raises:
            ImportError: If statsmodels is not installed.
            ValueError: If series is empty or invalid.
        """
        if seasonal_decompose is None:
            raise ImportError("statsmodels is required for seasonal_decompose_analysis.")
            
        if series is None or len(series) == 0:
            raise ValueError("Input series cannot be empty.")
            
        try:
            series = series.dropna()
            result = seasonal_decompose(series, model=model, period=period)
            return {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "resid": result.resid
            }
        except Exception as e:
            logger.error(f"Error in seasonal_decompose_analysis: {e}")
            raise

    def generate_arima_forecast_code(self, data_var: str, date_col: str, target_col: str, p: int = 1, d: int = 1, q: int = 1) -> str:
        """
        Generate Python code for ARIMA forecasting.

        Args:
            data_var (str): Name of the DataFrame variable in the generated code.
            date_col (str): Name of the date column.
            target_col (str): Name of the target variable column.
            p (int): AR order.
            d (int): Differencing degree.
            q (int): MA order.

        Returns:
            str: Generated Python code string.
        """
        code = f'''
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Ensure data is sorted by date and set as index
df = {data_var}.copy()
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
df = df.sort_values('{date_col}').set_index('{date_col}')

# Prepare time series
ts_data = df['{target_col}'].dropna()

# Fit ARIMA model
model = ARIMA(ts_data, order=({p}, {d}, {q}))
fitted_model = model.fit()

# Forecast the next 30 periods
forecast = fitted_model.forecast(steps=30)
print(fitted_model.summary())

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(ts_data.index, ts_data, label='Historical')
plt.plot(forecast.index, forecast, label='Forecast (ARIMA)', color='red')
plt.title('ARIMA Forecast for {target_col}')
plt.legend()
plt.show()
'''
        return code.strip()

    def generate_sarima_forecast_code(self, data_var: str, date_col: str, target_col: str, p: int = 1, d: int = 1, q: int = 1, P: int = 1, D: int = 1, Q: int = 1, s: int = 12) -> str:
        """
        Generate Python code for SARIMA forecasting.

        Args:
            data_var (str): Name of the DataFrame variable in the generated code.
            date_col (str): Name of the date column.
            target_col (str): Name of the target variable column.
            p (int): Non-seasonal AR order.
            d (int): Non-seasonal differencing degree.
            q (int): Non-seasonal MA order.
            P (int): Seasonal AR order.
            D (int): Seasonal differencing degree.
            Q (int): Seasonal MA order.
            s (int): Seasonal period.

        Returns:
            str: Generated Python code string.
        """
        code = f"""
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Ensure data is sorted by date and set as index
df = {data_var}.copy()
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
df = df.sort_values('{date_col}').set_index('{date_col}')

# Prepare time series
ts_data = df['{target_col}'].dropna()

# Fit SARIMA model
model = SARIMAX(ts_data, order=({p}, {d}, {q}), seasonal_order=({P}, {D}, {Q}, {s}))
fitted_model = model.fit(disp=False)

# Forecast the next 30 periods
forecast = fitted_model.forecast(steps=30)
print(fitted_model.summary())

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(ts_data.index, ts_data, label='Historical')
plt.plot(forecast.index, forecast, label='Forecast (SARIMA)', color='purple')
plt.title('SARIMA Forecast for {target_col}')
plt.legend()
plt.show()
"""
        return code.strip()


    def generate_prophet_forecast_code(self, data_var: str, date_col: str, target_col: str) -> str:
        """
        Generate Python code for Prophet forecasting.

        Args:
            data_var (str): Name of the DataFrame variable in the generated code.
            date_col (str): Name of the date column.
            target_col (str): Name of the target variable column.

        Returns:
            str: Generated Python code string.
        """
        code = f'''
import pandas as pd
try:
    from prophet import Prophet
except ImportError:
    print("Prophet is not installed. Please install it using 'pip install prophet'.")
    Prophet = None

if Prophet is not None:
    df = {data_var}.copy()
    
    # Prophet requires columns to be named 'ds' and 'y'
    prophet_df = df[['{date_col}', '{target_col}']].rename(columns={{'{date_col}': 'ds', '{target_col}': 'y'}})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Initialize and fit model
    model = Prophet()
    model.fit(prophet_df.dropna())
    
    # Make future dataframe for 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Plot forecast
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)
    
    import matplotlib.pyplot as plt
    plt.show()
'''
        return code.strip()

    def generate_exponential_smoothing_code(self, data_var: str, date_col: str, target_col: str, seasonal_periods: Optional[int] = None) -> str:
        """
        Generate Python code for Exponential Smoothing (Holt-Winters) forecasting.

        Args:
            data_var (str): Name of the DataFrame variable in the generated code.
            date_col (str): Name of the date column.
            target_col (str): Name of the target variable column.
            seasonal_periods (Optional[int]): Number of periods in a complete seasonal cycle.

        Returns:
            str: Generated Python code string.
        """
        seasonal_param = f", seasonal_periods={seasonal_periods}" if seasonal_periods else ""
        trend = "'add'"
        seasonal = "'add'" if seasonal_periods else "None"
        
        code = f"""
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

df = {data_var}.copy()
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
df = df.sort_values('{date_col}').set_index('{date_col}')

ts_data = df['{target_col}'].dropna()

# Fit Exponential Smoothing model
model = ExponentialSmoothing(
    ts_data, 
    trend={trend}, 
    seasonal={seasonal}{seasonal_param},
    initialization_method='estimated'
)
fitted_model = model.fit()

# Forecast next 30 periods
forecast = fitted_model.forecast(30)

plt.figure(figsize=(10, 5))
plt.plot(ts_data.index, ts_data, label='Historical')
plt.plot(forecast.index, forecast, label='Forecast (Exponential Smoothing)', color='orange')
plt.title('Exponential Smoothing Forecast for {target_col}')
plt.legend()
plt.show()
"""
        return code.strip()


    def generate_trendline_forecast_code(self, data_var: str, date_col: str, target_col: str, degree: int = 1) -> str:
        """Generate Python code for Polynomial/Linear trendline forecasting."""
        code = (
            f"import pandas as pd\n"

            f"import numpy as np\n"
            f"import matplotlib.pyplot as plt\n\n"
            f"df = {data_var}.copy()\n"
            f"df['{date_col}'] = pd.to_datetime(df['{date_col}'])\n"
            f"df = df.sort_values('{date_col}')\n"
            f"start_date = df['{date_col}'].min()\n"
            f"df['time_idx'] = (df['{date_col}'] - start_date).dt.days\n"
            f"valid_df = df.dropna(subset=['{target_col}'])\n"
            f"x = valid_df['time_idx'].values\n"
            f"y = valid_df['{target_col}'].values\n"
            f"coeffs = np.polyfit(x, y, {degree})\n"
            f"poly_eqn = np.poly1d(coeffs)\n"
            f"max_day = x.max()\n"
            f"future_x = np.arange(max_day + 1, max_day + 31)\n"
            f"future_dates = [start_date + pd.Timedelta(days=int(d)) for d in future_x]\n"
            f"future_y = poly_eqn(future_x)\n"
            f"plt.figure(figsize=(10, 5))\n"
            f"plt.scatter(df['{date_col}'], df['{target_col}'], label='Historical', color='blue', alpha=0.5)\n"
            f"plt.plot(valid_df['{date_col}'], poly_eqn(x), label='Trendline Fit', color='black')\n"
            f"plt.plot(future_dates, future_y, label='Forecast', color='green', linestyle='--')\n"
            f"plt.title('Trendline Forecast')\n"
            f"plt.legend()\n"
            f"plt.show()\n"
        )
        return code.strip()




# Global Singleton Instance
forecasting_engine = ForecastingEngine()
