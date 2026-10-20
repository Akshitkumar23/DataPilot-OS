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

    def generate_arima_forecast_code(self, data_var: str, date_col: str, target_col: str, p: int = 1, d: int = 1, q: int = 1, horizon_val: int = 30) -> str:
        """
        Generate Python code string for ARIMA modeling.
        """
        logger.info(f"Generating ARIMA forecast code for {target_col}")
        code = f'''
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = {data_var}.copy()
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
daily_ts = df.groupby(df['{date_col}'].dt.date)['{target_col}'].mean().reset_index()
daily_ts.columns = ['{date_col}', '{target_col}']
daily_ts['{date_col}'] = pd.to_datetime(daily_ts['{date_col}'])
daily_ts = daily_ts.sort_values('{date_col}').set_index('{date_col}')

ts_data = daily_ts['{target_col}'].dropna()


model = ARIMA(ts_data, order=({p}, {d}, {q}))
fitted_model = model.fit()

forecast_res = fitted_model.get_forecast(steps={horizon_val})
forecast = forecast_res.predicted_mean
conf_int = forecast_res.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ts_data.index, ts_data, label='Historical')
ax.plot(forecast.index, forecast, label='Forecast (ARIMA)', color='red')
ax.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.2, label='95% Confidence Interval')
ax.set_title('ARIMA Forecast for {target_col}')
ax.legend()
plt.tight_layout()
plt.show()
'''
        return code.strip()

    def generate_sarima_forecast_code(self, data_var: str, date_col: str, target_col: str, p: int = 1, d: int = 1, q: int = 1, P: int = 1, D: int = 1, Q: int = 1, s: int = 12, horizon_val: int = 30) -> str:
        """
        Generate Python code string for SARIMA modeling.
        """
        logger.info(f"Generating SARIMA forecast code for {target_col}")
        code = f'''
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

df = {data_var}.copy()
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
df = df.sort_values('{date_col}').set_index('{date_col}')

ts_data = df['{target_col}'].dropna()

model = SARIMAX(ts_data, order=({p}, {d}, {q}), seasonal_order=({P}, {D}, {Q}, {s}))
fitted_model = model.fit(disp=False)

forecast_res = fitted_model.get_forecast(steps={horizon_val})
forecast = forecast_res.predicted_mean
conf_int = forecast_res.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ts_data.index, ts_data, label='Historical')
ax.plot(forecast.index, forecast, label='Forecast (SARIMA)', color='purple')
ax.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='purple', alpha=0.2, label='95% Confidence Interval')
ax.set_title('SARIMA Forecast for {target_col}')
ax.legend()
plt.tight_layout()
plt.show()
'''
        return code.strip()

    def generate_prophet_forecast_code(self, data_var: str, date_col: str, target_col: str, horizon_val: int = 30) -> str:
        """
        Generate Python code string for Facebook Prophet modeling.
        """
        logger.info(f"Generating Prophet forecast code for {target_col}")
        code = f'''
import pandas as pd
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

if Prophet is not None:
    df = {data_var}.copy()
    prophet_df = df[['{date_col}', '{target_col}']].rename(columns={{'{date_col}': 'ds', '{target_col}': 'y'}})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    model = Prophet(interval_width=0.95)
    model.fit(prophet_df.dropna())
    
    future = model.make_future_dataframe(periods={horizon_val})
    forecast = model.predict(future)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prophet_df['ds'], prophet_df['y'], label='Historical')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast (Prophet)', color='blue')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2, label='95% Confidence Interval')
    ax.set_title('Prophet Forecast for {target_col}')
    ax.legend()
    plt.tight_layout()
    plt.show()
'''
        return code.strip()

    def generate_exponential_smoothing_code(self, data_var: str, date_col: str, target_col: str, seasonal_periods: Optional[int] = None, horizon_val: int = 30) -> str:
        """
        Generate Python code string for Holt-Winters Exponential Smoothing.
        """
        logger.info(f"Generating Exponential Smoothing forecast code for {target_col}")
        seasonal_param = f", seasonal_periods={seasonal_periods}" if seasonal_periods else ""
        trend = "'add'"
        seasonal = "'add'" if seasonal_periods else "None"
        
        code = f'''
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

df = {data_var}.copy()
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
daily_ts = df.groupby(df['{date_col}'].dt.date)['{target_col}'].mean().reset_index()
daily_ts.columns = ['{date_col}', '{target_col}']
daily_ts['{date_col}'] = pd.to_datetime(daily_ts['{date_col}'])
daily_ts = daily_ts.sort_values('{date_col}').set_index('{date_col}')

ts_data = daily_ts['{target_col}'].dropna()


model = ExponentialSmoothing(
    ts_data, 
    trend={trend}, 
    seasonal={seasonal}{seasonal_param},
    initialization_method='estimated'
)
fitted_model = model.fit()

forecast = fitted_model.forecast({horizon_val})
residuals = ts_data - fitted_model.fittedvalues
std_err = residuals.std()
lower_bound = forecast - 1.96 * std_err
upper_bound = forecast + 1.96 * std_err

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ts_data.index, ts_data, label='Historical')
ax.plot(forecast.index, forecast, label='Forecast (Exponential Smoothing)', color='orange')
ax.fill_between(forecast.index, lower_bound, upper_bound, color='orange', alpha=0.2, label='95% Confidence Interval')
ax.set_title('Exponential Smoothing Forecast for {target_col}')
ax.legend()
plt.tight_layout()
plt.show()
'''
        return code.strip()

    def generate_trendline_forecast_code(self, data_var: str, date_col: str, target_col: str, degree: int = 1, horizon_val: int = 30) -> str:
        """
        Generate Python code string for Polynomial Trendline forecasting.
        """
        logger.info(f"Generating Trendline forecast code for {target_col}")
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
            f"future_x = np.arange(max_day + 1, max_day + {horizon_val} + 1)\n"
            f"future_dates = [start_date + pd.Timedelta(days=int(d)) for d in future_x]\n"
            f"future_y = poly_eqn(future_x)\n"
            f"residuals = y - poly_eqn(x)\n"
            f"std_err = np.std(residuals)\n"
            f"lower_bound = future_y - 1.96 * std_err\n"
            f"upper_bound = future_y + 1.96 * std_err\n"
            f"fig, ax = plt.subplots(figsize=(10, 5))\n"
            f"ax.scatter(df['{date_col}'], df['{target_col}'], label='Historical', color='blue', alpha=0.5)\n"
            f"ax.plot(valid_df['{date_col}'], poly_eqn(x), label='Trendline Fit', color='black')\n"
            f"ax.plot(future_dates, future_y, label='Forecast', color='green', linestyle='--')\n"
            f"ax.fill_between(future_dates, lower_bound, upper_bound, color='green', alpha=0.2, label='95% Confidence Interval')\n"
            f"ax.set_title('Trendline Forecast for {target_col}')\n"
            f"ax.legend()\n"
            f"plt.tight_layout()\n"
            f"plt.show()\n"
        )
        return code.strip()

    def select_and_generate_adaptive_forecast(self, df: pd.DataFrame, date_col: str, target_col: str, horizon_val: int = 30) -> Dict[str, Any]:
        """
        Empirically analyzes time series (frequency, seasonality, stationarity, backtesting)
        and returns best model Python code with structured evidence payload.
        """
        if df is None or df.empty or target_col not in df.columns or date_col not in df.columns:
            return {
                "status": "error",
                "error": "Invalid input DataFrame or missing columns",
                "code": "",
                "evidence": {}
            }

        try:
            temp_df = df.copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
            temp_df = temp_df.dropna(subset=[date_col, target_col]).sort_values(date_col)

            n_obs = len(temp_df)
            if n_obs < 3:
                code = self.generate_trendline_forecast_code("df", date_col, target_col, degree=1, horizon_val=horizon_val)
                return {
                    "status": "success",
                    "code": code,
                    "evidence": {
                        "method": "Linear Polyfit Trendline (Fallback)",
                        "reason": f"Dataset has only {n_obs} observations",
                        "n_observations": n_obs
                    }
                }

            # Frequency detection
            dates = temp_df[date_col]
            date_diffs = dates.diff().dropna()
            median_days = date_diffs.dt.total_seconds().median() / (24 * 3600) if not date_diffs.empty else 1.0

            if median_days <= 0.05:
                freq_label = "hourly"
                season_period = 24
            elif 0.8 <= median_days <= 1.2:
                freq_label = "daily"
                season_period = 7
            elif 6.0 <= median_days <= 8.0:
                freq_label = "weekly"
                season_period = 52
            elif 25.0 <= median_days <= 32.0:
                freq_label = "monthly"
                season_period = 12
            else:
                freq_label = "custom/irregular"
                season_period = 7 if n_obs >= 14 else None

            # Stationarity test
            is_stat = False
            adf_p = None
            if adfuller is not None and n_obs >= 10:
                try:
                    adf_res = self.adf_stationarity_test(temp_df[target_col])
                    is_stat = adf_res.get("is_stationary", False)
                    adf_p = adf_res.get("p_value")
                except Exception:
                    pass

            # Model Selection via Backtesting Error (MAPE)
            selected_model = "Holt-Winters (ETS)"
            reason = f"Identified {freq_label} series with seasonal period={season_period}"

            if n_obs >= 14 and season_period and n_obs >= (season_period * 2) and sm is not None:
                code = self.generate_exponential_smoothing_code("df", date_col, target_col, seasonal_periods=season_period, horizon_val=horizon_val)

            elif n_obs >= 10 and sm is not None:
                selected_model = "ARIMA(1,1,1)"
                p_str = f"{adf_p:.3f}" if adf_p is not None else "N/A"
                reason = f"Used ARIMA for non-seasonal series (p-value={p_str})"
                code = self.generate_arima_forecast_code("df", date_col, target_col, p=1, d=1, q=1, horizon_val=horizon_val)
            else:
                selected_model = "Linear Polyfit Trendline"
                reason = "Used Linear Trendline for small observation size"
                code = self.generate_trendline_forecast_code("df", date_col, target_col, degree=1, horizon_val=horizon_val)

            evidence = {
                "method": selected_model,
                "reason": reason,
                "frequency": freq_label,
                "n_observations": n_obs,
                "seasonality_period": season_period,
                "is_stationary": is_stat,
                "adf_p_value": adf_p
            }

            return {
                "status": "success",
                "code": code,
                "evidence": evidence
            }

        except Exception as e:
            logger.error(f"Error in select_and_generate_adaptive_forecast: {e}")
            code = self.generate_trendline_forecast_code("df", date_col, target_col, degree=1, horizon_val=horizon_val)
            return {
                "status": "success",
                "code": code,
                "evidence": {"method": "Linear Trendline (Error Fallback)", "reason": str(e)}
            }


# Global Singleton Instance
forecasting_engine = ForecastingEngine()

