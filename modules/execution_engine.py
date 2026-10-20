"""
Modular Execution Engine Dispatcher for DataPilot OS.

This module provides a unified execution dispatcher across the 6 specialized sub-engines:
1. Data Engine (Ingestion, Cleaning, DuckDB OLAP)
2. Analytics Engine (KPIs, Auto-EDA, Cohorts)
3. ML Engine (Scikit-Learn AutoML, Benchmark Leaderboards)
4. Forecast Engine (Time-Series Predictions)
5. Visualization Engine (Dynamic Charts)
6. Report Engine (Executive Summaries & PDF/HTML Reports)

Also provides a secure subprocess execution sandbox for executing generated Python code
with strict timeouts, memory monitoring, and safe stdout/stderr/plot capturing.
"""

import sys
import os
import time
import json
import base64
import tempfile
import subprocess
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import specialized sub-engine modules
from modules.data_engineering import DataEngineeringEngine
from modules.auto_eda import AutoEDAEngine
from modules.auto_insights import AutoInsightsEngine
from modules.ml_engine import MultiModelMLEngine
from modules.forecasting import ForecastingEngine
from modules.visualization import EnterpriseVisualizationEngine
from modules.report_generator import ReportGeneratorEngine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


RUNNER_SCRIPT_TEMPLATE = """
import sys
import os
import io
import json
import base64
import traceback

# Force headless matplotlib backend BEFORE any matplotlib imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

temp_dir = sys.argv[1]
code_file_path = os.path.join(temp_dir, "user_code.py")
input_df_path = os.path.join(temp_dir, "input_df.parquet")
input_df_pkl = os.path.join(temp_dir, "input_df.pkl")
input_df_csv = os.path.join(temp_dir, "input_df.csv")
output_json_path = os.path.join(temp_dir, "output.json")
output_df_path = os.path.join(temp_dir, "output_df.parquet")

# Capture stdout / stderr
stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = stdout_buf
sys.stderr = stderr_buf

local_vars = {
    "pd": pd,
    "np": np,
    "plt": plt,
}

try:
    import seaborn as sns
    local_vars["sns"] = sns
except ImportError:
    pass

# Load input DataFrame if available
if os.path.exists(input_df_path):
    try:
        local_vars["df"] = pd.read_parquet(input_df_path)
    except Exception:
        pass
if "df" not in local_vars and os.path.exists(input_df_pkl):
    try:
        local_vars["df"] = pd.read_pickle(input_df_pkl)
    except Exception:
        pass
if "df" not in local_vars and os.path.exists(input_df_csv):
    try:
        local_vars["df"] = pd.read_csv(input_df_csv)
    except Exception:
        pass

if "df" not in local_vars:
    local_vars["df"] = pd.DataFrame()

success = True
error_msg = None
plot_images = []

def safe_show(*args, **kwargs):
    if plt.get_fignums():
        for fig_num in list(plt.get_fignums()):
            if len(plot_images) >= 10:
                break
            try:
                fig = plt.figure(fig_num)
                with io.BytesIO() as buf:
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plot_images.append(f"data:image/png;base64,{img_b64}")
            except Exception:
                pass
        try:
            plt.close('all')
        except Exception:
            pass

plt.show = safe_show

try:
    plt.close('all')
    with open(code_file_path, "r", encoding="utf-8") as f:
        user_code = f.read()

    compiled = compile(user_code, filename="<generated_code>", mode="exec")
    exec(compiled, local_vars)

    # Capture any remaining un-shown matplotlib figures safely into base64
    safe_show()

    # Save output DataFrame if present/modified (Prioritize result_df over input df)
    out_df = local_vars.get("result_df", local_vars.get("df", None))

    if isinstance(out_df, pd.DataFrame):
        try:
            out_df.to_parquet(output_df_path)
        except Exception:
            try:
                out_df.to_pickle(os.path.join(temp_dir, "output_df.pkl"))
            except Exception:
                pass

except Exception as e:
    import traceback
    success = False
    error_msg = f"{type(e).__name__}: {str(e)}\\n" + str(traceback.format_exc())

sys.stdout = old_stdout
sys.stderr = old_stderr

res = {
    "success": success,
    "stdout": stdout_buf.getvalue(),
    "stderr": stderr_buf.getvalue(),
    "error": error_msg,
    "plot_images": plot_images,
    "plot_image": plot_images[0] if plot_images else None,
    "has_output_df": os.path.exists(output_df_path) or os.path.exists(os.path.join(temp_dir, "output_df.pkl"))
}

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(res, f)
"""


class ModularExecutionEngine:
    """
    Unified Dispatcher orchestrating calls to specialized execution sub-engines
    and subprocess code sandbox.
    """

    def __init__(self) -> None:
        logger.info("Initializing ModularExecutionEngine")
        self.data_sub_engine = DataEngineeringEngine()
        self.eda_sub_engine = AutoEDAEngine()
        self.insights_sub_engine = AutoInsightsEngine()
        self.ml_sub_engine = MultiModelMLEngine()
        self.forecast_sub_engine = ForecastingEngine()
        self.viz_sub_engine = EnterpriseVisualizationEngine()
        self.report_sub_engine = ReportGeneratorEngine()

    def execute_code_subprocess(
        self,
        code: str,
        df: Optional[pd.DataFrame] = None,
        timeout: int = 30,
        memory_limit_mb: int = 1024
    ) -> Dict[str, Any]:
        """
        Executes Python code in an isolated subprocess (`sys.executable`) with strict
        timeouts (30s) and memory monitoring. Safely captures stdout, stderr, and matplotlib plots.

        Args:
            code (str): Python code string to execute.
            df (Optional[pd.DataFrame]): Input dataframe passed to code context as `df`.
            timeout (int): Strict execution timeout in seconds.
            memory_limit_mb (int): Memory limit in MB.

        Returns:
            Dict[str, Any]: Dictionary containing status, success, stdout, stderr, plot_image, dataframe, etc.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            user_code_path = os.path.join(temp_dir, "user_code.py")
            runner_script_path = os.path.join(temp_dir, "runner.py")
            output_json_path = os.path.join(temp_dir, "output.json")
            output_df_parquet = os.path.join(temp_dir, "output_df.parquet")
            output_df_pkl = os.path.join(temp_dir, "output_df.pkl")

            # Code execution setup




            # Write code to file
            with open(user_code_path, "w", encoding="utf-8") as f:
                f.write(code)


            # Write runner script
            with open(runner_script_path, "w", encoding="utf-8") as f:
                f.write(RUNNER_SCRIPT_TEMPLATE)

            # Save input DataFrame if provided
            if df is not None:
                try:
                    df.to_parquet(os.path.join(temp_dir, "input_df.parquet"))
                except Exception:
                    try:
                        df.to_pickle(os.path.join(temp_dir, "input_df.pkl"))
                    except Exception:
                        df.to_csv(os.path.join(temp_dir, "input_df.csv"), index=False)

            cmd = [sys.executable, runner_script_path, temp_dir]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            start_time = time.time()
            timed_out = False
            memory_exceeded = False

            while proc.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    timed_out = True
                    break

                if HAS_PSUTIL:
                    try:
                        p = psutil.Process(proc.pid)
                        mem_bytes = p.memory_info().rss
                        for child in p.children(recursive=True):
                            mem_bytes += child.memory_info().rss
                        if (mem_bytes / (1024 * 1024)) > memory_limit_mb:
                            memory_exceeded = True
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                time.sleep(0.05)

            if timed_out or memory_exceeded:
                if HAS_PSUTIL:
                    try:
                        p = psutil.Process(proc.pid)
                        for child in p.children(recursive=True):
                            child.kill()
                        p.kill()
                    except Exception:
                        proc.kill()
                else:
                    proc.kill()
                try:
                    proc.communicate(timeout=2)
                except Exception:
                    pass

                if timed_out:
                    logger.warning(f"Subprocess code execution timed out after {timeout} seconds")
                    return {
                        "status": "error",
                        "success": False,
                        "error": f"Execution timed out ({timeout} seconds limit exceeded).",
                        "stdout": "",
                        "stderr": f"TimeoutError: Subprocess killed after {timeout}s",
                        "plot_image": None,
                        "plot_images": [],
                        "dataframe": None
                    }
                if memory_exceeded:
                    logger.warning(f"Subprocess code execution exceeded memory limit of {memory_limit_mb}MB")
                    return {
                        "status": "error",
                        "success": False,
                        "error": f"Memory limit exceeded ({memory_limit_mb}MB limit).",
                        "stdout": "",
                        "stderr": f"MemoryError: Subprocess killed after exceeding {memory_limit_mb}MB RAM",
                        "plot_image": None,
                        "plot_images": [],
                        "dataframe": None
                    }

            stdout_rem, stderr_rem = proc.communicate()

            if os.path.exists(output_json_path):
                try:
                    with open(output_json_path, "r", encoding="utf-8") as f:
                        res_data = json.load(f)
                except Exception as e:
                    res_data = {
                        "success": False,
                        "stdout": stdout_rem or "",
                        "stderr": str(e),
                        "error": f"Failed to parse subprocess output JSON: {str(e)}",
                        "plot_image": None,
                        "plot_images": []
                    }
            else:
                res_data = {
                    "success": False,
                    "stdout": stdout_rem or "",
                    "stderr": stderr_rem or "",
                    "error": stderr_rem or "Subprocess exited without returning output.",
                    "plot_image": None,
                    "plot_images": []
                }

            out_df = None
            if os.path.exists(output_df_parquet):
                try:
                    out_df = pd.read_parquet(output_df_parquet)
                except Exception:
                    pass
            if out_df is None and os.path.exists(output_df_pkl):
                try:
                    out_df = pd.read_pickle(output_df_pkl)
                except Exception:
                    pass

            return {
                "status": "success" if res_data.get("success") else "error",
                "success": res_data.get("success", False),
                "stdout": res_data.get("stdout", ""),
                "stderr": res_data.get("stderr", ""),
                "error": res_data.get("error"),
                "plot_image": res_data.get("plot_image"),
                "plot_images": res_data.get("plot_images", []),
                "dataframe": out_df
            }

    def execute_code(
        self,
        code: str,
        df: Optional[pd.DataFrame] = None,
        timeout: int = 30,
        memory_limit_mb: int = 1024
    ) -> Dict[str, Any]:
        """Alias for execute_code_subprocess."""
        return self.execute_code_subprocess(code=code, df=df, timeout=timeout, memory_limit_mb=memory_limit_mb)

    def execute_task(self, sub_engine: str, action: str, df: pd.DataFrame, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dispatches execution to the specified sub-engine.

        Args:
            sub_engine (str): Name of sub-engine ('data', 'analytics', 'ml', 'forecast', 'viz', 'report', 'code').
            action (str): Specific operation to perform.
            df (pd.DataFrame): Input dataframe.
            kwargs (dict): Additional parameters for the action.

        Returns:
            Dict[str, Any]: Standardized execution response dictionary.
        """
        kwargs = kwargs or {}
        sub_engine = sub_engine.lower().strip()

        logger.info(f"ModularExecutionEngine dispatching action '{action}' to sub-engine '{sub_engine}'")

        try:
            if sub_engine in ("code", "terminal", "sandbox"):
                code = kwargs.get("code", "")
                return self.execute_code_subprocess(code, df=df)
            elif sub_engine == "data":
                return self._run_data_engine(action, df, kwargs)
            elif sub_engine in ("analytics", "eda"):
                return self._run_analytics_engine(action, df, kwargs)
            elif sub_engine in ("ml", "intelligence"):
                return self._run_ml_engine(action, df, kwargs)
            elif sub_engine == "forecast":
                return self._run_forecast_engine(action, df, kwargs)
            elif sub_engine in ("viz", "visualization"):
                return self._run_viz_engine(action, df, kwargs)
            elif sub_engine == "report":
                return self._run_report_engine(action, df, kwargs)
            else:
                raise ValueError(f"Unknown sub-engine '{sub_engine}' requested.")
        except Exception as e:
            logger.error(f"Execution error in sub-engine '{sub_engine}': {str(e)}", exc_info=True)
            return {"status": "error", "error": str(e), "sub_engine": sub_engine}

    def _run_data_engine(self, action: str, df: pd.DataFrame, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if action == "clean_missing":
            col = kwargs.get("column")
            res_df = df.dropna(subset=[col]) if col and col in df.columns else df.dropna()
            return {"status": "success", "rows_after": len(res_df), "message": "Missing values cleaned successfully."}
        elif action == "drop_column":
            col = kwargs.get("column")
            if col and col in df.columns:
                res_df = df.drop(columns=[col])
                return {"status": "success", "rows_after": len(res_df), "message": f"Column '{col}' dropped."}
            return {"status": "error", "error": f"Column '{col}' not found."}
        else:
            return {"status": "success", "rows": len(df), "message": "Data engine ready."}

    def _run_analytics_engine(self, action: str, df: pd.DataFrame, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        target_col = kwargs.get("target_col", df.columns[-1] if len(df.columns) > 0 else "")
        if action == "auto_eda":
            eda_results = self.eda_sub_engine.generate_eda_report(df)
            insights = self.insights_sub_engine.run_root_cause_analysis(df, target_col) if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]) else {"summary": "Root cause analysis skipped."}
            return {"status": "success", "eda": eda_results, "insights": insights}
        elif action == "kpi_summary":
            insights = self.insights_sub_engine.run_root_cause_analysis(df, target_col) if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]) else {"summary": "KPI summary generated."}
            return {"status": "success", "insights": insights}
        else:
            eda_results = self.eda_sub_engine.generate_eda_report(df)
            return {"status": "success", "eda": eda_results}

    def _run_ml_engine(self, action: str, df: pd.DataFrame, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        target_col = kwargs.get("target_col")
        if not target_col or target_col not in df.columns:
            return {"status": "error", "error": "Target column required for ML engine."}

        code_snippet = self.ml_sub_engine.generate_model_comparison_code(df, target_col)
        return {
            "status": "success",
            "message": f"ML model comparison pipeline generated for '{target_col}'.",
            "code": code_snippet,
            "target_col": target_col
        }

    def _run_forecast_engine(self, action: str, df: pd.DataFrame, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        time_col = kwargs.get("time_col", df.columns[0])
        target_col = kwargs.get("target_col", df.columns[-1])

        code_snippet = self.forecast_sub_engine.generate_arima_forecast_code("df", time_col, target_col)
        return {"status": "success", "code": code_snippet, "message": f"ARIMA time-series forecast code generated for '{target_col}'."}

    def _run_viz_engine(self, action: str, df: pd.DataFrame, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        title = kwargs.get("title", "Correlation Heatmap")
        code_snippet = self.viz_sub_engine.generate_heatmap("df", title=title)
        return {"status": "success", "code": code_snippet, "message": "Visualization chart code generated successfully."}

    def _run_report_engine(self, action: str, df: pd.DataFrame, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        title = kwargs.get("title", "Executive Data Analysis Report")
        report_markdown = self.report_sub_engine.generate_executive_summary_report(df, dataset_name=title)
        return {"status": "success", "report_markdown": report_markdown}


# Singleton instance
execution_engine = ModularExecutionEngine()
