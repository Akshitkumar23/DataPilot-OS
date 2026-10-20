"""
DataPilot OS — Isolated Python Execution Sandbox.
Runs generated code in an isolated subprocess with:
- Configurable timeout (default 60s)
- Memory monitoring via psutil
- Working-directory isolation
- Stdout/stderr capture
- Plot artifact generation (base64 PNG)
- Process cleanup on timeout/error

Execution States: QUEUED → VALIDATING → RUNNING → SUCCEEDED/FAILED/TIMED_OUT/CANCELLED
"""

import os
import sys
import json
import uuid
import time
import subprocess
import tempfile
import logging
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import pandas as pd

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

class ExecutionStatus(str, Enum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"

class SandboxConfig(BaseModel):
    timeout_sec: float = 60.0
    max_memory_mb: int = 1024
    working_dir: Optional[str] = None
    allowed_imports: List[str] = Field(default_factory=list)
    capture_plots: bool = True
    max_plot_count: int = 5

class SandboxResult(BaseModel):
    execution_id: str
    status: ExecutionStatus
    stdout: str
    stderr: str
    exit_code: Optional[int] = None
    execution_time_sec: float
    plots: List[str] = Field(default_factory=list)
    output_tables: List[Dict] = Field(default_factory=list)
    generated_files: List[str] = Field(default_factory=list)
    memory_peak_mb: Optional[float] = None
    error_message: Optional[str] = None

RUNNER_SCRIPT_TEMPLATE = """
import sys
import os
import io
import json
import base64
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

temp_dir = sys.argv[1]
code_file_path = os.path.join(temp_dir, "user_code.py")
input_df_path = os.path.join(temp_dir, "input_df.parquet")
output_json_path = os.path.join(temp_dir, "output.json")
output_df_path = os.path.join(temp_dir, "output_df.json")

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

if os.path.exists(input_df_path):
    try:
        local_vars["df"] = pd.read_parquet(input_df_path)
    except Exception:
        local_vars["df"] = pd.DataFrame()
else:
    local_vars["df"] = pd.DataFrame()

success = True
error_msg = None
plot_images = []
output_tables = []

def safe_show(*args, **kwargs):
    if plt.get_fignums():
        for fig_num in list(plt.get_fignums()):
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plot_images.append(f"data:image/png;base64,{img_b64}")
        plt.close('all')

plt.show = safe_show

try:
    plt.close('all')
    with open(code_file_path, "r", encoding="utf-8") as f:
        user_code = f.read()

    compiled = compile(user_code, filename="<generated_code>", mode="exec")
    exec(compiled, local_vars)
    
    safe_show()

    for k, v in local_vars.items():
        if k in ['result_df', 'output_df'] and isinstance(v, pd.DataFrame):
            output_tables.append(v.to_dict(orient='records'))
            
except Exception as e:
    success = False
    error_msg = f"{type(e).__name__}: {str(e)}"
    traceback.print_exc()

sys.stdout = old_stdout
sys.stderr = old_stderr

res = {
    "success": success,
    "stdout": stdout_buf.getvalue(),
    "stderr": stderr_buf.getvalue(),
    "error": error_msg,
    "plot_images": plot_images,
    "output_tables": output_tables
}

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(res, f)
"""

class PythonSandbox:
    def __init__(self):
        self._active_processes = {}

    def execute(self, code: str, dataset_context: dict, config: SandboxConfig = None) -> SandboxResult:
        if config is None:
            config = SandboxConfig()
            
        execution_id = str(uuid.uuid4())
        
        start_time = time.time()
        status = ExecutionStatus.RUNNING
        stdout = ""
        stderr = ""
        plots = []
        output_tables = []
        error_msg = None
        exit_code = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            user_code_path = os.path.join(temp_dir, "user_code.py")
            runner_script_path = os.path.join(temp_dir, "runner.py")
            output_json_path = os.path.join(temp_dir, "output.json")
            
            with open(user_code_path, "w", encoding="utf-8") as f:
                f.write(code)
                
            with open(runner_script_path, "w", encoding="utf-8") as f:
                f.write(RUNNER_SCRIPT_TEMPLATE)
                
            if "df" in dataset_context and isinstance(dataset_context["df"], pd.DataFrame):
                dataset_context["df"].to_parquet(os.path.join(temp_dir, "input_df.parquet"))
                
            cmd = [sys.executable, runner_script_path, temp_dir]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self._active_processes[execution_id] = proc
            
            timed_out = False
            mem_exceeded = False
            peak_memory_mb = 0.0
            
            while proc.poll() is None:
                if time.time() - start_time > config.timeout_sec:
                    timed_out = True
                    break
                try:
                    import psutil
                    p = psutil.Process(proc.pid)
                    mem_mb = p.memory_info().rss / (1024 * 1024)
                    if mem_mb > peak_memory_mb:
                        peak_memory_mb = mem_mb
                    if config.max_memory_mb and mem_mb > config.max_memory_mb:
                        mem_exceeded = True
                        break
                except Exception:
                    pass
                    
                time.sleep(0.1)
                
            if timed_out or mem_exceeded:
                proc.kill()
                proc.wait()
                if timed_out:
                    status = ExecutionStatus.TIMED_OUT
                    error_msg = f"Execution timed out after {config.timeout_sec}s"
                else:
                    status = ExecutionStatus.FAILED
                    error_msg = f"Memory limit exceeded ({peak_memory_mb:.1f}MB > {config.max_memory_mb}MB)"
            else:
                exit_code = proc.returncode
                if os.path.exists(output_json_path):
                    with open(output_json_path, "r", encoding="utf-8") as f:
                        res = json.load(f)
                    stdout = res.get("stdout", "")
                    stderr = res.get("stderr", "")
                    plots = res.get("plot_images", [])
                    output_tables = res.get("output_tables", [])
                    if not res.get("success"):
                        status = ExecutionStatus.FAILED
                        error_msg = res.get("error")
                    else:
                        status = ExecutionStatus.SUCCEEDED
                else:
                    out, err = proc.communicate()
                    stdout = out
                    stderr = err
                    status = ExecutionStatus.FAILED if exit_code != 0 else ExecutionStatus.SUCCEEDED
                    
            if execution_id in self._active_processes:
                del self._active_processes[execution_id]
                
        return SandboxResult(
            execution_id=execution_id,
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time_sec=time.time() - start_time,
            plots=plots,
            output_tables=output_tables,
            error_message=error_msg
        )

    def cancel(self, execution_id: str) -> bool:
        proc = self._active_processes.get(execution_id)
        if proc:
            try:
                proc.kill()
                return True
            except Exception:
                return False
        return False

python_sandbox = PythonSandbox()
