"""
Verification script for the packaged standalone Electron backend (http://127.0.0.1:8011).
Tests:
1. CORS behavior: localhost:5173 and 'null' origins accepted; unauthorized origin rejected.
2. Dataset upload & profile.
3. Governed operations: clip_iqr, impute_mean, yeo_johnson, one_hot_encode.
4. Natural Language ToolSpec parity:
   - stationarity_test (ADF / KPSS)
   - elbow_analysis (KMeans inertia/silhouette curve)
   - pca_analysis (PCA explained variance & projections)
   - autocorrelation (ACF / PACF)
   - decompose_time_series (trend, seasonal, residual)
"""

import requests
import json
import os
import pandas as pd
import numpy as np
import io

BASE_URL = "http://127.0.0.1:8011"

def get_auth_headers():
    appdata = os.environ.get("APPDATA", "")
    token_path = os.path.join(appdata, "DataPilot OS", ".env_token")
    if os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            token = f.read().strip()
            return {"Authorization": f"Bearer {token}", "Origin": "http://localhost:5173"}
    return {"Origin": "http://localhost:5173"}

def test_live_cors():
    print("[1/3] Testing Live CORS on Packaged Backend...")
    # 1. Allowed: localhost:5173
    r1 = requests.get(f"{BASE_URL}/health", headers={"Origin": "http://localhost:5173"})
    assert r1.headers.get("access-control-allow-origin") == "http://localhost:5173", f"Expected allowed origin, got {r1.headers}"
    print("  -> Origin 'http://localhost:5173' correctly ALLOWED")

    # 2. Allowed: null (packaged electron file:// or about:blank)
    r2 = requests.get(f"{BASE_URL}/health", headers={"Origin": "null"})
    assert r2.headers.get("access-control-allow-origin") == "null", f"Expected allowed null origin, got {r2.headers}"
    print("  -> Origin 'null' correctly ALLOWED")

    # 3. Disallowed: evil origin
    r3 = requests.get(f"{BASE_URL}/health", headers={"Origin": "http://malicious-website.com"})
    assert "access-control-allow-origin" not in r3.headers or r3.headers["access-control-allow-origin"] != "http://malicious-website.com", f"Evil origin was not rejected: {r3.headers}"
    print("  -> Origin 'http://malicious-website.com' correctly REJECTED")

def test_live_governed_operations():
    print("\n[2/3] Testing Governed Operations on Packaged Backend...")
    headers = get_auth_headers()
    # Generate test dataset
    df = pd.DataFrame({
        "temperature": [20.0, 22.5, 21.0, 100.0, None, 19.5, 23.0, 22.0, -50.0, 21.5],
        "category": ["A", "B", "A", "C", "B", "A", "B", "C", "A", "B"],
        "pressure": [101.3, 101.5, 101.2, 102.0, 101.4, 101.1, 101.6, 101.8, 101.2, 101.5]
    })
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    files = {"file": ("sensor_live.csv", csv_bytes, "text/csv")}
    
    # Upload requires Authorization header
    up_headers = {k: v for k, v in headers.items() if k != "Content-Type"}
    r_up = requests.post(f"{BASE_URL}/api/datasets/upload", files=files, headers=up_headers)
    assert r_up.status_code == 200, f"Upload failed: {r_up.text}"
    ds_meta = r_up.json()
    ds_id = ds_meta["dataset_id"]
    print(f"  -> Uploaded dataset {ds_id} ({ds_meta.get('rows', len(df))} rows)")

    # 1. impute_mean
    r_op1 = requests.post(
        f"{BASE_URL}/api/operations/apply",
        json={"dataset_id": ds_id, "operation": "impute_mean", "columns": ["temperature"]},
        headers=headers
    )
    assert r_op1.status_code == 200, f"impute_mean failed: {r_op1.text}"
    print("  -> Governed op 'impute_mean' SUCCEEDED")

    # 2. clip_iqr
    r_op2 = requests.post(
        f"{BASE_URL}/api/operations/apply",
        json={"dataset_id": ds_id, "operation": "clip_iqr", "columns": ["temperature"]},
        headers=headers
    )
    assert r_op2.status_code == 200, f"clip_iqr failed: {r_op2.text}"
    print("  -> Governed op 'clip_iqr' SUCCEEDED")

    # 3. yeo_johnson
    r_op3 = requests.post(
        f"{BASE_URL}/api/operations/apply",
        json={"dataset_id": ds_id, "operation": "yeo_johnson", "columns": ["pressure"]},
        headers=headers
    )
    assert r_op3.status_code == 200, f"yeo_johnson failed: {r_op3.text}"
    print("  -> Governed op 'yeo_johnson' SUCCEEDED")

    # 4. one_hot_encode
    r_op4 = requests.post(
        f"{BASE_URL}/api/operations/apply",
        json={"dataset_id": ds_id, "operation": "one_hot_encode", "columns": ["category"]},
        headers=headers
    )
    assert r_op4.status_code == 200, f"one_hot_encode failed: {r_op4.text}"
    print("  -> Governed op 'one_hot_encode' SUCCEEDED")

    # Check provenance audit trail
    r_audit = requests.get(f"{BASE_URL}/api/governance/audit?dataset_id={ds_id}", headers=headers)
    assert r_audit.status_code == 200
    events = r_audit.json().get("events", [])
    assert len(events) >= 4, f"Expected >= 4 events, got {len(events)}"
    print(f"  -> Provenance audit logged {len(events)} governed events dynamically")

def test_live_nl_parity():
    print("\n[3/3] Testing Natural Language ToolSpec Parity on Packaged Backend...")
    headers = get_auth_headers()
    # Time-series dataset
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    trend = np.linspace(10, 50, 100)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(100) / 7)
    noise = np.random.normal(0, 1, 100)
    sales = trend + seasonal + noise
    
    df_ts = pd.DataFrame({
        "order_date": dates.strftime("%Y-%m-%d"),
        "sales": sales,
        "visitors": sales * 1.5 + np.random.normal(0, 2, 100),
        "spend": sales * 0.4 + np.random.normal(0, 1, 100)
    })
    csv_bytes = df_ts.to_csv(index=False).encode('utf-8')
    files = {"file": ("store_ts.csv", csv_bytes, "text/csv")}
    up_headers = {k: v for k, v in headers.items() if k != "Content-Type"}
    r_up = requests.post(f"{BASE_URL}/api/datasets/upload", files=files, headers=up_headers)
    assert r_up.status_code == 200
    ts_id = r_up.json()["dataset_id"]
    print(f"  -> Uploaded time-series dataset {ts_id} (100 rows)")

    # 0. Ambiguity -> Clarification contract
    r_ambig = requests.post(f"{BASE_URL}/api/chat", json={
        "query": "Check if sales is stationary",
        "dataset_id": ts_id
    }, headers=headers)
    assert r_ambig.status_code == 422
    ambig_data = r_ambig.json().get("detail", r_ambig.json())
    assert ambig_data.get("status") == "NEEDS_CLARIFICATION"
    assert ambig_data.get("error", {}).get("code") == "CLARIFICATION_REQUIRED"
    print("  -> Ambiguity Interdict: correctly returned NEEDS_CLARIFICATION / CLARIFICATION_REQUIRED")

    # 1. Elbow Analysis
    cmd_elb = json.dumps({"commands": [{"tool": "elbow_analysis", "arguments": {"columns": ["sales", "visitors", "spend"]}}]})
    r_elb = requests.post(f"{BASE_URL}/api/chat", json={"query": cmd_elb, "dataset_id": ts_id}, headers=headers)
    assert r_elb.status_code == 200, f"Elbow failed: {r_elb.text}"
    elb_data = r_elb.json()
    assert elb_data["status"] == "SUCCEEDED"
    elb_inner = elb_data["result"].get("data", elb_data["result"])
    assert "curve" in elb_inner or "k_values" in elb_inner or "records" in elb_inner
    print("  -> Elbow Analysis SUCCEEDED")

    # 2. ACF / PACF Autocorrelation
    cmd_acf = json.dumps({"commands": [{"tool": "autocorrelation", "arguments": {"time_column": "order_date", "value_column": "sales", "max_lags": 10}}]})
    r_acf = requests.post(f"{BASE_URL}/api/chat", json={"query": cmd_acf, "dataset_id": ts_id}, headers=headers)
    assert r_acf.status_code == 200, f"ACF failed: {r_acf.text}"
    acf_data = r_acf.json()
    assert acf_data["status"] == "SUCCEEDED"
    acf_inner = acf_data["result"].get("data", acf_data["result"])
    assert "records" in acf_inner or "acf_values" in acf_inner
    print("  -> Autocorrelation (ACF / PACF) SUCCEEDED")

    # 3. Seasonal Decomposition
    cmd_dec = json.dumps({"commands": [{"tool": "decompose_time_series", "arguments": {"time_column": "order_date", "target_column": "sales", "period": 7, "model": "additive"}}]})
    r_dec = requests.post(f"{BASE_URL}/api/chat", json={"query": cmd_dec, "dataset_id": ts_id}, headers=headers)
    assert r_dec.status_code == 200, f"Decompose failed: {r_dec.text}"
    dec_data = r_dec.json()
    assert dec_data["status"] == "SUCCEEDED"
    dec_inner = dec_data["result"].get("data", dec_data["result"])
    assert "records" in dec_inner or "trend" in dec_inner
    print("  -> Decomposition (Trend, Seasonal, Residual) SUCCEEDED")

    # 4. PCA Analysis
    cmd_pca = json.dumps({"commands": [{"tool": "pca_analysis", "arguments": {"columns": ["sales", "visitors", "spend"], "n_components": 2}}]})
    r_pca = requests.post(f"{BASE_URL}/api/chat", json={"query": cmd_pca, "dataset_id": ts_id}, headers=headers)
    assert r_pca.status_code == 200, f"PCA failed: {r_pca.text}"
    pca_data = r_pca.json()
    assert pca_data["status"] == "SUCCEEDED", f"PCA failed: {pca_data}"
    print("  -> PCA Analysis SUCCEEDED")

    # 5. Stationarity (ADF / KPSS)
    cmd_stat = json.dumps({"commands": [{"tool": "stationarity_test", "arguments": {"target_column": "sales", "time_column": "order_date"}}]})
    r_stat = requests.post(f"{BASE_URL}/api/chat", json={"query": cmd_stat, "dataset_id": ts_id}, headers=headers)
    assert r_stat.status_code == 200, f"Stationarity failed: {r_stat.text}"
    stat_data = r_stat.json()
    assert stat_data["status"] == "SUCCEEDED", f"Stationarity failed: {stat_data}"
    print("  -> Stationarity Test (ADF / KPSS) SUCCEEDED")

def test_live_phase7_sql_and_reporting():
    print("\n[4/4] Testing Governed SQL Studio & Analytical Reporting on Packaged Backend...")
    headers = get_auth_headers()
    # 1. Prepare and upload a dedicated financial dataset
    np.random.seed(77)
    df_fin = pd.DataFrame({
        "account_id": [f"ACC_{i:04d}" for i in range(50)],
        "region": np.random.choice(["EMEA", "APAC", "AMER"], size=50),
        "balance": np.random.normal(50000, 15000, size=50).round(2),
        "risk_score": np.random.uniform(0.1, 0.9, size=50).round(3)
    })
    csv_bytes = df_fin.to_csv(index=False).encode('utf-8')
    files = {"file": ("financial_audit.csv", csv_bytes, "text/csv")}
    up_headers = {k: v for k, v in headers.items() if k != "Content-Type"}
    r_up = requests.post(f"{BASE_URL}/api/datasets/upload", files=files, headers=up_headers)
    assert r_up.status_code == 200
    ds_id = r_up.json()["dataset_id"]
    print(f"  -> Uploaded financial audit dataset {ds_id} (50 rows)")

    # 2. Test SQL Validation (Read-only check, schema check)
    r_val = requests.post(f"{BASE_URL}/api/sql/validate", json={
        "dataset_id": ds_id,
        "sql": "SELECT region, COUNT(*) AS count, AVG(balance) AS mean_bal FROM active_dataset GROUP BY region"
    }, headers=headers)
    assert r_val.status_code == 200, f"Validate failed: {r_val.text}"
    val_data = r_val.json()
    assert val_data["status"] == "VALID"
    print("  -> SQL Validation SUCCEEDED (read-only relation grounded)")

    # 3. Test SQL Execution via DuckDB in-process manager
    r_exec = requests.post(f"{BASE_URL}/api/sql/execute", json={
        "dataset_id": ds_id,
        "sql": "SELECT region, COUNT(*) AS count, ROUND(AVG(balance), 2) AS mean_bal FROM active_dataset GROUP BY region ORDER BY mean_bal DESC"
    }, headers=headers)
    assert r_exec.status_code == 200, f"Execute failed: {r_exec.text}"
    exec_data = r_exec.json()
    assert exec_data["status"] == "SUCCEEDED"
    assert exec_data["row_count"] == 3
    assert len(exec_data["rows"]) == 3
    print(f"  -> DuckDB SQL Execution SUCCEEDED ({exec_data['row_count']} rows returned in {exec_data.get('execution_time_ms', 0)}ms)")

    # 4. Test SQL Query History
    r_hist = requests.get(f"{BASE_URL}/api/sql/history?dataset_id={ds_id}", headers=headers)
    assert r_hist.status_code == 200
    hist_entries = r_hist.json().get("history", [])
    assert len(hist_entries) >= 1
    print(f"  -> Query History Retrieval SUCCEEDED ({len(hist_entries)} queries recorded)")

    # 5. Test NL-to-SQL Intent Compilation
    r_nl = requests.post(f"{BASE_URL}/api/sql/nl-intent", json={
        "dataset_id": ds_id,
        "intent": {
            "operation": "aggregate",
            "selected_columns": ["region"],
            "aggregations": [{"column": "balance", "func": "avg", "alias": "avg_balance"}],
            "group_by": ["region"],
            "order_by": [{"column": "avg_balance", "direction": "desc"}]
        }
    }, headers=headers)
    assert r_nl.status_code == 200
    nl_data = r_nl.json()
    assert "AVG(balance)" in nl_data["sql"]
    print(f"  -> NL-to-SQL Deterministic Compilation SUCCEEDED: {nl_data['sql']}")

    # 6. Test Analytical Report Generation (9 sections synthesized without hallucination)
    r_rep = requests.post(f"{BASE_URL}/api/reports/generate", json={
        "dataset_id": ds_id,
        "title": "Governed Financial Risk Audit Report"
    }, headers=headers)
    assert r_rep.status_code == 200, f"Report generation failed: {r_rep.text}"
    rep_data = r_rep.json()
    assert rep_data["report_id"].startswith("rep_")
    assert len(rep_data["sections"]) == 9
    assert len(rep_data["executive_summary"]) > 20
    print(f"  -> Report Generation SUCCEEDED (9 sections, report_id={rep_data['report_id']})")

    # 7. Test Standalone HTML Export with Print Media
    r_exp = requests.get(f"{BASE_URL}/api/reports/{rep_data['report_id']}/export?format=html", headers=headers)
    assert r_exp.status_code == 200
    html_content = r_exp.text
    assert "<!DOCTYPE html>" in html_content
    assert "@media print" in html_content
    print("  -> Standalone HTML Publication Export SUCCEEDED (verified print CSS)")

def main():
    print("=" * 60)
    print("VERIFYING PACKAGED STANDALONE ELECTRON RUNTIME (http://127.0.0.1:8011)")
    print("=" * 60)
    test_live_cors()
    test_live_governed_operations()
    test_live_nl_parity()
    test_live_phase7_sql_and_reporting()
    print("\n" + "=" * 60)
    print("ALL PACKAGED RUNTIME VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

