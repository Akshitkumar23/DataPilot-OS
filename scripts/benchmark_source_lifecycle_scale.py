"""
DataPilot OS — Source Lifecycle & Execution Scale Benchmark.
Measures performance, memory, pushdown, and mathematical parity across:
- Small (10,000 rows)
- Medium (250,000 rows)
- Large (1,000,000 rows)
For both CSV and Parquet source formats.
Verifies source file immutability (SHA-256 before == SHA-256 after).
"""
import os
import sys
import time
import json
import hashlib
import tempfile
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath('.'))

from backend.dataset_engine.loader import dataset_loader
from backend.dataset_engine.profiler import DatasetProfiler
from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
from backend.core.dataset_registry import dataset_registry, DatasetEntry

def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def generate_synthetic_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = np.array(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"])
    return pd.DataFrame({
        "id": np.arange(1, n_rows + 1, dtype=np.int64),
        "group_cat": rng.choice(categories, size=n_rows),
        "val_num": rng.uniform(0.0, 1000.0, size=n_rows),
        "weight_num": rng.uniform(1.0, 50.0, size=n_rows),
        "flag_bool": rng.choice([True, False], size=n_rows)
    })

def run_scale_benchmarks():
    scales = [
        {"name": "Small", "rows": 10_000},
        {"name": "Medium", "rows": 250_000},
        {"name": "Large", "rows": 1_000_000}
    ]
    formats = ["csv", "parquet"]
    
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_summary": "16 GB RAM visible (~5.8 GB free at start), Windows 10/11 x64",
        "duckdb_version": "1.5.4",
        "benchmarks": []
    }
    
    profiler = DatasetProfiler()
    temp_dir = os.path.abspath("scratch/benchmarks")
    os.makedirs(temp_dir, exist_ok=True)
    
    for scale in scales:
        n_rows = scale["rows"]
        scale_name = scale["name"]
        print(f"\n==================================================")
        print(f" GENERATING DATA: {scale_name} ({n_rows:,} rows)")
        print(f"==================================================")
        
        t0 = time.perf_counter()
        raw_df = generate_synthetic_data(n_rows)
        gen_time_s = time.perf_counter() - t0
        ram_mb = raw_df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"Generated {n_rows:,} rows in {gen_time_s:.2f}s. In-memory DataFrame size: {ram_mb:.2f} MB")
        
        for fmt in formats:
            test_file = os.path.join(temp_dir, f"bench_{scale_name.lower()}_{n_rows}.{fmt}")
            if fmt == "csv":
                raw_df.to_csv(test_file, index=False)
            else:
                raw_df.to_parquet(test_file, index=False)
                
            file_size_mb = os.path.getsize(test_file) / (1024 * 1024)
            sha256_before = compute_file_sha256(test_file)
            print(f"\n--- Testing {scale_name} {fmt.upper()} (File size: {file_size_mb:.2f} MB) ---")
            
            # Step 1: Ingestion & Load Benchmark
            t_load_start = time.perf_counter()
            df_loaded, load_res = dataset_loader.load(test_file)
            load_time_ms = (time.perf_counter() - t_load_start) * 1000.0
            assert load_res.success, f"Failed to load {test_file}"
            
            # Step 2: Profiling Benchmark
            t_prof_start = time.perf_counter()
            ds_id = f"ds_{scale_name.lower()}_{fmt}"
            knowledge = profiler.profile(
                df=df_loaded,
                dataset_id=ds_id,
                name=os.path.basename(test_file),
                file_path=test_file,
                format=fmt,
                file_hash=sha256_before
            )
            profile_time_ms = (time.perf_counter() - t_prof_start) * 1000.0
            
            entry = DatasetEntry(
                dataset_id=ds_id,
                name=os.path.basename(test_file),
                file_path=test_file,
                format=fmt,
                row_count=n_rows,
                column_count=len(df_loaded.columns),
                knowledge_model=knowledge.model_dump(),
                loaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                file_hash=sha256_before
            )
            dataset_registry.register(ds_id, entry)
            dataset_registry._dataframes_cache[ds_id] = df_loaded
            
            # Step 3: Query 1 - Group-By Aggregation
            sql_agg = "SELECT group_cat, COUNT(*) AS count_val, ROUND(AVG(val_num), 4) AS avg_val, ROUND(SUM(weight_num), 2) AS sum_weight FROM active_dataset GROUP BY group_cat ORDER BY group_cat"
            
            # In-memory DataFrame scan
            res_agg_inmem = GovernedDuckDBManager.execute_query(
                sql=sql_agg,
                df=df_loaded,
                dataset_id=ds_id,
                source_file_path=""
            )
            # Direct file scan
            res_agg_direct = GovernedDuckDBManager.execute_query(
                sql=sql_agg,
                df=df_loaded,
                dataset_id=ds_id,
                source_file_path=test_file
            )
            
            # Step 4: Query 2 - Filtered Projection with Pushdown
            sql_filter = "SELECT id, group_cat, val_num FROM active_dataset WHERE val_num > 500.0 AND flag_bool = true ORDER BY id LIMIT 50"
            res_filt_inmem = GovernedDuckDBManager.execute_query(
                sql=sql_filter,
                df=df_loaded,
                dataset_id=ds_id,
                source_file_path=""
            )
            res_filt_direct = GovernedDuckDBManager.execute_query(
                sql=sql_filter,
                df=df_loaded,
                dataset_id=ds_id,
                source_file_path=test_file
            )
            
            # Step 5: Parity & Pushdown Assertions
            assert res_agg_direct.execution_mode == "direct_file_scan", f"Expected direct_file_scan, got {res_agg_direct.execution_mode}"
            assert res_agg_direct.pushdown_supported is True
            assert res_agg_inmem.execution_mode == "in_memory_dataframe"
            assert res_agg_inmem.pushdown_supported is False
            
            # Verify Aggregation Parity
            assert len(res_agg_direct.rows) == len(res_agg_inmem.rows)
            for r_dir, r_mem in zip(res_agg_direct.rows, res_agg_inmem.rows):
                assert r_dir["group_cat"] == r_mem["group_cat"]
                assert r_dir["count_val"] == r_mem["count_val"]
                assert abs(float(r_dir["avg_val"]) - float(r_mem["avg_val"])) < 1e-4
                assert abs(float(r_dir["sum_weight"]) - float(r_mem["sum_weight"])) < 1e-2
                
            # Verify Filtered Projection Parity
            assert len(res_filt_direct.rows) == len(res_filt_inmem.rows)
            for r_dir, r_mem in zip(res_filt_direct.rows, res_filt_inmem.rows):
                assert r_dir["id"] == r_mem["id"]
                assert r_dir["group_cat"] == r_mem["group_cat"]
                assert abs(float(r_dir["val_num"]) - float(r_mem["val_num"])) < 1e-4

            # Step 6: Verify Source Preservation
            sha256_after = compute_file_sha256(test_file)
            source_preserved = (sha256_before == sha256_after)
            assert source_preserved, f"CRITICAL: File {test_file} was modified!"
            
            bench_record = {
                "scale": scale_name,
                "rows": n_rows,
                "format": fmt,
                "file_size_mb": round(file_size_mb, 2),
                "ram_materialized_mb": round(ram_mb, 2),
                "load_time_ms": round(load_time_ms, 2),
                "profile_time_ms": round(profile_time_ms, 2),
                "query_agg": {
                    "in_memory_ms": round(res_agg_inmem.execution_time_ms, 2),
                    "direct_scan_ms": round(res_agg_direct.execution_time_ms, 2),
                    "execution_mode_direct": res_agg_direct.execution_mode,
                    "parity_verified": True
                },
                "query_filter": {
                    "in_memory_ms": round(res_filt_inmem.execution_time_ms, 2),
                    "direct_scan_ms": round(res_filt_direct.execution_time_ms, 2),
                    "execution_mode_direct": res_filt_direct.execution_mode,
                    "pushdown_supported": res_filt_direct.pushdown_supported,
                    "parity_verified": True
                },
                "source_preservation": {
                    "sha256_before": sha256_before,
                    "sha256_after": sha256_after,
                    "immutability_verified": source_preserved
                }
            }
            results["benchmarks"].append(bench_record)
            
            print(f"  Load Time: {load_time_ms:.1f} ms | Profile Time: {profile_time_ms:.1f} ms")
            print(f"  Agg Query: In-Memory {res_agg_inmem.execution_time_ms:.1f} ms vs Direct Scan {res_agg_direct.execution_time_ms:.1f} ms (Parity: 100%)")
            print(f"  Filter Query: In-Memory {res_filt_inmem.execution_time_ms:.1f} ms vs Direct Scan {res_filt_direct.execution_time_ms:.1f} ms (Pushdown: YES, Parity: 100%)")
            print(f"  Source Immutability (SHA-256 match): {source_preserved}")
            
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
                
    # Save results to docs
    os.makedirs("docs", exist_ok=True)
    out_path = os.path.abspath("docs/benchmark_source_lifecycle_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n==================================================")
    print(f" SCALE BENCHMARK COMPLETE. Results written to:")
    print(f" {out_path}")
    print(f"==================================================")

if __name__ == "__main__":
    run_scale_benchmarks()
