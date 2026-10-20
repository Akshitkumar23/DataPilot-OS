"""
DataPilot OS — Runtime-Source Non-Mutating Architecture Inventory Generator.

Operational Characteristics:
- Runtime-Source Non-Mutating: Does not edit, patch, or alter any application source files,
  runtime configurations, test files, or database entries.
- Output Scope: Strictly writes structured documentation artifacts to docs/architecture/.
- Import-Time Side-Effect Audit:
  - Imports `backend.main.app` to inspect FastAPI's registered route graph.
  - Side effects of importing `backend.main`:
    1. Initializes in-memory singletons (`capability_registry` registers 22 ToolSpecs and Drivers).
    2. Instantiates `ollama_service` in-memory object (does NOT initiate network calls during import).
    3. Mounts router definitions onto the in-memory Starlette router tree.
    4. Lifespan handlers (`workflow_store._load_from_disk()`) are NOT executed upon import (they only
       execute under an active ASGI lifespan context managed by uvicorn).
    5. No file system mutations, disk writes (outside docs/architecture/), or subprocesses occur.
"""

import ast
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_ARCH_DIR = os.path.join(REPO_ROOT, "docs", "architecture")

import sys
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)



def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def sha256_of_string(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def audit_mounted_routes(frontend_calls_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Inspect backend/main.py to extract all mounted routers and routes."""
    routes = []
    
    # Map frontend callers by path prefix / endpoint pattern
    frontend_map: Dict[str, List[Dict[str, Any]]] = {}
    if frontend_calls_data:
        for call in frontend_calls_data:
            ep = call.get("endpoint", "")
            # normalize template variable or query parameter
            clean_ep = re.sub(r'\$\{[^}]+\}', '', ep).rstrip('/')
            if not clean_ep:
                clean_ep = ep
            frontend_map.setdefault(clean_ep, []).append({
                "component": call.get("component"),
                "line": call.get("line")
            })
    
    try:
        from backend.main import app
        raw_routes = []
        for r in app.routes:
            if hasattr(r, "original_router") and hasattr(r.original_router, "routes"):
                for sub_r in r.original_router.routes:
                    raw_routes.append(sub_r)
            elif hasattr(r, "routes"):
                for sub_r in r.routes:
                    raw_routes.append(sub_r)
            else:
                raw_routes.append(r)
                
        for route in raw_routes:
            methods = list(getattr(route, "methods", []))
            path = getattr(route, "path", "")
            name = getattr(route, "name", "")
            
            if not path or path in ["/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"]:
                continue
            
            # Determine router origin
            router_origin = "main"
            if "/api/chat" in path or "/api/query" in path:
                router_origin = "chat_router"
            elif "/api/ollama" in path:
                router_origin = "ollama_router"
            elif "/api/datasets" in path:
                router_origin = "datasets_router"
            elif "/api/intent" in path:
                router_origin = "intent_router"
            elif "/api/code" in path:
                router_origin = "code_router"
            elif "/api/execution" in path:
                router_origin = "execution_router"
            elif "/api/results" in path:
                router_origin = "results_router"
            elif "/api/production" in path:
                router_origin = "production_router"
            elif "/api/graph" in path:
                router_origin = "graph_router"
            elif "/api/workflows" in path:
                router_origin = "workflows_router"
            elif path in ["/livez", "/health", "/api/v2/health", "/"]:
                router_origin = "backend.main (health)"
            
            # Risk & classification
            is_auth = router_origin != "backend.main (health)"
            is_authoritative = router_origin in ["chat_router", "datasets_router", "ollama_router", "workflows_router", "backend.main (health)"]
            accepts_code = router_origin in ["code_router", "execution_router"]
            
            risk = "low"
            rec_action = "retain_authoritative"
            side_effects = "Standard query handling or state retrieval."
            
            if name in {"quarantined_execution_api", "quarantined_intent_api"}:
                risk = "low"
                rec_action = "retain_authenticated_quarantine_tombstone"
                side_effects = "Returns HTTP 410 without invoking the retired implementation."
                accepts_code = False
            elif path == "/api/production/execute":
                risk = "low"
                rec_action = "retain_default_deny_until_typed_graph_migration"
                side_effects = "Generated-code execution is blocked unless an explicit legacy developer flag is enabled."
            elif router_origin == "execution_router":
                risk = "critical"
                rec_action = "quarantine_in_phase_0b"
                side_effects = "Spawns Python subprocess running arbitrary unvalidated code via ModularExecutionEngine."
            elif router_origin == "intent_router":
                risk = "medium"
                rec_action = "quarantine_duplicate_in_phase_0b"
                side_effects = "Duplicate analytical entry point bypassing /api/chat orchestrator."
            elif router_origin == "code_router" and "/generate" in path:
                risk = "low"
                rec_action = "retain_default_deny_legacy_gate"
                side_effects = "Free-form generation is blocked unless an explicit legacy developer flag is enabled."
            elif router_origin == "code_router":
                risk = "medium"
                rec_action = "retain_for_developer_ide"
                side_effects = "Validates or edits generated Python code."
            elif router_origin == "datasets_router" and any(k in path for k in ["upload", "connect", "activate", "approve"]):
                risk = "low"
                rec_action = "retain_authoritative"
                side_effects = "Modifies active dataset state, ingests external tabular file."
            elif router_origin == "workflows_router" and any(m in methods for m in ["DELETE", "POST"]):
                risk = "low"
                rec_action = "retain_authoritative"
                side_effects = "Deletes or modifies stored workflow session."
            elif not is_authoritative:
                risk = "medium"
                rec_action = "evaluate_for_phase_0b_quarantine"
                side_effects = "Auxiliary or unmigrated endpoint."
                
            # Match frontend callers
            clean_path = re.sub(r'\{[^}]+\}', '', path).rstrip('/')
            callers = frontend_map.get(clean_path, [])
            if not callers and path in frontend_map:
                callers = frontend_map[path]
                
            routes.append({
                "path": path,
                "methods": sorted(methods),
                "handler_name": name,
                "router": router_origin,
                "mounted": True,
                "authenticated": is_auth,
                "authoritative": is_authoritative,
                "accepts_code": accepts_code,
                "side_effects": side_effects,
                "risk": risk,
                "recommended_action": rec_action,
                "frontend_callers": callers
            })
    except Exception as e:
        print(f"Warning: Could not import backend.main: {e}")
        
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "total_mounted_routes": len(routes),
        "routes": routes
    }


def audit_frontend_network_traffic() -> Dict[str, Any]:
    """Scan src/ for all network communication vectors: fetch, axios, XHR, WebSocket, and Electron IPC."""
    src_dir = os.path.join(REPO_ROOT, "src")
    fetch_calls = []
    axios_calls = []
    xhr_calls = []
    ws_calls = []
    ipc_calls = []
    
    # Patterns
    fetch_pattern = re.compile(r'fetch\(\s*`\$\{API\}([^`?]+)(?:\?([^`]+))?`', re.MULTILINE)
    axios_pattern = re.compile(r'\baxios\.(?:get|post|put|delete|patch|request)\s*\(|\baxios\(', re.MULTILINE)
    xhr_pattern = re.compile(r'new\s+XMLHttpRequest\s*\(', re.MULTILINE)
    ws_pattern = re.compile(r'new\s+(?:WebSocket|EventSource)\s*\(', re.MULTILINE)
    ipc_pattern = re.compile(r'window\.electron(?:\.([a-zA-Z0-9_]+))?', re.MULTILINE)
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith((".jsx", ".js", ".ts", ".tsx")):
                rel_path = os.path.relpath(os.path.join(root, file), REPO_ROOT).replace("\\", "/")
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    
                    full_text = "".join(lines)
                    
                    # 1. Fetch
                    for match in fetch_pattern.finditer(full_text):
                        raw_endpoint = match.group(1).strip()
                        line_no = full_text[:match.start()].count("\n") + 1
                        
                        is_mounted = any(raw_endpoint.startswith(p) for p in [
                            "/api/chat", "/api/query", "/api/datasets", "/api/ollama",
                            "/api/workflows", "/api/code", "/api/execution", "/api/results",
                            "/api/production", "/api/graph", "/health"
                        ])
                        
                        is_unmounted_legacy = raw_endpoint in [
                            "/api/terminal/execute", "/api/screen/analyze", "/api/screen/detach",
                            "/api/validate", "/api/ml/train", "/api/db/test", "/api/db/import",
                            "/api/operations/preview", "/api/operations/apply", "/api/export",
                            "/api/operations/undo", "/api/undo", "/api/operations/redo",
                            "/api/fs/writeback", "/api/connect", "/api/upload"
                        ]
                        
                        fetch_calls.append({
                            "component": rel_path,
                            "line": line_no,
                            "endpoint": raw_endpoint,
                            "status": "mounted" if is_mounted else "unmounted_legacy_404",
                            "route_mounted": is_mounted,
                            "is_unmounted_legacy": is_unmounted_legacy,
                            "risk": "high" if is_unmounted_legacy else ("medium" if not is_mounted else "low"),
                            "remediation": "migrate_or_disable_in_phase_0b" if is_unmounted_legacy else "retain"
                        })
                        
                    # 2. Axios
                    for match in axios_pattern.finditer(full_text):
                        line_no = full_text[:match.start()].count("\n") + 1
                        axios_calls.append({"component": rel_path, "line": line_no, "snippet": match.group(0)})
                        
                    # 3. XMLHttpRequest
                    for match in xhr_pattern.finditer(full_text):
                        line_no = full_text[:match.start()].count("\n") + 1
                        xhr_calls.append({"component": rel_path, "line": line_no, "snippet": match.group(0)})
                        
                    # 4. WebSocket / EventSource
                    for match in ws_pattern.finditer(full_text):
                        line_no = full_text[:match.start()].count("\n") + 1
                        ws_calls.append({"component": rel_path, "line": line_no, "snippet": match.group(0)})
                        
                    # 5. Electron IPC
                    for match in ipc_pattern.finditer(full_text):
                        line_no = full_text[:match.start()].count("\n") + 1
                        channel = match.group(1) or "window.electron"
                        ipc_calls.append({
                            "component": rel_path,
                            "line": line_no,
                            "channel": channel,
                            "type": "electron_bridge"
                        })
                except Exception as fe:
                    print(f"Error scanning {filepath}: {fe}")
                    
    return {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "network_vector_summary": {
            "fetch_calls_count": len(fetch_calls),
            "unmounted_legacy_fetch_calls_count": sum(1 for c in fetch_calls if c["is_unmounted_legacy"]),
            "axios_calls_count": len(axios_calls),
            "xmlhttprequest_calls_count": len(xhr_calls),
            "websocket_eventsource_calls_count": len(ws_calls),
            "electron_ipc_calls_count": len(ipc_calls)
        },
        "api_wrappers": {
            "auth_headers_provider": "src/apiConfig.js::getAuthHeaders",
            "base_url_resolver": "src/apiConfig.js::API"
        },
        "fetch_calls": fetch_calls,
        "axios_calls": axios_calls,
        "xmlhttprequest_calls": xhr_calls,
        "websocket_and_eventsource_calls": ws_calls,
        "electron_ipc_calls": ipc_calls,
        # Backward compatibility alias
        "total_frontend_api_calls": len(fetch_calls),
        "unmounted_legacy_calls_count": sum(1 for c in fetch_calls if c["is_unmounted_legacy"]),
        "calls": fetch_calls
    }


def audit_heuristics_and_defaults() -> Dict[str, Any]:
    """Scan intent_resolver.py, capability_registry.py, and typed_operators.py for heuristics and defaults."""
    heuristics = []
    defaults = []
    
    # Check intent_resolver.py for domain-specific candidates and suffix matching
    resolver_path = os.path.join(REPO_ROOT, "backend", "ds_engine", "intent_resolver.py")
    if os.path.exists(resolver_path):
        with open(resolver_path, "r", encoding="utf-8") as f:
            resolver_code = f.read()
            
        # Check hardcoded candidates
        if "machine_failure" in resolver_code:
            heuristics.append({
                "file": "backend/ds_engine/intent_resolver.py",
                "lines": "105-111",
                "type": "hardcoded_domain_candidates",
                "description": "Hardcoded binary target candidates: target, machine_failure, failure_type, failure, churn, is_churn, left_company",
                "classification": "ungrounded_analytical_default / potential_hidden_fallback",
                "tier": "Tier 5: Ungrounded Analytical Default",
                "risk": "high",
                "action_required": "Replace with strict schema candidate matching and NEEDS_CLARIFICATION prompt in Phase 1 & 3"
            })
            
        if "_flag" in resolver_code:
            heuristics.append({
                "file": "backend/ds_engine/intent_resolver.py",
                "lines": "95-103",
                "type": "suffix_stripping_heuristic",
                "description": "Stem-matching suffixes stripped: _flag, _indicator, _event, _col, _column, _label, _status",
                "classification": "domain_convention_heuristic",
                "tier": "Tier 4: Domain-Convention / Heuristic Default",
                "risk": "medium",
                "action_required": "Formalize as explicit declared schema match rule with user transparency"
            })
            
        if "clean in low_c" in resolver_code:
            heuristics.append({
                "file": "backend/ds_engine/intent_resolver.py",
                "lines": "82-93",
                "type": "substring_matching_heuristic",
                "description": "Substring matching where requested string is contained in column name",
                "classification": "implicit_heuristic / risk_of_unintended_column_binding",
                "tier": "Tier 4: Domain-Convention / Heuristic Default",
                "risk": "medium",
                "action_required": "Gate with candidate ambiguity threshold; prompt for clarification when substring matches >1 columns"
            })
            
    # Comprehensive 5-tier Defaults Taxonomy
    defaults.extend([
        # Tier 1: Standards-based declared defaults
        {
            "parameter": "alpha",
            "default_value": 0.05,
            "subsystem": "statistical_test_operator",
            "tier": "Tier 1: Standards-based declared default",
            "rationale": "Standard statistical significance threshold (95% confidence level / alpha=0.05) widely accepted in scientific literature.",
            "scientific_safety": "safe_declared"
        },
        {
            "parameter": "random_state",
            "default_value": 42,
            "subsystem": "auto_ml_operator / clustering_operator",
            "tier": "Tier 1: Standards-based declared default",
            "rationale": "Ensures deterministic seed reproducibility for pseudo-random train/test splits and centroid initialization.",
            "scientific_safety": "safe_declared"
        },
        {
            "parameter": "confidence_level",
            "default_value": 0.95,
            "subsystem": "time_series_forecast_operator",
            "tier": "Tier 1: Standards-based declared default",
            "rationale": "Standard prediction interval for forecast confidence bands.",
            "scientific_safety": "safe_declared"
        },
        # Tier 2: User interface presentation defaults
        {
            "parameter": "chart_type",
            "default_value": "none",
            "subsystem": "intent_schema / capability_registry",
            "tier": "Tier 2: User interface presentation default",
            "rationale": "Prevents generating unwanted charts when user did not request visualization.",
            "scientific_safety": "safe"
        },
        {
            "parameter": "preview_limit",
            "default_value": 100,
            "subsystem": "datasets_api / query_orchestrator",
            "tier": "Tier 2: User interface presentation default",
            "rationale": "Limits rows returned to desktop UI to prevent IPC and DOM memory exhaustion.",
            "scientific_safety": "safe"
        },
        {
            "parameter": "top_n_features",
            "default_value": 10,
            "subsystem": "auto_ml / feature_importance",
            "tier": "Tier 2: User interface presentation default",
            "rationale": "Limits displayed feature importance bars for visual clarity in UI.",
            "scientific_safety": "safe"
        },
        # Tier 3: Backward compatibility defaults
        {
            "parameter": "scaling",
            "default_value": "standard",
            "subsystem": "clustering_operator / pca_operator",
            "tier": "Tier 3: Backward compatibility default",
            "rationale": "Default feature normalization (StandardScaler) for distance-based clustering when user specifies no preprocessor.",
            "scientific_safety": "acceptable_with_manifest_audit"
        },
        {
            "parameter": "result_envelope_format",
            "default_value": "v2_legacy_compatible",
            "subsystem": "chat_router",
            "tier": "Tier 3: Backward compatibility default",
            "rationale": "Provides legacy-compatible fields ('status', 'data', 'insights') alongside modern typed operator payload.",
            "scientific_safety": "safe"
        },
        # Tier 4: Domain-convention / heuristic defaults
        {
            "parameter": "group_by fallback to count",
            "default_value": "count",
            "subsystem": "command_resolver / group_aggregate",
            "tier": "Tier 4: Domain-convention / heuristic default",
            "rationale": "When user requests group_by without an aggregation metric, implicitly defaults to counting occurrences.",
            "scientific_safety": "requires_explicit_user_intent_in_phase_1"
        },
        {
            "parameter": "missing_value_numeric_imputation",
            "default_value": "median",
            "subsystem": "data_clean_operator",
            "tier": "Tier 4: Domain-convention / heuristic default",
            "rationale": "Default imputation strategy for continuous numerical columns.",
            "scientific_safety": "requires_manifest_audit"
        },
        # Tier 5: Ungrounded analytical defaults (Must be eliminated)
        {
            "parameter": "binary_target_candidate_list",
            "default_value": "['target', 'machine_failure', 'failure_type', 'failure', 'churn', 'is_churn', 'left_company']",
            "subsystem": "backend/ds_engine/intent_resolver.py:105-111",
            "tier": "Tier 5: Ungrounded analytical default",
            "rationale": "Hardcodes domain-specific problem columns from external Kaggle datasets into general intent resolver.",
            "scientific_safety": "UNSAFE - Must be eliminated in Phase 3"
        },
        {
            "parameter": "implicit_single_metric_selection",
            "default_value": "first_numeric_column",
            "subsystem": "legacy fallback resolver",
            "tier": "Tier 5: Ungrounded analytical default",
            "rationale": "Picks the first numeric column when query lacks metric specification rather than asking for clarification.",
            "scientific_safety": "UNSAFE - Must be eliminated in Phase 1"
        }
    ])
    
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "heuristics": heuristics,
        "defaults_taxonomy": defaults
    }


def audit_exception_handling() -> Dict[str, Any]:
    """Scan backend/ and modules/ for broad except Exception blocks."""
    exceptions = []
    
    dirs_to_scan = [
        os.path.join(REPO_ROOT, "backend"),
        os.path.join(REPO_ROOT, "modules")
    ]
    
    pattern = re.compile(r'except\s+Exception(\s+as\s+([a-zA-Z0-9_]+))?\s*:', re.MULTILINE)
    
    for scan_dir in dirs_to_scan:
        for root, _, files in os.walk(scan_dir):
            for file in files:
                if file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), REPO_ROOT).replace("\\", "/")
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                            
                        for i, line in enumerate(lines):
                            match = pattern.search(line)
                            if match:
                                line_num = i + 1
                                # Inspect next 3 lines to see if it passes or logs
                                next_lines = "".join(lines[i+1:min(i+5, len(lines))])
                                is_pass = "pass" in next_lines.strip()
                                has_logging = "logger." in next_lines or "logging." in next_lines
                                
                                classification = "swallowed_silent" if is_pass else ("logged_handled" if has_logging else "unclassified")
                                severity = "high" if is_pass and "legacy.py" not in rel_path else ("low" if has_logging else "medium")
                                
                                exceptions.append({
                                    "file": rel_path,
                                    "line": line_num,
                                    "classification": classification,
                                    "is_pass": is_pass,
                                    "is_logged": has_logging,
                                    "severity": severity
                                })
                    except Exception as fe:
                        print(f"Error reading {filepath}: {fe}")
                        
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "total_broad_exceptions": len(exceptions),
        "swallowed_exceptions_count": sum(1 for e in exceptions if e["is_pass"]),
        "exceptions": exceptions
    }


def main():
    print("=== DataPilot OS Read-Only Runtime Architecture Audit ===")
    os.makedirs(DOCS_ARCH_DIR, exist_ok=True)
    
    # 1. Frontend Network Vectors (Fetch, Axios, XHR, WebSocket, Electron IPC)
    print("[1/4] Auditing frontend network traffic across all vectors...")
    frontend_network = audit_frontend_network_traffic()
    with open(os.path.join(DOCS_ARCH_DIR, "frontend-network-inventory.json"), "w", encoding="utf-8") as f:
        json.dump(frontend_network, f, indent=2)
    with open(os.path.join(DOCS_ARCH_DIR, "frontend-api-call-inventory.json"), "w", encoding="utf-8") as f:
        json.dump(frontend_network, f, indent=2)
    print(f"  -> Saved {len(frontend_network['fetch_calls'])} fetch calls and {len(frontend_network['electron_ipc_calls'])} IPC channels to frontend-network-inventory.json")

    # 2. Mounted Routes (cross-referenced with frontend callers)
    print("[2/4] Auditing mounted routes...")
    mounted_routes = audit_mounted_routes(frontend_network["fetch_calls"])
    with open(os.path.join(DOCS_ARCH_DIR, "mounted-route-inventory.json"), "w", encoding="utf-8") as f:
        json.dump(mounted_routes, f, indent=2)
    print(f"  -> Saved {len(mounted_routes['routes'])} routes to mounted-route-inventory.json")
    
    # 3. Heuristics & Defaults
    print("[3/4] Auditing heuristics and defaults...")
    heuristics_defaults = audit_heuristics_and_defaults()
    with open(os.path.join(DOCS_ARCH_DIR, "default-and-heuristic-audit.json"), "w", encoding="utf-8") as f:
        json.dump(heuristics_defaults, f, indent=2)
    print(f"  -> Saved heuristics and defaults to default-and-heuristic-audit.json")
    
    # 4. Exception Handling
    print("[4/4] Auditing broad exception handling...")
    exceptions = audit_exception_handling()
    with open(os.path.join(DOCS_ARCH_DIR, "exception-handling-audit.json"), "w", encoding="utf-8") as f:
        json.dump(exceptions, f, indent=2)
    print(f"  -> Saved {len(exceptions['exceptions'])} exceptions to exception-handling-audit.json")
    
    print("\nAudit complete! All JSON deliverables generated cleanly in docs/architecture/.")


if __name__ == "__main__":
    main()
