import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live_ollama: mark test as requiring a live Ollama server or live model connection"
    )


def pytest_collection_modifyitems(config, items):
    run_live = os.getenv("DATAPILOT_RUN_LIVE_OLLAMA", "").strip() in ("1", "true", "True", "yes")
    skip_live = pytest.mark.skip(
        reason="Skipping live Ollama test because DATAPILOT_RUN_LIVE_OLLAMA is not set to 1"
    )

    live_file_prefixes = (
        "test_live_ollama_",
        "test_v2_live_",
        "test_massive_ollama_",
        "test_live_toolspec_",
    )
    live_exact_files = {
        "test_broad_regression_matrix.py",
        "test_multi_intent_pipeline.py",
        "test_tristate_ordering.py",
        "test_ambiguity_and_synonyms.py",
        "test_shadow_mode_guard.py",
    }

    for item in items:
        # Auto-detect live Ollama test files by filename
        file_name = os.path.basename(str(item.fspath))
        if any(file_name.startswith(prefix) for prefix in live_file_prefixes) or file_name in live_exact_files:
            item.add_marker(pytest.mark.live_ollama)

        if "live_ollama" in item.keywords:
            if not run_live:
                item.add_marker(skip_live)
