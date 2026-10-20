"""
Phase 7 Test Suite: Natural Language to SQL Pipeline.
Verifies:
1. Deterministic compilation of structured SQLIntent into DuckDB SQL
2. Schema grounding & identifier quoting
3. Rejection of unknown columns without guessing or fuzzy matching
4. Integration with GovernedSQLOperator in TypedOperatorRegistry
"""
import pytest
import pandas as pd
import numpy as np

from backend.sql_engine.sql_models import SQLIntent
from backend.sql_engine.sql_builder import DeterministicSQLBuilder
from backend.ds_engine.typed_operators import TypedOperatorRegistry
from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily


@pytest.fixture
def hr_dataset():
    return pd.DataFrame({
        "employee_id": [101, 102, 103, 104, 105],
        "department_name": ["Sales", "Engineering", "Sales", "HR", "Engineering"],
        "annual_salary": [75000, 120000, 82000, 65000, 135000],
        "years_experience": [3, 8, 4, 2, 10],
        "performance_score": [3.8, 4.5, 4.0, 3.5, 4.9]
    })


def test_nl_intent_aggregation_compilation(hr_dataset):
    """Verifies that an intent for average salary by department compiles deterministically."""
    intent = SQLIntent(
        operation="aggregate",
        selected_columns=["department_name"],
        aggregations=[{"column": "annual_salary", "func": "avg", "alias": "avg_salary"}],
        group_by=["department_name"],
        order_by=[{"column": "avg_salary", "direction": "desc"}],
        limit=5
    )

    compiled_sql = DeterministicSQLBuilder.build(intent, hr_dataset)
    assert 'SELECT department_name, AVG(annual_salary) AS avg_salary FROM active_dataset' in compiled_sql
    assert 'GROUP BY department_name' in compiled_sql
    assert 'ORDER BY avg_salary DESC' in compiled_sql
    assert 'LIMIT 5' in compiled_sql


def test_nl_intent_filter_and_selection(hr_dataset):
    """Verifies that filter intents compile to grounded SQL with valid operators."""
    intent = SQLIntent(
        operation="select",
        selected_columns=["employee_id", "department_name", "annual_salary"],
        filters=[{"column": "years_experience", "operator": ">=", "value": 4}],
        order_by=[{"column": "annual_salary", "direction": "desc"}]
    )

    compiled_sql = DeterministicSQLBuilder.build(intent, hr_dataset)
    assert 'SELECT employee_id, department_name, annual_salary FROM active_dataset' in compiled_sql
    assert 'WHERE years_experience >= 4' in compiled_sql
    assert 'ORDER BY annual_salary DESC' in compiled_sql


def test_nl_intent_rejects_unknown_columns(hr_dataset):
    """Verifies that referencing a non-existent column raises structured ValueError without guessing."""
    intent = SQLIntent(
        operation="select",
        selected_columns=["employee_id", "bonus_amount"],  # 'bonus_amount' doesn't exist
    )

    with pytest.raises(ValueError) as excinfo:
        DeterministicSQLBuilder.build(intent, hr_dataset)
    
    msg = str(excinfo.value)
    assert "Column 'bonus_amount' does not exist in active dataset" in msg
    assert "Available columns:" in msg
    assert "employee_id" in msg
    assert "annual_salary" in msg


def test_typed_operator_registry_executes_sql_intent(hr_dataset):
    """Verifies that GovernedSQLOperator executes SQLIntent end-to-end via TypedOperatorRegistry."""
    intent_v2 = AnalysisIntentV2(
        task_family=TaskFamily.SQL_QUERY,
        arguments={
            "sql_intent": {
                "operation": "aggregate",
                "selected_columns": ["department_name"],
                "aggregations": [{"column": "annual_salary", "func": "avg", "alias": "mean_sal"}],
                "group_by": ["department_name"],
                "order_by": [{"column": "mean_sal", "direction": "desc"}]
            }
        }
    )

    output = TypedOperatorRegistry.execute_intent(hr_dataset, intent_v2)
    assert output.status == "SUCCEEDED"
    assert output.dataframe is not None
    assert len(output.dataframe) == 3  # Sales, Engineering, HR
    assert "mean_sal" in output.dataframe.columns
    assert output.manifest.operator == "sql_query"
