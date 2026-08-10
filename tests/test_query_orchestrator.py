import unittest
import pandas as pd
import json
from unittest.mock import patch, MagicMock

from modules.schema_analyzer import SchemaAnalyzer
from modules.models.semantic_plan import SemanticPlan, Operation, Visualization
from modules.models.execution_plan import ExecutionPlan, ExecutionStep
from modules.models.execution_result import ExecutionStatus
from modules.semantic_planner import SemanticPlanner
from modules.plan_normalizer import PlanNormalizer
from modules.plan_validator import PlanValidator
from modules.query_repair import QueryRepair
from modules.result_validator import ResultValidator
from modules.emergency_fallback import EmergencyFallback
from modules.query_orchestrator import QueryOrchestrator

class TestQueryOrchestratorArchitecture(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "Salary": [50000, 60000, 75000, 90000, 110000, 125000],
            "Department": ["HR", "IT", "Finance", "IT", "Finance", "HR"],
            "Age": [25, 30, 35, 40, 45, 50],
            "Experience": [2, 5, 8, 12, 15, 20]
        })

    def test_1_schema_analyzer_and_caching(self):
        """Verify schema extraction and hash caching."""
        schema1 = SchemaAnalyzer.analyze(self.df)
        self.assertEqual(schema1["rows"], 6)
        self.assertIn("Salary", schema1["columns"])
        self.assertEqual(schema1["columns"]["Salary"]["semantic_type"], "currency")

        # Second call should hit cache
        schema2 = SchemaAnalyzer.analyze(self.df)
        self.assertEqual(schema1, schema2)

    def test_2_semantic_meaning_without_keywords(self):
        """Verify natural language meaning extraction without explicit chart keywords."""
        plan = SemanticPlan(
            goal="analyze_salary_concentration",
            operations=[Operation(operation="distribution", target="Salary")]
        )
        exec_plan = PlanNormalizer.normalize(plan)
        self.assertEqual(len(exec_plan.steps), 1)
        self.assertEqual(exec_plan.steps[0].capability, "sig_hist")
        self.assertEqual(exec_plan.steps[0].parameters["target"], "Salary")

        val_res = PlanValidator.validate(exec_plan, df=self.df)
        self.assertTrue(val_res.is_valid)

    def test_3_self_repair_loop_invalid_column(self):
        """Verify Goal-Preserving Self-Repair fixes invalid column Salary_USD -> Salary."""
        invalid_plan = SemanticPlan(
            goal="analyze_salary_distribution",
            operations=[Operation(operation="distribution", target="Salary_USD")]
        )
        invalid_exec_plan = PlanNormalizer.normalize(invalid_plan)
        val_res = PlanValidator.validate(invalid_exec_plan, df=self.df)
        self.assertFalse(val_res.is_valid)

        # Mock Ollama repair returning corrected plan
        repaired_plan = SemanticPlan(
            goal="analyze_salary_distribution",
            operations=[Operation(operation="distribution", target="Salary")]
        )
        with patch.object(QueryRepair, '_call_ollama_repair', return_value=repaired_plan):
            repaired_exec_plan = QueryRepair.repair_loop(
                query_raw="Show salary distribution",
                initial_plan=invalid_plan,
                validation_res=val_res,
                schema=SchemaAnalyzer.analyze(self.df),
                df=self.df
            )
            self.assertIsNotNone(repaired_exec_plan)
            self.assertEqual(repaired_exec_plan.steps[0].parameters["target"], "Salary")

    def test_4_offline_ollama_fallback(self):
        """Verify offline Ollama triggers Emergency Recovery Safety Net."""
        with patch.object(SemanticPlanner, 'plan', return_value=None):
            res = QueryOrchestrator.process_query("how salaries are distributed", df=self.df)
            self.assertIsNotNone(res)
            self.assertIn(res.status, [ExecutionStatus.SUCCESS, ExecutionStatus.UNSUPPORTED])

    def test_5_result_validator_structural_checks(self):
        """Verify ResultValidator structural checks."""
        raw_res = {
            "status": "success",
            "response": "Histogram rendered for Salary",
            "code": "plt.show()",
            "plot_image": "base64_data_string"
        }
        res = ResultValidator.validate_result(raw_res)
        self.assertEqual(res.status, ExecutionStatus.SUCCESS)
        self.assertEqual(res.plot_image, "base64_data_string")

if __name__ == "__main__":
    unittest.main()
