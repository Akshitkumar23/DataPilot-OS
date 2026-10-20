"""
DataPilot OS — 7-Step Code Validation Pipeline.
Invalid code NEVER reaches the execution engine.

Validation pipeline:
1. Python syntax check
2. AST structural & safety analysis
3. Import whitelist check (AST-based, handles submodules & blocks relative imports)
4. Domain-agnostic dataset column reference validation (AST-based with dynamic temporal derivations)
5. Capability contract compliance
6. Output contract compliance
7. Security check (AST-based call & attribute blocker + regex fallback)
"""

import ast
import re
from typing import List, Optional, Set, Dict, Any
from pydantic import BaseModel

from .code_generator import CodeVersion

class ValidationIssue(BaseModel):
    step: int
    severity: str
    issue_type: str
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None

class CodeValidationResult(BaseModel):
    is_valid: bool
    issues: List[ValidationIssue]
    can_autofix: bool
    autofixed_code: Optional[str] = None
    steps_passed: List[int]
    steps_failed: List[int]

ALLOWED_TOP_MODULES = {
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'scipy',
    'statsmodels', 'plotly', 'polars', 'duckdb', 'pyarrow',
    'warnings', 'math', 'statistics', 'datetime', 'json', 're',
    'collections', 'itertools', 'functools', 'typing', 'pathlib',
    'io', 'base64', 'hashlib', 'copy', 'dateutil', 'typing_extensions',
    'dataclasses', 'time', 'string', 'random', 'os'
}

ALLOWED_EXPLICIT_SUBMODULES = {
    'os.path'
}

FORBIDDEN_CALL_NAMES = {
    'exec', 'eval', 'open', 'compile', '__import__',
    'globals', 'locals', 'input', 'breakpoint', 'exit', 'quit'
}

FORBIDDEN_ATTRIBUTES = {
    'os.system', 'os.popen', 'os.remove', 'os.unlink', 'os.rmdir',
    'os.environ', 'os.kill', 'shutil.rmtree', 'shutil.move',
    'pickle.loads', 'pickle.load', 'sys.exit', 'sys.modules'
}

FORBIDDEN_MODULE_ROOTS = {
    'subprocess', 'socket', 'requests', 'urllib', 'http', 'ftplib',
    'ctypes', 'marshal', 'pickle', 'telnetlib', 'smtplib', 'asyncio'
}

GENERIC_TEMPORAL_COLS = {
    'Year', 'Year_Month', 'Month', 'Day', 'Quarter', 'Date',
    'Hour', 'Minute', 'Second', 'DayOfWeek', 'Week', 'Trend', 'Order_Date_dt', 'Ship_Date_dt', 'temp_dt'
}


class ASTValidationVisitor(ast.NodeVisitor):
    def __init__(self, dataset_columns: List[str]):
        self.dataset_columns = set(dataset_columns)
        self.assigned_df_cols: Set[str] = set()
        self.accessed_df_cols: List[tuple[str, int]] = []
        self.import_violations: List[tuple[str, int]] = []
        self.security_violations: List[tuple[str, int]] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            mod = alias.name
            root = mod.split('.')[0]
            if mod not in ALLOWED_EXPLICIT_SUBMODULES and root not in ALLOWED_TOP_MODULES:
                self.import_violations.append((f"Import not allowed: {mod}", node.lineno))
            if root in FORBIDDEN_MODULE_ROOTS:
                self.security_violations.append((f"Forbidden module imported: {mod}", node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level and node.level > 0:
            self.import_violations.append(("Relative imports not allowed", node.lineno))
            return
        if node.module:
            mod = node.module
            root = mod.split('.')[0]
            if mod not in ALLOWED_EXPLICIT_SUBMODULES and root not in ALLOWED_TOP_MODULES:
                self.import_violations.append((f"Import not allowed: {mod}", node.lineno))
            if root in FORBIDDEN_MODULE_ROOTS:
                self.security_violations.append((f"Forbidden module imported: {mod}", node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALL_NAMES:
                self.security_violations.append((f"Forbidden function call: {node.func.id}()", node.lineno))
        elif isinstance(node.func, ast.Attribute):
            attr_full = self._get_full_attribute_name(node.func)
            if attr_full in FORBIDDEN_ATTRIBUTES:
                self.security_violations.append((f"Forbidden method call: {attr_full}()", node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            self._extract_df_assignment_targets(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.target:
            self._extract_df_assignment_targets(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._extract_df_assignment_targets(node.target)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == 'df':
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                self.accessed_df_cols.append((slice_node.value, node.lineno))
            elif isinstance(slice_node, ast.List):
                for elt in slice_node.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        self.accessed_df_cols.append((elt.value, node.lineno))
        self.generic_visit(node)

    def _extract_df_assignment_targets(self, target: ast.AST):
        if isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name) and target.value.id == 'df':
                if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                    self.assigned_df_cols.add(target.slice.value)
                elif isinstance(target.slice, ast.List):
                    for elt in target.slice.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            self.assigned_df_cols.add(elt.value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._extract_df_assignment_targets(elt)

    def _get_full_attribute_name(self, node: ast.Attribute) -> str:
        parts = [node.attr]
        curr = node.value
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
        return ".".join(reversed(parts))


class CodeValidator:
    def validate(self, code_version: CodeVersion, dataset_context: dict) -> CodeValidationResult:
        code = code_version.code
        dataset_columns = dataset_context.get("columns", [])
        issues: List[ValidationIssue] = []
        steps_passed: List[int] = []
        steps_failed: List[int] = []

        # Step 1: Syntax Check
        step1_issues = self._step1_syntax(code)
        if step1_issues:
            issues.extend(step1_issues)
            steps_failed.append(1)
            return CodeValidationResult(is_valid=False, issues=issues, can_autofix=False, steps_passed=steps_passed, steps_failed=steps_failed)
        steps_passed.append(1)

        # Step 2: AST Parse & Structural Analysis
        step2_issues, tree = self._step2_ast(code)
        if step2_issues:
            issues.extend(step2_issues)
            steps_failed.append(2)
            return CodeValidationResult(is_valid=False, issues=issues, can_autofix=False, steps_passed=steps_passed, steps_failed=steps_failed)
        steps_passed.append(2)

        # Run unified AST Visitor
        visitor = ASTValidationVisitor(dataset_columns)
        if tree:
            visitor.visit(tree)

        # Step 3: Imports Validation
        step3_issues = self._step3_imports_from_visitor(visitor)
        issues.extend(step3_issues)
        if any(i.severity == 'error' for i in step3_issues):
            steps_failed.append(3)
            return CodeValidationResult(is_valid=False, issues=issues, can_autofix=False, steps_passed=steps_passed, steps_failed=steps_failed)
        steps_passed.append(3)

        # Step 4: Column Reference Validation
        step4_issues = self._step4_column_refs_from_visitor(visitor, dataset_columns)
        issues.extend(step4_issues)
        can_autofix = False
        autofixed_code = None
        if any(i.severity == 'error' for i in step4_issues):
            steps_failed.append(4)
            fixed = self._autofix_column_names(code, dataset_columns)
            if fixed and fixed != code:
                can_autofix = True
                autofixed_code = fixed
            return CodeValidationResult(is_valid=False, issues=issues, can_autofix=can_autofix, autofixed_code=autofixed_code, steps_passed=steps_passed, steps_failed=steps_failed)
        steps_passed.append(4)

        # Step 5: Capability Contract Compliance
        step5_issues = self._step5_capability_contract(code_version)
        issues.extend(step5_issues)
        if any(i.severity == 'error' for i in step5_issues):
            steps_failed.append(5)
            return CodeValidationResult(is_valid=False, issues=issues, can_autofix=False, steps_passed=steps_passed, steps_failed=steps_failed)
        steps_passed.append(5)

        # Step 6: Output Contract Compliance
        step6_issues = self._step6_output_contract(code_version)
        issues.extend(step6_issues)
        if any(i.severity == 'error' for i in step6_issues):
            steps_failed.append(6)
            return CodeValidationResult(is_valid=False, issues=issues, can_autofix=False, steps_passed=steps_passed, steps_failed=steps_failed)
        steps_passed.append(6)

        # Step 7: Security Check
        step7_issues = self._step7_security_from_visitor(visitor, code)
        issues.extend(step7_issues)
        if any(i.severity == 'error' for i in step7_issues):
            steps_failed.append(7)
            return CodeValidationResult(is_valid=False, issues=issues, can_autofix=False, steps_passed=steps_passed, steps_failed=steps_failed)
        steps_passed.append(7)

        return CodeValidationResult(
            is_valid=True,
            issues=issues,
            can_autofix=False,
            steps_passed=steps_passed,
            steps_failed=steps_failed
        )

    def _step1_syntax(self, code: str) -> List[ValidationIssue]:
        try:
            compile(code, '<string>', 'exec')
            return []
        except (SyntaxError, ValueError) as e:
            lineno = getattr(e, 'lineno', None)
            return [ValidationIssue(step=1, severity='error', issue_type='SyntaxError', message=str(e), line_number=lineno)]

    def _step2_ast(self, code: str) -> tuple[List[ValidationIssue], Optional[ast.AST]]:
        try:
            tree = ast.parse(code)
            if not tree.body:
                return [ValidationIssue(step=2, severity='error', issue_type='ASTError', message="Empty code body")], None
            return [], tree
        except Exception as e:
            return [ValidationIssue(step=2, severity='error', issue_type='ASTError', message=str(e))], None

    def _step3_imports_from_visitor(self, visitor: ASTValidationVisitor) -> List[ValidationIssue]:
        return [
            ValidationIssue(step=3, severity='error', issue_type='ImportViolation', message=msg, line_number=line)
            for msg, line in visitor.import_violations
        ]

    def _step4_column_refs_from_visitor(self, visitor: ASTValidationVisitor, dataset_columns: List[str]) -> List[ValidationIssue]:
        issues = []
        raw_cols = set(dataset_columns)
        
        # Build dynamic derived column whitelist based on dataset columns
        derived_temporal_cols = {f"{c}_{suffix}" for c in raw_cols for suffix in ['dt', 'year', 'month', 'day', 'quarter', 'hour']}
        
        allowed_cols = raw_cols | visitor.assigned_df_cols | derived_temporal_cols | GENERIC_TEMPORAL_COLS

        for col, lineno in visitor.accessed_df_cols:
            if col not in allowed_cols:
                issues.append(ValidationIssue(
                    step=4,
                    severity='error',
                    issue_type='InvalidColumn',
                    message=f"Column '{col}' not found in dataset or derived columns.",
                    line_number=lineno
                ))
        return issues

    def _step5_capability_contract(self, code_version: CodeVersion) -> List[ValidationIssue]:
        issues = []
        code = code_version.code
        purpose = code_version.purpose.upper()
        if "FORECAST" in purpose and not any(k in code for k in ["date_range", "dt", "datetime", "iloc"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Forecast code missing temporal date range or indexing logic."))
        elif "AUTO_ML" in purpose and not any(k in code for k in ["fit(", "train_test_split", "score"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="AutoML code missing model fitting or scoring logic."))
        elif "GROUP" in purpose and not any(k in code for k in ["groupby", "agg(", "value_counts"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Aggregation code missing groupby or aggregation logic."))
        elif "CLEAN" in purpose and not any(k in code for k in ["drop_duplicates", "dropna", "fillna", "isnull"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Data cleaning code missing cleaning or null handling logic."))
        elif "AUTOCORRELATION" in purpose and not any(k in code for k in ["acf", "pacf", "plot_acf", "plot_pacf"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Autocorrelation code missing ACF/PACF logic."))
        elif "CORRELATION" in purpose and "AUTOCORRELATION" not in purpose and not any(k in code for k in ["corr(", "corr", "heatmap", "pearson"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Correlation code missing correlation matrix or heatmap logic."))
        elif "STATISTICAL_TEST" in purpose and not any(k in code for k in ["ttest", "pvalue", "f_oneway", "chi2", "shapiro", "stats"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Statistical test code missing testing logic (ttest, pvalue, etc.)."))
        elif "ANOMALY_DETECTION" in purpose and not any(k in code for k in ["IsolationForest", "LocalOutlierFactor", "zscore", "quantile", "outlier"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Anomaly detection code missing outlier detection logic."))
        elif "CLUSTERING" in purpose and not any(k in code for k in ["KMeans", "DBSCAN", "Agglomerative", "fit_predict", "cluster"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Clustering code missing clustering logic."))
        elif "PCA" in purpose and not any(k in code for k in ["PCA", "fit_transform", "components_"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="PCA code missing PCA logic."))
        elif "REGRESSION" in purpose and not any(k in code for k in ["LinearRegression", "Ridge", "Lasso", "RandomForestRegressor", "mean_squared_error", "r2_score"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Regression code missing regression model logic."))
        elif "CLASSIFICATION" in purpose and not any(k in code for k in ["LogisticRegression", "RandomForestClassifier", "accuracy_score", "predict", "confusion_matrix"]):
            issues.append(ValidationIssue(step=5, severity='error', issue_type='CapabilityContractViolation', message="Classification code missing classifier logic."))
        return issues

    def _step6_output_contract(self, code_version: CodeVersion) -> List[ValidationIssue]:
        issues = []
        code = code_version.code
        for contract in code_version.expected_outputs:
            c_type = contract.get("type")
            if c_type == "chart" and not any(k in code for k in ["plt.", "fig", "show()", "plot"]):
                issues.append(ValidationIssue(step=6, severity='error', issue_type='OutputContractViolation', message="Expected chart output declared but missing plot render instructions."))
            elif c_type in ("table", "metrics", "audit_summary") and not any(k in code for k in ["print(", "to_string(", "head(", "describe("]):
                issues.append(ValidationIssue(step=6, severity='error', issue_type='OutputContractViolation', message=f"Expected {c_type} output declared but missing print/display instructions."))
        return issues

    def _step7_security_from_visitor(self, visitor: ASTValidationVisitor, code: str) -> List[ValidationIssue]:
        issues = [
            ValidationIssue(step=7, severity='error', issue_type='SecurityViolation', message=msg, line_number=line)
            for msg, line in visitor.security_violations
        ]
        
        # Regex fallback for obfuscated builtins / reflection patterns
        dangerous_regexes = [
            (r'\b(exec|eval)\s*\(', "Dynamic execution pattern detected"),
            (r'\bopen\s*\(', "File system open() pattern detected"),
            (r'__builtins__|__subclasses__|getattr\s*\(', "Reflection / introspection bypass pattern detected"),
        ]
        for pattern, desc in dangerous_regexes:
            if re.search(pattern, code):
                issues.append(ValidationIssue(step=7, severity='error', issue_type='SecurityViolation', message=desc))
                
        return issues

    def _autofix_column_names(self, code: str, dataset_columns: List[str]) -> Optional[str]:
        fixed_code = code
        lower_cols = {col.lower(): col for col in dataset_columns}
        
        def repl(match):
            col = match.group(1)
            if col not in dataset_columns and col.lower() in lower_cols:
                correct = lower_cols[col.lower()]
                return f"df['{correct}']"
            return match.group(0)

        fixed_code = re.sub(r"df\[['\"](.*?)['\"]\]", repl, fixed_code)
        return fixed_code if fixed_code != code else None

code_validator = CodeValidator()
code_validator_instance = code_validator
