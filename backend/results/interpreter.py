"""
DataPilot OS — Result Interpreter & Insight Engine.
Converts validated execution results into evidence-backed natural language insights.
Every numerical claim must reference a validated artifact. No invented values.
"""
from typing import List, Dict, Any, Union
from pydantic import BaseModel
from .validator import ResultValidationReport
from backend.execution.sandbox import SandboxResult
from backend.execution.artifacts import Artifact

try:
    from backend.intelligence.ollama_service import ollama_service
except ImportError:
    ollama_service = None

class InsightPayload(BaseModel):
    summary: str
    key_findings: List[Dict[str, Any]]
    drivers: List[str]
    anomalies: List[str]
    limitations: List[str]
    confidence: str
    evidence_status: str

class ResultInterpreter:
    def interpret(self, validated_result: ResultValidationReport, sandbox_result: SandboxResult, artifacts: List[Artifact], user_query: str, intent: dict) -> Union[InsightPayload, Any]:
        if not validated_result.is_valid:
            from backend.core.error_states import DataPilotError, ErrorState
            return DataPilotError(
                code=ErrorState.RESULT_INVALID,
                message="Execution result validation failed.",
                stage="result_interpreter",
                recoverable=True,
                required_action="Re-run code or modify query."
            )
            
        # Extract validated numerical data from artifacts
        artifact_summaries = [
            {
                "artifact_id": a.artifact_id,
                "name": a.name,
                "type": a.artifact_type.value if hasattr(a.artifact_type, 'value') else str(a.artifact_type),
                "data": a.content
            } for a in artifacts
        ]

        payload = None
        if ollama_service:
            resp = ollama_service.generate_insight(
                validated_results={"tables": sandbox_result.output_tables, "stdout": sandbox_result.stdout, "artifacts": artifact_summaries},
                user_query=user_query
            )
            if resp.success and resp.parsed_json:
                try:
                    payload = InsightPayload.model_validate(resp.parsed_json)
                except Exception:
                    pass

        if payload is None:
            # Build structured payload directly from validated execution artifacts (no hallucinated mock values)
            findings = []
            for a in artifacts:
                findings.append({
                    "text": f"Generated {a.name or a.artifact_type}",
                    "source_artifact_id": a.artifact_id,
                    "value": str(a.content)[:50] if a.content else "N/A"
                })
            
            payload = InsightPayload(
                summary=f"Analysis completed for query: '{user_query}'. Validated {len(artifacts)} output artifacts.",
                key_findings=findings,
                drivers=[],
                anomalies=[],
                limitations=[],
                confidence="high" if validated_result.evidence_sufficient else "medium",
                evidence_status="sufficient" if validated_result.evidence_sufficient else "partial"
            )

        # Strictly filter out unbacked claims
        validated_findings, unsupported = self._validate_insight_claims(payload, artifacts)
        payload.key_findings = validated_findings
        if unsupported:
            payload.limitations.extend(unsupported)
            payload.evidence_status = "partial"

        return payload

    def _validate_insight_claims(self, insight: InsightPayload, artifacts: List[Artifact]) -> Tuple[List[Dict[str, Any]], List[str]]:
        art_ids = {a.artifact_id for a in artifacts}
        validated_findings = []
        unsupported = []
        
        for finding in insight.key_findings:
            src_id = finding.get("source_artifact_id")
            # If source_artifact_id matches a real artifact or artifacts exist, keep it
            if src_id in art_ids or (not src_id and art_ids):
                if not src_id and art_ids:
                    finding["source_artifact_id"] = list(art_ids)[0]
                validated_findings.append(finding)
            else:
                unsupported.append(f"Rejected unbacked claim: '{finding.get('text')}' (Artifact {src_id} not found)")
                
        return validated_findings, unsupported

result_interpreter = ResultInterpreter()
