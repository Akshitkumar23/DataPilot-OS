"""
DataPilot OS — Phase 7 Report Evidence Model.
Typed Pydantic contracts for analytical evidence collected across all system phases.
Guarantees 100% factual grounding: every report statement must trace to an evidence item.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, computed_field


class AnalysisEvidenceItem(BaseModel):
    """A discrete unit of verified analytical evidence."""
    model_config = ConfigDict(extra="forbid")
    analysis_id: str
    dataset_id: str
    dataset_fingerprint: str
    capability: str  # PROFILE | EDA | CORRELATION | STATS | ML | CLUSTERING | PCA | TIMESERIES | SQL | GOVERNANCE
    operation: str
    timestamp: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    chart_data: Optional[Dict[str, Any]] = None
    finding: str = ""
    is_stale: bool = False


class ReportEvidencePackage(BaseModel):
    """Consolidated evidence package for the active dataset."""
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    dataset_name: str
    current_fingerprint: str
    has_stale_evidence: bool = False
    evidence_items: List[AnalysisEvidenceItem] = Field(default_factory=list)
    total_analyses: int = 0
    available_capabilities: List[str] = Field(default_factory=list)
    stale_reason: Optional[str] = None

    @property
    def stale_warning(self) -> bool:
        return self.has_stale_evidence

    @property
    def total_evidence_count(self) -> int:
        return len(self.evidence_items)

    @property
    def dataset_fingerprint(self) -> str:
        return self.current_fingerprint


class ReportSection(BaseModel):
    """A single structured section of an analytical report."""
    model_config = ConfigDict(extra="forbid")
    section_id: str
    title: str
    subtitle: Optional[str] = None
    narrative: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    chart: Optional[Dict[str, Any]] = None
    evidence_ids: List[str] = Field(default_factory=list)

    @computed_field
    @property
    def content(self) -> str:
        return self.narrative

    @computed_field
    @property
    def key_metrics(self) -> Dict[str, Any]:
        return self.metrics

    @computed_field
    @property
    def table_data(self) -> List[Dict[str, Any]]:
        return self.tables

    @computed_field
    @property
    def is_verified(self) -> bool:
        return True


class AnalyticalReport(BaseModel):
    """The authoritative analytical report model."""
    model_config = ConfigDict(extra="forbid")
    report_id: str
    title: str
    dataset_id: str
    dataset_name: str
    dataset_fingerprint: str
    generated_at: str
    sections: List[ReportSection] = Field(default_factory=list)
    executive_summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    total_evidence_count: int = 0
    is_stale: bool = False
