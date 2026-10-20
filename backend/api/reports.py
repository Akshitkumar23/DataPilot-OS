"""
DataPilot OS — Phase 7 Governed Analytical Reporting API Router.
Endpoints:
- GET /api/reports/evidence: Collect all verified analytical evidence for the active dataset.
- POST /api/reports/generate: Synthesize verified evidence into an AnalyticalReport.
- GET /api/reports/latest: Retrieve the latest report for a dataset.
- POST /api/reports/export: Export the report as publication-grade HTML or JSON.
"""
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field
import pandas as pd

from backend.core.dataset_registry import dataset_registry
from backend.reporting.evidence_model import (
    ReportEvidencePackage,
    AnalyticalReport,
)
from backend.reporting.evidence_collector import ReportEvidenceCollector
from backend.reporting.report_generator import GovernedReportGenerator
from backend.reporting.report_exporter import ReportExporter

router = APIRouter(prefix="/api/reports", tags=["reports"])

# In-memory cache for latest generated reports per dataset and by report_id
_LATEST_REPORTS: Dict[str, AnalyticalReport] = {}
_REPORTS_BY_ID: Dict[str, AnalyticalReport] = {}


class ReportGenerateRequest(BaseModel):
    dataset_id: str
    title: Optional[str] = None


class ReportExportRequest(BaseModel):
    dataset_id: str
    title: Optional[str] = None
    format: str = "html"  # "html" | "json"


def _get_dataset_and_df(dataset_id: str):
    if not dataset_registry:
        raise HTTPException(status_code=500, detail="Dataset registry unavailable")
    entry = dataset_registry.get(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found in registry")
    df = dataset_registry.get_dataframe(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"DataFrame for dataset '{dataset_id}' not found in memory")
    return entry, df


@router.get("/evidence", response_model=ReportEvidencePackage)
def get_report_evidence(dataset_id: str = Query(...)):
    """Retrieve all verified evidence items for the active dataset."""
    entry, df = _get_dataset_and_df(dataset_id)
    return ReportEvidenceCollector.collect(dataset_id, df, entry.name)


@router.post("/generate", response_model=AnalyticalReport)
def generate_analytical_report(req: ReportGenerateRequest):
    """Generate a structured, evidence-grounded analytical report."""
    entry, df = _get_dataset_and_df(req.dataset_id)
    report = GovernedReportGenerator.generate(
        dataset_id=req.dataset_id,
        df=df,
        dataset_name=entry.name,
        title=req.title,
        record_provenance=True
    )
    _LATEST_REPORTS[req.dataset_id] = report
    _REPORTS_BY_ID[report.report_id] = report
    return report


@router.get("/latest", response_model=AnalyticalReport)
def get_latest_report(dataset_id: str = Query(...)):
    """Retrieve the most recent generated report for a dataset, or generate if none exists."""
    if dataset_id in _LATEST_REPORTS:
        return _LATEST_REPORTS[dataset_id]
    entry, df = _get_dataset_and_df(dataset_id)
    report = GovernedReportGenerator.generate(
        dataset_id=dataset_id,
        df=df,
        dataset_name=entry.name,
        record_provenance=True
    )
    _LATEST_REPORTS[dataset_id] = report
    _REPORTS_BY_ID[report.report_id] = report
    return report


@router.get("/{report_id}/export")
def export_report_by_id(report_id: str, format: str = Query("html")):
    """Export an existing report by its report_id as standalone HTML or structured JSON."""
    report = _REPORTS_BY_ID.get(report_id)
    if not report:
        for r in _LATEST_REPORTS.values():
            if r.report_id == report_id:
                report = r
                break
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    if format.lower() == "json":
        return report.model_dump()

    html_content = ReportExporter.to_html(report)
    safe_name = report.dataset_name.replace(".", "_")
    return Response(content=html_content, media_type="text/html", headers={
        "Content-Disposition": f"attachment; filename={safe_name}_report_{report_id}.html"
    })


@router.get("/export")
def export_report_by_dataset(dataset_id: str = Query(...), format: str = Query("html")):
    """Export the report for dataset_id as standalone HTML or structured JSON."""
    entry, df = _get_dataset_and_df(dataset_id)
    report = _LATEST_REPORTS.get(dataset_id)
    if not report:
        report = GovernedReportGenerator.generate(
            dataset_id=dataset_id,
            df=df,
            dataset_name=entry.name,
            record_provenance=False
        )
        _LATEST_REPORTS[dataset_id] = report
        _REPORTS_BY_ID[report.report_id] = report

    if format.lower() == "json":
        return report.model_dump()

    html_content = ReportExporter.to_html(report)
    safe_name = entry.name.replace(".", "_")
    return Response(content=html_content, media_type="text/html", headers={
        "Content-Disposition": f"attachment; filename={safe_name}_report.html"
    })


@router.post("/export")
def export_analytical_report(req: ReportExportRequest):
    """Export the report as standalone HTML or structured JSON."""
    entry, df = _get_dataset_and_df(req.dataset_id)
    report = _LATEST_REPORTS.get(req.dataset_id)
    if not report:
        report = GovernedReportGenerator.generate(
            dataset_id=req.dataset_id,
            df=df,
            dataset_name=entry.name,
            title=req.title,
            record_provenance=False
        )
        _LATEST_REPORTS[req.dataset_id] = report
        _REPORTS_BY_ID[report.report_id] = report

    if req.format.lower() == "json":
        return report.model_dump()

    html_content = ReportExporter.to_html(report)
    safe_name = entry.name.replace(".", "_")
    return Response(content=html_content, media_type="text/html", headers={
        "Content-Disposition": f"attachment; filename={safe_name}_report.html"
    })
