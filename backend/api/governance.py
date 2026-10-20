"""
DataPilot OS — Governance & Audit API Router (Phase 2B).
Exposes immutable provenance records for governed data operations.
Completely dataset-agnostic and domain-agnostic.
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from backend.persistence.provenance_store import provenance_store

router = APIRouter(prefix="/api/governance", tags=["governance"])


@router.get("/audit")
def get_audit_trail(
    dataset_id: Optional[str] = Query(None, description="Optional dataset ID filter"),
    limit: int = Query(50, ge=1, le=1000, description="Max events to return")
):
    """
    Returns the immutable audit log of governed data cleaning and transformation operations.
    Each event contains before/after state snapshots, fingerprints, and affected columns.
    """
    events = provenance_store.get_events(dataset_id=dataset_id)
    # Return most recent first
    recent_events = list(reversed(events))[:limit]
    return {
        "status": "success",
        "total": len(events),
        "count": len(recent_events),
        "dataset_id": dataset_id,
        "events": [e.model_dump() for e in recent_events]
    }


@router.get("/audit/{event_id}")
def get_audit_event(event_id: str):
    """Returns a specific provenance event by its unique ID."""
    events = provenance_store.get_events()
    event = next((e for e in events if e.event_id == event_id), None)
    if not event:
        raise HTTPException(status_code=404, detail=f"Provenance event '{event_id}' not found")
    return event.model_dump()
