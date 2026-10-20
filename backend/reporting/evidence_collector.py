"""
DataPilot OS — Phase 7 Report Evidence Collector.
Harvests verified analytical evidence from the active DataFrame and the immutable provenance store.
Detects stale evidence if the dataset state or cryptographic fingerprint has diverged.
"""
from typing import List, Optional
import numpy as np
import pandas as pd

from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.persistence.provenance_store import provenance_store
from backend.reporting.evidence_model import AnalysisEvidenceItem, ReportEvidencePackage


class ReportEvidenceCollector:
    """Collects and organizes verified evidence across all phases for reporting."""

    @classmethod
    def collect(
        cls,
        dataset_id: str,
        df: pd.DataFrame,
        dataset_name: str = "active_dataset",
        profile: Optional[Any] = None,
        stale_fingerprint_check: Optional[str] = None
    ) -> ReportEvidencePackage:
        if not isinstance(dataset_name, str):
            profile = dataset_name
            dataset_name = getattr(profile, "dataset_name", "active_dataset")

        current_fp = DatasetProfiler.compute_hash(df)
        items: List[AnalysisEvidenceItem] = []
        has_stale = False
        stale_reason = None

        if stale_fingerprint_check and stale_fingerprint_check != current_fp:
            has_stale = True
            stale_reason = f"Dataset was mutated: fingerprint mismatch ({current_fp} != {stale_fingerprint_check})."

        # 1. Base Dataset Profile Evidence (always live & verified from current DataFrame)
        row_count = int(len(df))
        col_count = int(len(df.columns))
        total_cells = max(1, row_count * col_count)
        null_count = int(df.isna().sum().sum())
        null_pct = round((null_count / total_cells) * 100.0, 2)
        try:
            dup_count = int(df.astype(str).duplicated().sum())
        except Exception:
            dup_count = 0
        dup_pct = round((dup_count / max(1, row_count)) * 100.0, 2)

        num_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns]
        cat_cols = [str(c) for c in df.select_dtypes(exclude=[np.number]).columns]

        profile_ev = AnalysisEvidenceItem(
            analysis_id=f"ev_profile_{current_fp[:8]}",
            dataset_id=dataset_id,
            dataset_fingerprint=current_fp,
            capability="PROFILE",
            operation="DATASET_PROFILE",
            timestamp=pd.Timestamp.now().isoformat(),
            parameters={"columns": list(df.columns)},
            metrics={
                "row_count": row_count,
                "column_count": col_count,
                "total_cells": total_cells,
                "null_count": null_count,
                "null_percentage": null_pct,
                "duplicate_count": dup_count,
                "duplicate_percentage": dup_pct,
                "numeric_column_count": len(num_cols),
                "categorical_column_count": len(cat_cols),
                "numeric_columns": num_cols,
                "categorical_columns": cat_cols,
            },
            finding=f"Dataset contains {row_count:,} rows and {col_count} columns ({len(num_cols)} numeric, {len(cat_cols)} categorical). Missingness: {null_pct}%, Duplicates: {dup_pct}%.",
            is_stale=False
        )
        items.append(profile_ev)

        # 2. Collect Historical Provenance Events
        prov_events = provenance_store.get_events(dataset_id=dataset_id)
        for ev in prov_events:
            if ev.status not in ("SUCCESS", "SUCCEEDED"):
                continue

            # Check fingerprint alignment
            ev_input_fp = getattr(ev, "input_dataset_version", "")
            ev_output_fp = getattr(ev, "output_dataset_version", "")
            is_stale = (ev_input_fp != current_fp and ev_output_fp != current_fp)
            if is_stale:
                has_stale = True

            # Map to canonical reporting capability
            raw_cap = (ev.capability or "").upper()
            raw_op = (ev.operation or "").upper()

            if "CORRELAT" in raw_cap or "CORRELAT" in raw_op:
                rep_cap = "CORRELATION"
            elif "STAT" in raw_cap or "STAT" in raw_op or "ASSOCIATION" in raw_op:
                rep_cap = "STATS"
            elif "ML" in raw_cap or "TRAIN" in raw_op or "AUTOML" in raw_op or "PREDICT" in raw_op:
                rep_cap = "ML"
            elif "CLUSTER" in raw_cap or "KMEANS" in raw_op or "ELBOW" in raw_op:
                rep_cap = "CLUSTERING"
            elif "PCA" in raw_cap or "PCA" in raw_op or "DIMENSIONALITY" in raw_cap:
                rep_cap = "PCA"
            elif "TIME" in raw_cap or "SERIES" in raw_cap or "AUTOCORRELAT" in raw_op or "DECOMPOS" in raw_op or "FORECAST" in raw_op:
                rep_cap = "TIMESERIES"
            elif "SQL" in raw_cap or "SQL" in raw_op:
                rep_cap = "SQL"
            elif "CLEAN" in raw_cap or "TRANSFORM" in raw_cap or "DROP" in raw_op or "IMPUTE" in raw_op or "SCALE" in raw_op:
                rep_cap = "GOVERNANCE"
            else:
                rep_cap = "EDA"

            finding_text = ev.finding or (ev.result_summary.get("finding") if isinstance(ev.result_summary, dict) else "")
            if not finding_text and isinstance(ev.result_summary, dict):
                finding_text = f"Executed {ev.operation} on {', '.join(ev.affected_columns or ev.columns or [])}."

            item = AnalysisEvidenceItem(
                analysis_id=ev.event_id,
                dataset_id=dataset_id,
                dataset_fingerprint=ev_output_fp or ev_input_fp or current_fp,
                capability=rep_cap,
                operation=ev.operation,
                timestamp=ev.timestamp,
                parameters=ev.parameters or {},
                metrics=ev.result_summary if isinstance(ev.result_summary, dict) else {},
                finding=finding_text,
                is_stale=is_stale
            )
            items.append(item)

        caps = list(dict.fromkeys([i.capability for i in items]))

        return ReportEvidencePackage(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            current_fingerprint=current_fp,
            has_stale_evidence=has_stale,
            stale_reason=stale_reason,
            evidence_items=items,
            total_analyses=len(items),
            available_capabilities=caps
        )

    harvest_evidence = collect

