"""
DataPilot OS — Multi-Dataset Relationship Detection.
Detects potential relationships between connected datasets based on evidence.
NEVER automatically joins datasets. Returns proposals for user approval only.
DataPilot must never silently join datasets or modify production data.
"""

from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
import pandas as pd
from datetime import datetime

class RelationshipType(str, Enum):
    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY"
    MANY_TO_MANY = "MANY_TO_MANY"
    UNKNOWN = "UNKNOWN"

class JoinColumn(BaseModel):
    dataset_id: str
    column_name: str
    semantic_type: str
    unique_ratio: float
    null_ratio: float

class RelationshipProposal(BaseModel):
    proposal_id: str
    left_dataset_id: str
    right_dataset_id: str
    left_column: str
    right_column: str
    relationship_type: RelationshipType
    confidence: float
    evidence: List[str]
    overlap_ratio: float
    requires_user_approval: bool = True
    approved: bool = False
    approved_at: Optional[str] = None

class RelationshipDetector:
    def detect(self, datasets: Dict[str, pd.DataFrame], profiles: Dict[str, Any]) -> List[RelationshipProposal]:
        proposals = []
        dataset_ids = list(datasets.keys())
        for i in range(len(dataset_ids)):
            for j in range(i + 1, len(dataset_ids)):
                ds1_id = dataset_ids[i]
                ds2_id = dataset_ids[j]
                df1 = datasets[ds1_id]
                df2 = datasets[ds2_id]
                profile1 = profiles.get(ds1_id, {})
                profile2 = profiles.get(ds2_id, {})
                
                candidate_pairs = self._find_candidate_pairs(df1, df2, profile1, profile2)
                for col1, col2, conf in candidate_pairs:
                    overlap = self._compute_value_overlap(df1[col1], df2[col2])
                    rel_type = self._determine_relationship_type(df1, col1, df2, col2)
                    evidence = [
                        f"Column names match: {col1} and {col2}",
                        f"Value overlap: {overlap:.2%}",
                        f"Relationship Type: {rel_type.value}"
                    ]
                    
                    proposal = RelationshipProposal(
                        proposal_id=f"{ds1_id}_{col1}-{ds2_id}_{col2}",
                        left_dataset_id=ds1_id,
                        right_dataset_id=ds2_id,
                        left_column=col1,
                        right_column=col2,
                        relationship_type=rel_type,
                        confidence=conf * overlap,
                        evidence=evidence,
                        overlap_ratio=overlap
                    )
                    proposals.append(proposal)
                    
        return sorted(proposals, key=lambda p: p.confidence, reverse=True)

    def _find_candidate_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame, profile1: dict, profile2: dict) -> List[Tuple[str, str, float]]:
        candidates = []
        for col1 in df1.columns:
            for col2 in df2.columns:
                if col1.lower() == col2.lower() and df1[col1].dtype == df2[col2].dtype:
                    candidates.append((col1, col2, 0.8))
                elif col1.lower().endswith('id') and col2.lower().endswith('id') and df1[col1].dtype == df2[col2].dtype:
                    candidates.append((col1, col2, 0.6))
        return candidates

    def _compute_value_overlap(self, series1: pd.Series, series2: pd.Series) -> float:
        s1 = series1.dropna().head(10000)
        s2 = series2.dropna().head(10000)
        if len(s1) == 0:
            return 0.0
        s1_set = set(s1)
        s2_set = set(s2)
        overlap = len(s1_set.intersection(s2_set)) / len(s1_set)
        return overlap

    def _determine_relationship_type(self, df1: pd.DataFrame, col1: str, df2: pd.DataFrame, col2: str) -> RelationshipType:
        s1_unique = df1[col1].is_unique
        s2_unique = df2[col2].is_unique
        
        if s1_unique and s2_unique:
            return RelationshipType.ONE_TO_ONE
        elif s1_unique and not s2_unique:
            return RelationshipType.ONE_TO_MANY
        elif not s1_unique and s2_unique:
            return RelationshipType.ONE_TO_MANY
        else:
            return RelationshipType.MANY_TO_MANY

    def approve_relationship(self, proposal_id: str, proposals: List[RelationshipProposal]) -> RelationshipProposal:
        for proposal in proposals:
            if proposal.proposal_id == proposal_id:
                proposal.approved = True
                proposal.approved_at = datetime.utcnow().isoformat()
                return proposal
        raise ValueError(f"Proposal {proposal_id} not found")

    def generate_join_code(self, proposal: RelationshipProposal, left_var: str, right_var: str) -> str:
        if not proposal.approved:
            raise ValueError(f"Proposal {proposal.proposal_id} must be approved by user before generating join code.")
        
        how = "inner"
        if proposal.relationship_type == RelationshipType.ONE_TO_MANY:
            if proposal.left_dataset_id == left_var: 
                how = "left"
        return f"merged_df = {left_var}.merge({right_var}, left_on='{proposal.left_column}', right_on='{proposal.right_column}', how='{how}')"

relationship_detector = RelationshipDetector()
