"""
DataPilot OS — Graph Edge with Engine-Controlled Routing.
Edges define valid transitions between graph nodes.
The routing predicate is deterministic — LLM cannot override it.
"""

from enum import Enum
from typing import Optional, Tuple, List
from pydantic import BaseModel

class EdgeType(str, Enum):
    SEQUENTIAL = "SEQUENTIAL"
    CONDITIONAL = "CONDITIONAL"
    PARALLEL = "PARALLEL"
    EVIDENCE_DRIVEN = "EVIDENCE_DRIVEN"

class GraphEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    condition: Optional[str] = None
    is_active: bool = True
    label: Optional[str] = None

FORBIDDEN_EDGES = [
    ("load_dataset", "production_execute")
]

def validate_edge(edge: GraphEdge, all_nodes: list) -> Tuple[bool, Optional[str]]:
    node_ids = {node.id for node in all_nodes} if all_nodes else set()
    
    if all_nodes:
        if edge.source_node_id not in node_ids:
            return False, f"Source node {edge.source_node_id} not found."
        if edge.target_node_id not in node_ids:
            return False, f"Target node {edge.target_node_id} not found."
            
    if (edge.source_node_id, edge.target_node_id) in FORBIDDEN_EDGES:
        return False, f"Edge from {edge.source_node_id} to {edge.target_node_id} is forbidden."
        
    return True, None
