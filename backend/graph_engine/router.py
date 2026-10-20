"""
DataPilot OS — Engine-Controlled Graph Router.
Routing is deterministic and engine-controlled.
The LLM proposes node sequences — the router validates every edge.
No LLM output can bypass router validation.
"""
from typing import Tuple, Optional, List, Dict, Any
from backend.graph_engine.graph import ExecutionGraph
from backend.graph_engine.state import GraphExecutionState
from backend.graph_engine.edge import FORBIDDEN_EDGES

class GraphRouter:
    def validate_transition(self, graph: ExecutionGraph, from_node_id: str, to_node_id: str, state: GraphExecutionState) -> Tuple[bool, Optional[str]]:
        edge = next((e for e in graph.edges if e.source_node_id == from_node_id and e.target_node_id == to_node_id), None)
        if not edge:
            return False, "Edge does not exist in graph"
            
        for src, tgt in FORBIDDEN_EDGES:
            if edge.source_node_id == src and edge.target_node_id == tgt:
                return False, f"Forbidden edge: {src} -> {tgt}"
                
        return True, None

    def get_next_nodes(self, graph: ExecutionGraph, completed_node_id: str, state: GraphExecutionState) -> List[str]:
        next_nodes = []
        for edge in graph.edges:
            if edge.source_node_id == completed_node_id:
                next_nodes.append(edge.target_node_id)
        return next_nodes

    def validate_proposed_graph(self, proposed_nodes: List[str], proposed_edges: List[Dict]) -> Tuple[bool, List[str]]:
        violations = []
        for edge in proposed_edges:
            src = edge.get("source") or edge.get("source_node_id")
            tgt = edge.get("target") or edge.get("target_node_id")
            for forbidden_src, forbidden_tgt in FORBIDDEN_EDGES:
                if src == forbidden_src and tgt == forbidden_tgt:
                    violations.append(f"Forbidden edge {src}->{tgt}")
        return len(violations) == 0, violations

graph_router = GraphRouter()
