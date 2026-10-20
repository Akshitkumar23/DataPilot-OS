"""
DataPilot OS — Typed Execution Graph.
The graph is the authoritative execution model for DataPilot.
It defines WHAT happens, in what order, under what conditions.
The frontend only visualizes this — it has no execution authority.
"""
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, ConfigDict
import datetime

from backend.graph_engine.node import GraphNode
from backend.graph_engine.edge import GraphEdge, FORBIDDEN_EDGES
from backend.graph_engine.state import GraphExecutionState, NodeStatus

class ExecutionGraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_id: str
    workflow_id: str
    version: int
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    entry_node_id: str
    exit_node_ids: List[str]
    created_at: str
    replan_reason: Optional[str] = None
    parent_version: Optional[int] = None

    def add_node(self, node: GraphNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists in graph.")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        for forbidden in FORBIDDEN_EDGES:
            if edge.source == forbidden.get("source") and edge.target == forbidden.get("target"):
                raise ValueError(f"Edge {edge.source} -> {edge.target} is forbidden.")
        self.edges.append(edge)

    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if self.entry_node_id not in self.nodes:
            errors.append("Entry node missing.")
        for ext in self.exit_node_ids:
            if ext not in self.nodes:
                errors.append(f"Exit node {ext} missing.")
        for edge in self.edges:
            if edge.source not in self.nodes:
                errors.append(f"Edge source {edge.source} missing.")
            if edge.target not in self.nodes:
                errors.append(f"Edge target {edge.target} missing.")
            
            for forbidden in FORBIDDEN_EDGES:
                if edge.source == forbidden.get("source") and edge.target == forbidden.get("target"):
                    errors.append(f"Edge {edge.source} -> {edge.target} is forbidden.")
        
        connected = {self.entry_node_id}
        changed = True
        while changed:
            changed = False
            for edge in self.edges:
                if edge.source in connected and edge.target not in connected:
                    connected.add(edge.target)
                    changed = True
        
        for n in self.nodes:
            if n not in connected:
                errors.append(f"Node {n} is disconnected from entry.")

        return (len(errors) == 0, errors)

    def get_ready_nodes(self, state: GraphExecutionState) -> List[GraphNode]:
        ready = []
        for node_id, node in self.nodes.items():
            node_state = state.node_states.get(node_id)
            if not node_state or node_state.status != NodeStatus.PENDING:
                continue
            
            predecessors = [e.source for e in self.edges if e.target == node_id]
            all_succ = True
            for p in predecessors:
                p_state = state.node_states.get(p)
                if not p_state or p_state.status != NodeStatus.SUCCEEDED:
                    all_succ = False
                    break
            
            if all_succ or not predecessors:
                if node_id == self.entry_node_id or all_succ:
                    ready.append(node)
        return ready

    def get_execution_order(self) -> List[List[str]]:
        in_degree = {n: 0 for n in self.nodes}
        adj = {n: [] for n in self.nodes}
        for e in self.edges:
            adj[e.source].append(e.target)
            in_degree[e.target] += 1
        
        queue = [n for n in self.nodes if in_degree[n] == 0]
        order = []
        
        while queue:
            order.append(list(queue))
            next_q = []
            for u in queue:
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        next_q.append(v)
            queue = next_q
            
        return order

    def to_dict(self) -> dict:
        return self.model_dump()

    def get_ui_representation(self) -> dict:
        return {
            "nodes": [{"id": n.node_id, "type": n.node_type.value if hasattr(n.node_type, 'value') else n.node_type} for n in self.nodes.values()],
            "edges": [{"source": e.source, "target": e.target} for e in self.edges]
        }
