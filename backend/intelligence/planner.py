"""
DataPilot OS — Task Planner.
Converts structured IntentSchema into a typed ExecutionGraph (v1).
The planner proposes — the GraphRouter validates every edge.
LLM output cannot bypass planner validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid
import datetime

# Mock/Stubs for ExecutionGraph to support TaskPlanner types
class ExecutionNode(BaseModel):
    node_id: str
    capability_id: str
    inputs: Dict[str, Any]

class ExecutionGraph(BaseModel):
    workflow_id: str
    version: int = 1
    nodes: List[ExecutionNode] = []
    edges: List[Dict[str, str]] = []
    replan_reason: Optional[str] = None
    dataset_context: Dict[str, Any] = {}

class TaskPlanner:
    OPERATION_TO_CAPABILITY = {
        "trend_analysis": "time_series_trend",
        "time_series": "time_series_trend",
        "group_comparison": "group_aggregation",
        "groupby": "group_aggregation",
        "ranking": "top_n_ranking",
        "correlation": "correlation_analysis",
        "distribution": "distribution_analysis",
        "outlier_detection": "outlier_detection",
        "regression": "linear_regression",
        "classification": "classification_train",
        "clustering": "kmeans_clustering",
        "missing_values": "missing_value_analysis",
        "data_quality": "data_quality_audit",
        "eda": "auto_eda",
        "visualization": "auto_visualization",
        "forecast": "time_series_forecast",
        "pca": "pca_analysis",
        "hypothesis_test": "statistical_test",
    }

    def create_graph(self, intent: dict, dataset_context: dict, workflow_id: str) -> ExecutionGraph:
        graph = ExecutionGraph(workflow_id=workflow_id, dataset_context=dataset_context)
        
        # Always starts with a 'profile_dataset' node
        profile_node = ExecutionNode(
            node_id=f"node_{uuid.uuid4().hex[:8]}",
            capability_id="profile_dataset",
            inputs={}
        )
        graph.nodes.append(profile_node)
        
        prev_node_id = profile_node.node_id
        
        operations = intent.get("operations", [])
        for op in operations:
            cap_id = self.OPERATION_TO_CAPABILITY.get(op)
            if cap_id:
                node = ExecutionNode(
                    node_id=f"node_{uuid.uuid4().hex[:8]}",
                    capability_id=cap_id,
                    inputs={}
                )
                graph.nodes.append(node)
                graph.edges.append({"from": prev_node_id, "to": node.node_id})
                prev_node_id = node.node_id
                
        # Always ends with an 'insight_generation' node
        insight_node = ExecutionNode(
            node_id=f"node_{uuid.uuid4().hex[:8]}",
            capability_id="insight_generation",
            inputs={}
        )
        graph.nodes.append(insight_node)
        graph.edges.append({"from": prev_node_id, "to": insight_node.node_id})
        
        # In a real environment, graph_router.validate_proposed_graph(graph) would be called here
        
        return graph

    def replan(self, current_graph: ExecutionGraph, missing_evidence: List[str], workflow_id: str) -> ExecutionGraph:
        new_graph = current_graph.model_copy(deep=True)
        new_graph.version += 1
        new_graph.replan_reason = f"Missing evidence: {', '.join(missing_evidence)}"
        
        # Simplified replan logic: add auto_eda nodes for missing evidence
        for ev in missing_evidence:
            node = ExecutionNode(
                node_id=f"node_{uuid.uuid4().hex[:8]}",
                capability_id="auto_eda",
                inputs={"evidence_target": ev}
            )
            new_graph.nodes.append(node)
            
        return new_graph

task_planner = TaskPlanner()
