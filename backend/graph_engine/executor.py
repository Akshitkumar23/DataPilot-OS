import logging
from typing import Dict, Any, List, Optional

from backend.graph_engine.graph import ExecutionGraph
from backend.graph_engine.state import GraphExecutionState, NodeStatus
from backend.graph_engine.loop_engine import loop_controller, LoopDecision
from backend.core.workflow_state import WorkflowState
from backend.core.error_states import DataPilotError, ErrorState

logger = logging.getLogger(__name__)

class GraphExecutor:
    """
    DataPilot OS — Graph Executor.
    Deterministic execution engine for the typed ExecutionGraph.
    The Loop Controller sits INSIDE the executor as a control layer.
    """

    def execute_graph(
        self,
        graph: ExecutionGraph,
        state: GraphExecutionState,
        workflow_state: WorkflowState
    ) -> GraphExecutionState:
        """Executes the graph layer by layer using topological ordering."""
        logger.info(f"Executing graph {graph.graph_id} (version {graph.version})")
        layers = graph.get_execution_order()
        
        for layer in layers:
            nodes_in_layer = [graph.nodes[nid] for nid in layer if nid in graph.nodes]
            self._execute_parallel_layer(nodes_in_layer, state, workflow_state)
            
            # Stop graph if error reached
            if state.error or state.phase == "FAILED":
                break
                
        return state

    def _execute_node(
        self,
        node: Any,
        state: GraphExecutionState,
        workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """Executes a single graph node, updates state, and triggers Loop Controller observation."""
        state.set_node_status(node.id if hasattr(node, 'id') else str(node), NodeStatus.RUNNING)
        result = {}
        
        try:
            if hasattr(node, 'execute'):
                result = node.execute(state.shared_context)
            else:
                result = {"status": "success", "node_id": str(node)}
                
            state.set_node_status(node.id if hasattr(node, 'id') else str(node), NodeStatus.SUCCEEDED)
        except Exception as err:
            logger.error(f"Node execution failed: {err}")
            state.set_node_status(node.id if hasattr(node, 'id') else str(node), NodeStatus.FAILED)
            result = {"status": "failed", "error": str(err)}

        decision = self._observe(node, result, state, workflow_state)
        result["loop_decision"] = decision.action if hasattr(decision, 'action') else "continue"
        return result

    def _observe(
        self,
        node: Any,
        result: Dict[str, Any],
        state: GraphExecutionState,
        workflow_state: WorkflowState
    ) -> LoopDecision:
        """
        LOOP CONTROL LAYER — Called after every node execution.
        Evaluates node output via LoopController and returns decision.
        """
        node_id = getattr(node, 'id', str(node))
        decision = loop_controller.evaluate(state, result, loop_id=node_id)
        
        action = decision.action if hasattr(decision, 'action') else "continue"
        
        if action == "stop_no_progress":
            state.error = "LOOP_NO_PROGRESS: Proposed action is identical to previous failed attempt."
            state.phase = "FAILED"
        elif action == "stop_budget":
            state.error = "LOOP_BUDGET_EXCEEDED: Maximum loop iterations exceeded."
            state.phase = "FAILED"
            
        return decision

    def _execute_parallel_layer(
        self,
        nodes: List[Any],
        state: GraphExecutionState,
        workflow_state: WorkflowState
    ) -> List[Dict]:
        results = []
        for node in nodes:
            res = self._execute_node(node, state, workflow_state)
            results.append(res)
        return results

graph_executor = GraphExecutor()
