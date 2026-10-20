"""
DataPilot OS — Graph State Checkpoint.
Checkpoints allow workflow recovery after failures.
Every state transition is checkpointable.
"""
import os
import json
import uuid
from typing import Dict, Optional, List, Any

from backend.graph_engine.state import GraphExecutionState

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CheckpointManager:
    def __init__(self):
        self.checkpoint_dir = os.path.join(BASE_DIR, 'backend', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, state: GraphExecutionState, workflow_state: dict) -> str:
        checkpoint_id = str(uuid.uuid4())
        data = {
            "checkpoint_id": checkpoint_id,
            "graph_state": state.model_dump() if hasattr(state, "model_dump") else state.dict(),
            "workflow_state": workflow_state
        }
        filepath = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return checkpoint_id

    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        filepath = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            return json.load(f)

    def list_checkpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        checkpoints = []
        for fname in os.listdir(self.checkpoint_dir):
            if fname.endswith(".json"):
                filepath = os.path.join(self.checkpoint_dir, fname)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if data.get("graph_state", {}).get("workflow_id") == workflow_id:
                            checkpoints.append(data)
                except Exception:
                    continue
        return checkpoints

    def delete(self, checkpoint_id: str) -> bool:
        filepath = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

checkpoint_manager = CheckpointManager()
