"""
DataPilot OS — Version Store.
Tracks code versions and graph versions per workflow.
"""
import uuid
from typing import Dict, List, Any

class VersionStore:
    def __init__(self):
        # In-memory store for versions: workflow_id -> {"code": [...], "graph": [...]}
        self._versions: Dict[str, Dict[str, List[Any]]] = {}

    def _ensure_workflow(self, workflow_id: str):
        if workflow_id not in self._versions:
            self._versions[workflow_id] = {"code": [], "graph": []}

    def save_code_version(self, code_version_dict: Dict[str, Any]) -> str:
        version_id = str(uuid.uuid4())
        workflow_id = code_version_dict.get("workflow_id", "default")
        self._ensure_workflow(workflow_id)
        
        record = {
            "version_id": version_id,
            **code_version_dict
        }
        self._versions[workflow_id]["code"].append(record)
        return version_id

    def save_graph_version(self, graph_version_dict: Dict[str, Any]) -> str:
        version_id = str(uuid.uuid4())
        workflow_id = graph_version_dict.get("workflow_id", "default")
        self._ensure_workflow(workflow_id)
        
        record = {
            "version_id": version_id,
            **graph_version_dict
        }
        self._versions[workflow_id]["graph"].append(record)
        return version_id

    def get_versions_for_workflow(self, workflow_id: str) -> Dict[str, List[Any]]:
        return self._versions.get(workflow_id, {"code": [], "graph": []})

version_store = VersionStore()
