"""Contracts for native Ollama tool calling in the semantic planner path."""

import json
from unittest.mock import patch

from backend.intelligence.ollama_service import OllamaRequest, OllamaService


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_native_tool_calls_are_converted_to_command_contract():
    service = OllamaService()
    service._active_model = "qwen3:8b"
    tool = {
        "type": "function",
        "function": {
            "name": "correlate_data",
            "description": "Correlation analysis",
            "parameters": {"type": "object"},
        },
    }
    server_payload = {
        "message": {
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "correlate_data",
                    "arguments": {"columns": ["temperature", "pressure"]},
                }
            }],
        }
    }

    with patch("backend.intelligence.ollama_service.urllib.request.urlopen", return_value=_Response(server_payload)) as open_mock:
        response = service.structured_request(
            OllamaRequest(prompt="Find correlation", tools=[tool], think=False)
        )

    request_payload = json.loads(open_mock.call_args.args[0].data.decode("utf-8"))
    assert request_payload["tools"] == [tool]
    assert request_payload["think"] is False
    assert "format" not in request_payload
    assert request_payload["options"]["num_predict"] == 1024
    assert response.parsed_json == {
        "commands": [{"tool": "correlate_data", "arguments": {"columns": ["temperature", "pressure"]}}]
    }


def test_native_tool_call_is_not_erased_by_accompanying_plain_text():
    service = OllamaService()
    service._active_model = "qwen3:8b"
    server_payload = {
        "message": {
            "content": "I will train the requested model.",
            "tool_calls": [{
                "function": {
                    "name": "train_model",
                    "arguments": {"target_column": "outcome"},
                }
            }],
        }
    }

    with patch("backend.intelligence.ollama_service.urllib.request.urlopen", return_value=_Response(server_payload)):
        response = service.structured_request(
            OllamaRequest(prompt="Train", tools=[{"type": "function", "function": {"name": "train_model", "parameters": {}}}])
        )

    assert response.parsed_json == {
        "commands": [{"tool": "train_model", "arguments": {"target_column": "outcome"}}]
    }
