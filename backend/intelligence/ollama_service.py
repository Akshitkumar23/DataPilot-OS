import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180.0"))
OLLAMA_HEALTH_TIMEOUT = float(os.getenv("OLLAMA_HEALTH_TIMEOUT", "3.0"))

DEFAULT_PLANNER_MODEL = os.getenv("OLLAMA_PLANNER_MODEL", "qwen2.5:3b-instruct")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
DEFAULT_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5:3b-instruct")

EXPLICIT_COMPATIBLE_MODELS = [
    # Prefer an installed instruction-tuned tool model with enough capacity to
    # follow a multi-field ToolSpec contract.  This is availability ranking,
    # not a dataset/query fallback: an explicit OLLAMA_PLANNER_MODEL still wins
    # and any selected model must pass the same strict schema validation.
    "qwen2.5:3b-instruct", "qwen2.5-coder:3b-instruct",
    # Other models can only be chosen explicitly through OLLAMA_PLANNER_MODEL.
    # 2. Gemma Multimodal Series
    "gemma3:1b", "gemma3:4b", "gemma4:e2b", "gemma2:2b",
    # 3. Llama 3.2 Series
    "llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "mistral:7b", "phi3:mini"
]


class OllamaModel(BaseModel):
    name: str
    size_gb: Optional[float] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None

class OllamaStatus(BaseModel):
    is_running: bool
    ollama_url: str
    available_models: List[OllamaModel] = Field(default_factory=list)
    active_model: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

class OllamaReadinessStatus(BaseModel):
    ready: bool
    ollama_url: str = ""
    model: str = ""
    model_available: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class OllamaRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    response_schema: Optional[Union[Dict[str, Any], str]] = None
    temperature: float = 0.0
    top_p: float = 0.1
    timeout_sec: float = OLLAMA_TIMEOUT
    model: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    think: Optional[bool] = None
    max_tokens: int = 1024

class OllamaResponse(BaseModel):
    success: bool
    content: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    prompt_eval_count: int = 0
    eval_count: int = 0
    load_duration_ms: float = 0.0
    prompt_eval_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
    model_used: str = ""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class OllamaService:
    def __init__(self):
        self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_PLANNER_MODEL)
        self._active_model: Optional[str] = None
        self._model_cache: List[OllamaModel] = []

    def verify_ready(self) -> OllamaReadinessStatus:
        status = self.detect_server()
        if not status.is_running:
            return OllamaReadinessStatus(
                ready=False,
                ollama_url=self.OLLAMA_URL,
                model=self.DEFAULT_MODEL,
                model_available=False,
                error_code="INTENT_PLANNER_UNAVAILABLE",
                error_message=status.error or f"Ollama server is unreachable at {self.OLLAMA_URL}."
            )

        avail_names = [m.name for m in status.available_models]
        
        selected_model = None
        if self._active_model and any(m == self._active_model for m in avail_names):
            selected_model = self._active_model

        if not selected_model:
            # An explicit planner setting is an operator decision and must
            # take precedence over automatic capability ranking.  Previously
            # OLLAMA_PLANNER_MODEL was ignored during readiness selection.
            requested_model = os.getenv("OLLAMA_PLANNER_MODEL") or os.getenv("OLLAMA_MODEL")
            if requested_model:
                selected_model = next(
                    (m for m in avail_names if m == requested_model or m.startswith(requested_model.split(':')[0])),
                    None,
                )

        if not selected_model:
            # Search strictly within explicit compatible allowlist
            for pref in EXPLICIT_COMPATIBLE_MODELS:
                found = next((m for m in avail_names if m == pref or m.startswith(pref.split(':')[0])), None)
                if found:
                    selected_model = found
                    break

        if not selected_model and avail_names:
            selected_model = avail_names[0]

        if not selected_model:
            return OllamaReadinessStatus(
                ready=False,
                ollama_url=self.OLLAMA_URL,
                model=self.DEFAULT_MODEL,
                model_available=False,
                error_code="OLLAMA_MODEL_NOT_FOUND",
                error_message=f"No models found on Ollama server at {self.OLLAMA_URL}."
            )

        self._active_model = selected_model
        return OllamaReadinessStatus(
            ready=True,
            ollama_url=self.OLLAMA_URL,
            model=selected_model,
            model_available=True,
            error_code=None,
            error_message=None
        )

    def detect_server(self) -> OllamaStatus:
        """
        Detects if the Ollama server is running and fetches available models.
        Returns a typed status object instead of throwing exceptions.
        """
        url = f"{self.OLLAMA_URL}/api/tags"
        start_time = time.time()
        
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=OLLAMA_HEALTH_TIMEOUT) as response:
                latency_ms = (time.time() - start_time) * 1000
                data = json.loads(response.read().decode("utf-8"))
                
                models = []
                for m in data.get("models", []):
                    size_gb = m.get("size", 0) / (1024**3) if "size" in m else None
                    models.append(OllamaModel(
                        name=m.get("name", ""),
                        size_gb=size_gb,
                        modified_at=m.get("modified_at"),
                        digest=m.get("digest")
                    ))
                
                self._model_cache = models
                return OllamaStatus(
                    is_running=True,
                    ollama_url=self.OLLAMA_URL,
                    available_models=models,
                    active_model=self.get_active_model(),
                    latency_ms=latency_ms,
                    error=None
                )
        except urllib.error.URLError as e:
            # Auto-recovery: If local Ollama daemon is offline, attempt to spawn it in background
            if "localhost" in self.OLLAMA_URL or "127.0.0.1" in self.OLLAMA_URL:
                try:
                    import subprocess
                    subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    )
                    time.sleep(0.8)
                    req = urllib.request.Request(url, method="GET")
                    with urllib.request.urlopen(req, timeout=OLLAMA_HEALTH_TIMEOUT) as response:
                        latency_ms = (time.time() - start_time) * 1000
                        data = json.loads(response.read().decode("utf-8"))
                        models = []
                        for m in data.get("models", []):
                            size_gb = m.get("size", 0) / (1024**3) if "size" in m else None
                            models.append(OllamaModel(
                                name=m.get("name", ""),
                                size_gb=size_gb,
                                modified_at=m.get("modified_at"),
                                digest=m.get("digest")
                            ))
                        self._model_cache = models
                        return OllamaStatus(
                            is_running=True,
                            ollama_url=self.OLLAMA_URL,
                            available_models=models,
                            active_model=self.get_active_model(),
                            latency_ms=latency_ms,
                            error=None
                        )
                except Exception:
                    pass

            return OllamaStatus(
                is_running=False,
                ollama_url=self.OLLAMA_URL,
                error=f"OLLAMA_UNAVAILABLE: Ollama server not running at {self.OLLAMA_URL}. Detail: {str(e)}"
            )
        except Exception as e:
            return OllamaStatus(
                is_running=False,
                ollama_url=self.OLLAMA_URL,
                error=f"OLLAMA_UNAVAILABLE: Unexpected error checking Ollama at {self.OLLAMA_URL}. Detail: {str(e)}"
            )

    def list_models(self) -> List[OllamaModel]:
        """
        Calls detect_server and updates the local model cache, returning the list of available models.
        """
        status = self.detect_server()
        if status.is_running:
            self._model_cache = status.available_models
            return self._model_cache
        return []

    def get_active_model(self) -> Optional[str]:
        """
        Returns the active model, or falls back to default fast model if unassigned.
        """
        if self._active_model:
            return self._active_model
        
        return os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)

    def set_active_model(self, model_name: str) -> bool:
        """
        Sets the active model if it exists in the model cache.
        """
        if not self._model_cache:
            self.list_models()
            
        for model in self._model_cache:
            if model.name == model_name:
                self._active_model = model_name
                return True
                
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Executes a multi-turn chat completion request to /api/chat.
        """
        readiness = self.verify_ready()
        if not readiness.ready:
            raise RuntimeError(readiness.error_message or "Ollama server is not ready")

        selected_model = model or self._active_model or readiness.model or self.DEFAULT_MODEL
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)

        payload = {
            "model": selected_model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9
            }
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.OLLAMA_URL}/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            msg = res_json.get("message", {})
            content = msg.get("content", "")
            return {"content": content, "raw": res_json}

    def embed_texts(
        self,
        texts: List[str],
        model: str,
        timeout_sec: float = 15.0,
    ) -> Optional[List[List[float]]]:
        """Embed a small batch for semantic capability retrieval.

        Retrieval is deliberately separate from query execution: a failed
        embedding request yields ``None`` so the caller can use its explicitly
        configured lexical retriever and still fail closed at planning time.
        """
        if not texts:
            return []
        payload = {"model": model, "input": texts}
        try:
            req = urllib.request.Request(
                f"{self.OLLAMA_URL}/api/embed",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                embeddings = json.loads(response.read().decode("utf-8")).get("embeddings")
            if (
                isinstance(embeddings, list)
                and len(embeddings) == len(texts)
                and all(isinstance(vector, list) and vector for vector in embeddings)
            ):
                return embeddings
            logger.warning("OLLAMA_EMBEDDING_INVALID_RESPONSE | Model: %s", model)
        except Exception as exc:
            logger.warning("OLLAMA_EMBEDDING_UNAVAILABLE | Model: %s | Error: %s", model, exc)
        return None

    def structured_request(self, request: OllamaRequest) -> OllamaResponse:
        """
        Makes a request to Ollama, requesting structured JSON output if a schema is provided.
        Includes bounded retry (max 2 attempts) ONLY for transient communication failures.
        Retry is NOT a fallback (it still calls Ollama).
        """
        if not self._active_model:
            self.verify_ready()
        model = request.model or self._active_model or self.DEFAULT_MODEL
        url = f"{self.OLLAMA_URL}/api/chat"
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
        options = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_ctx": num_ctx,
            "num_predict": request.max_tokens,
        }
        num_threads = os.getenv("OLLAMA_NUM_THREADS")
        if num_threads:
            try:
                options["num_thread"] = int(num_threads)
            except ValueError:
                pass

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
            "options": options,
        }

        if request.tools:
            # Native function calling gives the model the live ToolSpec schema
            # directly, instead of asking it to imitate a JSON envelope.
            payload["tools"] = request.tools
        if request.response_schema:
            # Ollama's chat API supports a JSON schema in `format` alongside
            # tools.  Native tool_calls remain authoritative when supplied;
            # the format contract makes the content path deterministic for
            # smaller local models which elect to emit JSON instead.
            payload["format"] = request.response_schema

        # Explicitly control reasoning thinking mode (default False for latency)
        payload["think"] = request.think if request.think is not None else False
            
        max_attempts = 2
        last_error = None

        for attempt in range(1, max_attempts + 1):
            start_time = time.time()
            try:
                req = urllib.request.Request(
                    url, 
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=request.timeout_sec) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    data = json.loads(response.read().decode("utf-8"))
                    
                    message = data.get("message", {})
                    content = message.get("content", "")
                    prompt_eval_count = data.get("prompt_eval_count", 0)
                    eval_count = data.get("eval_count", 0)
                    load_duration_ms = float(data.get("load_duration", 0) or 0) / 1e6
                    prompt_eval_duration_ms = float(data.get("prompt_eval_duration", 0) or 0) / 1e6
                    eval_duration_ms = float(data.get("eval_duration", 0) or 0) / 1e6
                    parsed_json = None
                    tool_calls = message.get("tool_calls") or []
                    if tool_calls:
                        commands = []
                        for call in tool_calls:
                            function = call.get("function", {}) if isinstance(call, dict) else {}
                            name = function.get("name")
                            arguments = function.get("arguments", {})
                            if isinstance(arguments, str):
                                try:
                                    arguments = json.loads(arguments)
                                except json.JSONDecodeError:
                                    arguments = {}
                            if name:
                                commands.append({"tool": name, "arguments": arguments})
                        if commands:
                            parsed_json = {"commands": commands}
                            # Preserve a durable audit payload even when native
                            # tool calling returns an empty content field.
                            content = content or json.dumps(parsed_json)
                    # Native tool calls are the API's structured contract and
                    # must take precedence over any accompanying prose.  Some
                    # local models emit a valid tool_calls array *and* a short
                    # textual explanation; reparsing that text used to erase
                    # the already-valid command and left the planner with no
                    # tool name.
                    if content and parsed_json is None:
                        import re
                        cleaned = content.strip()
                        if "```" in cleaned:
                            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
                        cleaned_no_comments = re.sub(r'//.*', '', cleaned)
                        try:
                            parsed_json = json.loads(cleaned_no_comments)
                        except Exception:
                            try:
                                parsed_json = json.loads(cleaned)
                            except Exception:
                                match = re.search(r"\{.*\}", cleaned_no_comments, re.DOTALL)
                                if match:
                                    try:
                                        parsed_json = json.loads(match.group(0))
                                    except Exception:
                                        parsed_json = None

                                if not parsed_json:
                                    # Strip Gemma token delimiters e.g. <|"|> or <|...|>
                                    cleaned_tokens = re.sub(r'<\|[^>]*\|>', '', cleaned).strip()
                                    fn_match = re.match(r'([a-zA-Z0-9_]+)\s*(\{.*\})', cleaned_tokens, re.DOTALL)
                                    if fn_match:
                                        tool_name = fn_match.group(1)
                                        raw_args = fn_match.group(2)
                                        try:
                                            # Fix unquoted keys if any
                                            fixed_args = re.sub(r'([a-zA-Z0-9_]+)\s*:', r'"\1":', raw_args)
                                            args_dict = json.loads(fixed_args)
                                            parsed_json = {"tool": tool_name, "arguments": args_dict}
                                        except Exception:
                                            pass

                    return OllamaResponse(
                        success=True,
                        content=content,
                        parsed_json=parsed_json,
                        latency_ms=latency_ms,
                        prompt_eval_count=prompt_eval_count,
                        eval_count=eval_count,
                        load_duration_ms=load_duration_ms,
                        prompt_eval_duration_ms=prompt_eval_duration_ms,
                        eval_duration_ms=eval_duration_ms,
                        model_used=model,
                        tool_calls=tool_calls,
                    )
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8") if hasattr(e, 'read') else str(e)
                logger.error(f"OLLAMA_HTTP_ERROR | Code: {e.code} | Body: {err_body}")
                code_name = "OLLAMA_MODEL_NOT_FOUND" if e.code == 404 else "OLLAMA_INVALID_RESPONSE"
                return OllamaResponse(
                    success=False,
                    latency_ms=(time.time() - start_time) * 1000,
                    model_used=model,
                    error_code=code_name,
                    error_message=f"HTTP {e.code}: {err_body}"
                )
            except urllib.error.URLError as e:
                last_error = f"URLError on attempt #{attempt}: {str(e)}"
                logger.warning(f"OLLAMA_TRANSIENT_RETRY | Attempt #{attempt}/{max_attempts} failed: {e}")
                if attempt < max_attempts:
                    time.sleep(0.5)
            except Exception as e:
                last_error = f"Unexpected error on attempt #{attempt}: {str(e)}"
                logger.warning(f"OLLAMA_TRANSIENT_RETRY | Attempt #{attempt}/{max_attempts} failed: {e}")
                if attempt < max_attempts:
                    time.sleep(0.5)

        return OllamaResponse(
            success=False,
            latency_ms=0.0,
            model_used=model,
            error_code="INTENT_PLANNER_UNAVAILABLE",
            error_message=f"Ollama Service Unavailable after {max_attempts} attempts. Detail: {last_error}"
        )

    def generate_intent(self, user_query: str, dataset_context: dict, conversation_history: list) -> OllamaResponse:
        """
        Generates a structured user intent based on the query and dataset context.
        """
        system_prompt = (
            "You are the Intent Analysis engine for DataPilot OS. "
            "Your job is to analyze the user query and dataset context to produce a structured intent. "
            "You must map user queries precisely to the provided dataset context."
        )
        
        prompt = (
            f"User Query: {user_query}\n"
            f"Dataset Context: {json.dumps(dataset_context)}\n"
            f"Conversation History: {json.dumps(conversation_history)}"
        )
        
        schema = {
            "type": "object",
            "properties": {
                "objective": {"type": "string"},
                "analysis_type": {"type": "string"},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "dimensions": {"type": "array", "items": {"type": "string"}},
                "time_dimension": {"type": ["string", "null"]},
                "operations": {"type": "array", "items": {"type": "string"}},
                "visualizations": {"type": "array", "items": {"type": "string"}},
                "requires_clarification": {"type": "boolean"},
                "clarification_questions": {"type": "array", "items": {"type": "string"}},
                "intent_complete": {"type": "boolean"},
                "ambiguities": {"type": "array", "items": {"type": "string"}},
                "assumptions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["objective", "analysis_type", "operations", "requires_clarification", "intent_complete"]
        }
        
        req = OllamaRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=schema,
            temperature=0.1
        )
        return self.structured_request(req)

    def generate_code(self, plan: dict, dataset_context: dict, code_requirements: dict) -> OllamaResponse:
        """
        [DEPRECATED] DataPilot DS Engine owns all code generation deterministically.
        Ollama is strictly restricted to Intent extraction ('WHAT' only).
        """
        return OllamaResponse(
            success=False,
            latency_ms=0.0,
            model_used=self.get_active_model() or self.DEFAULT_MODEL,
            error_code="CODE_GENERATION_DEPRECATED",
            error_message="Raw LLM code generation is deprecated. All executable Python code is compiled deterministically by DataPilot DS Engine capability drivers."
        )

    def analyze_error(self, code: str, error_message: str, dataset_context: dict) -> OllamaResponse:
        """
        Analyzes a runtime execution error and returns structured diagnostic root cause.
        Does NOT generate raw Python code (DataPilot DS Engine owns code repair).
        """
        system_prompt = (
            "You are a Data Science Diagnostic Analyst for DataPilot OS. Analyze the failed execution and error message, "
            "and output a structured JSON diagnostic summary explaining the root cause."
        )
        
        prompt = (
            f"Failed Execution Context:\n{code[:500]}\n\n"
            f"Error Message:\n{error_message}\n\n"
            f"Dataset Context: {json.dumps(dataset_context)}"
        )
        
        schema = {
            "type": "object",
            "properties": {
                "root_cause": {"type": "string"},
                "error_category": {"type": "string"},
                "suggested_repair_action": {"type": "string"},
                "explanation": {"type": "string"}
            },
            "required": ["root_cause", "error_category", "suggested_repair_action", "explanation"]
        }
        
        req = OllamaRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=schema,
            temperature=0.1
        )
        return self.structured_request(req)

    def generate_insight(self, validated_results: dict, user_query: str) -> OllamaResponse:
        """
        Summarizes validated execution results into natural language insights.
        """
        system_prompt = (
            "Generate insight from validated data analysis results. Every numerical claim must "
            "reference the provided data. Do NOT invent values."
        )
        
        prompt = (
            f"User Query: {user_query}\n"
            f"Validated Results: {json.dumps(validated_results)}"
        )
        
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_findings": {"type": "array", "items": {"type": "string"}},
                "drivers": {"type": "array", "items": {"type": "string"}},
                "anomalies": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
            },
            "required": ["summary", "key_findings", "confidence"]
        }
        
        req = OllamaRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=schema,
            temperature=0.2
        )
        return self.structured_request(req)

ollama_service = OllamaService()
