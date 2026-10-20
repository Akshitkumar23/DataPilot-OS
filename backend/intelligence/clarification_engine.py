"""
DataPilot OS — Clarification Engine.
When user intent is incomplete or ambiguous, generates structured
clarification requests instead of guessing or using fallbacks.

Clarification is NOT fallback. It is a normal intent-resolution state.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from backend.intelligence.intent_engine import IntentSchema

logger = logging.getLogger(__name__)

class ClarificationQuestion(BaseModel):
    question_id: str
    question_text: str
    question_type: str  # 'single_choice', 'multi_choice', 'free_text'
    options: Optional[List[str]]  # For choice questions
    required: bool
    relates_to: str  # Which intent field this resolves (e.g., 'time_dimension', 'metrics')

class ClarificationRequest(BaseModel):
    request_id: str
    questions: List[ClarificationQuestion]
    partial_intent: Optional[IntentSchema]
    context_summary: str  # brief summary of what was understood
    options_provided: bool  # True if options are given (not free-form)

class ClarificationEngine:
    def generate_clarification(
        self, 
        partial_intent: IntentSchema, 
        ambiguities: List[str], 
        dataset_context: dict
    ) -> ClarificationRequest:
        questions = []
        options_provided = False
        
        available_columns = [col.get("name", "") for col in dataset_context.get("columns", [])]
        
        import uuid
        request_id = str(uuid.uuid4())
        
        for idx, ambiguity in enumerate(ambiguities):
            q_id = f"q_{idx}"
            
            # Simple heuristic for demonstration; can be expanded via LLM call
            if "not found in dataset" in ambiguity.lower() and "column" in ambiguity.lower():
                questions.append(
                    ClarificationQuestion(
                        question_id=q_id,
                        question_text=f"{ambiguity} Please select the correct column.",
                        question_type="single_choice",
                        options=available_columns,
                        required=True,
                        relates_to="metrics" if "metric" in ambiguity.lower() else ("dimensions" if "dimension" in ambiguity.lower() else "time_dimension")
                    )
                )
                options_provided = True
            elif "analysis type" in ambiguity.lower():
                questions.append(
                    ClarificationQuestion(
                        question_id=q_id,
                        question_text="What type of analysis would you like to perform?",
                        question_type="single_choice",
                        options=["time_series", "comparison", "correlation", "distribution", "other"],
                        required=True,
                        relates_to="analysis_type"
                    )
                )
                options_provided = True
            elif "time range" in ambiguity.lower():
                 questions.append(
                    ClarificationQuestion(
                        question_id=q_id,
                        question_text="Please specify the time range for the analysis.",
                        question_type="free_text",
                        options=None,
                        required=True,
                        relates_to="filters"
                    )
                )
            else:
                questions.append(
                    ClarificationQuestion(
                        question_id=q_id,
                        question_text=f"Could you clarify: {ambiguity}?",
                        question_type="free_text",
                        options=None,
                        required=True,
                        relates_to="ambiguities"
                    )
                )
        
        context_summary = f"Understood objective: {partial_intent.objective}. However, some details are missing or ambiguous."
        
        return ClarificationRequest(
            request_id=request_id,
            questions=questions,
            partial_intent=partial_intent,
            context_summary=context_summary,
            options_provided=options_provided
        )

    def resolve_clarification(
        self, 
        clarification_request: ClarificationRequest, 
        user_response: Dict[str, Any], 
        partial_intent: IntentSchema
    ) -> IntentSchema:
        """
        Takes user's answers to clarification questions.
        user_response format: {question_id: answer}
        """
        for question in clarification_request.questions:
            answer = user_response.get(question.question_id)
            if not answer:
                continue
                
            if question.relates_to == "metrics":
                if isinstance(answer, list):
                    partial_intent.metrics.extend(answer)
                else:
                    partial_intent.metrics.append(str(answer))
            elif question.relates_to == "dimensions":
                if isinstance(answer, list):
                    partial_intent.dimensions.extend(answer)
                else:
                    partial_intent.dimensions.append(str(answer))
            elif question.relates_to == "time_dimension":
                partial_intent.time_dimension = str(answer)
            elif question.relates_to == "analysis_type":
                partial_intent.analysis_type = str(answer)
            
            # Remove resolved ambiguity if tracked
            # Note: A real implementation might ask the LLM to merge
            
        # Re-evaluate completeness
        if partial_intent.metrics and partial_intent.analysis_type:
            partial_intent.requires_clarification = False
            partial_intent.intent_complete = True
            partial_intent.ambiguities = []
        else:
            partial_intent.requires_clarification = True
            partial_intent.intent_complete = False
            
        return partial_intent

clarification_engine = ClarificationEngine()
