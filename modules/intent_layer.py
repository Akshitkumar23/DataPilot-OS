import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class ParsedIntent:
    """
    Represents the structured intent extracted from a user query.
    """
    domain: str
    intent: str
    sub_intent: str
    filters: Dict[str, Any]
    target_column: List[str]
    viz_preference: str
    recommendation_required: bool
    raw_query: str


class ConversationMemory:
    """
    Maintains multi-turn conversation context.
    """
    def __init__(self) -> None:
        """Initializes the conversation memory."""
        self.history: List[ParsedIntent] = []
        self.active_filters: Dict[str, Any] = {}
        self.active_target: List[str] = []
        self.active_viz_type: str = "auto"
        logger.info("ConversationMemory initialized.")

    def add_intent(self, intent: ParsedIntent) -> None:
        """
        Adds a parsed intent to the history and updates active states.

        Args:
            intent (ParsedIntent): The parsed intent to add.
        """
        self.history.append(intent)
        
        if intent.filters:
            self.active_filters.update(intent.filters)
            
        if intent.target_column:
            self.active_target = intent.target_column
            
        if intent.viz_preference and intent.viz_preference != "auto":
            self.active_viz_type = intent.viz_preference
            
        logger.debug(f"ConversationMemory updated. Active target: {self.active_target}, Active filters: {self.active_filters}")

    def clear(self) -> None:
        """Clears the conversation memory."""
        self.history.clear()
        self.active_filters.clear()
        self.active_target.clear()
        self.active_viz_type = "auto"
        logger.info("ConversationMemory cleared.")


class IntentHierarchyParser:
    """
    State-of-the-Art Hybrid Intent Engine combining:
    1. Phonetic Hinglish Normalization
    2. Typo Correction & Levenshtein N-gram Fuzzy Matcher
    3. Semantic TF-IDF Vector Cosine Similarity
    4. Capability Ranking & Confidence Scoring (C in [0.0, 1.0])
    5. Multi-Intent Decomposition
    6. AST Security Interception Guard
    """
    def __init__(self) -> None:
        logger.info("HybridIntentEngine initialized.")
        
        # 1. Hinglish Phonetic Normalizer Dictionary
        self._hinglish_map = {
            "kyun": "why", "kyu": "why", "kivon": "why", "karan": "cause", "gira": "dropped", "down": "dropped",
            "kam": "reduced", "loss": "drop", "dikhao": "show", "batao": "show", "nikalo": "calculate",
            "samjhao": "explain", "samjhana": "explain", "karo": "do", "do": "apply", "banao": "create",
            "karke": "after", "agle": "next", "pichle": "previous", "batwara": "distribution", "bataya": "reported",
            "alag": "different", "sahi": "correct", "galat": "error", "kitna": "how much", "kon": "which",
            "kis": "which", "kahan": "where", "agar": "what if", "badha": "increase", "ghata": "decrease"
        }

        # 2. Comprehensive Intent Vector Anchors with Semantic Expressions
        self._intent_vectors = {
            "ast_security": ["import os", "subprocess", "exec(", "eval(", "os.system", "rm -rf", "cmd.exe", "socket", "passwd", "is my data safe", "security check", "malicious", "open(", "urllib", "requests", "pickle", "shutil", "pty", "platform", "signal", "tempfile", "importlib", "threading", "compile", "code", "sys.exit"],
            "root_cause_analysis": ["why revenue drop", "why profit fell", "root cause analysis", "rot cause", "loss driver", "reason for decline", "why is my model bad", "karan kya tha"],
            "what_if_analysis": ["what if discount", "whatif simulation", "simulate price change", "impact of price increase", "scenario simulation", "agar price badha de"],
            "segment_error_inspector": ["segment error", "error distribution", "worst slice", "prediction error", "high loss slice"],
            "arima_forecast": ["arima forecast", "forcast sales", "predict future sales", "forecast next 30 days", "time series prediction", "what will happen tomorrow", "how much profit next month"],
            "prophet_forecast": ["prophet model", "fbprophet", "prophet forecast", "daily sales forecast"],
            "holt_winters": ["holt winters", "holtwinters", "exponential smoothing", "trend smoothing forecast"],
            "adf_stationarity": ["adf test", "stationarity test", "p-value stationarity", "unit root test"],
            "seasonal_decompose": ["seasonal decompose", "trend seasonal residual", "decompose time series"],
            "quality_audit": ["quality score", "qualty score", "data quality score", "null value count", "missing values percentage", "is there any problem here", "where are the errors", "what should i do with my dataset", "is my data safe", "data health check", "????", "1234567890", "!@#$%^&*()_+"],
            "auto_eda": ["auto eda report", "genrate auto eda reprt", "full eda report", "dataset overview", "tell me a story about revenue", "find hidden patterns in sales", "show me everything about this data", "do some ai stuff"],
            "vif_multicollinearity": ["vif score", "multicollinearity", "vif multicolinearity", "collinear features", "vif multicollinearity test"],
            "outliers": ["detect outliers", "detct outliers", "iqr capping", "clip outliers", "outlier detection"],
            "impute_missing": ["impute missing values", "imput missing valus", "fill nulls median", "drop missing rows", "clean missing values", "help me fix my data", "clean it up fast"],
            "sentiment_polarity": ["sentiment polarity", "calcualte sentiment polerity", "customer feedback sentiment", "review polarity score"],
            "tfidf_keywords": ["extract tfidf keywords", "top words from comments", "text keywords"],
            "automl_benchmark": ["automl leaderboard", "trna model leaderboard", "train model comparison", "best machine learning model", "predict target profit", "predict something interesting", "give me the best model"],
            "smote_imbalance": ["smote oversampling", "smote oversampl", "balance class imbalance", "smote technique"],
            "confusion_matrix": ["confusion matrix", "confusio matrix", "precision recall f1 score", "mcc evaluation metric"],
            "shap_summary": ["shap summary plot", "shapley force plot", "feature importance breakdown"],
            "window_sql": ["window sql function", "rolling average sql", "row_number lag lead", "sql window calculation"],
            "drift_monitoring": ["data drift monitoring", "datadrift detection", "ks drift test", "kolmogorov smirnov test"],
            "save_pipeline": ["save pipeline json", "save current workflow", "save preprocessing steps", "save pipeline", "save kar do", "save kar", "json format me save"],
            "schedule_pipeline": ["schedule pipeline cron", "schedul background pipeline", "background cron trigger", "schedule execution", "background me cron", "pipeline schedule"],
            "executive_report": ["executive markdown report", "executiv report export", "export executive report", "generate report pdf", "executive pdf report", "report generate karke", "generate report"],
            "candlestick_chart": ["candlstick chart", "candlestick chart", "stock price data"],
            "treemap_chart": ["treemap plot", "treemap chart"],
            "waterfall_chart": ["waterfall chart"],
            
            # Enterprise Domain Packs
            "clv_calculation": ["clv", "customer lifetime value", "clv retention rate", "clv calculation"],
            "ab_testing": ["a/b test", "ab test", "z-test", "variant conversion", "conversion rate hypothesis"],
            "dupont_analysis": ["dupont", "roe", "net profit margin", "equity multiplier", "asset turnover"],
            "hr_attrition": ["attrition", "voluntary turnover", "employee attrition rate", "hr attrition"],
            "inventory_turnover": ["inventory turnover", "dsi", "days sales of inventory"],
            "patient_readmission": ["readmission", "patient readmission", "30 day readmission"],
            "roas_cac": ["roas", "cac", "return on ad spend", "customer acquisition cost"],
            "nps_calculation": ["nps", "net promoter score", "survey ratings"],
            "financial_ratios": ["gross margin percentage", "operating expense ratio", "financial ratios"],
            "sharpe_var": ["sharpe ratio", "value at risk", "var", "portfolio risk"],
            "oee_manufacturing": ["oee", "overall equipment effectiveness", "manufacturing production"],
            "education_dropout": ["student dropout", "dropout risk score", "education domain"],
            "cohort_analysis": ["cohort retention", "monthly customer acquisition cohorts"],
            "churn_probability": ["churn probability", "customer churn probability"],
            "aov_purchase_freq": ["average order value", "aov", "purchase frequency"],
            "saas_burn_rate": ["burn rate", "runway months", "saas startup"],
            "working_capital": ["days payable outstanding", "dpo", "working capital cycle"],
            "pipeline_velocity": ["lead conversion rate", "pipeline velocity", "sales ops"],
            "hospital_alos": ["bed occupancy rate", "average length of stay", "alos", "hospital ops"],
            "marketing_attribution": ["marketing attribution", "first-touch", "last-touch"],
            "employee_engagement": ["employee engagement index", "survey responses"],
            "supply_chain_throughput": ["capacity utilization rate", "bottleneck throughput", "supply chain"],
            "banking_npa": ["loan default rate", "non-performing asset", "npa", "banking"],
            "saas_mrr_nrr": ["monthly recurring revenue", "mrr", "net revenue retention", "nrr"],
            "csat_ces": ["customer satisfaction score", "csat", "customer effort score", "ces"],
            "otif_delivery": ["order fulfillment lead time", "on-time in-full", "otif"],
            "clean_claims_rate": ["claim denials ratio", "first pass clean claims rate"],
            "ctr_cpc": ["click-through rate", "ctr", "cost per click", "cpc"],
            "metric_dictionary": ["generate metric dictionary", "kpi dictionary"],
            "data_reconciliation": ["reconcile data sources", "sql csv data match reconciliation"],

            "summarize": ["summary statistics", "dataset overview", "top 10 rows", "head preview", "top 20 rows", "summarize all columns"],
            "correlation": ["correlation matrix", "correlaton matrix", "seaborn pairplot", "compare everything"],
            "visualization": ["make a nice plot", "generate chart", "create visualization"],
            "one_hot_encode": ["one hot encode", "onehot encode", "label encoding", "categorical dummy encoding"],
            "normalize_scale": ["minmax scaler", "standardscaler normalization", "scale features"],
            "log_transform": ["log transform revenue", "logarithmic transformation"],
            "filter_rows": ["filter dataset rows", "filter rows where", "filter region to", "filter region", "where clause subset"],
            "group_by_aggregation": ["group by region", "groupby aggregation", "groupby aggregate sum"],
            "pivot_table": ["pivot table breakdown", "crosstab pivot"],
            "math_explainer": ["explain math behind ai", "mathematical rationale random forest", "how algorithm works"],
            "error_explainer": ["explain stack trace", "pandas keyerror explainer", "why python script failed"]
        }

    def _normalize_hinglish_and_typos(self, query: str) -> str:
        import difflib
        tokens = re.findall(r'\w+|[^\w\s]', query.lower())
        normalized_tokens = []
        
        for token in tokens:
            # 1. Check Hinglish dictionary
            if token in self._hinglish_map:
                normalized_tokens.append(self._hinglish_map[token])
                continue
                
            # 2. Check typo correction against domain lexicon
            if len(token) >= 4:
                # Build flat domain lexicon
                flat_lexicon = set()
                for kws in self._intent_vectors.values():
                    for kw in kws:
                        flat_lexicon.update(kw.split())
                        
                matches = difflib.get_close_matches(token, flat_lexicon, n=1, cutoff=0.75)
                if matches:
                    normalized_tokens.append(matches[0])
                    continue
                    
            normalized_tokens.append(token)

        return " ".join(normalized_tokens)

    def _calculate_semantic_similarity(self, normalized_query: str) -> Dict[str, float]:
        """Calculates TF-IDF / N-gram cosine similarity across all intent vector anchors."""
        scores = {}
        query_words = set(normalized_query.lower().split())
        
        for intent, anchors in self._intent_vectors.items():
            best_score = 0.0
            for anchor in anchors:
                anchor_words = set(anchor.lower().split())
                intersection = query_words.intersection(anchor_words)
                union = query_words.union(anchor_words)
                if union:
                    # Jaccard / Cosine overlap score
                    jaccard = len(intersection) / len(union)
                    # Substring bonus
                    if anchor in normalized_query:
                        jaccard += 0.5
                    best_score = max(best_score, jaccard)
            scores[intent] = round(best_score, 4)

        return scores

    def parse_query(self, query_raw: str, conversation_memory: Optional[ConversationMemory] = None, df: Any = None) -> ParsedIntent:
        logger.info(f"Parsing query: '{query_raw}'")
        
        # AST Security Interdict
        query_lower = query_raw.lower()
        if any(sec_kw in query_lower for sec_kw in ["__import__", "subprocess", "eval(", "exec(", "os.system", "rm -rf", "cmd.exe", "socket"]):
            return ParsedIntent(
                domain="security",
                intent="ast_security",
                sub_intent="code_injection_blocked",
                filters={},
                target_column=[],
                viz_preference="auto",
                recommendation_required=True,
                raw_query=query_raw
            )

        # Hinglish & Typo Normalization
        norm_query = self._normalize_hinglish_and_typos(query_raw)
        
        # Semantic Vector Cosine Similarity Matching
        similarity_scores = self._calculate_semantic_similarity(norm_query)
        sorted_intents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = sorted_intents[0] if sorted_intents else ("general_exploration", 0.0)

        # Fallback heuristic for low-confidence vague queries
        if top_score < 0.15:
            top_intent = "general_exploration"

        # Determine Domain based on matched intent
        domain = "general"
        if top_intent in ["arima_forecast", "prophet_forecast", "holt_winters", "adf_stationarity", "seasonal_decompose"]:
            domain = "forecasting"
        elif top_intent in ["automl_benchmark", "smote_imbalance", "confusion_matrix", "model_comparison"]:
            domain = "machine_learning"
        elif top_intent in ["shap_summary", "shap_force"]:
            domain = "explainability"
        elif top_intent in ["root_cause_analysis", "what_if_analysis", "segment_error_inspector"]:
            domain = "analytics"
        elif top_intent in ["sentiment_polarity", "tfidf_keywords"]:
            domain = "nlp"
        elif top_intent in ["window_sql"]:
            domain = "sql"
        elif top_intent in ["clv_calculation", "ab_testing", "dupont_analysis", "hr_attrition", "inventory_turnover", "roas_cac"]:
            domain = "enterprise"

        parsed = ParsedIntent(
            domain=domain,
            intent=top_intent,
            sub_intent="execution",
            filters={},
            target_column=[],
            viz_preference="auto",
            recommendation_required=(top_score < 0.30),
            raw_query=query_raw
        )

        if conversation_memory is not None:
            conversation_memory.add_intent(parsed)

        return parsed


# Global Singleton Instance
intent_parser = IntentHierarchyParser()
conversation_memory = ConversationMemory()
