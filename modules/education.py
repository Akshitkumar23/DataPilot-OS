import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EducationEngine:
    """
    An educational module designed to provide pedagogical explanations for data science methods,
    generate interview questions, and explain stack trace errors to students.
    """

    def __init__(self) -> None:
        """
        Initializes the EducationEngine with the knowledge base.
        """
        logger.info("Initializing EducationEngine")
        
        self._knowledge_base: Dict[str, Dict[str, Any]] = {
            "mean_vs_median_imputation": {
                "description": "Mean vs Median Imputation",
                "why": "Missing values can cause machine learning algorithms to fail. Imputation fills these gaps.",
                "when": "Use mean for normally distributed data without outliers. Use median for skewed data or data with outliers.",
                "advantages": ["Simple to implement", "Fast computation"],
                "disadvantages": ["Reduces variance", "Mean is sensitive to outliers"],
                "mathematical_rationale": "Mean: $\\mu = \\frac{1}{N} \\sum_{i=1}^N x_i$. Median: The middle value when data is sorted.",
                "python_code_snippet": "import pandas as pd\n# Mean imputation\ndf['col'] = df['col'].fillna(df['col'].mean())\n# Median imputation\ndf['col'] = df['col'].fillna(df['col'].median())"
            },
            "random_forest_vs_xgboost": {
                "description": "Random Forest vs XGBoost",
                "why": "Both are powerful ensemble learning methods for classification and regression.",
                "when": "Use Random Forest for a robust, out-of-the-box model. Use XGBoost when you need maximum performance and can tune hyperparameters.",
                "advantages": ["RF: Less prone to overfitting, easy to use", "XGB: High performance, handles missing data"],
                "disadvantages": ["RF: Can be large and slow to predict", "XGB: Prone to overfitting if not tuned, computationally expensive"],
                "mathematical_rationale": "RF uses Bagging (Bootstrap Aggregating) to train independent trees. XGBoost uses Gradient Boosting to train trees sequentially, minimizing a loss function using gradient descent.",
                "python_code_snippet": "from sklearn.ensemble import RandomForestClassifier\nfrom xgboost import XGBClassifier\n\nrf = RandomForestClassifier()\nrf.fit(X_train, y_train)\n\nxgb = XGBClassifier()\nxgb.fit(X_train, y_train)"
            },
            "iqr_capping": {
                "description": "IQR (Interquartile Range) Capping for Outliers",
                "why": "Outliers can skew statistical analyses and model training. Capping limits their effect.",
                "when": "When you have continuous numerical data with extreme values that you don't want to drop entirely.",
                "advantages": ["Preserves data points (no data loss)", "Reduces the impact of extreme values"],
                "disadvantages": ["Alters the true distribution", "Arbitrary cutoff points"],
                "mathematical_rationale": "$IQR = Q3 - Q1$. Lower Bound = $Q1 - 1.5 \\times IQR$. Upper Bound = $Q3 + 1.5 \\times IQR$. Values outside are capped at these bounds.",
                "python_code_snippet": "import numpy as np\n\nq1 = df['col'].quantile(0.25)\nq3 = df['col'].quantile(0.75)\niqr = q3 - q1\nlower_bound = q1 - 1.5 * iqr\nupper_bound = q3 + 1.5 * iqr\n\ndf['col'] = np.where(df['col'] < lower_bound, lower_bound, df['col'])\ndf['col'] = np.where(df['col'] > upper_bound, upper_bound, df['col'])"
            }
        }
        
        self._interview_questions: Dict[str, Dict[str, List[Dict[str, str]]]] = {
            "DS": {
                "easy": [
                    {"question": "What is the difference between supervised and unsupervised learning?", "answer": "Supervised learning uses labeled data to train a model, while unsupervised learning finds patterns in unlabeled data."}
                ],
                "medium": [
                    {"question": "Explain the bias-variance tradeoff.", "answer": "Bias is error from erroneous assumptions, leading to underfitting. Variance is error from sensitivity to small fluctuations, leading to overfitting. The tradeoff seeks to minimize total error."}
                ]
            },
            "SQL": {
                "easy": [
                    {"question": "What is the difference between INNER JOIN and LEFT JOIN?", "answer": "INNER JOIN returns only matching rows from both tables. LEFT JOIN returns all rows from the left table and matched rows from the right table, with NULLs for unmatched rows."}
                ]
            },
            "Python": {
                "medium": [
                    {"question": "How do decorators work in Python?", "answer": "Decorators are functions that modify the behavior of another function or class. They take a function as input and return a new function."}
                ]
            }
        }

    def explain_code_and_math(self, method_name: str) -> Dict[str, Any]:
        """
        Provides a pedagogical explanation for Data Science methods.

        Args:
            method_name (str): The name of the method to explain 
                               (e.g., 'mean_vs_median_imputation', 'random_forest_vs_xgboost').

        Returns:
            Dict[str, Any]: A dictionary containing explanations including Why, When,
                            Advantages, Disadvantages, Mathematical Rationale, and Python Code snippet.
                            
        Raises:
            ValueError: If the method_name is not found in the knowledge base.
        """
        logger.info(f"Requesting explanation for method: {method_name}")
        
        try:
            # Normalize method name for lookup
            normalized_name = method_name.lower().replace(" ", "_")
            
            if normalized_name not in self._knowledge_base:
                logger.warning(f"Method '{method_name}' not found in knowledge base.")
                raise ValueError(f"Explanation for '{method_name}' is not currently available.")
                
            return self._knowledge_base[normalized_name]
            
        except Exception as e:
            logger.error(f"Error while fetching explanation for '{method_name}': {e}")
            raise

    def generate_interview_questions(self, domain: str, difficulty: str) -> List[Dict[str, str]]:
        """
        Generates interactive DS/SQL/Python interview questions and sample answers.

        Args:
            domain (str): The domain of the interview questions (e.g., 'DS', 'SQL', 'Python').
            difficulty (str): The difficulty level (e.g., 'easy', 'medium', 'hard').

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing 'question' and 'answer' keys.
            
        Raises:
            ValueError: If the domain or difficulty is invalid.
        """
        logger.info(f"Generating interview questions for domain: {domain}, difficulty: {difficulty}")
        
        try:
            valid_domains = ["DS", "SQL", "Python"]
            valid_difficulties = ["easy", "medium", "hard"]
            
            domain_key = next((d for d in valid_domains if d.lower() == domain.lower()), None)
            diff_key = difficulty.lower()
            
            if not domain_key:
                raise ValueError(f"Invalid domain: '{domain}'. Valid domains are: {', '.join(valid_domains)}")
                
            if diff_key not in valid_difficulties:
                raise ValueError(f"Invalid difficulty: '{difficulty}'. Valid difficulties are: {', '.join(valid_difficulties)}")
                
            questions = self._interview_questions.get(domain_key, {}).get(diff_key, [])
            
            if not questions:
                logger.info(f"No questions found for {domain} - {difficulty}.")
                return [{"question": "No questions available for this category yet.", "answer": "N/A"}]
                
            return questions
            
        except Exception as e:
            logger.error(f"Error generating interview questions: {e}")
            raise

    def explain_stack_trace_error(self, error_msg: str) -> Dict[str, str]:
        """
        Explains Python/Pandas error tracebacks in clear, simple terms with step-by-step resolution advice.

        Args:
            error_msg (str): The raw stack trace or error message string.

        Returns:
            Dict[str, str]: A dictionary containing a simplified explanation and resolution steps.
            
        Raises:
            ValueError: If the provided error_msg is not a valid string.
        """
        logger.info("Explaining stack trace error")
        
        try:
            if not error_msg or not isinstance(error_msg, str):
                raise ValueError("Error message must be a non-empty string.")
                
            error_lower = error_msg.lower()
            
            # Simple heuristic-based matching for common errors
            if "keyerror" in error_lower and "pandas" in error_lower:
                explanation = "A KeyError in Pandas usually means you are trying to access a column name or row index that does not exist in the DataFrame."
                resolution = "1. Check the spelling of the column name.\\n2. Print df.columns to see all available columns.\\n3. Ensure there are no hidden leading/trailing spaces in the column name (e.g., ' column_name ')."
            elif "typeerror" in error_lower:
                explanation = "A TypeError occurs when an operation or function is applied to an object of inappropriate type."
                resolution = "1. Check the data types of the variables involved using type().\\n2. If operating on a Pandas series, check df['col'].dtype.\\n3. Convert the variable to the correct type (e.g., using astype() or int())."
            elif "indexerror" in error_lower:
                explanation = "An IndexError means you are trying to access an element from a list or array using an index that is out of bounds (e.g., too large)."
                resolution = "1. Check the length of the list/array using len().\\n2. Remember that Python uses 0-based indexing (the first element is at index 0, the last is at len - 1)."
            elif "modulenotfounderror" in error_lower or "importerror" in error_lower:
                explanation = "Python cannot find the library or module you are trying to import."
                resolution = "1. Verify the spelling of the module name.\\n2. Ensure the library is installed in your current environment (e.g., run `pip install module_name`).\\n3. Check if you are using the correct Python virtual environment."
            else:
                explanation = "This error is not immediately recognized by the standard heuristics, but it generally indicates a runtime failure."
                resolution = "1. Read the very last line of the error trace carefully; it often summarizes the issue.\\n2. Look at the specific line of your code mentioned just before the error summary.\\n3. Search the exact error message online or consult the library documentation."
                
            return {
                "explanation": explanation,
                "resolution": resolution
            }
            
        except Exception as e:
            logger.error(f"Error while explaining stack trace: {e}")
            raise


# Global Singleton Instance
education_engine = EducationEngine()
