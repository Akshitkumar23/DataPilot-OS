import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelExplainabilityEngine:
    """
    Engine to generate Python code for model explainability techniques like Feature Importance,
    SHAP, Partial Dependence Plots, and single instance breakdown.
    """

    def generate_feature_importance_code(self) -> str:
        """
        Generates Python code to plot a Feature Importance Bar Chart for a trained model.
        
        Returns:
            str: The generated Python code.
        """
        try:
            logger.info("Generating Feature Importance code.")
            code = """
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_feature_importance(model: Any, feature_names: List[str]) -> None:
    \"\"\"
    Plot feature importance for a given model.
    \"\"\"
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).ravel()
        else:
            raise ValueError("Model does not have 'feature_importances_' or 'coef_' attribute.")
            
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df_imp)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise
"""
            return code
        except Exception as e:
            logger.error(f"Error generating feature importance code: {str(e)}")
            raise

    def generate_shap_explanation_code(self) -> str:
        """
        Generates Python code to plot a SHAP Summary Plot.
        
        Returns:
            str: The generated Python code.
        """
        try:
            logger.info("Generating SHAP explanation code.")
            code = """
import logging
import shap
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_shap_summary(model: Any, X_train: pd.DataFrame) -> None:
    \"\"\"
    Plot SHAP summary plot for a given model and training data.
    \"\"\"
    try:
        # Initialize JS visualization code
        shap.initjs()
        
        # Create object that can calculate shap values
        try:
            explainer = shap.Explainer(model, X_train)
        except Exception:
            explainer = shap.TreeExplainer(model)
        
        # Calculate Shap values
        shap_values = explainer.shap_values(X_train)
        
        # Plot summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_train)
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting SHAP summary: {str(e)}")
        raise
"""
            return code
        except Exception as e:
            logger.error(f"Error generating SHAP explanation code: {str(e)}")
            raise

    def generate_pdp_plot_code(self, feature_name: str) -> str:
        """
        Generates Python code to plot a Partial Dependence Plot for a specific feature.
        
        Args:
            feature_name (str): The name of the feature to plot.
            
        Returns:
            str: The generated Python code.
        """
        try:
            logger.info(f"Generating PDP plot code for feature: {feature_name}")
            code = f"""
import logging
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from typing import Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_partial_dependence(model: Any, X_train: pd.DataFrame, feature_name: str = '{feature_name}') -> None:
    \"\"\"
    Plot Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE) for a specific feature.
    \"\"\"
    try:
        if feature_name not in X_train.columns:
            raise ValueError(f"Feature '{{feature_name}}' not found in features.")
            
        plt.figure(figsize=(8, 6))
        display = PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X_train,
            features=[feature_name],
            kind='both'
        )
        plt.title(f'PDP and ICE Plot for {{feature_name}}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting PDP/ICE: {{str(e)}}")
        raise
"""
            return code.strip()
        except Exception as e:
            logger.error(f"Error generating PDP plot code: {str(e)}")
            raise

    def generate_prediction_breakdown_code(self) -> str:
        """
        Generates Python code for a single instance prediction breakdown (e.g., using SHAP force plot).
        
        Returns:
            str: The generated Python code.
        """
        try:
            logger.info("Generating prediction breakdown code.")
            code = """
import logging
import shap
import pandas as pd
from typing import Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_prediction_breakdown(model: Any, instance: pd.DataFrame, X_train: pd.DataFrame) -> Any:
    \"\"\"
    Plot prediction breakdown (SHAP force plot) for a single instance.
    \"\"\"
    try:
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance)
        
        # Check if shap_values is a list (multiclass) or array (binary/regression)
        if isinstance(shap_values, list):
            # Taking the first class for breakdown as default
            expected_value = explainer.expected_value[0]
            shap_val = shap_values[0]
        else:
            expected_value = explainer.expected_value
            shap_val = shap_values
            
        display = shap.force_plot(expected_value, shap_val, instance)
        return display
    except Exception as e:
        logger.error(f"Error plotting prediction breakdown: {str(e)}")
        raise
"""
            return code
        except Exception as e:
            logger.error(f"Error generating prediction breakdown code: {str(e)}")
            raise


# Global Singleton Instance
explainability_engine = ModelExplainabilityEngine()

