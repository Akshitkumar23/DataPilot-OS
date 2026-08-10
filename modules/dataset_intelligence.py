import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DatasetIntelligence:
    """
    Intelligent analysis of datasets to detect domain and audit data quality.
    """
    
    def __init__(self) -> None:
        """Initialize the DatasetIntelligence class."""
        self.domain_keywords = {
            "Retail Sales": ["price", "revenue", "sales", "discount", "customer", "order", "product", "sku", "qty", "quantity", "store"],
            "HR Analytics": ["employee", "salary", "department", "hire", "attrition", "performance", "manager", "role", "tenure", "bonus"],
            "Financial Analytics": ["transaction", "balance", "account", "credit", "debit", "loan", "interest", "investment", "expense", "profit"],
            "Manufacturing & Operations": ["inventory", "supplier", "factory", "machine", "production", "downtime", "defect", "yield", "material"],
            "Healthcare Analytics": ["patient", "diagnosis", "treatment", "doctor", "hospital", "prescription", "blood", "pressure", "disease", "symptom"],
            "Marketing Analytics": ["campaign", "click", "conversion", "impression", "lead", "cpc", "ctr", "ad", "bounce", "audience"]
        }

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes column names and dtypes to classify dataset domain.

        Args:
            df (pd.DataFrame): The dataset to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the detected domain and confidence score (0-100%).
        """
        try:
            logger.info("Starting domain detection.")
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame.")

            if df.empty:
                logger.warning("Empty DataFrame provided for domain detection.")
                return {"domain": "General Analytics", "confidence_score": 0.0}

            columns_lower = [str(col).lower() for col in df.columns]
            domain_scores = {domain: 0 for domain in self.domain_keywords.keys()}

            for col in columns_lower:
                for domain, keywords in self.domain_keywords.items():
                    if any(keyword in col for keyword in keywords):
                        domain_scores[domain] += 1

            max_score = max(domain_scores.values()) if domain_scores else 0
            
            if max_score == 0:
                return {"domain": "General Analytics", "confidence_score": 100.0}

            best_domain = max(domain_scores, key=domain_scores.get)
            
            # Simple confidence calculation based on proportion of matching columns
            confidence = (max_score / len(columns_lower)) * 100
            confidence = min(confidence * 2, 100.0) # Boost confidence slightly but cap at 100

            logger.info(f"Domain detected: {best_domain} with {confidence:.2f}% confidence.")
            return {
                "domain": best_domain,
                "confidence_score": round(confidence, 2)
            }
            
        except Exception as e:
            logger.error(f"Error in detect_domain: {str(e)}")
            return {"domain": "General Analytics", "confidence_score": 0.0, "error": str(e)}

    def audit_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates a Data Quality Score based on missing %, duplicate %, outlier concentration, 
        dtype consistency, and invalid formats.

        Args:
            df (pd.DataFrame): The dataset to audit.

        Returns:
            Dict[str, Any]: Detailed score breakdown and proactive recommendations.
        """
        try:
            logger.info("Starting data quality audit.")
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame.")

            if df.empty:
                logger.warning("Empty DataFrame provided for data quality audit.")
                return {"quality_score": 0.0, "recommendations": ["Provide a non-empty dataset."]}

            total_rows = len(df)
            total_cells = df.size

            # 1. Missing % (Weight: 30%)
            missing_count = df.isnull().sum().sum()
            missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0
            missing_score = max(0, 100 - (missing_pct * 2))

            # 2. Duplicate % (Weight: 20%)
            duplicate_count = df.duplicated().sum()
            duplicate_pct = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
            duplicate_score = max(0, 100 - (duplicate_pct * 2))

            # 3. Outlier concentration (Weight: 20%)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_pct = 0
            if len(numeric_cols) > 0:
                q1 = df[numeric_cols].quantile(0.25)
                q3 = df[numeric_cols].quantile(0.75)
                iqr = q3 - q1
                outlier_condition = ((df[numeric_cols] < (q1 - 1.5 * iqr)) | (df[numeric_cols] > (q3 + 1.5 * iqr)))
                outlier_count = outlier_condition.sum().sum()
                total_numeric_cells = len(numeric_cols) * total_rows
                outlier_pct = (outlier_count / total_numeric_cells) * 100 if total_numeric_cells > 0 else 0
            outlier_score = max(0, 100 - (outlier_pct * 2))

            # 4. Dtype consistency & Invalid formats (Weight: 30%)
            # Simplified: checking if object columns have mixed types (e.g. string and int)
            mixed_type_cols = 0
            for col in df.select_dtypes(include=['object']).columns:
                types_in_col = df[col].dropna().apply(type).nunique()
                if types_in_col > 1:
                    mixed_type_cols += 1
            
            consistency_pct = (mixed_type_cols / len(df.columns)) * 100 if len(df.columns) > 0 else 0
            consistency_score = max(0, 100 - (consistency_pct * 5))

            # Overall Score
            overall_score = (missing_score * 0.3) + (duplicate_score * 0.2) + (outlier_score * 0.2) + (consistency_score * 0.3)

            # Recommendations
            recommendations = []
            if missing_pct > 5:
                recommendations.append(f"High missing data ({missing_pct:.1f}%). Consider imputation or dropping rows/columns with excessive nulls.")
            if duplicate_pct > 1:
                recommendations.append(f"Found duplicate rows ({duplicate_pct:.1f}%). Consider dropping duplicates.")
            if outlier_pct > 5:
                recommendations.append(f"Significant outliers detected ({outlier_pct:.1f}% in numeric columns). Consider capping or investigating anomalies.")
            if mixed_type_cols > 0:
                recommendations.append(f"Found {mixed_type_cols} column(s) with mixed data types. Standardize types for better processing.")
            
            if not recommendations:
                recommendations.append("Data quality looks excellent. No immediate actions required.")

            logger.info(f"Data quality audit complete. Score: {overall_score:.2f}")

            return {
                "quality_score": round(overall_score, 2),
                "breakdown": {
                    "missing_score": round(missing_score, 2),
                    "missing_percentage": round(missing_pct, 2),
                    "duplicate_score": round(duplicate_score, 2),
                    "duplicate_percentage": round(duplicate_pct, 2),
                    "outlier_score": round(outlier_score, 2),
                    "outlier_percentage": round(outlier_pct, 2),
                    "consistency_score": round(consistency_score, 2)
                },
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in audit_data_quality: {str(e)}")
            return {"quality_score": 0.0, "error": str(e), "recommendations": ["Error analyzing dataset."]}


# Global Singleton Instance
dataset_intelligence = DatasetIntelligence()
