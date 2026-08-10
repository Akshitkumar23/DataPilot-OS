import logging
import pandas as pd
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportGeneratorEngine:
    """
    Engine for generating professional executive reports in various formats.
    """
    
    def generate_executive_report_markdown(self, df: pd.DataFrame, dataset_name: str, quality_score: int, insights: List[str], code_snippets: List[str]) -> str:
        """
        Generates a professional 1-click Markdown/HTML executive report containing data profile, 
        quality score badge, key insights, embedded code, and recommended actions.
        
        Args:
            df (pd.DataFrame): The dataset being reported on.
            dataset_name (str): The name of the dataset.
            quality_score (int): The data quality score (0-100).
            insights (List[str]): A list of key insights.
            code_snippets (List[str]): A list of Python code snippets.
            
        Returns:
            str: The generated markdown report.
        """
        logger.info(f"Generating executive report for dataset: {dataset_name}")
        try:
            badge_color = "green" if quality_score >= 80 else ("yellow" if quality_score >= 60 else "red")
            
            report = f"# Executive Data Report: {dataset_name}\n\n"
            report += f"![Data Quality Score](https://img.shields.io/badge/Quality_Score-{quality_score}%25-{badge_color}?style=for-the-badge)\n\n"
            
            report += "## 📊 Data Profile\n"
            report += f"- **Total Rows:** {df.shape[0]:,}\n"
            report += f"- **Total Columns:** {df.shape[1]:,}\n"
            report += f"- **Missing Cells:** {df.isna().sum().sum():,}\n"
            report += f"- **Duplicate Rows:** {df.duplicated().sum():,}\n\n"
            
            report += "## 💡 Key Insights\n"
            if insights:
                for insight in insights:
                    report += f"- {insight}\n"
            else:
                report += "- No key insights generated.\n"
            report += "\n"
            
            report += "## 🛠️ Code Snippets\n"
            if code_snippets:
                for code in code_snippets:
                    report += f"```python\n{code}\n```\n"
            else:
                report += "- No code snippets included.\n"
            report += "\n"
            
            report += "## 📋 Recommended Actions\n"
            report += "- Review missing or duplicate values if present.\n"
            report += "- Execute the provided code snippets to reproduce the analysis.\n"
            
            logger.info("Executive report generated successfully.")
            return report
        except Exception as e:
            logger.error(f"Error generating executive report: {e}")
            raise


# Global Singleton Instance
report_generator = ReportGeneratorEngine()
