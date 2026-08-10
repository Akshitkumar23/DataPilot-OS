"""
Domain Packs Module for DataPilot OS.
This module provides domain-specific KPI calculations and chart code generators.
"""

import logging
from typing import Dict, Any

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DomainPacksEngine:
    """
    Engine for generating domain-specific KPIs and visualizations.
    
    Provides specialized methods for calculating metrics and generating
    Python code for charts across different domains like Retail, HR,
    Finance, Operations, Healthcare, Marketing, and Education.
    """

    def __init__(self) -> None:
        """Initialize the DomainPacksEngine."""
        logger.info("Initializing DomainPacksEngine.")

    def calculate_retail_pack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Retail KPIs and generate visualization code.
        
        Metrics: Revenue, AOV, RFM Segmentation, Basket Size, Inventory Turnover.
        
        Args:
            data (Dict[str, Any]): Dictionary containing retail data.
            
        Returns:
            Dict[str, Any]: Dictionary containing calculated KPIs and chart generation code.
        """
        logger.info("Calculating Retail Pack KPIs.")
        try:
            kpis = {
                "Revenue": data.get("revenue", 0.0),
                "AOV": data.get("aov", 0.0),
                "RFM_Segmentation": data.get("rfm", {}),
                "Basket_Size": data.get("basket_size", 0.0),
                "Inventory_Turnover": data.get("inventory_turnover", 0.0),
            }
            chart_code = '''
import matplotlib.pyplot as plt
def plot_retail_kpis(kpis):
    fig, ax = plt.subplots()
    ax.bar(['Revenue', 'AOV', 'Basket Size'], [kpis['Revenue'], kpis['AOV'], kpis['Basket_Size']])
    plt.title('Retail KPIs')
    return fig
'''
            return {"kpis": kpis, "chart_code": chart_code.strip()}
        except Exception as e:
            logger.error(f"Error calculating Retail KPIs: {e}")
            raise

    def calculate_hr_pack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate HR KPIs and generate visualization code.
        
        Metrics: Employee Attrition Rate %, Salary Band distribution, Tenure, Promotion Rate.
        
        Args:
            data (Dict[str, Any]): Dictionary containing HR data.
            
        Returns:
            Dict[str, Any]: Dictionary containing calculated KPIs and chart generation code.
        """
        logger.info("Calculating HR Pack KPIs.")
        try:
            kpis = {
                "Employee_Attrition_Rate_Pct": data.get("attrition_rate", 0.0),
                "Salary_Band_Distribution": data.get("salary_bands", {}),
                "Tenure": data.get("tenure", 0.0),
                "Promotion_Rate": data.get("promotion_rate", 0.0),
            }
            chart_code = '''
import matplotlib.pyplot as plt
def plot_hr_kpis(kpis):
    fig, ax = plt.subplots()
    labels = list(kpis['Salary_Band_Distribution'].keys())
    values = list(kpis['Salary_Band_Distribution'].values())
    if labels and values:
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        plt.title('Salary Band Distribution')
    return fig
'''
            return {"kpis": kpis, "chart_code": chart_code.strip()}
        except Exception as e:
            logger.error(f"Error calculating HR KPIs: {e}")
            raise

    def calculate_finance_pack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Finance KPIs and generate visualization code.
        
        Metrics: Gross Margin %, EBITDA Margin %, ROE, ROA, Debt-to-Equity, Cash Flow.
        
        Args:
            data (Dict[str, Any]): Dictionary containing finance data.
            
        Returns:
            Dict[str, Any]: Dictionary containing calculated KPIs and chart generation code.
        """
        logger.info("Calculating Finance Pack KPIs.")
        try:
            kpis = {
                "Gross_Margin_Pct": data.get("gross_margin", 0.0),
                "EBITDA_Margin_Pct": data.get("ebitda_margin", 0.0),
                "ROE": data.get("roe", 0.0),
                "ROA": data.get("roa", 0.0),
                "Debt_to_Equity": data.get("debt_to_equity", 0.0),
                "Cash_Flow": data.get("cash_flow", 0.0),
            }
            chart_code = '''
import matplotlib.pyplot as plt
def plot_finance_kpis(kpis):
    fig, ax = plt.subplots()
    metrics = ['Gross Margin %', 'EBITDA Margin %', 'ROE', 'ROA']
    values = [kpis.get('Gross_Margin_Pct', 0), kpis.get('EBITDA_Margin_Pct', 0), kpis.get('ROE', 0), kpis.get('ROA', 0)]
    ax.bar(metrics, values, color='green')
    plt.title('Financial Metrics')
    return fig
'''
            return {"kpis": kpis, "chart_code": chart_code.strip()}
        except Exception as e:
            logger.error(f"Error calculating Finance KPIs: {e}")
            raise

    def calculate_operations_pack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Operations KPIs and generate visualization code.
        
        Metrics: OEE %, Machine Downtime, Reject Rate %, Lead Time, SLA Fill Rate %.
        
        Args:
            data (Dict[str, Any]): Dictionary containing operations data.
            
        Returns:
            Dict[str, Any]: Dictionary containing calculated KPIs and chart generation code.
        """
        logger.info("Calculating Operations Pack KPIs.")
        try:
            kpis = {
                "OEE_Pct": data.get("oee", 0.0),
                "Machine_Downtime": data.get("downtime", 0.0),
                "Reject_Rate_Pct": data.get("reject_rate", 0.0),
                "Lead_Time": data.get("lead_time", 0.0),
                "SLA_Fill_Rate_Pct": data.get("sla_fill_rate", 0.0),
            }
            chart_code = '''
import matplotlib.pyplot as plt
def plot_operations_kpis(kpis):
    fig, ax = plt.subplots()
    metrics = ['OEE %', 'Reject Rate %', 'SLA Fill Rate %']
    values = [kpis.get('OEE_Pct', 0), kpis.get('Reject_Rate_Pct', 0), kpis.get('SLA_Fill_Rate_Pct', 0)]
    ax.bar(metrics, values, color='orange')
    plt.title('Operations Efficiency')
    return fig
'''
            return {"kpis": kpis, "chart_code": chart_code.strip()}
        except Exception as e:
            logger.error(f"Error calculating Operations KPIs: {e}")
            raise

    def calculate_healthcare_pack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Healthcare KPIs and generate visualization code.
        
        Metrics: Patient Readmission Rate %, Length of Stay LOS, Bed Utilization.
        
        Args:
            data (Dict[str, Any]): Dictionary containing healthcare data.
            
        Returns:
            Dict[str, Any]: Dictionary containing calculated KPIs and chart generation code.
        """
        logger.info("Calculating Healthcare Pack KPIs.")
        try:
            kpis = {
                "Patient_Readmission_Rate_Pct": data.get("readmission_rate", 0.0),
                "Length_of_Stay_LOS": data.get("los", 0.0),
                "Bed_Utilization": data.get("bed_utilization", 0.0),
            }
            chart_code = '''
import matplotlib.pyplot as plt
def plot_healthcare_kpis(kpis):
    fig, ax = plt.subplots()
    metrics = ['Readmission Rate %', 'Bed Utilization']
    values = [kpis.get('Patient_Readmission_Rate_Pct', 0), kpis.get('Bed_Utilization', 0)]
    ax.bar(metrics, values, color='purple')
    plt.title('Healthcare Facility Usage')
    return fig
'''
            return {"kpis": kpis, "chart_code": chart_code.strip()}
        except Exception as e:
            logger.error(f"Error calculating Healthcare KPIs: {e}")
            raise

    def calculate_marketing_pack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Marketing KPIs and generate visualization code.
        
        Metrics: CTR %, CPA, CAC, ROAS, Customer Lifetime Value LTV.
        
        Args:
            data (Dict[str, Any]): Dictionary containing marketing data.
            
        Returns:
            Dict[str, Any]: Dictionary containing calculated KPIs and chart generation code.
        """
        logger.info("Calculating Marketing Pack KPIs.")
        try:
            kpis = {
                "CTR_Pct": data.get("ctr", 0.0),
                "CPA": data.get("cpa", 0.0),
                "CAC": data.get("cac", 0.0),
                "ROAS": data.get("roas", 0.0),
                "Customer_Lifetime_Value_LTV": data.get("ltv", 0.0),
            }
            chart_code = '''
import matplotlib.pyplot as plt
def plot_marketing_kpis(kpis):
    fig, ax = plt.subplots()
    metrics = ['CPA', 'CAC', 'ROAS']
    values = [kpis.get('CPA', 0), kpis.get('CAC', 0), kpis.get('ROAS', 0)]
    ax.bar(metrics, values, color='red')
    plt.title('Marketing Cost & Return')
    return fig
'''
            return {"kpis": kpis, "chart_code": chart_code.strip()}
        except Exception as e:
            logger.error(f"Error calculating Marketing KPIs: {e}")
            raise

    def calculate_education_pack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Education KPIs and generate visualization code.
        
        Metrics: Student Score percentiles, Pass Rate %, Grade distribution.
        
        Args:
            data (Dict[str, Any]): Dictionary containing education data.
            
        Returns:
            Dict[str, Any]: Dictionary containing calculated KPIs and chart generation code.
        """
        logger.info("Calculating Education Pack KPIs.")
        try:
            kpis = {
                "Student_Score_Percentiles": data.get("score_percentiles", {}),
                "Pass_Rate_Pct": data.get("pass_rate", 0.0),
                "Grade_Distribution": data.get("grade_distribution", {}),
            }
            chart_code = '''
import matplotlib.pyplot as plt
def plot_education_kpis(kpis):
    fig, ax = plt.subplots()
    labels = list(kpis.get('Grade_Distribution', {}).keys())
    values = list(kpis.get('Grade_Distribution', {}).values())
    if labels and values:
        ax.bar(labels, values, color='blue')
        plt.title('Grade Distribution')
    return fig
'''
            return {"kpis": kpis, "chart_code": chart_code.strip()}
        except Exception as e:
            logger.error(f"Error calculating Education KPIs: {e}")
            raise


# Global Singleton Instance
domain_packs = DomainPacksEngine()
