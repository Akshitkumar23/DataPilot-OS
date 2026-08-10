"""
DataPilot OS - Jinja2 Template Engine for Code Generation
Provides centralized, clean rendering of Python code templates with error handling and fallbacks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

logger = logging.getLogger(__name__)


class TemplateEngine:
    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            # Default to templates/ relative to project root
            base_dir = Path(__file__).resolve().parent.parent
            templates_dir = base_dir / "templates"
        
        self.templates_dir = templates_dir
        self.env: Optional[jinja2.Environment] = None
        
        if HAS_JINJA2 and self.templates_dir.exists():
            try:
                loader = jinja2.FileSystemLoader(str(self.templates_dir))
                self.env = jinja2.Environment(
                    loader=loader,
                    trim_blocks=True,
                    lstrip_blocks=True,
                    keep_trailing_newline=True,
                    autoescape=False
                )
            except Exception as e:
                logger.error(f"Failed to initialize Jinja2 environment: {e}")
                self.env = None
        else:
            if not HAS_JINJA2:
                logger.warning("jinja2 package is not installed. Code generation will fallback to default strings.")
            elif not self.templates_dir.exists():
                logger.warning(f"Templates directory not found at {self.templates_dir}.")

    def render(self, template_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Renders a Jinja2 template with context. If rendering fails, returns a fallback code snippet.
        """
        context = context or {}
        if "df_var" not in context:
            context["df_var"] = "df"

        if not self.env:
            logger.error(f"Jinja2 environment uninitialized when rendering '{template_name}'.")
            return f"# [FALLBACK] Unable to render {template_name}: Jinja2 environment not ready."

        try:
            # Ensure .j2 extension if omitted
            if not template_name.endswith(".j2"):
                template_name = f"{template_name}.j2"
                
            tmpl = self.env.get_template(template_name)
            rendered = tmpl.render(**context)
            return rendered.strip()
        except Exception as e:
            logger.error(f"Error rendering template '{template_name}': {e}", exc_info=True)
            return f"# [ERROR] Failed to render template '{template_name}': {e}"

    # ── Specialized Wrapper Methods ──

    def render_eda_dashboard(
        self,
        best_cat: Optional[str] = None,
        best_num1: Optional[str] = None,
        best_num2: Optional[str] = None,
        df_var: str = "df"
    ) -> str:
        return self.render("eda_dashboard.j2", {
            "best_cat": best_cat,
            "best_num1": best_num1,
            "best_num2": best_num2,
            "df_var": df_var
        })

    def render_forecast(
        self,
        dt_col: str,
        y_col: str,
        horizon: int = 30,
        is_forecast: bool = True,
        df_var: str = "df"
    ) -> str:
        return self.render("forecast.j2", {
            "dt_col": dt_col,
            "y_col": y_col,
            "horizon": horizon,
            "is_forecast": is_forecast,
            "df_var": df_var
        })

    def render_correlation_heatmap(
        self,
        num_cols: Optional[List[str]] = None,
        title: str = "Feature Correlation Heatmap",
        df_var: str = "df"
    ) -> str:
        return self.render("correlation_heatmap.j2", {
            "num_cols": num_cols,
            "title": title,
            "df_var": df_var
        })

    def render_ml_training(
        self,
        target: str,
        model_cls: str = "RandomForestClassifier",
        is_feature_eng: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
        df_var: str = "df"
    ) -> str:
        return self.render("ml_training.j2", {
            "target": target,
            "model_cls": model_cls,
            "is_feature_eng": is_feature_eng,
            "test_size": test_size,
            "random_state": random_state,
            "df_var": df_var
        })

    def render_cleaning(
        self,
        target_col: Optional[str] = None,
        strategy: Optional[str] = None,
        null_count: int = 0,
        skewness: float = 0.0,
        is_audit: bool = False,
        df_var: str = "df"
    ) -> str:
        return self.render("cleaning.j2", {
            "target_col": target_col,
            "strategy": strategy,
            "null_count": null_count,
            "skewness": skewness,
            "is_audit": is_audit,
            "df_var": df_var
        })

    def render_distribution_hist(
        self,
        col: str,
        bins: int = 30,
        color: str = "#0284c7",
        df_var: str = "df"
    ) -> str:
        return self.render("distribution_hist.j2", {
            "col": col,
            "bins": bins,
            "color": color,
            "df_var": df_var
        })

    def render_outlier_boxplot(
        self,
        y_col: str,
        x_col: Optional[str] = None,
        df_var: str = "df"
    ) -> str:
        return self.render("outlier_boxplot.j2", {
            "y_col": y_col,
            "x_col": x_col,
            "df_var": df_var
        })

    def render_scatter_plot(
        self,
        x_col: str,
        y_col: str,
        hue_col: Optional[str] = None,
        df_var: str = "df"
    ) -> str:
        return self.render("scatter_plot.j2", {
            "x_col": x_col,
            "y_col": y_col,
            "hue_col": hue_col,
            "df_var": df_var
        })

    def render_top_n_bar(
        self,
        col: str,
        val_col: Optional[str] = None,
        n: int = 10,
        df_var: str = "df"
    ) -> str:
        return self.render("top_n_bar.j2", {
            "col": col,
            "val_col": val_col,
            "n": n,
            "df_var": df_var
        })

    def render_pca(
        self,
        n_components: int = 2,
        df_var: str = "df"
    ) -> str:
        return self.render("pca.j2", {
            "n_components": n_components,
            "df_var": df_var
        })

    def render_kmeans_elbow(
        self,
        k_min: int = 2,
        k_max: int = 8,
        df_var: str = "df"
    ) -> str:
        return self.render("kmeans_elbow.j2", {
            "k_min": k_min,
            "k_max": k_max,
            "df_var": df_var
        })

    def render_data_profiling(
        self,
        df_var: str = "df"
    ) -> str:
        return self.render("data_profiling.j2", {
            "df_var": df_var
        })

    def render_encoding_scaling(
        self,
        mode: str = "one_hot",
        col: str = "",
        df_var: str = "df"
    ) -> str:
        return self.render("encoding_scaling.j2", {
            "mode": mode,
            "col": col,
            "df_var": df_var
        })


# Global Singleton Instance
template_engine = TemplateEngine()
