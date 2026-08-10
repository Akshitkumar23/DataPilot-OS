import os
import json
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
import joblib
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

def compute_knowledge_hash(db_path: str) -> Optional[str]:
    import hashlib
    if os.path.exists(db_path):
        try:
            with open(db_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None
    return None

def build_and_train_ai_brain(force: bool = False):
    """
    Builds the Master Data Science Knowledge Database (ds_pdf_knowledge_db.json)
    and trains the Machine Learning Meta-Classifier Model artifacts (models/datapilot_model.pkl).
    
    This brain is self-contained and pre-trained on Data Science & Analytics domain rules.
    It does not depend on any external PDF or local file path at runtime.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(base_dir, 'ds_pdf_knowledge_db.json')
    models_dir = os.path.join(base_dir, "models")
    model_file = os.path.join(models_dir, "datapilot_model.pkl")
    vec_file = os.path.join(models_dir, "vectorizer.pkl")
    vec_file_alt = os.path.join(models_dir, "datapilot_vectorizer.pkl")
    hash_file = os.path.join(models_dir, "knowledge_db.hash")

    if not force and os.path.exists(kb_path) and os.path.exists(model_file) and (os.path.exists(vec_file) or os.path.exists(vec_file_alt)) and os.path.exists(hash_file):
        current_hash = compute_knowledge_hash(kb_path)
        try:
            with open(hash_file, "r", encoding="utf-8") as f:
                stored_hash = f.read().strip()
            if current_hash and current_hash == stored_hash:
                print("[CACHE HIT] Knowledge DB hash unchanged. Pre-trained classifier artifacts loaded from cache.")
                return
        except Exception:
            pass

    print("=========================================================================")
    print("DATAPILOT OS: INITIALIZING MASTER AI BRAIN & TRAINING META-CLASSIFIER")
    print("=========================================================================")
    
    # Master Domain Rules (Learned & Self-Contained Knowledge Base)
    ds_knowledge_rules = [
        {
            "method": "Mean Imputation",
            "category": "Data Cleaning & Imputation",
            "when_to_apply": "When numerical column has missing null values and skewness is low (between -1.0 and +1.0).",
            "why_to_apply": "Preserves dataset sample size without altering mean distribution for normally distributed data.",
            "how_to_apply": "df['col'] = pd.to_numeric(df['col'], errors='coerce').fillna(df['col'].mean().round(2))"
        },
        {
            "method": "Median Imputation",
            "category": "Data Cleaning & Imputation",
            "when_to_apply": "When numerical column has missing null values and heavy skewness (> 1.0 or < -1.0) or extreme outliers.",
            "why_to_apply": "Median is robust against extreme outliers and skewed distributions, unlike Mean.",
            "how_to_apply": "df['col'] = pd.to_numeric(df['col'], errors='coerce').fillna(df['col'].median().round(2))"
        },
        {
            "method": "Mode Imputation",
            "category": "Data Cleaning & Imputation",
            "when_to_apply": "When categorical (string/object) column has missing null values.",
            "why_to_apply": "Replaces missing categories with the most frequent value without introducing invalid levels.",
            "how_to_apply": "df['col'] = df['col'].fillna(df['col'].mode()[0])"
        },
        {
            "method": "Interquartile Range (IQR) Outlier Capping",
            "category": "Outlier Detection & Capping",
            "when_to_apply": "When continuous numeric features contain extreme high/low values beyond [Q1 - 1.5*IQR, Q3 + 1.5*IQR].",
            "why_to_apply": "Prevents extreme outliers from corrupting Machine Learning model weights while retaining row count.",
            "how_to_apply": "Q1 = df['col'].quantile(0.25)\nQ3 = df['col'].quantile(0.75)\nIQR = Q3 - Q1\ndf['col'] = df['col'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)"
        },
        {
            "method": "One-Hot Encoding",
            "category": "Feature Engineering & Encoding",
            "when_to_apply": "When categorical feature has nominal (unordered) text categories and unique levels < 20.",
            "why_to_apply": "Converts categorical strings into numerical binary dummy columns required by linear & tree models.",
            "how_to_apply": "df = pd.get_dummies(df, columns=['col'], drop_first=True, dtype=int)"
        },
        {
            "method": "Standard Scaling (Z-Score Normalization)",
            "category": "Feature Scaling",
            "when_to_apply": "When features have varying numeric scales before training Distance/Gradient models (KNN, SVM, Linear Regression).",
            "why_to_apply": "Transforms features to mean=0 and variance=1 so large magnitude features do not dominate model learning.",
            "how_to_apply": "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)"
        },
        {
            "method": "Random Forest Classifier & Regressor",
            "category": "Supervised Machine Learning",
            "when_to_apply": "When target label is discrete categorical (Classification) or continuous numeric float (Regression).",
            "why_to_apply": "Ensemble of decision trees that reduces variance, resists overfitting, and measures feature importances.",
            "how_to_apply": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)"
        },
        {
            "method": "Histogram & Distribution Plotting",
            "category": "Exploratory Data Analysis & Matplotlib",
            "when_to_apply": "When analyzing frequency distribution, bell-curve normality, and skewness of a numeric column.",
            "why_to_apply": "Visually confirms whether a feature is Gaussian or skewed before deciding imputation strategy.",
            "how_to_apply": "import matplotlib.pyplot as plt\nplt.hist(df['col'], bins=20, color='skyblue', edgecolor='black')\nplt.title('Distribution of col')\nplt.show()"
        },
        {
            "method": "Boxplot Outlier Visualization",
            "category": "Exploratory Data Analysis & Seaborn",
            "when_to_apply": "When identifying extreme outlier points, median, and interquartile range boundaries.",
            "why_to_apply": "Displays five-number summary (Min, Q1, Median, Q3, Max) and isolated outlier dots.",
            "how_to_apply": "import seaborn as sns\nsns.boxplot(x=df['col'], color='teal')\nplt.title('Boxplot of col')\nplt.show()"
        },
        {
            "method": "Scatter Plot & Correlation Heatmap",
            "category": "Exploratory Data Analysis & Seaborn",
            "when_to_apply": "When assessing linear/non-linear relationships between two continuous numeric features.",
            "why_to_apply": "Discovers collinearity and strong predictor candidates for target variable.",
            "how_to_apply": "import seaborn as sns\nsns.heatmap(df.corr(), annot=True, cmap='coolwarm')\nplt.show()"
        },
        {
            "method": "K-Means Unsupervised Clustering",
            "category": "Unsupervised Machine Learning",
            "when_to_apply": "When discovering natural groupings, customer segments, or un-labeled clusters in data.",
            "why_to_apply": "Partitions observations into K distinct clusters minimizing within-cluster variance (inertia).",
            "how_to_apply": "from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=3, random_state=42)\ndf['Cluster'] = kmeans.fit_predict(X_scaled)"
        },
        {
            "method": "Principal Component Analysis (PCA)",
            "category": "Dimensionality Reduction",
            "when_to_apply": "When dataset contains high-dimensional collinear features (20+ columns) that need compression for visualization.",
            "why_to_apply": "Projects high-dimensional space onto 2 principal orthogonal axes retaining maximum variance.",
            "how_to_apply": "from sklearn.decomposition import PCA\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_scaled)"
        },
        {
            "method": "VIF Multicollinearity Check",
            "category": "Statistical Diagnostics",
            "when_to_apply": "When multiple numerical predictor columns show strong inter-correlations (VIF > 5.0).",
            "why_to_apply": "Detects redundant predictor variables that destabilize linear regression coefficients.",
            "how_to_apply": "from statsmodels.stats.outliers_influence import variance_inflation_factor\nvif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]"
        },
        {
            "method": "Yeo-Johnson Power Transformation",
            "category": "Advanced Feature Engineering",
            "when_to_apply": "When numeric features exhibit strong non-Gaussian skewness and contain zero or negative values.",
            "why_to_apply": "Transforms non-normal skewed feature distributions into Gaussian bell curves to satisfy linear assumptions.",
            "how_to_apply": "from sklearn.preprocessing import PowerTransformer\npt = PowerTransformer(method='yeo-johnson')\ndf['col'] = pt.fit_transform(df[['col']])"
        },
        {
            "method": "Datetime Decomposition Extraction",
            "category": "Feature Engineering & Time Series",
            "when_to_apply": "When dataset contains timestamp strings or datetime objects.",
            "why_to_apply": "Extracts granular temporal components (Year, Month, Day, Hour, DayOfWeek, Is_Weekend) for ML algorithms.",
            "how_to_apply": "df['year'] = pd.to_datetime(df['date']).dt.year\ndf['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek >= 5"
        },
        {
            "method": "SelectKBest Feature Selection",
            "category": "Feature Selection & Reduction",
            "when_to_apply": "When dataset has hundreds of feature columns and only top K informative features are needed.",
            "why_to_apply": "Ranks features using ANOVA F-score or Mutual Information to eliminate uninformative noise.",
            "how_to_apply": "from sklearn.feature_selection import SelectKBest, f_classif\nselector = SelectKBest(score_func=f_classif, k=10)\nX_new = selector.fit_transform(X, y)"
        },
        {
            "method": "Annotated Seaborn Bar Chart",
            "category": "Publication Data Visualization",
            "when_to_apply": "When visualizing aggregated group totals or percentages with exact value labels on top of bars.",
            "why_to_apply": "Displays clean, publication-ready bar plots with high-contrast text annotations and custom color palettes.",
            "how_to_apply": "import seaborn as sns\nimport matplotlib.pyplot as plt\nax = sns.barplot(data=df, x='x_col', y='y_col', palette='Blues_r')\nfor p in ax.patches:\n    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')\nplt.show()"
        },
        {
            "method": "Shapiro-Wilk Normality Test",
            "category": "Statistical Diagnostics",
            "when_to_apply": "When testing whether a continuous numeric column follows a normal Gaussian bell curve distribution.",
            "why_to_apply": "Computes p-value to statistically confirm Gaussian normality before parametric hypothesis testing.",
            "how_to_apply": "from scipy.stats import shapiro\nstat, p_val = shapiro(df['col'].dropna())\nprint(f'Shapiro Statistic: {stat:.4f}, p-value: {p_val:.4e}')"
        },
        {
            "method": "RFE Recursive Feature Elimination",
            "category": "Feature Selection & Reduction",
            "when_to_apply": "When iteratively selecting top K features by recursively pruning non-predictive variables.",
            "why_to_apply": "Recursively fits a model and removes the weakest features to optimize generalization score.",
            "how_to_apply": "from sklearn.feature_selection import RFE\nfrom sklearn.ensemble import RandomForestClassifier\nselector = RFE(RandomForestClassifier(), n_features_to_select=5)\nselector.fit(X, y)"
        },
        {
            "method": "Lasso L1 Feature Selection",
            "category": "Feature Selection & Reduction",
            "when_to_apply": "When driving coefficients of uninformative collinear features exactly to zero.",
            "why_to_apply": "Applies L1 regularization penalty to perform automated feature pruning.",
            "how_to_apply": "from sklearn.linear_model import LassoCV\nlasso = LassoCV().fit(X, y)\nselected_features = X.columns[lasso.coef_ != 0]"
        }
    ]

    kb_output = {
        "knowledge_source": "DataPilot OS Master AI Brain Domain Knowledge",
        "methods_knowledge_count": len(ds_knowledge_rules),
        "ds_methods_kb": ds_knowledge_rules
    }

    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb_output, f, indent=2)

    print(f"[1/3] Compiled {len(ds_knowledge_rules)} Master Data Science Domain Rules.")
    print(f"[2/3] Saved Master Knowledge Database -> '{kb_path}'")

    # Train/Test NLP Queries
    dataset_map = [
        # Cat 0: Mean Imputation
        ([
            "column has 5% missing numeric values normal distribution skew 0.1",
            "impute missing null values in numerical column with mean",
            "fill na in age column low skewness normal distribution",
            "numerical feature has null values skewness 0.2 fillmean",
            "missing float integers fill mean average strategy"
        ], [
            "replace missing numeric entries with column mean value",
            "normal distribution missing data mean imputation pandas"
        ], 0),

        # Cat 1: Median Imputation
        ([
            "column has 15% missing numeric values highly skewed distribution skew 2.4",
            "heavy skewness outliers fill missing with median",
            "impute nulls in income column median strategy fillna",
            "skewed data distribution fill median robust outliers",
            "missing numerical salary values median imputation"
        ], [
            "replace nulls in skewed salary column with median",
            "robust missing value imputation for outlier dataset"
        ], 1),

        # Cat 2: Mode Imputation
        ([
            "categorical column has 8% missing text values mode available",
            "fill na in gender string category mode most frequent",
            "impute missing text entries mode value fillna",
            "object string null values fill most common mode"
        ], [
            "fill missing text categories with mode value",
            "impute missing country labels using most frequent value"
        ], 2),

        # Cat 3: One-Hot Encoding
        ([
            "categorical text column with 3 unique distinct levels low cardinality",
            "one-hot encode categorical feature low cardinality",
            "convert text column into binary dummy variables get_dummies",
            "pandas get_dummies on category feature onehot"
        ], [
            "create dummy binary columns from category text",
            "encode categorical gender column with pd get_dummies"
        ], 3),

        # Cat 4: IQR Outlier Capping
        ([
            "numeric column with 12 extreme outliers outside 1.5 IQR boundary",
            "extreme outliers outside 1.5 iqr boundary interquartile range",
            "clip extreme values zscore iqr capping quantile",
            "detect and cap outliers using iqr quantile 0.25 0.75"
        ], [
            "cap extreme price outliers using 1.5 IQR quantile formula",
            "clip values exceeding upper and lower interquartile bounds"
        ], 4),

        # Cat 5: Supervised ML
        ([
            "target column categorical binary outcome label prediction classification",
            "predict target variable label outcome classification random forest",
            "supervised machine learning model accuracy f1 score",
            "train random forest classifier regression model fit score"
        ], [
            "fit random forest model on target column to predict churn",
            "supervised learning pipeline train_test_split fit evaluate score"
        ], 5),

        # Cat 6: Histogram
        ([
            "visualize frequency distribution histogram bell curve normality check",
            "histplot Gaussian kernel density kde plot",
            "plot distribution of numeric feature histogram frequency"
        ], [
            "draw histogram to check bell curve normality of price column",
            "matplotlib plt hist distribution frequency plot"
        ], 6),

        # Cat 7: Boxplot
        ([
            "box plot outlier visualization interquartile range",
            "five number summary median boxplot seaborn fliersize",
            "sns boxplot feature outlier dots quantile box"
        ], [
            "plot seaborn boxplot to detect outlier dots in salary",
            "visualize five number summary and median with boxplot"
        ], 7),

        # Cat 8: Correlation Heatmap
        ([
            "correlation heatmap scatter plot seaborn heatmap",
            "plot feature correlation matrix coolwarm blues",
            "bivariate scatter plot correlation heatmap seaborn"
        ], [
            "generate correlation heatmap matrix using seaborn sns heatmap",
            "check linear relationships between features with heatmap corr"
        ], 8),

        # Cat 9: K-Means
        ([
            "unsupervised k-means clustering customer segmentation elbow method",
            "kmeans n_clusters elbow method inertia fit_predict",
            "cluster observations un-labeled data kmeans centroids"
        ], [
            "run k-means clustering to segment customers into 3 clusters",
            "unsupervised learning kmeans n_clusters fit_predict"
        ], 9),

        # Cat 10: PCA
        ([
            "pca dimensionality reduction 2d projection feature compression",
            "principal component analysis compression high dimension",
            "pca n_components 2 orthogonal axes fit_transform"
        ], [
            "compress 30 continuous columns into 2 PCA principal components",
            "sklearn decomposition PCA fit_transform 2d projection"
        ], 10),

        # Cat 12: VIF Multicollinearity Check
        ([
            "calculate variance inflation factor vif multicollinearity score predictor features",
            "vif check multicollinearity redundant predictor variables regression",
            "statsmodels variance_inflation_factor vif multicollinearity test"
        ], [
            "run VIF diagnostic to check feature multicollinearity",
            "calculate variance inflation factor score for features"
        ], 12),

        # Cat 13: Yeo-Johnson Power Transform
        ([
            "yeo johnson power transformer skewness normalization gaussian curve",
            "transform non normal skewed distribution to bell curve yeo-johnson",
            "power transformer yeo johnson skew reduction"
        ], [
            "apply yeo johnson transformation to normalize skewed features",
            "sklearn preprocessing PowerTransformer yeo-johnson fit_transform"
        ], 13),

        # Cat 14: Datetime Decomposition
        ([
            "extract datetime components year month day hour is_weekend",
            "parse date timestamp string datetime feature engineering dt",
            "pd to_datetime extract date parts dayofweek"
        ], [
            "extract year month day and weekend flag from date column",
            "decompose timestamp into temporal components"
        ], 14),

        # Cat 15: SelectKBest Feature Selection
        ([
            "select k best feature selection anova f_classif mutual info",
            "rank top k features using selectkbest feature_selection",
            "eliminate uninformative feature noise selectkbest k=10"
        ], [
            "select top 10 most informative features using SelectKBest",
            "sklearn feature_selection SelectKBest fit_transform"
        ], 15),

        # Cat 16: Annotated Seaborn Bar Chart
        ([
            "annotated seaborn bar chart with percentage labels text on bars",
            "sns barplot annotate percentage values high contrast",
            "publication ready bar chart with value annotations"
        ], [
            "draw seaborn bar plot with exact percentage annotations on top",
            "annotate bar height values on seaborn bar chart"
        ], 16)
    ]

    X_train_text = []
    y_train = []
    X_test_text = []
    y_test = []

    for tr_list, te_list, label in dataset_map:
        for text in tr_list:
            for noise_id in range(12):
                X_train_text.append(f"{text} n_{np.random.randint(10, 99)} id_{noise_id}")
                y_train.append(label)
        for text in te_list:
            for noise_id in range(4):
                X_test_text.append(f"{text} t_{noise_id}")
                y_test.append(label)

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    clf.fit(X_train_vec, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_vec))
    test_acc = accuracy_score(y_test, clf.predict(X_test_vec))

    print("[3/3] Training Machine Learning Meta-Classifier Model...")
    print(f"      Model Training Accuracy: {round(train_acc * 100, 2)}%")
    print(f"      Model Out-of-Sample Test Accuracy: {round(test_acc * 100, 2)}%")

    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(models_dir, "datapilot_model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "datapilot_vectorizer.pkl"))

    current_hash = compute_knowledge_hash(kb_path)
    if current_hash:
        with open(os.path.join(models_dir, "knowledge_db.hash"), "w", encoding="utf-8") as f:
            f.write(current_hash)

    print("=========================================================================")
    print("SUCCESS: SELF-CONTAINED AI BRAIN TRAINING COMPLETE (STANDALONE & READY)")
    print("=========================================================================")


class DataScienceKnowledgeEngine:
    """
    Data Science Knowledge & Inference Engine powering DataPilot OS.
    Utilizes embedded statistical domain rules and trained ML Meta-Classifier artifacts.
    Self-contained: Does not depend on external file paths at runtime.
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(base_dir, "ds_pdf_knowledge_db.json")
        self.model_path = os.path.join(base_dir, "models", "datapilot_model.pkl")
        self.vec_path = os.path.join(base_dir, "models", "vectorizer.pkl")
        self.vec_path_alt = os.path.join(base_dir, "models", "datapilot_vectorizer.pkl")
        self.hash_path = os.path.join(base_dir, "models", "knowledge_db.hash")

        cache_valid = False
        if os.path.exists(self.db_path) and os.path.exists(self.model_path) and (os.path.exists(self.vec_path) or os.path.exists(self.vec_path_alt)) and os.path.exists(self.hash_path):
            current_hash = compute_knowledge_hash(self.db_path)
            try:
                with open(self.hash_path, "r", encoding="utf-8") as f:
                    stored_hash = f.read().strip()
                if current_hash and current_hash == stored_hash:
                    cache_valid = True
            except Exception:
                cache_valid = False

        if not cache_valid:
            try:
                build_and_train_ai_brain(force=True)
            except Exception as e:
                print("Self-training warning:", e)
        else:
            print("[CACHE HIT] Loading pre-trained classifier and vectorizer directly from disk.")

        self.knowledge_db = {}
        self.clf_model = None
        self.vectorizer = None

        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.knowledge_db = json.load(f)
            except Exception as e:
                print(f"Error loading knowledge DB from '{self.db_path}', attempting fallback rebuild: {e}")
                try:
                    build_and_train_ai_brain(force=True)
                    if os.path.exists(self.db_path):
                        with open(self.db_path, "r", encoding="utf-8") as f:
                            self.knowledge_db = json.load(f)
                except Exception as rebuild_err:
                    print("Error rebuilding knowledge DB:", rebuild_err)
                    self.knowledge_db = {}

        if os.path.exists(self.model_path):
            try:
                self.clf_model = joblib.load(self.model_path)
            except Exception as e:
                print("Error loading ML model:", e)

        v_path = self.vec_path if os.path.exists(self.vec_path) else self.vec_path_alt
        if os.path.exists(v_path):
            try:
                self.vectorizer = joblib.load(v_path)
            except Exception as e:
                print("Error loading vectorizer:", e)

        # Initialize Enterprise Modular Architecture Components
        try:
            from modules.capability_registry import capability_registry
            from modules.intent_layer import intent_parser, conversation_memory
            from modules.dataset_intelligence import dataset_intelligence
            from modules.security_cache import security_guard, execution_cache
            from modules.planning_agent import planning_agent
            from modules.auto_insights import auto_insights

            self.registry = capability_registry
            self.intent_parser = intent_parser
            self.memory = conversation_memory
            self.dataset_intelligence = dataset_intelligence
            self.security_guard = security_guard
            self.execution_cache = execution_cache
            self.planning_agent = planning_agent
            self.auto_insights = auto_insights
        except Exception as mod_err:
            print("Warning initializing enterprise modules:", mod_err)
            self.registry = None
            self.intent_parser = None
            self.memory = None
            self.dataset_intelligence = None
            self.security_guard = None
            self.execution_cache = None
            self.planning_agent = None
            self.auto_insights = None


    def analyze_dataset_and_recommend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes an incoming pandas DataFrame and returns automated Data Science solutions.
        """
        rows, cols_count = df.shape
        problems_detected = []
        recommended_goals = []
        code_solutions = []

        schema_summary = {}
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            null_cnt = int(df[col].isnull().sum())
            null_pct = round((null_cnt / rows) * 100, 2) if rows > 0 else 0
            unique_cnt = int(df[col].nunique())
            schema_summary[str(col)] = {
                "dtype": dtype_str,
                "null_count": null_cnt,
                "null_percentage": null_pct,
                "unique_count": unique_cnt
            }

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Rule 1: Handling Null Values (Mean vs Median vs Mode)
        for col in df.columns:
            null_pct = schema_summary[str(col)]["null_percentage"]
            if null_pct > 0:
                if col in num_cols:
                    skewness = df[col].skew() if len(df[col].dropna()) > 3 else 0
                    if abs(skewness) > 1.0:
                        problems_detected.append(f"Column '{col}' has {null_pct}% missing values with high skewness ({round(skewness, 2)}).")
                        recommended_goals.append(f"Impute missing values in '{col}' using Median (Robust Imputation).")
                        code_solutions.append({
                            "id": f"impute_median_{col}",
                            "title": f"Median Imputation for '{col}'",
                            "problem": f"Heavy skewness ({round(skewness, 2)}) in '{col}' with {null_pct}% nulls.",
                            "code": f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce').fillna(df['{col}'].median().round(2))",
                            "op": "impute",
                            "col": col
                        })
                    else:
                        problems_detected.append(f"Column '{col}' has {null_pct}% missing values with low skewness ({round(skewness, 2)}).")
                        recommended_goals.append(f"Impute missing values in '{col}' using Mean (Gaussian Imputation).")
                        code_solutions.append({
                            "id": f"impute_mean_{col}",
                            "title": f"Mean Imputation for '{col}'",
                            "problem": f"Low skewness ({round(skewness, 2)}) in '{col}' with {null_pct}% nulls.",
                            "code": f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce').fillna(df['{col}'].mean().round(2))",
                            "op": "impute",
                            "col": col
                        })
                else:
                    problems_detected.append(f"Categorical column '{col}' has {null_pct}% missing text entries.")
                    recommended_goals.append(f"Impute missing categories in '{col}' using Mode (Most Frequent).")
                    code_solutions.append({
                        "id": f"impute_mode_{col}",
                        "title": f"Mode Imputation for '{col}'",
                        "problem": f"Missing text categories in '{col}' ({null_pct}% nulls).",
                        "code": f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])",
                        "op": "impute",
                        "col": col
                    })

        # Rule 2: One-Hot Encoding for Low Cardinality Categoricals
        for col in cat_cols:
            u_cnt = schema_summary[str(col)]["unique_count"]
            if 1 < u_cnt < 20:
                recommended_goals.append(f"One-Hot Encode categorical feature '{col}' ({u_cnt} levels).")
                code_solutions.append({
                    "id": f"encode_{col}",
                    "title": f"One-Hot Encode '{col}'",
                    "problem": f"Categorical string feature '{col}' requires dummy encoding for ML models.",
                    "code": f"df = pd.get_dummies(df, columns=['{col}'], drop_first=True, dtype=int)",
                    "op": "encode",
                    "col": col
                })

        # Rule 3: IQR Outlier Detection
        for col in num_cols:
            num_series = df[col].dropna()
            if len(num_series) > 10:
                q1 = num_series.quantile(0.25)
                q3 = num_series.quantile(0.75)
                iqr = q3 - q1
                outliers = num_series[(num_series < (q1 - 1.5 * iqr)) | (num_series > (q3 + 1.5 * iqr))]
                if len(outliers) > 0:
                    recommended_goals.append(f"Cap {len(outliers)} outliers in column '{col}' using Interquartile Range (IQR).")
                    code_solutions.append({
                        "id": f"clip_outliers_{col}",
                        "title": f"IQR Outlier Capping for '{col}'",
                        "problem": f"Detected {len(outliers)} extreme outlier values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] bounds.",
                        "code": f"Q1 = df['{col}'].quantile(0.25)\nQ3 = df['{col}'].quantile(0.75)\nIQR = Q3 - Q1\ndf['{col}'] = df['{col}'].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)",
                        "op": "clip",
                        "col": col
                    })

        # Rule 4: ML Model Selection & Training Inference
        target_candidate = df.columns[-1] if len(df.columns) > 0 else None
        if target_candidate and target_candidate in df.columns:
            target_is_cat = not pd.api.types.is_numeric_dtype(df[target_candidate]) or df[target_candidate].nunique() < 10
            model_type_name = "RandomForestClassifier" if target_is_cat else "RandomForestRegressor"
            metric_name = "Classification Accuracy & F1-Score" if target_is_cat else "Mean Squared Error (MSE) & R2-Score"

            recommended_goals.append(f"Train {model_type_name} model to predict target '{target_candidate}' and evaluate {metric_name}.")
            code_solutions.append({
                "id": "train_model",
                "title": f"Train {model_type_name} Model on '{target_candidate}'",
                "problem": f"Target column '{target_candidate}' ready for supervised ML model training.",
                "code": f"# Train-Test Split and Model Fitting\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import {model_type_name}\n\nX = pd.get_dummies(df.drop(columns=['{target_candidate}']), drop_first=True, dtype=int).fillna(0)\ny = df['{target_candidate}']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nmodel = {model_type_name}(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\nprint('Model Performance Score:', round(model.score(X_test, y_test) * 100, 2), '%')",
                "op": "train",
                "col": target_candidate
            })

        return {
            "source": "DataPilot OS Embedded AI Knowledge Engine",
            "total_rows": rows,
            "total_columns": cols_count,
            "target_column": target_candidate,
            "schema_summary": schema_summary,
            "problems_detected": problems_detected,
            "recommended_goals": recommended_goals,
            "code_solutions": code_solutions
        }

    def get_business_glossary(self) -> Dict[str, Dict[str, str]]:
        """Returns the Enterprise Business Glossary & Metric Dictionary."""
        return {
            "Revenue": {
                "category": "Finance",
                "definition": "Total gross income generated from goods or services sold.",
                "formula": "Sum of Sales Amount"
            },
            "Gross Margin": {
                "category": "Finance",
                "definition": "Percentage of total sales revenue retained after incurring direct costs.",
                "formula": "(Revenue - Cost of Goods Sold) / Revenue"
            },
            "Customer Churn Rate": {
                "category": "SaaS & Retail",
                "definition": "Percentage of customers who discontinue subscriptions or purchases within a given timeframe.",
                "formula": "(Lost Customers at End of Period / Total Customers at Start of Period) * 100"
            },
            "Customer Lifetime Value (CLV)": {
                "category": "Marketing & SaaS",
                "definition": "Total net profit attributed to the entire future relationship with a customer.",
                "formula": "Average Order Value * Purchase Frequency * Customer Value Lifetime"
            },
            "Hospital Readmission Rate": {
                "category": "Healthcare",
                "definition": "Percentage of patients unexpectedly readmitted to hospital within 30 days of discharge.",
                "formula": "(Readmitted Patients within 30 days / Total Discharged Patients) * 100"
            }
        }

    def get_domain_rules(self, domain: str = "all") -> List[Dict[str, Any]]:
        """Returns domain-specific analytical rules."""
        rules = [
            {"domain": "Finance", "rule": "Flag transactions > 3 standard deviations above user rolling mean as potential anomaly/fraud."},
            {"domain": "Retail", "rule": "Calculate safety stock levels using 95% service level normal z-score (1.65)."},
            {"domain": "Healthcare", "rule": "Enforce strict HIPAA de-identification guard on PII columns (SSN, Phone, Address)."},
            {"domain": "SaaS", "rule": "Compute Monthly Recurrent Revenue (MRR) expansion vs contraction cohorts."}
        ]
        if domain != "all":
            return [r for r in rules if r["domain"].lower() == domain.lower()]
        return rules

    analyze_dataset_with_knowledge = analyze_dataset_and_recommend

ds_engine = DataScienceKnowledgeEngine()

if __name__ == "__main__":
    build_and_train_ai_brain()

