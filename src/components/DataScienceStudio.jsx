import { useState } from 'react';
import { 
  BarChart2, 
  Cpu, 
  Wrench, 
  Layers, 
  Sparkles, 
  Play, 
  CheckCircle2, 
  Database,
  TrendingUp,
  Table,
  Sliders
} from 'lucide-react';

const API = import.meta.env.VITE_API_URL || window.ENV_API_URL || 'http://localhost:8000';

export default function DataScienceStudio({ schema, rowCount, pipelineSteps, onRunCode, onRefresh }) {
  const [activeTab, setActiveTab] = useState('eda');
  const [selectedTarget, setSelectedTarget] = useState('');
  const [selectedModel, setSelectedModel] = useState('auto');
  const [trainingResult, setTrainingResult] = useState(null);
  const [isTraining, setIsTraining] = useState(false);

  const columns = schema ? Object.keys(schema) : [];

  const handleTrainModel = async () => {
    if (!selectedTarget) {
      alert("Please select a target column to train the Machine Learning model.");
      return;
    }
    setIsTraining(true);
    setTrainingResult(null);

    try {
      const resp = await fetch(`${API}/api/ml/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_column: selectedTarget,
          algorithm: selectedModel,
          test_size: 0.2
        })
      });
      const data = await resp.json();
      if (data.success || data.model_path) {
        setTrainingResult(data);
        if (onRefresh) onRefresh();
      } else {
        alert(`Training failed: ${data.detail || data.error || 'Unknown error'}`);
      }
    } catch (err) {
      alert(`Network error during ML model training: ${err.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: '#090d16',
      color: '#f8fafc',
      fontFamily: 'var(--font-sans, system-ui, sans-serif)',
      overflow: 'hidden'
    }}>
      {/* Studio Header */}
      <div style={{
        padding: '14px 20px',
        background: '#0f172a',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            background: 'linear-gradient(135deg, #0284c7, #0ea5e9)',
            padding: 8,
            borderRadius: 8,
            display: 'flex'
          }}>
            <Cpu size={20} color="#ffffff" />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: 16, fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.01em' }}>
              Master Data Science & AutoML Studio
            </h2>
            <p style={{ margin: 0, fontSize: 11, color: '#94a3b8' }}>
              Statistical Diagnostics, Feature Engineering, Publication Plots & ML Training Pipeline
            </p>
          </div>
        </div>

        {/* Dataset Meta Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          background: '#1e293b',
          padding: '6px 14px',
          borderRadius: 20,
          border: '1px solid #334155',
          fontSize: 12,
          fontWeight: 600,
          color: '#38bdf8'
        }}>
          <Database size={14} color="#38bdf8" />
          <span>{(rowCount || 0).toLocaleString()} Rows</span>
          <span style={{ color: '#475569' }}>|</span>
          <span>{columns.length} Columns</span>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div style={{
        display: 'flex',
        background: '#0c121e',
        borderBottom: '1px solid #1e293b',
        padding: '0 16px',
        gap: 4,
        flexShrink: 0
      }}>
        {[
          { id: 'eda', label: '📊 Exploratory Diagnostics (EDA)', icon: BarChart2 },
          { id: 'prep', label: '🧹 Cleaning & Imputation', icon: Wrench },
          { id: 'feature', label: '⚡ Feature Engineering', icon: Sliders },
          { id: 'ml', label: '🤖 AutoML & Model Studio', icon: Sparkles },
          { id: 'tree', label: '🌲 Pipeline History Tree', icon: Layers }
        ].map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '12px 18px',
                background: isActive ? '#0f172a' : 'transparent',
                border: 'none',
                borderBottom: isActive ? '2px solid #0ea5e9' : '2px solid transparent',
                color: isActive ? '#38bdf8' : '#94a3b8',
                fontWeight: isActive ? 700 : 500,
                fontSize: 13,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              <Icon size={16} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Studio Content Area */}
      <div style={{ flex: 1, padding: 20, overflowY: 'auto', background: '#090d16' }}>
        
        {/* TAB 1: EDA & Diagnostics */}
        {activeTab === 'eda' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, color: '#38bdf8' }}>
                <TrendingUp size={18} />
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700 }}>Correlation & Heatmap Analysis</h3>
              </div>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Generate a publication-ready Seaborn correlation heatmap to discover linear relationships and collinearity among numerical features.
              </p>
              <button
                onClick={() => onRunCode && onRunCode(`import seaborn as sns\nimport matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6), dpi=150)\nsns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Feature Correlation Matrix', fontsize=14, pad=12)\nplt.tight_layout()\nplt.show()`)}
                style={{
                  width: '100%',
                  background: '#0284c7',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6
                }}
              >
                <Play size={14} /> Run Correlation Matrix Plot
              </button>
            </div>

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, color: '#38bdf8' }}>
                <Table size={18} />
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700 }}>VIF Multicollinearity Check</h3>
              </div>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Calculate Variance Inflation Factor (VIF) scores for continuous numerical features to prune redundant collinear predictor variables.
              </p>
              <button
                onClick={() => onRunCode && onRunCode(`from statsmodels.stats.outliers_influence import variance_inflation_factor\nnum_df = df.select_dtypes(include=['number']).dropna()\nif not num_df.empty:\n    vif_data = pd.DataFrame()\n    vif_data['Feature'] = num_df.columns\n    vif_data['VIF Score'] = [variance_inflation_factor(num_df.values, i) for i in range(num_df.shape[1])]\n    print(vif_data.sort_values(by='VIF Score', ascending=False))`)}
                style={{
                  width: '100%',
                  background: '#0ea5e9',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6
                }}
              >
                <Play size={14} /> Run VIF Diagnostic
              </button>
            </div>

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, color: '#38bdf8' }}>
                <BarChart2 size={18} />
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700 }}>Shapiro-Wilk Normality & Skewness Check</h3>
              </div>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Inspect skewness and statistical bell-curve normality across continuous numeric variables to determine Gaussian vs Median imputation.
              </p>
              <button
                onClick={() => onRunCode && onRunCode(`import pandas as pd\nskew_df = pd.DataFrame({'Skewness': df.select_dtypes(include=['number']).skew()})\nskew_df['Status'] = skew_df['Skewness'].apply(lambda s: 'Highly Skewed (>1.0)' if abs(s) > 1.0 else 'Gaussian / Low Skew')\nprint(skew_df)`)}
                style={{
                  width: '100%',
                  background: '#0284c7',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6
                }}
              >
                <Play size={14} /> Run Skewness & Normality Analysis
              </button>
            </div>
          </div>
        )}

        {/* TAB 2: Cleaning & Preprocessing */}
        {activeTab === 'prep' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>
                IQR Outlier Capping (1.5x Boundary)
              </h3>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Automatically detect and clip extreme values beyond upper and lower interquartile bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
              </p>
              <button
                onClick={() => onRunCode && onRunCode(`for col in df.select_dtypes(include=['number']).columns:\n    q1 = df[col].quantile(0.25)\n    q3 = df[col].quantile(0.75)\n    iqr = q3 - q1\n    df[col] = df[col].clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)\nprint('[✓ SUCCESS] Capped all numeric outliers using IQR')`)}
                style={{
                  width: '100%',
                  background: '#10b981',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: 'pointer'
                }}
              >
                Execute IQR Outlier Capping
              </button>
            </div>

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>
                Automated Null Value Imputation
              </h3>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Impute missing numeric values using Mean (for Gaussian) or Median (for Skewed), and Mode for text categoricals.
              </p>
              <button
                onClick={() => onRunCode && onRunCode(`for col in df.columns:\n    if df[col].isnull().sum() > 0:\n        if pd.api.types.is_numeric_dtype(df[col]):\n            val = df[col].median() if abs(df[col].skew()) > 1.0 else df[col].mean()\n            df[col] = df[col].fillna(val)\n        else:\n            df[col] = df[col].fillna(df[col].mode()[0])\nprint('[✓ SUCCESS] Imputed all missing null values')`)}
                style={{
                  width: '100%',
                  background: '#0284c7',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: 'pointer'
                }}
              >
                Execute Smart Null Imputation
              </button>
            </div>
          </div>
        )}

        {/* TAB 3: Feature Engineering */}
        {activeTab === 'feature' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>
                Yeo-Johnson Skewness Transformation
              </h3>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Transform heavily non-normal numeric features into Gaussian bell curves to satisfy linear and distance model assumptions.
              </p>
              <button
                onClick={() => onRunCode && onRunCode(`from sklearn.preprocessing import PowerTransformer\npt = PowerTransformer(method='yeo-johnson')\nnum_cols = df.select_dtypes(include=['number']).columns\nif len(num_cols) > 0:\n    df[num_cols] = pt.fit_transform(df[num_cols])\nprint('[✓ SUCCESS] Transformed skewed features using Yeo-Johnson PowerTransformer')`)}
                style={{
                  width: '100%',
                  background: '#0ea5e9',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: 'pointer'
                }}
              >
                Apply Power Transformation
              </button>
            </div>

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>
                One-Hot Categorical Dummy Encoding
              </h3>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Convert categorical text columns into numerical binary dummy columns for Machine Learning models.
              </p>
              <button
                onClick={() => onRunCode && onRunCode(`cat_cols = df.select_dtypes(include=['object', 'category']).columns\nif len(cat_cols) > 0:\n    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)\nprint('[✓ SUCCESS] One-hot encoded categorical features')`)}
                style={{
                  width: '100%',
                  background: '#0284c7',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: 'pointer'
                }}
              >
                Execute One-Hot Dummy Encoding
              </button>
            </div>
          </div>
        )}

        {/* TAB 4: AutoML & Model Studio */}
        {activeTab === 'ml' && (
          <div style={{ maxWidth: 800, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 22 }}>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 800, color: '#f8fafc', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                <Sparkles size={18} color="#0ea5e9" /> Supervised AutoML & Model Fitting Studio
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                    Select Target Column (y)
                  </label>
                  <select
                    value={selectedTarget}
                    onChange={(e) => setSelectedTarget(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#1e293b',
                      color: '#f8fafc',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '10px 12px',
                      fontSize: 13,
                      fontWeight: 600
                    }}
                  >
                    <option value="">-- Choose Target --</option>
                    {columns.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                    Select Machine Learning Algorithm
                  </label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#1e293b',
                      color: '#f8fafc',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '10px 12px',
                      fontSize: 13,
                      fontWeight: 600
                    }}
                  >
                    <option value="auto">🏆 AutoML Candidate Tournament (Auto-Select Best)</option>
                    <option value="random_forest">🌲 Random Forest Ensemble</option>
                    <option value="gradient_boosting">⚡ Gradient Boosting Classifier/Regressor</option>
                    <option value="extra_trees">🌳 Extra Trees Ensemble</option>
                    <option value="knn">📍 K-Nearest Neighbors (KNN)</option>
                    <option value="linear">📈 Linear / Logistic Regression</option>
                  </select>
                </div>
              </div>

              <button
                onClick={handleTrainModel}
                disabled={isTraining || !selectedTarget}
                style={{
                  width: '100%',
                  background: isTraining ? '#334155' : 'linear-gradient(135deg, #0284c7, #0ea5e9)',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: 8,
                  padding: '12px 18px',
                  fontWeight: 800,
                  fontSize: 14,
                  cursor: isTraining || !selectedTarget ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  boxShadow: '0 4px 14px rgba(14,165,233,0.3)'
                }}
              >
                {isTraining ? (
                  <>Training Machine Learning Model & Evaluating Metrics...</>
                ) : (
                  <><Play size={16} /> Train Model & Compute Feature Importance</>
                )}
              </button>
            </div>

            {/* Training Results Presentation */}
            {trainingResult && (
              <div style={{ background: '#0f172a', border: '1px solid #10b981', borderRadius: 10, padding: 22 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14, color: '#10b981' }}>
                  <CheckCircle2 size={20} />
                  <h3 style={{ margin: 0, fontSize: 16, fontWeight: 800 }}>
                    ML Model Training Complete! ({trainingResult.algorithm})
                  </h3>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 18 }}>
                  <div style={{ background: '#1e293b', padding: 12, borderRadius: 6, textAlign: 'center' }}>
                    <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Primary Score</span>
                    <strong style={{ fontSize: 18, color: '#38bdf8' }}>
                      {typeof trainingResult.accuracy === 'number' ? `${(trainingResult.accuracy * 100).toFixed(2)}%` : `${(trainingResult.r2 * 100).toFixed(2)}%`}
                    </strong>
                  </div>
                  <div style={{ background: '#1e293b', padding: 12, borderRadius: 6, textAlign: 'center' }}>
                    <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Problem Type</span>
                    <strong style={{ fontSize: 15, color: '#f8fafc', textTransform: 'capitalize' }}>
                      {trainingResult.problem_type}
                    </strong>
                  </div>
                  <div style={{ background: '#1e293b', padding: 12, borderRadius: 6, textAlign: 'center' }}>
                    <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Model Artifact</span>
                    <strong style={{ fontSize: 12, color: '#10b981' }}>
                      {trainingResult.model_path ? trainingResult.model_path.split('\\').pop() : 'Saved .pkl'}
                    </strong>
                  </div>
                </div>

                {/* Feature Importances Tree */}
                {trainingResult.feature_importances && Object.keys(trainingResult.feature_importances).length > 0 && (
                  <div>
                    <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                      🌲 Feature Importance Ranking (Tree Weights)
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {Object.entries(trainingResult.feature_importances).slice(0, 7).map(([feat, score]) => (
                        <div key={feat} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12 }}>
                          <span style={{ width: 140, color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {feat}
                          </span>
                          <div style={{ flex: 1, background: '#1e293b', height: 8, borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{
                              width: `${Math.min(score * 100, 100)}%`,
                              background: '#0ea5e9',
                              height: '100%',
                              borderRadius: 4
                            }} />
                          </div>
                          <span style={{ width: 50, color: '#38bdf8', fontWeight: 700, textAlign: 'right' }}>
                            {(score * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* TAB 5: Pipeline History Tree */}
        {activeTab === 'tree' && (
          <div style={{ maxWidth: 800, margin: '0 auto' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 22 }}>
              <h3 style={{ margin: '0 0 16px 0', fontSize: 15, fontWeight: 800, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 8 }}>
                <Layers size={18} color="#0ea5e9" /> Transformation Pipeline Step Tree
              </h3>

              {pipelineSteps && pipelineSteps.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {pipelineSteps.map((step, idx) => (
                    <div key={step.step_id || idx} style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 12,
                      background: '#1e293b',
                      padding: 14,
                      borderRadius: 8,
                      borderLeft: '4px solid #0ea5e9'
                    }}>
                      <div style={{
                        background: '#0ea5e9',
                        color: '#fff',
                        borderRadius: '50%',
                        width: 22,
                        height: 22,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 11,
                        fontWeight: 800
                      }}>
                        {idx + 1}
                      </div>
                      <div style={{ flex: 1 }}>
                        <strong style={{ fontSize: 13, color: '#f8fafc' }}>
                          {step.operation || step.title || 'Pipeline Operation'}
                        </strong>
                        <p style={{ margin: '4px 0 0 0', fontSize: 11, color: '#94a3b8' }}>
                          {step.description || JSON.stringify(step.params || {})}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', margin: '20px 0' }}>
                  No transformations applied yet. Upload a dataset or execute an operation to see the pipeline tree!
                </p>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
