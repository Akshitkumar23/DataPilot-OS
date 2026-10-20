import React, { useState } from 'react';
import { 
  Sliders, 
  DollarSign, 
  CheckCircle2, 
  AlertTriangle, 
  ChevronDown, 
  ChevronUp, 
  RefreshCw, 
  TrendingDown, 
  ShieldCheck,
  Scale,
  Info
} from 'lucide-react';
import { API, getAuthHeaders } from '../apiConfig';

export default function MLStudio({
  datasetId,
  trainingResult,
  onRefresh
}) {
  const isBinaryClassification = 
    trainingResult && 
    trainingResult.task === 'classification' && 
    trainingResult.metrics?.confusion_matrix?.classes?.length === 2;

  const classes = trainingResult?.metrics?.confusion_matrix?.classes || [];
  const defaultPosClass = classes.length === 2 ? classes[1] : '';

  const [positiveClass, setPositiveClass] = useState(defaultPosClass);
  const [fpCost, setFpCost] = useState('1.0');
  const [fnCost, setFnCost] = useState('5.0');
  const [tpCost, setTpCost] = useState('0.0');
  const [tnCost, setTnCost] = useState('0.0');
  const [calibrationPolicy, setCalibrationPolicy] = useState('auto');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  if (!isBinaryClassification) {
    return null;
  }

  const handleOptimize = async () => {
    setIsOptimizing(true);
    setErrorMsg(null);

    const fp = parseFloat(fpCost);
    const fn = parseFloat(fnCost);
    const tp = parseFloat(tpCost) || 0.0;
    const tn = parseFloat(tnCost) || 0.0;

    if (isNaN(fp) || isNaN(fn) || fp <= 0 || fn <= 0) {
      setErrorMsg('False Positive and False Negative costs must be strictly positive numeric values (> 0).');
      setIsOptimizing(false);
      return;
    }

    const payload = {
      dataset_id: datasetId,
      target: trainingResult.target,
      positive_class: positiveClass || classes[1],
      false_positive_cost: fp,
      false_negative_cost: fn,
      true_positive_cost: tp,
      true_negative_cost: tn,
      features: trainingResult.features,
      algorithm: trainingResult.algorithm || 'auto',
      calibration_policy: calibrationPolicy,
      grid_resolution: 100,
      test_size: 0.2,
      cv_folds: 5
    };

    try {
      const resp = await fetch(`${API}/api/ml/optimize-threshold`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload)
      });
      const data = await resp.json();

      if (resp.ok && data.status === 'SUCCEEDED') {
        setOptimizationResult(data.result);
        if (onRefresh) onRefresh();
      } else {
        const detail = data.detail || {};
        const code = detail.error_code || 'OPTIMIZATION_ERROR';
        const msg = detail.message || data.message || 'Cost-sensitive threshold optimization failed.';
        setErrorMsg(`[${code}]: ${msg}`);
      }
    } catch (err) {
      setErrorMsg(`Network error: ${err.message}`);
    } finally {
      setIsOptimizing(false);
    }
  };

  // Helper to render mini SVG cost curve
  const renderCostCurve = (curveData, optTau, baseTau) => {
    if (!curveData || curveData.length < 2) return null;
    const width = 420;
    const height = 150;
    const padding = 30;

    const costs = curveData.map(d => d.cost);
    const minCost = Math.min(...costs);
    const maxCost = Math.max(...costs);
    const costRange = maxCost - minCost || 1.0;

    const getX = tau => padding + ((tau - 0.0) / 1.0) * (width - 2 * padding);
    const getY = c => height - padding - ((c - minCost) / costRange) * (height - 2 * padding);

    const points = curveData.map(d => `${getX(d.threshold).toFixed(1)},${getY(d.cost).toFixed(1)}`).join(' ');

    const optX = getX(optTau);
    const optY = getY(curveData.reduce((prev, curr) => Math.abs(curr.threshold - optTau) < Math.abs(prev.threshold - optTau) ? curr : prev).cost);

    const baseX = getX(baseTau);
    const baseY = getY(curveData.reduce((prev, curr) => Math.abs(curr.threshold - baseTau) < Math.abs(prev.threshold - baseTau) ? curr : prev).cost);

    return (
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible', background: '#0a0f1d', borderRadius: 8, border: '1px solid #1e293b' }}>
        {/* Grid lines */}
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#334155" strokeWidth="1" />

        {/* Cost Polyline */}
        <polyline fill="none" stroke="#38bdf8" strokeWidth="2.5" strokeLinejoin="round" points={points} />

        {/* Baseline tau=0.50 Marker */}
        <line x1={baseX} y1={padding} x2={baseX} y2={height - padding} stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="3 3" />
        <circle cx={baseX} cy={baseY} r="4" fill="#94a3b8" />
        <text x={baseX} y={padding - 5} fill="#94a3b8" fontSize="9" textAnchor="middle" fontWeight="bold">τ=0.50</text>

        {/* Optimal Marker */}
        <line x1={optX} y1={padding} x2={optX} y2={height - padding} stroke="#10b981" strokeWidth="2" />
        <circle cx={optX} cy={optY} r="5" fill="#10b981" stroke="#0f172a" strokeWidth="1.5" />
        <text x={optX} y={padding - 5} fill="#34d399" fontSize="10" textAnchor="middle" fontWeight="bold">τ*={optTau.toFixed(2)}</text>

        {/* Axis labels */}
        <text x={padding} y={height - 8} fill="#64748b" fontSize="9">0.0</text>
        <text x={width - padding} y={height - 8} fill="#64748b" fontSize="9" textAnchor="end">1.0 (Threshold)</text>
        <text x={padding + 5} y={padding + 10} fill="#64748b" fontSize="9">Min Cost: ${minCost.toFixed(0)}</text>
        <text x={padding + 5} y={padding + 22} fill="#64748b" fontSize="9">Max Cost: ${maxCost.toFixed(0)}</text>
      </svg>
    );
  };

  return (
    <div style={{ marginTop: 20, background: '#0a0f1d', border: '1px solid #1e293b', borderRadius: 12, padding: 20 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ background: 'rgba(56, 189, 248, 0.15)', padding: 8, borderRadius: 8, color: '#38bdf8' }}>
            <Sliders size={18} />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 6 }}>
              Dynamic Asymmetric Cost-Matrix Optimization
              <span style={{ fontSize: 10, background: '#0284c7', color: '#fff', padding: '1px 6px', borderRadius: 4, fontWeight: 600 }}>FW-02</span>
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: 11, color: '#94a3b8' }}>
              Minimize expected business financial loss under analyst cost policy with calibrated probability support.
            </p>
          </div>
        </div>
      </div>

      {/* Cost Matrix Specification Inputs */}
      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 16, marginBottom: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14, alignItems: 'flex-end' }}>
          <div>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>
              Positive Class Target
            </label>
            <select
              value={positiveClass}
              onChange={e => setPositiveClass(e.target.value)}
              style={{
                width: '100%',
                background: '#1e293b',
                color: '#f8fafc',
                border: '1px solid #334155',
                borderRadius: 6,
                padding: '6px 10px',
                fontSize: 12
              }}
            >
              {classes.map(c => (
                <option key={String(c)} value={String(c)}>
                  {String(c)} (Positive)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>
              Cost of False Positive ($C_&#123;FP&#125;$)
            </label>
            <div style={{ position: 'relative' }}>
              <input
                type="number"
                step="0.5"
                min="0.01"
                value={fpCost}
                onChange={e => setFpCost(e.target.value)}
                style={{
                  width: '100%',
                  background: '#1e293b',
                  color: '#f8fafc',
                  border: '1px solid #334155',
                  borderRadius: 6,
                  padding: '6px 10px',
                  fontSize: 12,
                  boxSizing: 'border-box'
                }}
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>
              Cost of False Negative ($C_&#123;FN&#125;$)
            </label>
            <div style={{ position: 'relative' }}>
              <input
                type="number"
                step="0.5"
                min="0.01"
                value={fnCost}
                onChange={e => setFnCost(e.target.value)}
                style={{
                  width: '100%',
                  background: '#1e293b',
                  color: '#f8fafc',
                  border: '1px solid #334155',
                  borderRadius: 6,
                  padding: '6px 10px',
                  fontSize: 12,
                  boxSizing: 'border-box'
                }}
              />
            </div>
          </div>

          <div>
            <button
              onClick={handleOptimize}
              disabled={isOptimizing}
              style={{
                width: '100%',
                background: isOptimizing ? '#334155' : 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                padding: '7px 14px',
                fontSize: 12,
                fontWeight: 600,
                cursor: isOptimizing ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6
              }}
            >
              {isOptimizing ? <RefreshCw size={14} className="animate-spin" /> : <Scale size={14} />}
              {isOptimizing ? 'Optimizing...' : 'Optimize Threshold'}
            </button>
          </div>
        </div>

        {/* Advanced Toggle */}
        <div style={{ marginTop: 10 }}>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            style={{
              background: 'none',
              border: 'none',
              color: '#38bdf8',
              fontSize: 11,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              padding: 0
            }}
          >
            {showAdvanced ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            {showAdvanced ? 'Hide 4-Cell Matrix & Calibration Controls' : 'Show 4-Cell Matrix & Calibration Controls'}
          </button>

          {showAdvanced && (
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #1e293b', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
              <div>
                <label style={{ display: 'block', fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>
                  True Positive Benefit/Cost ($C_&#123;TP&#125;$)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={tpCost}
                  onChange={e => setTpCost(e.target.value)}
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    color: '#f8fafc',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '5px 8px',
                    fontSize: 11,
                    boxSizing: 'border-box'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>
                  True Negative Benefit/Cost ($C_&#123;TN&#125;$)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={tnCost}
                  onChange={e => setTnCost(e.target.value)}
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    color: '#f8fafc',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '5px 8px',
                    fontSize: 11,
                    boxSizing: 'border-box'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>
                  Calibration Policy
                </label>
                <select
                  value={calibrationPolicy}
                  onChange={e => setCalibrationPolicy(e.target.value)}
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    color: '#f8fafc',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '5px 8px',
                    fontSize: 11
                  }}
                >
                  <option value="auto">Auto (Model-Aware)</option>
                  <option value="sigmoid">Sigmoid (Platt Scaling)</option>
                  <option value="isotonic">Isotonic (N &ge; 2000)</option>
                  <option value="none">None (Raw Probabilities)</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {errorMsg && (
          <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: 6, color: '#fca5a5', fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}>
            <AlertTriangle size={14} />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      {/* Results Section */}
      {optimizationResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Key Impact Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Optimal Decision Cutoff</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#10b981', display: 'flex', alignItems: 'center', gap: 8 }}>
                &tau;* = {optimizationResult.threshold_selection.optimized_threshold.toFixed(2)}
                <span style={{ fontSize: 11, fontWeight: 400, color: '#64748b' }}>
                  (vs. 0.50 default)
                </span>
              </div>
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>
                Bayes: {optimizationResult.cost_policy.theoretical_bayes_threshold.toFixed(4)} | Ratio: {optimizationResult.cost_policy.loss_ratio}x
              </div>
            </div>

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Net Financial Savings</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 8 }}>
                {optimizationResult.business_decision_quality.percentage_cost_reduction.toFixed(1)}%
                <span style={{ fontSize: 11, fontWeight: 400, color: '#34d399' }}>
                  (-${optimizationResult.business_decision_quality.absolute_cost_reduction.toFixed(1)})
                </span>
              </div>
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>
                ${optimizationResult.business_decision_quality.baseline_cost.toFixed(0)} &rarr; ${optimizationResult.business_decision_quality.optimized_cost.toFixed(0)} on held-out test
              </div>
            </div>

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Calibration & Stability</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 6 }}>
                <ShieldCheck size={14} color="#10b981" />
                {optimizationResult.calibration.applied ? `${optimizationResult.calibration.method.toUpperCase()} Calibrated` : 'Raw Uncalibrated'}
              </div>
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>
                Brier: {optimizationResult.calibration.brier_score} | CV &sigma;: &plusmn;{optimizationResult.threshold_selection.stability.std.toFixed(3)}
              </div>
            </div>
          </div>

          {/* Detailed Visualizations: Confusion Matrix Comparison & Cost Curve */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            {/* Confusion Matrices Before/After */}
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 16 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#f8fafc', marginBottom: 12, display: 'flex', justifyContent: 'space-between' }}>
                <span>Decision Matrix Comparison</span>
                <span style={{ fontSize: 10, color: '#94a3b8' }}>Held-out Test (N={optimizationResult.sample_counts.test})</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {/* Baseline Matrix */}
                <div style={{ background: '#0a0f1d', borderRadius: 6, padding: 10, border: '1px solid #1e293b' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6, textAlign: 'center' }}>
                    Baseline (&tau; = 0.50)
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 10, textAlign: 'center' }}>
                    <div style={{ background: 'rgba(16, 185, 129, 0.15)', padding: 6, borderRadius: 4, color: '#34d399' }}>
                      TN: {optimizationResult.business_decision_quality.baseline_confusion_matrix.tn}
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.15)', padding: 6, borderRadius: 4, color: '#fca5a5' }}>
                      FP: {optimizationResult.business_decision_quality.baseline_confusion_matrix.fp}
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.25)', padding: 6, borderRadius: 4, color: '#f87171' }}>
                      FN: {optimizationResult.business_decision_quality.baseline_confusion_matrix.fn}
                    </div>
                    <div style={{ background: 'rgba(16, 185, 129, 0.15)', padding: 6, borderRadius: 4, color: '#34d399' }}>
                      TP: {optimizationResult.business_decision_quality.baseline_confusion_matrix.tp}
                    </div>
                  </div>
                  <div style={{ fontSize: 10, color: '#64748b', marginTop: 6, textAlign: 'center' }}>
                    Cost: ${optimizationResult.business_decision_quality.baseline_cost.toFixed(0)}
                  </div>
                </div>

                {/* Optimized Matrix */}
                <div style={{ background: '#0a0f1d', borderRadius: 6, padding: 10, border: '1px solid #10b981' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#34d399', marginBottom: 6, textAlign: 'center' }}>
                    Optimized (&tau;* = {optimizationResult.threshold_selection.optimized_threshold.toFixed(2)})
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 10, textAlign: 'center' }}>
                    <div style={{ background: 'rgba(16, 185, 129, 0.15)', padding: 6, borderRadius: 4, color: '#34d399' }}>
                      TN: {optimizationResult.business_decision_quality.optimized_confusion_matrix.tn}
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.15)', padding: 6, borderRadius: 4, color: '#fca5a5' }}>
                      FP: {optimizationResult.business_decision_quality.optimized_confusion_matrix.fp}
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.25)', padding: 6, borderRadius: 4, color: '#f87171' }}>
                      FN: {optimizationResult.business_decision_quality.optimized_confusion_matrix.fn}
                    </div>
                    <div style={{ background: 'rgba(16, 185, 129, 0.15)', padding: 6, borderRadius: 4, color: '#34d399' }}>
                      TP: {optimizationResult.business_decision_quality.optimized_confusion_matrix.tp}
                    </div>
                  </div>
                  <div style={{ fontSize: 10, color: '#34d399', fontWeight: 600, marginTop: 6, textAlign: 'center' }}>
                    Cost: ${optimizationResult.business_decision_quality.optimized_cost.toFixed(0)}
                  </div>
                </div>
              </div>
            </div>

            {/* Cost vs. Threshold Curve */}
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 16 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#f8fafc', marginBottom: 12, display: 'flex', justifyContent: 'space-between' }}>
                <span>Empirical Business Loss Curve</span>
                <span style={{ fontSize: 10, color: '#94a3b8' }}>Out-of-Sample Cost Sweep</span>
              </div>
              {renderCostCurve(
                optimizationResult.cost_curve,
                optimizationResult.threshold_selection.optimized_threshold,
                optimizationResult.threshold_selection.baseline_threshold
              )}
            </div>
          </div>

          {/* Warnings & Methodological Disclaimer */}
          {optimizationResult.warnings && optimizationResult.warnings.length > 0 && (
            <div style={{ background: 'rgba(245, 158, 11, 0.1)', border: '1px solid #f59e0b', borderRadius: 6, padding: '8px 12px', fontSize: 11, color: '#fbbf24' }}>
              {optimizationResult.warnings.map((w, idx) => (
                <div key={idx}><strong>Notice [{w.code}]:</strong> {w.message}</div>
              ))}
            </div>
          )}

          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, padding: '8px 12px', background: 'rgba(51, 65, 85, 0.3)', borderRadius: 6, fontSize: 10, color: '#94a3b8' }}>
            <Info size={14} style={{ flexShrink: 0, marginTop: 1, color: '#38bdf8' }} />
            <span>
              <strong>Methodological Protocol:</strong> Threshold optimization minimizes direct financial loss under analyst cost policy. 
              Standard metrics like F1 and Accuracy are non-linear under asymmetric penalties and should not be used as primary operational financial targets.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
