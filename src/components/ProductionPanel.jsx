import React, { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle, Lock, Play, RefreshCw, ShieldCheck, XCircle } from 'lucide-react';

export default function ProductionPanel({
  workflowId,
  datasets = [],
  activeDatasetId,
  onValidate,
  onExecuteProduction,
  onDatasetChange,
  validationResult,
  isChecking = false,
  isExecuting = false
}) {
  const availableDatasets = useMemo(() => datasets.map((dataset, index) => ({
    id: dataset.id || dataset.dataset_id || `dataset-${index}`,
    name: dataset.name || 'Unnamed dataset',
    rowCount: dataset.rowCount ?? dataset.row_count ?? 0
  })), [datasets]);
  const [productionDatasetId, setProductionDatasetId] = useState(activeDatasetId || '');
  const [approved, setApproved] = useState(false);

  useEffect(() => {
    if (activeDatasetId) setProductionDatasetId(activeDatasetId);
  }, [activeDatasetId]);

  if (!workflowId) {
    return (
      <div className="card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>Run and validate an analysis before preparing it for production.</p>
      </div>
    );
  }

  const checks = validationResult?.checks || [];
  const allPassed = validationResult?.is_compatible === true;
  const selected = availableDatasets.find(dataset => dataset.id === productionDatasetId);

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '16px', borderBottom: '1px solid var(--border)', background: '#1e293b', color: '#fff' }}>
        <h2 style={{ fontSize: '14px', fontWeight: 800, margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Lock size={16} color="#fbbf24" /> PRODUCTION WORKFLOW
        </h2>
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div>
          <h3 style={{ fontSize: '13px', fontWeight: 700, marginBottom: '12px', color: 'var(--text-primary)' }}>1. Select Connected Production Dataset</h3>
          {availableDatasets.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', margin: 0 }}>Connect the production dataset before validating this workflow.</p>
          ) : (
            <select
              className="form-input"
              value={productionDatasetId}
              onChange={(event) => {
                setProductionDatasetId(event.target.value);
                setApproved(false);
                onDatasetChange?.();
              }}
              style={{ width: '100%' }}
            >
              <option value="">Select a connected dataset</option>
              {availableDatasets.map(dataset => (
                <option key={dataset.id} value={dataset.id}>{dataset.name} ({dataset.rowCount.toLocaleString()} rows)</option>
              ))}
            </select>
          )}
        </div>

        {productionDatasetId && (
          <div>
            <h3 style={{ fontSize: '13px', fontWeight: 700, marginBottom: '12px', color: 'var(--text-primary)' }}>2. Compatibility Check</h3>
            {!validationResult ? (
              <button className="btn btn--primary" onClick={() => onValidate?.(productionDatasetId)} disabled={isChecking}>
                {isChecking ? <RefreshCw size={16} className="spinner" /> : <ShieldCheck size={16} />} Run Compatibility Check
              </button>
            ) : (
              <div style={{ border: '1px solid var(--border)', borderRadius: '6px', padding: '16px' }}>
                {checks.map((check) => (
                  <div key={check.check_name} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 10, fontSize: 13 }}>
                    <span>{check.message || check.check_name}</span>
                    {check.passed ? <CheckCircle size={16} color="var(--success)" /> : <XCircle size={16} color="var(--danger)" />}
                  </div>
                ))}
                {(validationResult.warnings || []).map((warning) => (
                  <div key={warning} style={{ marginTop: 8, color: 'var(--warning)', fontSize: 12 }}>{warning}</div>
                ))}
                {!allPassed && (
                  <div style={{ marginTop: 16, padding: 10, background: 'var(--danger-dim)', color: 'var(--danger)', fontSize: 12, borderRadius: 4, display: 'flex', gap: 8 }}>
                    <AlertTriangle size={16} /> Production is blocked until all checks pass.
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {allPassed && selected && (
          <div>
            <h3 style={{ fontSize: '13px', fontWeight: 700, marginBottom: '12px', color: 'var(--text-primary)' }}>3. Approve & Execute</h3>
            <div style={{ background: 'var(--bg-panel)', padding: 16, borderRadius: 6, fontSize: 13, marginBottom: 16, color: 'var(--text-secondary)' }}>
              <div><strong>Target:</strong> {selected.name}</div>
              <div><strong>Rows:</strong> {selected.rowCount.toLocaleString()}</div>
            </div>
            <label style={{ display: 'flex', alignItems: 'flex-start', gap: 10, cursor: 'pointer', marginBottom: 20, fontSize: 13 }}>
              <input type="checkbox" checked={approved} onChange={event => setApproved(event.target.checked)} style={{ marginTop: 2 }} />
              <span>I confirm that this validated workflow may run against the selected production dataset.</span>
            </label>
            <button
              className="btn"
              style={{ width: '100%', justifyContent: 'center', background: approved ? '#dc2626' : '#e2e8f0', color: approved ? '#fff' : '#94a3b8', borderColor: approved ? '#dc2626' : '#cbd5e1', padding: 12, fontSize: 15 }}
              disabled={!approved || isExecuting}
              onClick={() => onExecuteProduction?.(productionDatasetId)}
            >
              {isExecuting ? <RefreshCw size={18} className="spinner" /> : <Play size={18} />}
              {isExecuting ? 'Executing...' : 'Run on Production Dataset'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
