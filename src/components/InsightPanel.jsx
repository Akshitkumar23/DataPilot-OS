import React from 'react';
import { Lightbulb, RefreshCw, AlertCircle } from 'lucide-react';

export default function InsightPanel({ insight, isGenerating, onRegenerate }) {
  if (isGenerating) {
    return (
      <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px' }}>
        <RefreshCw size={32} className="spinner" color="var(--accent)" />
        <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Analyzing results...</span>
      </div>
    );
  }

  if (!insight) {
    return (
      <div className="card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px', textAlign: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>Insights will appear after successful result validation.</p>
      </div>
    );
  }

  const confColor = insight.confidence === 'high' ? 'var(--success)' : insight.confidence === 'medium' ? 'var(--warning)' : 'var(--danger)';
  const confBg = insight.confidence === 'high' ? 'var(--success-dim)' : insight.confidence === 'medium' ? 'var(--warning-dim)' : 'var(--danger-dim)';

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '16px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-panel)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Lightbulb size={18} color="var(--accent)" />
          <h2 style={{ fontSize: '14px', fontWeight: 800, margin: 0 }}>AI INSIGHTS</h2>
          <span style={{ fontSize: '11px', background: confBg, color: confColor, padding: '2px 8px', borderRadius: '12px', fontWeight: 700, marginLeft: '8px' }}>
            {insight.confidence?.toUpperCase()} CONFIDENCE
          </span>
        </div>
        <div style={{ fontSize: '11px', fontWeight: 600, color: insight.evidence_status === 'sufficient' ? 'var(--success)' : 'var(--warning)' }}>
          Evidence: {insight.evidence_status === 'sufficient' ? 'Sufficient ✔' : 'Partial ⚠'}
        </div>
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: '20px' }}>
        <p style={{ fontSize: '15px', lineHeight: 1.6, color: 'var(--text-primary)', marginBottom: '24px' }}>
          {insight.summary}
        </p>

        {insight.key_findings?.length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <h3 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: 700, marginBottom: '12px', letterSpacing: '0.05em' }}>Key Findings</h3>
            <ol style={{ paddingLeft: '20px', margin: 0, color: 'var(--text-primary)' }}>
              {insight.key_findings.map((finding, idx) => (
                <li key={idx} style={{ marginBottom: '10px', fontSize: '14px', lineHeight: 1.5 }}>
                  {finding.text}
                  {finding.value !== undefined && (
                    <span style={{ marginLeft: '8px', fontWeight: 600, color: finding.value >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                      {finding.value > 0 ? '+' : ''}{finding.value}
                    </span>
                  )}
                  {finding.source_artifact_id && (
                    <sup style={{ marginLeft: '4px', background: 'var(--bg-hover)', color: 'var(--text-secondary)', padding: '1px 4px', borderRadius: '4px', fontSize: '9px', cursor: 'pointer' }} title={`Source: ${finding.source_artifact_id}`}>
                      [{finding.source_artifact_id}]
                    </sup>
                  )}
                </li>
              ))}
            </ol>
          </div>
        )}

        {insight.drivers?.length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <h3 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: 700, marginBottom: '8px', letterSpacing: '0.05em' }}>Key Drivers</h3>
            <ul style={{ paddingLeft: '20px', margin: 0, color: 'var(--text-secondary)', fontSize: '13px' }}>
              {insight.drivers.map((driver, idx) => <li key={idx} style={{ marginBottom: '4px' }}>{driver}</li>)}
            </ul>
          </div>
        )}

        {insight.anomalies?.length > 0 && (
          <div style={{ marginBottom: '24px', background: 'var(--warning-dim)', padding: '12px', borderRadius: '6px', border: '1px solid var(--warning)' }}>
            <h3 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--warning)', fontWeight: 700, marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <AlertCircle size={14} /> Anomalies Detected
            </h3>
            <ul style={{ paddingLeft: '20px', margin: 0, color: 'var(--warning)', fontSize: '13px' }}>
              {insight.anomalies.map((anomaly, idx) => <li key={idx} style={{ marginBottom: '4px' }}>{anomaly}</li>)}
            </ul>
          </div>
        )}

        {insight.limitations?.length > 0 && (
          <details style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 600, marginBottom: '8px' }}>Limitations & Caveats</summary>
            <ul style={{ paddingLeft: '20px', margin: 0, marginTop: '8px' }}>
              {insight.limitations.map((lim, idx) => <li key={idx} style={{ marginBottom: '4px' }}>{lim}</li>)}
            </ul>
          </details>
        )}
      </div>

      <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border)', background: 'var(--bg-panel)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>All insights are based on validated execution results only.</span>
        <button onClick={onRegenerate} className="btn btn--ghost btn--sm">
          <RefreshCw size={14} /> Regenerate Insights
        </button>
      </div>
    </div>
  );
}
