import { useState } from 'react';
import {
  Database, Table2, ChevronRight, ChevronDown,
  CheckCircle2, Circle, Clock, FlaskConical
} from 'lucide-react';

const LAB_STEPS = [
  { id: 'profile',   label: 'Profile Dataset',         module: 'clean'     },
  { id: 'nulls',     label: 'Handle Missing Values',   module: 'clean'     },
  { id: 'outliers',  label: 'Detect Outliers',         module: 'clean'     },
  { id: 'encode',    label: 'Encode Categoricals',     module: 'transform' },
  { id: 'normalize', label: 'Normalize Numerics',      module: 'transform' },
  { id: 'train',     label: 'Train ML Model',          module: 'ml'        },
  { id: 'evaluate',  label: 'Evaluate & Export',       module: 'ml'        },
];

export default function ObjectExplorer({
  connections,
  activeDataset,
  onSelectDataset,
  onModuleChange,
  labStatus,
  activeLabStep,
  pipelineSteps,
}) {
  const [connOpen, setConnOpen] = useState(true);
  const [labOpen, setLabOpen]   = useState(true);
  const [pipeOpen, setPipeOpen] = useState(false);

  const completedCount = LAB_STEPS.filter(s => labStatus?.[s.id]).length;

  return (
    <div className="object-explorer">
      <div className="explorer-header">
        <Database size={12} />
        Object Explorer
      </div>

      <div className="explorer-body">
        {/* ── Connections ── */}
        <div className="explorer-section">
          <div
            className="explorer-section-header"
            onClick={() => setConnOpen(o => !o)}
          >
            {connOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <Database size={12} />
            Connections
            <span className="explorer-item__badge">{(connections || []).length}</span>
          </div>

          {connOpen && (connections || []).map((conn, i) => (
            <div key={i}>
              <div
                className={`explorer-item ${activeDataset?.name === conn.name ? 'active' : ''}`}
                onClick={() => onSelectDataset(conn)}
                style={{ paddingLeft: 26 }}
              >
                <span style={{ fontSize: 14 }}>
                  {conn.type === 'csv' ? '📄' : conn.type === 'excel' ? '📊' : '🗄️'}
                </span>
                <span className="explorer-item__name">{conn.name}</span>
                {conn.rowCount && (
                  <span className="explorer-item__badge">{(conn.rowCount / 1000).toFixed(0)}K</span>
                )}
              </div>
              {conn.columns?.slice(0, 5).map(col => (
                <div
                  key={col}
                  className="explorer-item"
                  style={{ paddingLeft: 38, fontSize: 11, color: 'var(--text-muted)' }}
                >
                  <Table2 size={10} />
                  {col}
                </div>
              ))}
              {conn.columns?.length > 5 && (
                <div style={{ paddingLeft: 38, fontSize: 10, color: 'var(--text-muted)', padding: '2px 10px 2px 38px' }}>
                  +{conn.columns.length - 5} more columns
                </div>
              )}
            </div>
          ))}

          {(connections || []).length === 0 && connOpen && (
            <div style={{ padding: '8px 14px', fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>
              No connections. Use Connect tab.
            </div>
          )}
        </div>

        {/* ── Lab Guide ── */}
        {activeDataset && (
          <div className="lab-guide">
            <div
              className="lab-guide-header"
              style={{ cursor: 'pointer' }}
              onClick={() => setLabOpen(o => !o)}
            >
              {labOpen ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
              <FlaskConical size={11} />
              Guided Lab
              <span className="explorer-item__badge" style={{ marginLeft: 'auto' }}>
                {completedCount}/{LAB_STEPS.length}
              </span>
            </div>

            {labOpen && LAB_STEPS.map((step) => {
              const done   = labStatus?.[step.id];
              const active = activeLabStep === step.id;
              return (
                <div
                  key={step.id}
                  className={`lab-step ${done ? 'done' : ''} ${active ? 'active' : ''}`}
                  onClick={() => onModuleChange(step.module)}
                >
                  <div className="lab-step__check">
                    {done ? '✓' : active ? '→' : ''}
                  </div>
                  <span className="lab-step__label">{step.label}</span>
                </div>
              );
            })}

            {labOpen && (
              <div style={{ padding: '6px 10px' }}>
                <div className="progress">
                  <div
                    className="progress__fill"
                    style={{ width: `${(completedCount / LAB_STEPS.length) * 100}%` }}
                  />
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3, textAlign: 'right' }}>
                  {Math.round((completedCount / LAB_STEPS.length) * 100)}% complete
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Pipeline ── */}
        {pipelineSteps?.length > 0 && (
          <div className="explorer-section" style={{ borderTop: '1px solid var(--border)', marginTop: 4 }}>
            <div
              className="explorer-section-header"
              onClick={() => setPipeOpen(o => !o)}
            >
              {pipeOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <Clock size={12} />
              Pipeline
              <span className="explorer-item__badge">{pipelineSteps.length}</span>
            </div>
            {pipeOpen && pipelineSteps.map((step, i) => (
              <div
                key={i}
                className="explorer-item"
                style={{ paddingLeft: 26, fontSize: 11, cursor: 'pointer' }}
                onClick={() => onSelectDataset && onSelectDataset(step)}
              >
                <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 10 }}>
                  {i + 1}.
                </span>
                <span className="explorer-item__name">{step.description || step.operation}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
