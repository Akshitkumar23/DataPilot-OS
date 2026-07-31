import { useState, useEffect } from 'react';
import {
  Eraser, Shuffle, Play, CheckCircle2, AlertCircle,
  ArrowRight, RefreshCw, Eye
} from 'lucide-react';

const API = import.meta.env.VITE_API_URL || window.ENV_API_URL || 'http://localhost:8000';

function formatApiError(detail, fallback = 'Operation failed') {
  if (Array.isArray(detail)) {
    return detail.map(err => {
      const loc = Array.isArray(err.loc) ? err.loc.filter(l => l !== 'body').join('.') : '';
      return loc ? `${loc}: ${err.msg}` : (err.msg || JSON.stringify(err));
    }).join('; ');
  }
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (typeof detail === 'object' && detail !== null) return JSON.stringify(detail);
  return fallback;
}

// ── Clean Operations ──────────────────────────
const CLEAN_OPS = [
  {
    id: 'fill_mean',     label: 'Fill Nulls → Mean',       icon: '📊',
    desc: 'Replace missing numeric values with the column mean.',
    params: ['column'],
  },
  {
    id: 'fill_median',   label: 'Fill Nulls → Median',     icon: '📈',
    desc: 'Replace missing values with the column median (outlier-robust).',
    params: ['column'],
  },
  {
    id: 'fill_mode',     label: 'Fill Nulls → Mode',       icon: '🔢',
    desc: 'Replace missing values with the most frequent value.',
    params: ['column'],
  },
  {
    id: 'fill_constant', label: 'Fill Nulls → Constant',   icon: '✏️',
    desc: 'Replace missing values with a custom constant.',
    params: ['column', 'fill_value'],
  },
  {
    id: 'drop_nulls',    label: 'Drop Null Rows',          icon: '🗑️',
    desc: 'Remove rows that have null values in the selected column.',
    params: ['column'],
  },
  {
    id: 'drop_duplicates', label: 'Drop Duplicates',       icon: '🔄',
    desc: 'Remove duplicate rows from the dataset.',
    params: [],
  },
  {
    id: 'clip_iqr',      label: 'IQR Outlier Capping (1.5x)', icon: '🛡️',
    desc: 'Cap extreme values beyond [Q1 - 1.5*IQR, Q3 + 1.5*IQR] bounds.',
    params: ['column'],
  },
  {
    id: 'drop_column',   label: 'Drop Column',             icon: '❌',
    desc: 'Permanently remove a column from the dataset.',
    params: ['column'],
  },
  {
    id: 'rename_column', label: 'Rename Column',           icon: '✏️',
    desc: 'Rename a column to a new name.',
    params: ['column', 'new_name'],
  },
];

// ── Transform Operations ──────────────────────
const TRANSFORM_OPS = [
  {
    id: 'one_hot_encode', label: 'One-Hot Encode',         icon: '🔤',
    desc: 'Convert categorical column into binary dummy columns.',
    params: ['column'],
  },
  {
    id: 'label_encode',   label: 'Label Encode',           icon: '🏷️',
    desc: 'Assign integer codes to each unique category.',
    params: ['column'],
  },
  {
    id: 'normalize_minmax', label: 'Normalize (Min-Max)',  icon: '📐',
    desc: 'Scale numeric values to [0, 1] range.',
    params: ['column'],
  },
  {
    id: 'standardize',   label: 'Standardize (Z-score)',   icon: '📏',
    desc: 'Subtract mean and divide by standard deviation (μ=0, σ=1).',
    params: ['column'],
  },
  {
    id: 'yeo_johnson',   label: 'Yeo-Johnson Skew Power',  icon: '⚡',
    desc: 'Transform non-normal skewed feature distribution to Gaussian curve.',
    params: ['column'],
  },
  {
    id: 'log_transform', label: 'Log Transform',           icon: '🔢',
    desc: 'Apply log(1+x) transformation to reduce skewness.',
    params: ['column'],
  },
  {
    id: 'bin_column',    label: 'Bin / Discretize',        icon: '📦',
    desc: 'Convert continuous numeric column into bins/categories.',
    params: ['column', 'n_bins'],
  },
  {
    id: 'extract_datetime', label: 'Extract Date Parts',   icon: '📅',
    desc: 'Extract year, month, day, hour from a datetime column.',
    params: ['column'],
  },
  {
    id: 'filter_rows',   label: 'Filter Rows',             icon: '🔍',
    desc: 'Keep only rows matching a condition.',
    params: ['column', 'operator', 'filter_value'],
  },
];

const OPERATORS = ['==', '!=', '>', '>=', '<', '<=', 'contains', 'startswith'];

export default function OperationPanel({ mode = 'clean', schema, onApplied }) {
  const [selectedOp, setSelectedOp]   = useState(null);
  const [column, setColumn]           = useState('');
  const [fillValue, setFillValue]     = useState('');
  const [newName, setNewName]         = useState('');
  const [nBins, setNBins]             = useState(5);
  const [operator, setOperator]       = useState('==');
  const [filterValue, setFilterValue] = useState('');
  const [preview, setPreview]         = useState(null);
  const [previewLoading, setPL]       = useState(false);
  const [applying, setApplying]       = useState(false);
  const [lastApplied, setLastApplied] = useState(null);
  const [error, setError]             = useState('');

  const ops = mode === 'clean' ? CLEAN_OPS : TRANSFORM_OPS;
  const op  = ops.find(o => o.id === selectedOp);

  const allCols     = schema?.map(c => c.name) ?? [];
  const numericCols = schema?.filter(c => c.dtype?.includes('int') || c.dtype?.includes('float')).map(c => c.name) ?? [];
  const catCols     = schema?.filter(c => c.dtype === 'object' || c.dtype === 'category').map(c => c.name) ?? [];

  // Auto-select column based on op type
  useEffect(() => {
    setColumn('');
    setFillValue('');
    setNewName('');
    setPreview(null);
    setError('');
    setLastApplied(null);
  }, [selectedOp]);

  const buildPayload = () => ({
    operation: selectedOp,
    column: column || undefined,
    fill_value: fillValue || undefined,
    new_name: newName || undefined,
    n_bins: nBins || undefined,
    operator: operator || undefined,
    filter_value: filterValue || undefined,
  });

  const handlePreview = async () => {
    if (!selectedOp) return;
    setPL(true);
    setError('');
    try {
      const res = await fetch(`${API}/api/operations/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload()),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data.detail, 'Preview failed'));
      setPreview(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setPL(false);
    }
  };

  const handleApply = async () => {
    if (!selectedOp) return;
    setApplying(true);
    setError('');
    try {
      const res = await fetch(`${API}/api/operations/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload()),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data.detail, 'Apply failed'));
      setLastApplied(op?.label);
      setPreview(null);
      setSelectedOp(null);
      onApplied?.(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setApplying(false);
    }
  };

  const colsForOp = () => {
    if (!op) return allCols;
    if (['normalize_minmax', 'standardize', 'log_transform', 'bin_column', 'fill_mean', 'fill_median'].includes(op.id)) return numericCols;
    if (['one_hot_encode', 'label_encode'].includes(op.id)) return catCols;
    return allCols;
  };

  return (
    <div className="op-panel fade-in">
      {/* Header */}
      <div className="op-panel__header">
        {mode === 'clean'
          ? <Eraser size={18} color="var(--warning)" />
          : <Shuffle size={18} color="var(--accent2)" />
        }
        <div>
          <div className="op-panel__title">
            {mode === 'clean' ? 'Data Cleaning' : 'Data Transformation'}
          </div>
          <div className="op-panel__subtitle">
            {mode === 'clean'
              ? 'Fix quality issues — nulls, duplicates, outliers'
              : 'Reshape data — encode, scale, bin, filter'
            }
          </div>
        </div>
        {lastApplied && (
          <div style={{ marginLeft: 'auto' }}>
            <span className="badge badge--green">✓ {lastApplied} applied</span>
          </div>
        )}
      </div>

      <div className="op-panel__body">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, height: '100%' }}>

          {/* Left — Op selector + params */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* Guided activity banner */}
            {op && (
              <div className="lab-activity fade-in">
                <div className="lab-activity-banner">
                  <div className="lab-step-badge">{mode.toUpperCase()}</div>
                  <div>
                    <div className="lab-activity-title">{op.icon} {op.label}</div>
                    <div className="lab-activity-desc">{op.desc}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Operation selector */}
            <div className="card">
              <div className="card-header">
                <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  Select Operation
                </span>
              </div>
              <div className="card-body" style={{ padding: 8 }}>
                <div className="op-cards">
                  {ops.map(o => (
                    <div
                      key={o.id}
                      className={`op-card ${selectedOp === o.id ? 'selected' : ''}`}
                      onClick={() => setSelectedOp(o.id)}
                    >
                      <div className="op-card__icon">{o.icon}</div>
                      <div className="op-card__label">{o.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Params */}
            {op && (
              <div className="card fade-in">
                <div className="card-header">
                  <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    Parameters
                  </span>
                </div>
                <div className="card-body">
                  {op.params.includes('column') && (
                    <div className="form-group">
                      <label className="form-label">Target Column</label>
                      <select className="form-select" value={column} onChange={e => setColumn(e.target.value)}>
                        <option value="">— Select column —</option>
                        {colsForOp().map(c => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                  )}
                  {op.params.includes('fill_value') && (
                    <div className="form-group">
                      <label className="form-label">Fill With</label>
                      <input className="form-input" value={fillValue} onChange={e => setFillValue(e.target.value)} placeholder="0 or 'Unknown'" />
                    </div>
                  )}
                  {op.params.includes('new_name') && (
                    <div className="form-group">
                      <label className="form-label">New Column Name</label>
                      <input className="form-input" value={newName} onChange={e => setNewName(e.target.value)} placeholder="new_column_name" />
                    </div>
                  )}
                  {op.params.includes('n_bins') && (
                    <div className="form-group">
                      <label className="form-label">Number of Bins: {nBins}</label>
                      <input type="range" min="2" max="20" value={nBins} onChange={e => setNBins(parseInt(e.target.value))} />
                    </div>
                  )}
                  {op.params.includes('operator') && (
                    <div className="form-group">
                      <label className="form-label">Condition</label>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 100px 1fr', gap: 6 }}>
                        <select className="form-select" value={column} onChange={e => setColumn(e.target.value)}>
                          <option value="">Column</option>
                          {allCols.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                        <select className="form-select" value={operator} onChange={e => setOperator(e.target.value)}>
                          {OPERATORS.map(op => <option key={op} value={op}>{op}</option>)}
                        </select>
                        <input className="form-input" value={filterValue} onChange={e => setFilterValue(e.target.value)} placeholder="value" />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right — Preview */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="card" style={{ flex: 1 }}>
              <div className="card-header">
                <Eye size={13} color="var(--text-secondary)" />
                <span className="card-title">Before / After Preview</span>
                {op && (
                  <button
                    className="btn btn--ghost btn--sm"
                    onClick={handlePreview}
                    disabled={previewLoading || !column && op?.params.includes('column')}
                    style={{ marginLeft: 'auto' }}
                  >
                    {previewLoading ? <span className="spinner" /> : <RefreshCw size={12} />}
                    Preview
                  </button>
                )}
              </div>
              <div className="card-body" style={{ flex: 1, overflow: 'auto' }}>
                {preview ? (
                  <div className="preview-compare fade-in">
                    {/* Before */}
                    <div className="preview-box">
                      <div className="preview-box-header before">Before</div>
                      <div className="preview-box-body">
                        {(preview.before ?? []).slice(0, 10).map((val, i) => (
                          <div key={i} style={{
                            padding: '2px 0',
                            color: val === null || val === undefined ? 'var(--text-muted)' : 'var(--text-secondary)',
                            fontStyle: val === null ? 'italic' : 'normal',
                          }}>
                            {val === null || val === undefined ? 'null' : String(val)}
                          </div>
                        ))}
                        {preview.before_stats && (
                          <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--border)', fontSize: 10, color: 'var(--text-muted)' }}>
                            Nulls: {preview.before_stats.nulls} · Rows: {preview.before_stats.rows}
                          </div>
                        )}
                      </div>
                    </div>
                    {/* After */}
                    <div className="preview-box">
                      <div className="preview-box-header after">After</div>
                      <div className="preview-box-body">
                        {(preview.after ?? []).slice(0, 10).map((val, i) => (
                          <div key={i} style={{
                            padding: '2px 0',
                            color: 'var(--success)',
                          }}>
                            {val === null || val === undefined ? 'null' : String(val)}
                          </div>
                        ))}
                        {preview.after_stats && (
                          <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--border)', fontSize: 10, color: 'var(--text-muted)' }}>
                            Nulls: {preview.after_stats.nulls} · Rows: {preview.after_stats.rows}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="empty-state" style={{ height: 'auto', padding: '30px 0' }}>
                    <div className="empty-state__icon" style={{ fontSize: 32 }}>👁</div>
                    <div className="empty-state__sub">
                      {op
                        ? 'Set parameters and click Preview to see what changes will be made'
                        : 'Select an operation from the panel to get started'
                      }
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="alert alert--danger fade-in">
                <AlertCircle size={13} />
                {error}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="op-panel__footer">
        {op && (
          <>
            <button
              className="btn btn--ghost btn--sm"
              onClick={handlePreview}
              disabled={previewLoading}
            >
              <Eye size={13} />
              Preview Changes
            </button>
            <button
              className="btn btn--primary"
              onClick={handleApply}
              disabled={applying || (!column && op?.params.includes('column'))}
            >
              {applying ? <span className="spinner" style={{ width: 12, height: 12 }} /> : <Play size={13} />}
              {applying ? 'Applying…' : 'Apply Operation'}
              <ArrowRight size={13} />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
