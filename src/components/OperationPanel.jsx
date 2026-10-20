import { useState, useEffect, useMemo } from 'react';
import {
  Eraser, Shuffle, Play, AlertCircle,
  ArrowRight, RefreshCw, Eye, ShieldCheck,
  BarChart2, TrendingUp, Hash, Edit3, Trash2,
  Repeat, Maximize2, Sliders, Binary, Zap, RotateCcw, CheckCircle2
} from 'lucide-react';
import { API, getAuthHeaders, formatApiError } from '../apiConfig';

const OPERATORS = ['==', '!=', '>', '>=', '<', '<='];

// ── Governed Clean Operations ───────
const CLEAN_OPS = [
  {
    id: 'fill_mean',
    label: 'Impute Mean',
    icon: BarChart2,
    desc: 'Replace missing numeric values with the calculated column arithmetic mean.',
    params: ['column'],
    requiresType: 'numeric'
  },
  {
    id: 'fill_median',
    label: 'Impute Median',
    icon: TrendingUp,
    desc: 'Replace missing values with the 50th percentile median (outlier-robust).',
    params: ['column'],
    requiresType: 'numeric'
  },
  {
    id: 'fill_mode',
    label: 'Impute Mode',
    icon: Hash,
    desc: 'Replace missing values with the most frequent observed category or value.',
    params: ['column'],
    requiresType: 'any'
  },
  {
    id: 'fill_constant',
    label: 'Impute Constant',
    icon: Edit3,
    desc: 'Replace missing values with a user-specified constant literal or numeric value.',
    params: ['column', 'fill_value'],
    requiresType: 'any'
  },
  {
    id: 'drop_nulls',
    label: 'Drop Missing Rows',
    icon: Trash2,
    desc: 'Remove rows containing null values in the selected column.',
    params: ['column'],
    requiresType: 'any'
  },
  {
    id: 'drop_duplicates',
    label: 'Deduplicate Rows',
    icon: Repeat,
    desc: 'Remove exact duplicate observations across all dataset columns.',
    params: [],
    requiresType: 'dataset'
  },
  {
    id: 'clip_iqr',
    label: 'Clip Outliers (1.5x IQR)',
    icon: ShieldCheck,
    desc: 'Cap extreme values beyond Tukey bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR].',
    params: ['column'],
    requiresType: 'numeric'
  },
];

// ── Governed Transform Operations ───
const TRANSFORM_OPS = [
  {
    id: 'normalize_minmax',
    label: 'Min-Max Scaling',
    icon: Maximize2,
    desc: 'Linearly transform numeric features into a closed [0, 1] interval.',
    params: ['column'],
    requiresType: 'numeric'
  },
  {
    id: 'standardize',
    label: 'Standardize (Z-Score)',
    icon: Sliders,
    desc: 'Center feature distribution around mean 0 with unit variance (μ=0, σ=1).',
    params: ['column'],
    requiresType: 'numeric'
  },
  {
    id: 'one_hot_encode',
    label: 'One-Hot Encode',
    icon: Binary,
    desc: 'Convert discrete categorical values into binary dummy indicator columns.',
    params: ['column'],
    requiresType: 'categorical'
  },
  {
    id: 'yeo_johnson',
    label: 'Yeo-Johnson Transform',
    icon: Zap,
    desc: 'Apply parametric power transformation to stabilize variance and minimize skew.',
    params: ['column'],
    requiresType: 'numeric'
  },
  {
    id: 'log_transform',
    label: 'Log Transform',
    icon: TrendingUp,
    desc: 'Apply natural logarithmic transform ln(1 + x) to compress right-skewed scales.',
    params: ['column'],
    requiresType: 'numeric'
  },
  {
    id: 'create_lag',
    label: 'Create Lag Feature',
    icon: Repeat,
    desc: 'Generate chronological lag x_(t-k) respecting entity boundaries and preserving missing values.',
    params: ['column', 'time_column', 'group_column', 'lag_order', 'new_name'],
    requiresType: 'any'
  },
  {
    id: 'create_rolling',
    label: 'Trailing Rolling Window',
    icon: TrendingUp,
    desc: 'Compute strictly trailing rolling statistics (mean, std, min, max, sum) over historical window.',
    params: ['column', 'time_column', 'group_column', 'window_size', 'aggregation', 'new_name'],
    requiresType: 'numeric'
  },
  {
    id: 'create_difference',
    label: 'Differencing',
    icon: Shuffle,
    desc: 'Compute discrete change x_t - x_(t-k) to induce stationarity within entity series.',
    params: ['column', 'time_column', 'group_column', 'lag_order', 'new_name'],
    requiresType: 'numeric'
  },
  {
    id: 'create_calendar_features',
    label: 'Calendar Extraction',
    icon: Sliders,
    desc: 'Extract non-constant calendar features (year, month, day, dayofweek, is_weekend, quarter, hour) from timestamps.',
    params: ['time_column'],
    requiresType: 'any'
  },
];

export default function OperationPanel({ mode = 'clean', schema, datasetId, onApplied }) {
  const [selectedOp, setSelectedOp]   = useState(null);
  const [column, setColumn]           = useState('');
  const [fillValue, setFillValue]     = useState('');
  const [newName, setNewName]         = useState('');
  const [nBins, setNBins]             = useState(5);
  const [operator, setOperator]       = useState('==');
  const [filterValue, setFilterValue] = useState('');
  const [timeColumn, setTimeColumn]   = useState('');
  const [groupColumn, setGroupColumn] = useState('');
  const [lagOrder, setLagOrder]       = useState(1);
  const [windowSize, setWindowSize]   = useState(3);
  const [aggregation, setAggregation] = useState('mean');
  const [preview, setPreview]         = useState(null);
  const [previewLoading, setPL]       = useState(false);
  const [applying, setApplying]       = useState(false);
  const [lastApplied, setLastApplied] = useState(null);
  const [lastEvent, setLastEvent]     = useState(null);
  const [activeRightTab, setActiveRightTab] = useState('preview'); // 'preview' | 'audit'
  const [auditEvents, setAuditEvents] = useState([]);
  const [loadingAudit, setLoadingAudit] = useState(false);
  const [error, setError]             = useState('');

  const ops = mode === 'clean' ? CLEAN_OPS : TRANSFORM_OPS;
  const op  = ops.find(o => o.id === selectedOp);

  const loadAuditTrail = async () => {
    if (!datasetId) return;
    setLoadingAudit(true);
    try {
      const res = await fetch(`${API}/api/governance/audit?dataset_id=${encodeURIComponent(datasetId)}`, {
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        setAuditEvents(data.events || []);
      }
    } catch (e) {
      console.warn('Failed to load audit trail:', e);
    } finally {
      setLoadingAudit(false);
    }
  };

  useEffect(() => {
    if (activeRightTab === 'audit' && datasetId) {
      loadAuditTrail();
    }
  }, [activeRightTab, datasetId]);

  const normalizedCols = useMemo(() => {
    if (!schema) return [];
    if (Array.isArray(schema)) {
      return schema.map(c => {
        if (typeof c === 'string') return { name: c, dtype: 'string' };
        return {
          name: c?.name || c?.column_name || c?.column || '',
          dtype: c?.dtype || c?.type || 'string'
        };
      });
    }
    if (typeof schema === 'object') {
      return Object.entries(schema).map(([name, dtype]) => ({
        name,
        dtype: typeof dtype === 'string' ? dtype : (dtype?.dtype || dtype?.type || 'string')
      }));
    }
    return [];
  }, [schema]);

  const allCols     = normalizedCols.map(c => c.name);
  const numericCols = normalizedCols.filter(c => {
    const dt = String(c.dtype || '').toLowerCase();
    return dt.includes('int') || dt.includes('float') || dt.includes('num') || dt.includes('double');
  }).map(c => c.name);
  const catCols     = normalizedCols.filter(c => {
    const dt = String(c.dtype || '').toLowerCase();
    return dt.includes('str') || dt.includes('obj') || dt.includes('cat') || dt.includes('text');
  }).map(c => c.name);

  // Auto-reset state when switching operation
  useEffect(() => {
    setColumn('');
    setFillValue('');
    setNewName('');
    setTimeColumn('');
    setGroupColumn('');
    setLagOrder(1);
    setWindowSize(3);
    setAggregation('mean');
    setPreview(null);
    setError('');
    setLastApplied(null);
  }, [selectedOp]);

  const handlePreview = async () => {
    if (!selectedOp) return;
    setPL(true);
    setError('');
    try {
      const payload = {
        dataset_id: datasetId,
        operation: selectedOp,
        column: column || undefined,
        fill_value: fillValue || undefined,
        new_name: newName || undefined,
        time_column: timeColumn || undefined,
        group_column: groupColumn || undefined,
        lag_order: lagOrder ? parseInt(lagOrder) : undefined,
        window_size: windowSize ? parseInt(windowSize) : undefined,
        aggregation: aggregation || undefined,
      };
      const res = await fetch(`${API}/api/operations/preview`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) {
        const errorMsg = data?.detail?.message
          || (typeof data?.detail === 'string' ? data.detail : null)
          || data?.message
          || formatApiError(data, res.status);
        throw new Error(errorMsg);
      }
      setPreview(data.preview);
    } catch (err) {
      setError(err.message || 'Operation preview failed.');
      setPreview(null);
    } finally {
      setPL(false);
    }
  };

  const handleApply = async () => {
    if (!selectedOp) return;
    setApplying(true);
    setError('');
    try {
      const payload = {
        dataset_id: datasetId,
        operation: selectedOp,
        column: column || undefined,
        fill_value: fillValue || undefined,
        new_name: newName || undefined,
        time_column: timeColumn || undefined,
        group_column: groupColumn || undefined,
        lag_order: lagOrder ? parseInt(lagOrder) : undefined,
        window_size: windowSize ? parseInt(windowSize) : undefined,
        aggregation: aggregation || undefined,
      };
      const res = await fetch(`${API}/api/operations/apply`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) {
        const errorMsg = data?.detail?.message
          || (typeof data?.detail === 'string' ? data.detail : null)
          || data?.message
          || formatApiError(data, res.status);
        throw new Error(errorMsg);
      }
      setLastApplied(op.label);
      if (data?.provenance_event) {
        setLastEvent(data.provenance_event);
      }
      setPreview(null);
      if (onApplied) {
        onApplied(data);
      }
      if (datasetId) {
        loadAuditTrail();
      }
    } catch (err) {
      setError(err.message || 'Operation apply failed.');
    } finally {
      setApplying(false);
    }
  };

  const handleReset = () => {
    setSelectedOp(null);
    setColumn('');
    setFillValue('');
    setNewName('');
    setTimeColumn('');
    setGroupColumn('');
    setLagOrder(1);
    setWindowSize(3);
    setAggregation('mean');
    setPreview(null);
    setError('');
    setLastApplied(null);
  };

  const colsForOp = () => {
    if (!op) return allCols;
    if (['normalize_minmax', 'standardize', 'yeo_johnson', 'log_transform', 'fill_mean', 'fill_median', 'clip_iqr', 'create_rolling', 'create_difference'].includes(op.id)) {
      return numericCols.length > 0 ? numericCols : allCols;
    }
    if (['one_hot_encode'].includes(op.id)) {
      return catCols.length > 0 ? catCols : allCols;
    }
    return allCols;
  };

  const isColMissing = Boolean(
    op && (
      (op.params.includes('column') && !column) ||
      (op.params.includes('time_column') && !timeColumn)
    )
  );
  const IconComponent = op ? op.icon : (mode === 'clean' ? Eraser : Shuffle);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: '#090d16',
      color: '#f8fafc',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      overflow: 'hidden'
    }}>
      {/* Header Bar */}
      <div style={{
        padding: '12px 20px',
        background: '#0f172a',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            background: mode === 'clean' ? '#0369a122' : '#05966922',
            padding: 8,
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: `1px solid ${mode === 'clean' ? '#0284c744' : '#05966944'}`
          }}>
            {mode === 'clean'
              ? <Eraser size={18} color="#38bdf8" />
              : <Shuffle size={18} color="#34d399" />
            }
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#f8fafc' }}>
              {mode === 'clean' ? 'Data Cleaning' : 'Data Transformation'}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 1 }}>
              {mode === 'clean'
                ? 'Governed quality operations — null imputation, deduplication, and outlier clipping'
                : 'Governed feature transformations — scaling, one-hot encoding, and variance stabilization'
              }
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {lastApplied && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: '#052e16',
              border: '1px solid #166534',
              borderRadius: 4,
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 600,
              color: '#86efac'
            }}>
              <CheckCircle2 size={13} />
              <span>{lastApplied} applied</span>
            </div>
          )}
          {lastEvent?.event_id && (
            <span
              style={{
                fontSize: 10,
                fontFamily: 'monospace',
                background: '#090d16',
                color: '#38bdf8',
                padding: '4px 8px',
                borderRadius: 4,
                border: '1px solid #1e293b'
              }}
              title={`Audit Event: ${lastEvent.event_id}\nFingerprint: ${lastEvent.after_state?.fingerprint || ''}`}
            >
              Event: {lastEvent.event_id.slice(0, 14)}…
            </span>
          )}
        </div>
      </div>

      {/* Main Content: Two Columns */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: 'minmax(320px, 380px) 1fr',
        gap: 16,
        padding: 16,
        overflow: 'hidden'
      }}>
        {/* Left Column: Operation Selection & Parameters */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          overflowY: 'auto',
          paddingRight: 4
        }}>
          {/* Operations List */}
          <div style={{
            background: '#0f172a',
            border: '1px solid #1e293b',
            borderRadius: 8,
            overflow: 'hidden'
          }}>
            <div style={{
              padding: '10px 14px',
              background: '#0d131f',
              borderBottom: '1px solid #1e293b',
              fontSize: 11,
              fontWeight: 700,
              color: '#94a3b8',
              letterSpacing: '0.05em',
              textTransform: 'uppercase'
            }}>
              Available Operations ({ops.length})
            </div>
            <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
              {ops.map(o => {
                const ItemIcon = o.icon;
                const isSelected = selectedOp === o.id;
                return (
                  <div
                    key={o.id}
                    onClick={() => setSelectedOp(o.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                      padding: '8px 12px',
                      borderRadius: 6,
                      cursor: 'pointer',
                      background: isSelected ? '#0284c71a' : '#090d16',
                      border: isSelected ? '1px solid #0ea5e9' : '1px solid #1e293b',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    <div style={{
                      padding: 6,
                      borderRadius: 4,
                      background: isSelected ? '#0284c733' : '#1e293b',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <ItemIcon size={14} color={isSelected ? '#38bdf8' : '#94a3b8'} />
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        fontSize: 12,
                        fontWeight: isSelected ? 700 : 500,
                        color: isSelected ? '#f8fafc' : '#cbd5e1'
                      }}>
                        {o.label}
                      </div>
                      <div style={{
                        fontSize: 10,
                        color: '#64748b',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {o.desc}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Parameters Form */}
          {op && (
            <div style={{
              background: '#0f172a',
              border: '1px solid #1e293b',
              borderRadius: 8,
              padding: 14,
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: '#94a3b8',
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase'
                }}>
                  Parameters: {op.label}
                </span>
                <button
                  onClick={handleReset}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: '#64748b',
                    cursor: 'pointer',
                    fontSize: 11,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4
                  }}
                  title="Clear selection"
                >
                  <RotateCcw size={11} /> Reset
                </button>
              </div>

              {op.params.includes('column') && (
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    Target Column {op.requiresType === 'numeric' && <span style={{ color: '#38bdf8' }}>(numeric)</span>}
                  </label>
                  <select
                    value={column}
                    onChange={e => setColumn(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#090d16',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      color: '#f8fafc',
                      fontSize: 12,
                      outline: 'none'
                    }}
                  >
                    <option value="">— Select column —</option>
                    {colsForOp().map(c => {
                      const colMeta = normalizedCols.find(nc => nc.name === c);
                      return (
                        <option key={c} value={c}>
                          {c} {colMeta?.dtype ? `[${colMeta.dtype}]` : ''}
                        </option>
                      );
                    })}
                  </select>
                </div>
              )}

              {op.params.includes('fill_value') && (
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    Replacement Constant
                  </label>
                  <input
                    value={fillValue}
                    onChange={e => setFillValue(e.target.value)}
                    placeholder="Enter fill value (e.g. 0 or 'None')"
                    style={{
                      width: '100%',
                      background: '#090d16',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      color: '#f8fafc',
                      fontSize: 12,
                      outline: 'none',
                      boxSizing: 'border-box'
                    }}
                  />
                </div>
              )}

              {op.params.includes('new_name') && (
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    New Column Name
                  </label>
                  <input
                    value={newName}
                    onChange={e => setNewName(e.target.value)}
                    placeholder="target_column_transformed"
                    style={{
                      width: '100%',
                      background: '#090d16',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      color: '#f8fafc',
                      fontSize: 12,
                      outline: 'none',
                      boxSizing: 'border-box'
                    }}
                  />
                </div>
              )}

              {op.params.includes('n_bins') && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    <span>Histogram Bins</span>
                    <span style={{ color: '#38bdf8' }}>{nBins}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="20"
                    value={nBins}
                    onChange={e => setNBins(parseInt(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              )}

              {op.params.includes('operator') && (
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    Filter Condition
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 70px 1fr', gap: 6 }}>
                    <select
                      value={column}
                      onChange={e => setColumn(e.target.value)}
                      style={{
                        background: '#090d16',
                        border: '1px solid #334155',
                        borderRadius: 6,
                        padding: '6px 8px',
                        color: '#f8fafc',
                        fontSize: 11
                      }}
                    >
                      <option value="">Column</option>
                      {allCols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                    <select
                      value={operator}
                      onChange={e => setOperator(e.target.value)}
                      style={{
                        background: '#090d16',
                        border: '1px solid #334155',
                        borderRadius: 6,
                        padding: '6px 8px',
                        color: '#f8fafc',
                        fontSize: 11
                      }}
                    >
                      {OPERATORS.map(opSym => <option key={opSym} value={opSym}>{opSym}</option>)}
                    </select>
                    <input
                      value={filterValue}
                      onChange={e => setFilterValue(e.target.value)}
                      placeholder="Value"
                      style={{
                        background: '#090d16',
                        border: '1px solid #334155',
                        borderRadius: 6,
                        padding: '6px 8px',
                        color: '#f8fafc',
                        fontSize: 11
                      }}
                    />
                  </div>
                </div>
              )}

              {op.params.includes('time_column') && (
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    Time / Sequence Column <span style={{ color: '#38bdf8' }}>*</span>
                  </label>
                  <select
                    value={timeColumn}
                    onChange={e => setTimeColumn(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#090d16',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      color: '#f8fafc',
                      fontSize: 12,
                      outline: 'none'
                    }}
                  >
                    <option value="">— Select time/sequence column —</option>
                    {allCols.map(c => {
                      const colMeta = normalizedCols.find(nc => nc.name === c);
                      return (
                        <option key={c} value={c}>
                          {c} {colMeta?.dtype ? `[${colMeta.dtype}]` : ''}
                        </option>
                      );
                    })}
                  </select>
                </div>
              )}

              {op.params.includes('group_column') && (
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    Entity / Group Column <span style={{ color: '#64748b', fontWeight: 400 }}>(optional, prevents cross-entity leakage)</span>
                  </label>
                  <select
                    value={groupColumn}
                    onChange={e => setGroupColumn(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#090d16',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      color: '#f8fafc',
                      fontSize: 12,
                      outline: 'none'
                    }}
                  >
                    <option value="">— None (Single global series) —</option>
                    {allCols.map(c => {
                      const colMeta = normalizedCols.find(nc => nc.name === c);
                      return (
                        <option key={c} value={c}>
                          {c} {colMeta?.dtype ? `[${colMeta.dtype}]` : ''}
                        </option>
                      );
                    })}
                  </select>
                </div>
              )}

              {op.params.includes('lag_order') && (
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    Lag Order (k &gt;= 1) <span style={{ color: '#38bdf8' }}>*</span>
                  </label>
                  <input
                    type="number"
                    min="1"
                    value={lagOrder}
                    onChange={e => setLagOrder(Math.max(1, parseInt(e.target.value) || 1))}
                    style={{
                      width: '100%',
                      background: '#090d16',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      color: '#f8fafc',
                      fontSize: 12,
                      outline: 'none',
                      boxSizing: 'border-box'
                    }}
                  />
                </div>
              )}

              {op.params.includes('window_size') && (
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    Rolling Window Size (W &gt;= 2) <span style={{ color: '#38bdf8' }}>*</span>
                  </label>
                  <input
                    type="number"
                    min="2"
                    value={windowSize}
                    onChange={e => setWindowSize(Math.max(2, parseInt(e.target.value) || 2))}
                    style={{
                      width: '100%',
                      background: '#090d16',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      color: '#f8fafc',
                      fontSize: 12,
                      outline: 'none',
                      boxSizing: 'border-box'
                    }}
                  />
                </div>
              )}

              {op.params.includes('aggregation') && (
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>
                    Window Aggregation Statistic <span style={{ color: '#38bdf8' }}>*</span>
                  </label>
                  <select
                    value={aggregation}
                    onChange={e => setAggregation(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#090d16',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      color: '#f8fafc',
                      fontSize: 12,
                      outline: 'none'
                    }}
                  >
                    <option value="mean">Mean (Average)</option>
                    <option value="std">Std (Standard Deviation)</option>
                    <option value="min">Min (Minimum)</option>
                    <option value="max">Max (Maximum)</option>
                    <option value="sum">Sum (Total)</option>
                  </select>
                </div>
              )}

              <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.4, marginTop: 4 }}>
                {op.desc}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Preview & Governance Audit Trail */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          background: '#0f172a',
          border: '1px solid #1e293b',
          borderRadius: 8,
          overflow: 'hidden'
        }}>
          {/* Tabs Toolbar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '8px 14px',
            background: '#0d131f',
            borderBottom: '1px solid #1e293b',
            flexShrink: 0
          }}>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                onClick={() => setActiveRightTab('preview')}
                style={{
                  padding: '5px 10px',
                  fontSize: 11,
                  fontWeight: 600,
                  borderRadius: 4,
                  cursor: 'pointer',
                  border: 'none',
                  background: activeRightTab === 'preview' ? '#1e293b' : 'transparent',
                  color: activeRightTab === 'preview' ? '#38bdf8' : '#94a3b8',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5
                }}
              >
                <Eye size={13} /> Change Preview
              </button>
              <button
                onClick={() => {
                  setActiveRightTab('audit');
                  loadAuditTrail();
                }}
                style={{
                  padding: '5px 10px',
                  fontSize: 11,
                  fontWeight: 600,
                  borderRadius: 4,
                  cursor: 'pointer',
                  border: 'none',
                  background: activeRightTab === 'audit' ? '#1e293b' : 'transparent',
                  color: activeRightTab === 'audit' ? '#34d399' : '#94a3b8',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5
                }}
              >
                <ShieldCheck size={13} /> Governance Audit Trail ({auditEvents.length})
              </button>
            </div>

            {activeRightTab === 'audit' && (
              <button
                onClick={loadAuditTrail}
                disabled={loadingAudit}
                style={{
                  background: '#1e293b',
                  border: '1px solid #334155',
                  color: '#94a3b8',
                  borderRadius: 4,
                  padding: '4px 8px',
                  fontSize: 11,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4
                }}
              >
                <RefreshCw size={11} className={loadingAudit ? 'animate-spin' : ''} /> Refresh
              </button>
            )}
          </div>

          {/* Tab Body */}
          <div style={{ flex: 1, overflowY: 'auto', padding: 14 }}>
            {activeRightTab === 'audit' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {auditEvents.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '40px 0', color: '#64748b' }}>
                    <ShieldCheck size={32} style={{ margin: '0 auto 8px', display: 'block', opacity: 0.4 }} />
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#94a3b8' }}>No Audit Events Recorded</div>
                    <div style={{ fontSize: 11, marginTop: 4 }}>Applied cleaning and transform operations will be logged here with cryptographic hashes.</div>
                  </div>
                ) : (
                  auditEvents.map(evt => (
                    <div
                      key={evt.event_id}
                      style={{
                        padding: '10px 12px',
                        background: '#090d16',
                        border: '1px solid #1e293b',
                        borderRadius: 6,
                        fontSize: 11
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <span style={{ fontWeight: 700, color: '#38bdf8', fontFamily: 'monospace' }}>
                          {evt.event_id}
                        </span>
                        <span style={{ color: '#64748b', fontSize: 10 }}>
                          {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : ''}
                        </span>
                      </div>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
                        <span style={{
                          fontSize: 10,
                          fontWeight: 700,
                          padding: '2px 6px',
                          borderRadius: 3,
                          background: '#064e3b',
                          color: '#6ee7b7'
                        }}>
                          {evt.operation}
                        </span>
                        {evt.affected_columns?.length > 0 && (
                          <span style={{ color: '#94a3b8' }}>
                            [{evt.affected_columns.join(', ')}]
                          </span>
                        )}
                        <span style={{ marginLeft: 'auto', color: '#64748b', fontSize: 10 }}>
                          {evt.execution_time_ms ? `${evt.execution_time_ms.toFixed(1)}ms` : ''}
                        </span>
                      </div>
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: 8,
                        padding: '6px 8px',
                        background: '#0d131f',
                        borderRadius: 4,
                        fontSize: 10,
                        color: '#94a3b8'
                      }}>
                        <div>
                          <span style={{ color: '#f87171', fontWeight: 600 }}>Before: </span>
                          {evt.before_state?.row_count} rows · {evt.before_state?.null_count} nulls
                          <div style={{ fontFamily: 'monospace', color: '#64748b', fontSize: 9 }}>
                            hash: {evt.before_state?.fingerprint?.substring(0, 10)}…
                          </div>
                        </div>
                        <div>
                          <span style={{ color: '#34d399', fontWeight: 600 }}>After: </span>
                          {evt.after_state?.row_count} rows · {evt.after_state?.null_count} nulls
                          <div style={{ fontFamily: 'monospace', color: '#64748b', fontSize: 9 }}>
                            hash: {evt.after_state?.fingerprint?.substring(0, 10)}…
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            ) : preview ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: 12
                }}>
                  {/* Before Box */}
                  <div style={{
                    background: '#090d16',
                    border: '1px solid #1e293b',
                    borderRadius: 6,
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      padding: '6px 10px',
                      background: '#0d131f',
                      borderBottom: '1px solid #1e293b',
                      fontSize: 10,
                      fontWeight: 700,
                      color: '#f87171',
                      letterSpacing: '0.05em',
                      textTransform: 'uppercase'
                    }}>
                      Original Sample (Before)
                    </div>
                    <div style={{ padding: 8, fontFamily: 'monospace', fontSize: 11, maxHeight: 220, overflowY: 'auto' }}>
                      {(preview.before ?? []).slice(0, 12).map((val, i) => (
                        <div key={i} style={{
                          padding: '3px 6px',
                          color: val === null || val === undefined ? '#64748b' : '#cbd5e1',
                          fontStyle: val === null ? 'italic' : 'normal',
                          borderBottom: '1px solid #1e293b22'
                        }}>
                          {val === null || val === undefined ? 'null' : String(val)}
                        </div>
                      ))}
                    </div>
                    {preview.before_stats && (
                      <div style={{
                        padding: '6px 10px',
                        background: '#0d131f',
                        borderTop: '1px solid #1e293b',
                        fontSize: 10,
                        color: '#94a3b8'
                      }}>
                        Nulls: <strong style={{ color: '#f8fafc' }}>{preview.before_stats.nulls}</strong> · Rows: <strong style={{ color: '#f8fafc' }}>{preview.before_stats.rows}</strong>
                      </div>
                    )}
                  </div>

                  {/* After Box */}
                  <div style={{
                    background: '#090d16',
                    border: '1px solid #1e293b',
                    borderRadius: 6,
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      padding: '6px 10px',
                      background: '#0d131f',
                      borderBottom: '1px solid #1e293b',
                      fontSize: 10,
                      fontWeight: 700,
                      color: '#34d399',
                      letterSpacing: '0.05em',
                      textTransform: 'uppercase'
                    }}>
                      Transformed Sample (After)
                    </div>
                    <div style={{ padding: 8, fontFamily: 'monospace', fontSize: 11, maxHeight: 220, overflowY: 'auto' }}>
                      {(preview.after ?? []).slice(0, 12).map((val, i) => (
                        <div key={i} style={{
                          padding: '3px 6px',
                          color: '#34d399',
                          borderBottom: '1px solid #1e293b22'
                        }}>
                          {val === null || val === undefined ? 'null' : String(val)}
                        </div>
                      ))}
                    </div>
                    {preview.after_stats && (
                      <div style={{
                        padding: '6px 10px',
                        background: '#0d131f',
                        borderTop: '1px solid #1e293b',
                        fontSize: 10,
                        color: '#94a3b8'
                      }}>
                        Nulls: <strong style={{ color: '#f8fafc' }}>{preview.after_stats.nulls}</strong> · Rows: <strong style={{ color: '#f8fafc' }}>{preview.after_stats.rows}</strong>
                      </div>
                    )}
                  </div>
                </div>

                {preview.columns && (
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>
                    Affected columns: <span style={{ color: '#38bdf8', fontFamily: 'monospace' }}>{preview.columns.join(', ')}</span>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '60px 0', color: '#64748b' }}>
                <Eye size={36} style={{ margin: '0 auto 10px', display: 'block', opacity: 0.3 }} />
                <div style={{ fontSize: 13, fontWeight: 600, color: '#94a3b8' }}>
                  {op ? 'Preview Changes Available' : 'Select an Operation'}
                </div>
                <div style={{ fontSize: 11, marginTop: 4, maxWidth: 320, margin: '4px auto 0' }}>
                  {op
                    ? 'Configure parameters on the left and click Preview to inspect the mutation before committing.'
                    : 'Choose an operation from the left panel to configure parameters and preview results.'
                  }
                </div>
              </div>
            )}
          </div>

          {/* Structured Error Display */}
          {error && (
            <div style={{
              margin: '0 14px 14px',
              padding: '8px 12px',
              borderRadius: 6,
              background: '#450a0a',
              border: '1px solid #7f1d1d',
              color: '#fca5a5',
              fontSize: 11,
              display: 'flex',
              alignItems: 'flex-start',
              gap: 8
            }}>
              <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong style={{ display: 'block', marginBottom: 2 }}>Operation Error</strong>
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer Actions */}
      <div style={{
        padding: '10px 20px',
        background: '#0f172a',
        borderTop: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        gap: 10,
        flexShrink: 0
      }}>
        {op && (
          <>
            <button
              onClick={handlePreview}
              disabled={previewLoading || isColMissing}
              style={{
                background: '#1e293b',
                color: isColMissing ? '#64748b' : '#f8fafc',
                border: '1px solid #334155',
                borderRadius: 6,
                padding: '7px 14px',
                fontSize: 12,
                fontWeight: 600,
                cursor: (previewLoading || isColMissing) ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 6
              }}
            >
              {previewLoading ? <RefreshCw size={13} className="animate-spin" /> : <Eye size={13} />}
              {previewLoading ? 'Generating Preview…' : 'Preview Changes'}
            </button>

            <button
              onClick={handleApply}
              disabled={applying || isColMissing}
              style={{
                background: isColMissing ? '#1e293b' : '#0284c7',
                color: isColMissing ? '#64748b' : '#ffffff',
                border: 'none',
                borderRadius: 6,
                padding: '7px 16px',
                fontSize: 12,
                fontWeight: 700,
                cursor: (applying || isColMissing) ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 6
              }}
            >
              {applying ? <RefreshCw size={13} className="animate-spin" /> : <Play size={13} />}
              {applying ? 'Applying Operation…' : 'Apply Operation'}
              <ArrowRight size={13} />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
