export default function PropertiesPanel({ selectedColumn, schema }) {
  if (!selectedColumn || !schema) {
    return (
      <div className="props-panel">
        <div className="props-header">Properties</div>
        <div className="props-body">
          <div style={{ color: 'var(--text-muted)', fontSize: 11, fontStyle: 'italic', padding: '8px 0' }}>
            Click a column header to inspect it.
          </div>
        </div>
      </div>
    );
  }

  const col = schema.find(c => c.name === selectedColumn);
  if (!col) return null;

  const stats = col.stats || {};
  const mean = stats.mean ?? col.mean;
  const min = stats.min ?? col.min;
  const max = stats.max ?? col.max;
  const std = stats.std ?? col.std;
  const topValues = stats.top_values ?? col.top_values;

  const totalCount = stats.total ?? col.total;
  const nullCount = stats.null_count ?? col.null_count;
  const uniqueCount = stats.unique_count ?? col.unique_count;

  const nullPct   = nullCount != null ? ((nullCount / (totalCount || 1)) * 100).toFixed(1) : '—';
  const nullClass = parseFloat(nullPct) > 20 ? 'bad' : parseFloat(nullPct) > 5 ? 'warn' : 'good';
  const uniquePct = uniqueCount != null ? ((uniqueCount / (totalCount || 1)) * 100).toFixed(1) : '—';

  const isNumeric = col.dtype?.includes('int') || col.dtype?.includes('float');
  const isCat     = col.dtype === 'object' || col.dtype === 'category';

  const typeColor = isNumeric ? '#93c5fd' : isCat ? '#c4b5fd' : 'var(--text-secondary)';
  const typeLabel = isNumeric ? 'Numeric' : isCat ? 'Categorical' : col.dtype;

  const topList = Array.isArray(topValues)
    ? topValues
    : typeof topValues === 'object' && topValues !== null
      ? Object.entries(topValues)
      : [];

  return (
    <div className="props-panel">
      <div className="props-header">Properties</div>
      <div className="props-body">
        {/* Column identity */}
        <div className="props-group">
          <div className="props-group-title">Column</div>
          <div style={{
            padding: '8px',
            background: 'var(--bg-widget)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border)',
            marginBottom: 6,
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 3 }}>
              {col.name}
            </div>
            <span className="badge" style={{ background: 'rgba(147,197,253,0.12)', color: typeColor, fontSize: 10 }}>
              {typeLabel}
            </span>
          </div>
        </div>

        {/* Stats */}
        <div className="props-group">
          <div className="props-group-title">Statistics</div>
          <div className="props-row">
            <span className="props-key">Total rows</span>
            <span className="props-val">{totalCount?.toLocaleString() ?? '—'}</span>
          </div>
          <div className="props-row">
            <span className="props-key">Null values</span>
            <span className={`props-val ${nullClass}`}>{nullCount ?? 0}</span>
          </div>
          <div className="props-row">
            <span className="props-key">Null %</span>
            <span className={`props-val ${nullClass}`}>{nullPct}%</span>
          </div>

          {/* Null fill bar */}
          <div className="props-bar">
            <div
              className={`props-bar__fill ${nullClass !== 'good' ? nullClass : ''}`}
              style={{ width: `${nullPct}%` }}
            />
          </div>

          <div className="props-row">
            <span className="props-key">Unique</span>
            <span className="props-val">{uniqueCount?.toLocaleString() ?? '—'}</span>
          </div>
          <div className="props-row">
            <span className="props-key">Unique %</span>
            <span className="props-val">{uniquePct}%</span>
          </div>
        </div>

        {/* Numeric-specific */}
        {isNumeric && (
          <div className="props-group">
            <div className="props-group-title">Distribution & Diagnostics</div>
            {mean != null && (
              <div className="props-row">
                <span className="props-key">Mean</span>
                <span className="props-val">{Number(mean).toFixed(2)}</span>
              </div>
            )}
            {std != null && (
              <div className="props-row">
                <span className="props-key">Std Dev</span>
                <span className="props-val">{Number(std).toFixed(2)}</span>
              </div>
            )}
            {min != null && (
              <div className="props-row">
                <span className="props-key">Min</span>
                <span className="props-val">{Number(min).toFixed(2)}</span>
              </div>
            )}
            {max != null && (
              <div className="props-row">
                <span className="props-key">Max</span>
                <span className="props-val">{Number(max).toFixed(2)}</span>
              </div>
            )}
            {/* Skewness Diagnostic */}
            <div className="props-row" style={{ marginTop: 4 }}>
              <span className="props-key">Skewness</span>
              <span className="props-val" style={{ color: '#38bdf8', fontWeight: 700 }}>
                {stats.skew != null ? Number(stats.skew).toFixed(2) : (col.skew != null ? Number(col.skew).toFixed(2) : '0.00')}
              </span>
            </div>
            <div style={{ marginTop: 6, fontSize: 10, padding: '4px 8px', background: '#0f172a', border: '1px solid #1e293b', borderRadius: 4, color: '#94a3b8' }}>
              {Math.abs(stats.skew ?? col.skew ?? 0) > 1.0 ? '⚡ Highly Skewed (Median Recommended)' : '✓ Normal Gaussian Curve'}
            </div>
          </div>
        )}

        {/* Categorical-specific */}
        {isCat && topList.length > 0 && (
          <div className="props-group">
            <div className="props-group-title">Top Values</div>
            {topList.slice(0, 5).map(([val, cnt]) => (
              <div key={val} className="props-row">
                <span className="props-key" style={{ fontSize: 10, maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {String(val)}
                </span>
                <span className="props-val">{cnt}</span>
              </div>
            ))}
          </div>
        )}

        {/* Suggested operations */}
        <div className="props-group">
          <div className="props-group-title">Suggest</div>
          {parseFloat(nullPct) > 0 && (
            <div className="alert alert--warning" style={{ marginBottom: 5 }}>
              <span style={{ fontSize: 12 }}>⚠</span>
              <span style={{ fontSize: 11 }}>{nullPct}% missing values — consider imputation</span>
            </div>
          )}
          {isNumeric && parseFloat(uniquePct) < 5 && (
            <div className="alert alert--info">
              <span style={{ fontSize: 12 }}>ℹ</span>
              <span style={{ fontSize: 11 }}>Low cardinality — may be categorical</span>
            </div>
          )}
          {isCat && parseFloat(uniquePct) > 80 && (
            <div className="alert alert--info">
              <span style={{ fontSize: 12 }}>ℹ</span>
              <span style={{ fontSize: 11 }}>High cardinality — consider hashing or dropping</span>
            </div>
          )}
          {parseFloat(nullPct) === 0 && (
            <div className="alert alert--success">
              <span style={{ fontSize: 12 }}>✓</span>
              <span style={{ fontSize: 11 }}>No missing values</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
