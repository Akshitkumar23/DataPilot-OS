import React, { useState, useMemo } from 'react';
import { Table, Search, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, FileSpreadsheet, Database, Filter, Hash, Type, RefreshCw, Layers, AlertTriangle } from 'lucide-react';
import { API, getAuthHeaders } from '../apiConfig';

export default function DataPreview({
  tableData = [],
  schema = [],
  rowCount = 0,
  activeDataset = null,
  profile = null,
  onRefresh,
  onSelectColumn
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [isSwitchingSheet, setIsSwitchingSheet] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const availableSheets = activeDataset?.available_sheets || profile?.available_sheets || [];
  const currentSheet = activeDataset?.sheet_name || profile?.sheet_name || (availableSheets.length > 0 ? availableSheets[0] : '');
  const isSourceNative = Boolean(activeDataset?.is_source_native ?? profile?.is_source_native ?? (activeDataset?.source_storage === 'source_native'));
  const sourceStorage = activeDataset?.source_storage || profile?.source_storage || (isSourceNative ? 'source_native' : 'managed_upload');
  const safeDisplayPath = activeDataset?.safe_display_path || profile?.safe_display_path || activeDataset?.name || '';
  const driftDetected = Boolean(activeDataset?.drift_detected || profile?.drift_detected);
  const driftReason = activeDataset?.drift_reason || profile?.drift_reason || '';

  const handleSheetChange = async (e) => {
    const newSheet = e.target.value;
    const dsId = activeDataset?.id || activeDataset?.dataset_id;
    if (!dsId || !newSheet) return;
    setIsSwitchingSheet(true);
    try {
      const res = await fetch(`${API}/api/datasets/${dsId}/select-sheet`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ sheet_name: newSheet })
      });
      if (res.ok) {
        if (onRefresh) onRefresh();
      } else {
        const err = await res.json().catch(() => null);
        alert(err?.detail || err?.message || 'Failed to switch worksheet');
      }
    } catch (err) {
      console.error('Worksheet selection failed:', err);
    } finally {
      setIsSwitchingSheet(false);
    }
  };

  const handleRefreshSource = async () => {
    const dsId = activeDataset?.id || activeDataset?.dataset_id;
    if (!dsId) return;
    setIsRefreshing(true);
    try {
      const res = await fetch(`${API}/api/datasets/${dsId}/refresh`, {
        method: 'POST',
        headers: getAuthHeaders()
      });
      if (res.ok) {
        if (onRefresh) onRefresh();
      } else {
        const err = await res.json().catch(() => null);
        alert(err?.detail || err?.message || 'Failed to refresh dataset from disk');
      }
    } catch (err) {
      console.error('Dataset refresh failed:', err);
    } finally {
      setIsRefreshing(false);
    }
  };

  // Filter rows based on search term
  const filteredData = useMemo(() => {
    if (!searchTerm.trim()) return tableData;
    const term = searchTerm.toLowerCase();
    return tableData.filter((row) =>
      Object.values(row).some((val) =>
        val !== null && val !== undefined && String(val).toLowerCase().includes(term)
      )
    );
  }, [tableData, searchTerm]);

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(filteredData.length / pageSize));
  const pageData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredData.slice(start, start + pageSize);
  }, [filteredData, currentPage, pageSize]);

  // Extract columns: prioritize real computed profile, fallback to schema, then tableData
  const columns = useMemo(() => {
    if (profile && Array.isArray(profile.columns) && profile.columns.length > 0) {
      return profile.columns.map(c => ({
        name: c.name,
        type: c.normalized_type || c.dtype || 'STR',
        dtype: c.dtype,
        nullCount: c.null_count ?? 0,
        nullPct: c.null_percentage !== undefined ? Number(c.null_percentage).toFixed(1) : '0.0',
        uniqueCount: c.unique_count ?? 0,
        uniquePct: c.unique_percentage !== undefined ? Number(c.unique_percentage).toFixed(1) : '0.0'
      }));
    }

    if (schema && Array.isArray(schema) && schema.length > 0) {
      return schema.map(s => {
        const colName = typeof s === 'string' ? s : (s?.column_name || s?.name || s?.column || '');
        const colType = typeof s === 'object' && s !== null
          ? (s?.normalized_type || s?.semantic_type || s?.dtype || s?.type || 'object')
          : 'object';
        return {
          name: colName,
          type: colType,
          nullCount: (typeof s === 'object' && (s?.null_count ?? s?.nullCount)) ?? 0,
          nullPct: (typeof s === 'object' && (s?.null_percentage ?? s?.null_pct ?? s?.nullPct))
            ? Number(s.null_percentage ?? s.null_pct ?? s.nullPct).toFixed(1)
            : ((typeof s === 'object' && s?.null_ratio !== undefined) ? Number(s.null_ratio * 100).toFixed(1) : '0.0'),
          uniqueCount: (typeof s === 'object' && (s?.unique_count ?? s?.uniqueCount ?? s?.cardinality)) ?? 0
        };
      });
    }

    if (schema && typeof schema === 'object' && !Array.isArray(schema) && Object.keys(schema).length > 0) {
      return Object.entries(schema).map(([name, dtype]) => {
        const info = typeof dtype === 'object' && dtype !== null ? dtype : { dtype };
        return {
          name,
          type: info.normalized_type || info.semantic_role || info.dtype || info.type || 'object',
          nullCount: info.null_count ?? 0,
          nullPct: info.null_percentage !== undefined
            ? Number(info.null_percentage).toFixed(1)
            : (info.null_ratio ? Number(info.null_ratio * 100).toFixed(1) : '0.0'),
          uniqueCount: info.cardinality ?? info.unique_count ?? 0
        };
      });
    }

    if (tableData && tableData.length > 0) {
      return Object.keys(tableData[0]).filter(k => k !== '_row_idx').map(col => ({
        name: col,
        type: typeof tableData[0][col] === 'number' ? 'NUM' : 'STR',
        nullCount: 0,
        nullPct: '0.0',
        uniqueCount: 0
      }));
    }
    return [];
  }, [profile, schema, tableData]);

  const getTypeBadge = (typeStr) => {
    const t = String(typeStr).toUpperCase();
    if (t === 'NUM' || t.includes('INT') || t.includes('FLOAT') || t.includes('DOUBLE') || t.includes('NUMBER')) {
      return <span style={{ background: 'rgba(14, 165, 233, 0.15)', color: '#38bdf8', padding: '2px 6px', borderRadius: 4, fontSize: 10, fontWeight: 700 }}>NUM</span>;
    }
    if (t === 'DATE' || t.includes('TIME') || t.includes('TIMESTAMP')) {
      return <span style={{ background: 'rgba(168, 85, 247, 0.15)', color: '#c084fc', padding: '2px 6px', borderRadius: 4, fontSize: 10, fontWeight: 700 }}>DATE</span>;
    }
    if (t === 'BOOL' || t.includes('BOOLEAN')) {
      return <span style={{ background: 'rgba(234, 179, 8, 0.15)', color: '#facc15', padding: '2px 6px', borderRadius: 4, fontSize: 10, fontWeight: 700 }}>BOOL</span>;
    }
    return <span style={{ background: 'rgba(148, 163, 184, 0.15)', color: '#cbd5e1', padding: '2px 6px', borderRadius: 4, fontSize: 10, fontWeight: 700 }}>STR</span>;
  };

  // Derive trustworthy dataset-level metrics from real active dataset profile
  const effectiveRowCount = profile?.row_count ?? rowCount ?? filteredData.length;
  const missingCells = profile?.missing_cells;
  const missingPct = profile?.missing_percentage !== undefined
    ? Number(profile.missing_percentage).toFixed(1)
    : (profile?.total_missing_ratio !== undefined ? Number(profile.total_missing_ratio * 100).toFixed(1) : null);

  const duplicateRows = profile?.duplicate_rows ?? profile?.duplicate_count;
  const duplicatePct = profile?.duplicate_percentage !== undefined
    ? Number(profile.duplicate_percentage).toFixed(1)
    : (effectiveRowCount > 0 && duplicateRows !== undefined ? Number((duplicateRows / effectiveRowCount) * 100).toFixed(1) : null);

  const schemaStatus = profile?.schema_status || {
    status: 'Valid',
    message: 'All columns type-consistent',
    is_valid: true
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      width: '100%',
      background: '#090d16',
      color: '#f8fafc',
      fontFamily: 'Inter, system-ui, sans-serif'
    }}>
      {/* Header Toolbar */}
      <div style={{
        padding: '10px 16px',
        background: '#0f172a',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 12
      }}>
        {/* Title & Info */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ background: '#10b981', padding: 6, borderRadius: 6, display: 'flex', alignItems: 'center' }}>
            <FileSpreadsheet size={16} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 800, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span>{activeDataset?.name || 'Active Dataset Preview Grid'}</span>
              <span style={{ fontSize: 11, background: '#1e293b', color: '#0ea5e9', padding: '2px 8px', borderRadius: 12, fontWeight: 600 }}>
                {effectiveRowCount.toLocaleString()} Rows × {columns.length} Columns
              </span>
              {activeDataset?.format && (
                <span style={{ fontSize: 10, background: '#1e293b', color: '#94a3b8', padding: '2px 8px', borderRadius: 12, fontWeight: 700, textTransform: 'uppercase', border: '1px solid #334155' }}>
                  {activeDataset.format}
                </span>
              )}
              {isSourceNative ? (
                <span
                  style={{
                    fontSize: 10,
                    background: 'rgba(16, 185, 129, 0.15)',
                    color: '#34d399',
                    padding: '2px 8px',
                    borderRadius: 12,
                    fontWeight: 700,
                    border: '1px solid rgba(16, 185, 129, 0.3)'
                  }}
                  title="Loaded in-place directly from original local disk path (zero-copy to uploads)"
                >
                  SOURCE-NATIVE
                </span>
              ) : (
                <span
                  style={{
                    fontSize: 10,
                    background: 'rgba(148, 163, 184, 0.15)',
                    color: '#94a3b8',
                    padding: '2px 8px',
                    borderRadius: 12,
                    fontWeight: 700,
                    border: '1px solid #334155'
                  }}
                >
                  {sourceStorage === 'promoted_result' ? 'SAVED RESULT' : 'MANAGED UPLOAD'}
                </span>
              )}
            </div>
            <div style={{ fontSize: 10, color: '#64748b', display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
              <span>Live Pandas DataFrame Sample Grid</span>
              {safeDisplayPath && (
                <>
                  <span>•</span>
                  <span style={{ color: '#94a3b8' }} title={`Source path: ${safeDisplayPath}`}>Source: {safeDisplayPath}</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Controls: Sheet Selector, Refresh, Search & Page Size */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          {/* Worksheet Selector if Excel */}
          {availableSheets.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#1e293b', padding: '3px 8px', borderRadius: 6, border: '1px solid #334155' }}>
              <Layers size={13} color="#38bdf8" />
              <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Sheet:</span>
              <select
                value={currentSheet}
                onChange={handleSheetChange}
                disabled={isSwitchingSheet}
                style={{
                  background: 'transparent',
                  color: '#38bdf8',
                  border: 'none',
                  fontSize: 11,
                  fontWeight: 700,
                  outline: 'none',
                  cursor: isSwitchingSheet ? 'wait' : 'pointer'
                }}
              >
                {availableSheets.map(s => (
                  <option key={s} value={s} style={{ background: '#0f172a', color: '#f8fafc' }}>{s}</option>
                ))}
              </select>
            </div>
          )}

          {/* Reload from Source Button */}
          <button
            onClick={handleRefreshSource}
            disabled={isRefreshing}
            title="Reload pristine data directly from source file on disk"
            style={{
              background: '#1e293b',
              color: '#94a3b8',
              border: '1px solid #334155',
              borderRadius: 6,
              padding: '4px 8px',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
              cursor: isRefreshing ? 'wait' : 'pointer'
            }}
          >
            <RefreshCw size={12} className={isRefreshing ? 'animate-spin' : ''} />
            <span>{isRefreshing ? 'Reloading...' : 'Reload'}</span>
          </button>

          {/* Search Bar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: 6,
            padding: '4px 10px',
            gap: 6
          }}>
            <Search size={14} color="#94a3b8" />
            <input
              type="text"
              placeholder="Search dataset rows..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              style={{
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: '#f8fafc',
                fontSize: 12,
                width: 160
              }}
            />
          </div>

          {/* Rows Per Page Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#94a3b8' }}>
            <span>Rows:</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setCurrentPage(1);
              }}
              style={{
                background: '#1e293b',
                color: '#38bdf8',
                border: '1px solid #334155',
                borderRadius: 4,
                padding: '3px 8px',
                fontSize: 11,
                fontWeight: 700,
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              <option value={15}>15</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
        </div>
      </div>

      {/* Drift Warning Banner */}
      {driftDetected && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.15)',
          borderBottom: '1px solid rgba(239, 68, 68, 0.4)',
          color: '#fca5a5',
          padding: '8px 16px',
          fontSize: 11,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 8
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertTriangle size={15} color="#ef4444" />
            <span><strong>Source File Drift Detected:</strong> {driftReason || 'Original disk file has changed or moved.'}</span>
          </div>
          <button
            onClick={handleRefreshSource}
            style={{
              background: '#ef4444',
              color: '#ffffff',
              border: 'none',
              borderRadius: 4,
              padding: '3px 10px',
              fontSize: 11,
              fontWeight: 700,
              cursor: 'pointer'
            }}
          >
            Reload Source File
          </button>
        </div>
      )}

      {/* Data Quality Health Score Gauges */}
      {columns.length > 0 && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          padding: '10px 16px',
          background: '#0a0f1d',
          borderBottom: '1px solid #1e293b',
          fontSize: 12
        }}>
          <span style={{ fontWeight: 700, color: '#38bdf8' }}>📊 Data Quality Health Gauges:</span>

          {/* Missingness Gauge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#0f172a', padding: '4px 10px', borderRadius: 6, border: '1px solid #1e293b' }}>
            <span style={{ color: '#94a3b8', fontWeight: 600 }}>Missingness:</span>
            {missingPct === null ? (
              <span style={{ color: '#64748b', fontWeight: 700 }}>Calculating...</span>
            ) : Number(missingPct) === 0 ? (
              <span style={{ color: '#10b981', fontWeight: 700 }}>0.0% Clean</span>
            ) : (
              <span style={{ color: '#f87171', fontWeight: 700 }}>
                {missingPct}% ({missingCells !== undefined ? `${missingCells.toLocaleString()} cells` : 'missing'})
              </span>
            )}
          </div>

          {/* Duplicates Gauge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#0f172a', padding: '4px 10px', borderRadius: 6, border: '1px solid #1e293b' }}>
            <span style={{ color: '#94a3b8', fontWeight: 600 }}>Duplicates:</span>
            {duplicateRows === undefined || duplicateRows === null ? (
              <span style={{ color: '#64748b', fontWeight: 700 }}>Calculating...</span>
            ) : duplicateRows === 0 ? (
              <span style={{ color: '#10b981', fontWeight: 700 }}>0 Duplicates (0.0%)</span>
            ) : (
              <span style={{ color: '#f59e0b', fontWeight: 700 }}>
                {duplicateRows.toLocaleString()} Duplicates ({duplicatePct || '0.0'}%)
              </span>
            )}
          </div>

          {/* Schema Type Safety Gauge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#0f172a', padding: '4px 10px', borderRadius: 6, border: '1px solid #1e293b' }}>
            <span style={{ color: '#94a3b8', fontWeight: 600 }}>Schema Type Safety:</span>
            {schemaStatus.is_valid ? (
              <span style={{ color: '#10b981', fontWeight: 700 }}>Schema: Valid</span>
            ) : (
              <span style={{ color: '#f59e0b', fontWeight: 700 }} title={schemaStatus.message}>
                {schemaStatus.message || 'Warning: Type Inconsistency'}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Main Grid View */}
      <div style={{ flex: 1, overflow: 'auto', background: '#090d16' }}>
        {columns.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b', gap: 10 }}>
            <Database size={32} color="#334155" />
            <span style={{ fontSize: 13 }}>No dataset active. Load a CSV or Excel file to preview data.</span>
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
            <thead>
              <tr style={{ background: '#0f172a', borderBottom: '2px solid #1e293b', position: 'sticky', top: 0, zIndex: 10 }}>
                <th style={{ padding: '8px 12px', width: 50, color: '#64748b', borderRight: '1px solid #1e293b', textAlign: 'center' }}>#</th>
                {columns.map((col, idx) => (
                  <th
                    key={idx}
                    style={{ padding: '8px 14px', borderRight: '1px solid #1e293b', whiteSpace: 'nowrap', cursor: 'pointer' }}
                    onClick={() => onSelectColumn && onSelectColumn(col.name)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                      <div>
                        <div style={{ fontWeight: 700, color: '#e2e8f0' }}>{col.name}</div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 9 }}>
                          {col.nullCount > 0 ? (
                            <span style={{ color: '#f87171', fontWeight: 600 }}>
                              {col.nullCount.toLocaleString()} nulls ({col.nullPct}%)
                            </span>
                          ) : (
                            <span style={{ color: '#10b981', fontWeight: 500 }}>
                              0 nulls
                            </span>
                          )}
                          <span style={{ color: '#64748b', fontWeight: 500 }}>
                            {col.uniqueCount.toLocaleString()} unique
                          </span>
                        </div>
                      </div>
                      {getTypeBadge(col.type)}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageData.length === 0 ? (
                <tr>
                  <td colSpan={columns.length + 1} style={{ textAlign: 'center', padding: 24, color: '#64748b' }}>
                    No matching rows found for "{searchTerm}"
                  </td>
                </tr>
              ) : (
                pageData.map((row, rIdx) => {
                  const absoluteIndex = (currentPage - 1) * pageSize + rIdx + 1;
                  return (
                    <tr
                      key={rIdx}
                      style={{
                        background: rIdx % 2 === 0 ? '#090d16' : '#0f172a',
                        borderBottom: '1px solid #1e293b',
                        transition: 'background 0.15s ease'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.background = '#1e293b'}
                      onMouseLeave={(e) => e.currentTarget.style.background = rIdx % 2 === 0 ? '#090d16' : '#0f172a'}
                    >
                      <td style={{ padding: '6px 12px', color: '#64748b', borderRight: '1px solid #1e293b', textAlign: 'center', fontWeight: 600 }}>
                        {absoluteIndex}
                      </td>
                      {columns.map((col, cIdx) => {
                        const val = row[col.name];
                        const displayVal = val === null || val === undefined ? 'NaN' : String(val);
                        const isNull = val === null || val === undefined || displayVal === 'NaN';

                        return (
                          <td
                            key={cIdx}
                            style={{
                              padding: '6px 14px',
                              borderRight: '1px solid #1e293b',
                              color: isNull ? '#ef4444' : '#cbd5e1',
                              fontStyle: isNull ? 'italic' : 'normal',
                              whiteSpace: 'nowrap',
                              maxWidth: 300,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis'
                            }}
                          >
                            {displayVal}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        )}
      </div>

      {/* Footer Pagination Bar */}
      <div style={{
        padding: '8px 16px',
        background: '#0f172a',
        borderTop: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: 11,
        color: '#94a3b8'
      }}>
        <div>
          Showing <span style={{ color: '#38bdf8', fontWeight: 700 }}>{filteredData.length > 0 ? (currentPage - 1) * pageSize + 1 : 0}</span> to{' '}
          <span style={{ color: '#38bdf8', fontWeight: 700 }}>{Math.min(currentPage * pageSize, filteredData.length)}</span> of{' '}
          <span style={{ color: '#f8fafc', fontWeight: 700 }}>{filteredData.length.toLocaleString()}</span> entries
        </div>

        {/* Pagination Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button
            onClick={() => setCurrentPage(1)}
            disabled={currentPage === 1}
            style={{ background: '#1e293b', border: 'none', color: '#f8fafc', padding: 4, borderRadius: 4, cursor: currentPage === 1 ? 'not-allowed' : 'pointer', opacity: currentPage === 1 ? 0.4 : 1 }}
          >
            <ChevronsLeft size={14} />
          </button>
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            style={{ background: '#1e293b', border: 'none', color: '#f8fafc', padding: 4, borderRadius: 4, cursor: currentPage === 1 ? 'not-allowed' : 'pointer', opacity: currentPage === 1 ? 0.4 : 1 }}
          >
            <ChevronLeft size={14} />
          </button>

          <span style={{ padding: '0 8px', fontWeight: 700, color: '#38bdf8' }}>
            Page {currentPage} of {totalPages}
          </span>

          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            style={{ background: '#1e293b', border: 'none', color: '#f8fafc', padding: 4, borderRadius: 4, cursor: currentPage === totalPages ? 'not-allowed' : 'pointer', opacity: currentPage === totalPages ? 0.4 : 1 }}
          >
            <ChevronRight size={14} />
          </button>
          <button
            onClick={() => setCurrentPage(totalPages)}
            disabled={currentPage === totalPages}
            style={{ background: '#1e293b', border: 'none', color: '#f8fafc', padding: 4, borderRadius: 4, cursor: currentPage === totalPages ? 'not-allowed' : 'pointer', opacity: currentPage === totalPages ? 0.4 : 1 }}
          >
            <ChevronsRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
