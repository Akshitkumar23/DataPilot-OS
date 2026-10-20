import React, { useState, useMemo } from 'react';
import { 
  Table, 
  Search, 
  ChevronLeft, 
  ChevronRight, 
  ChevronsLeft, 
  ChevronsRight, 
  FileSpreadsheet, 
  Database, 
  Download,
  Filter,
  ArrowUpDown
} from 'lucide-react';

export default function DataGrid({ 
  tableData = [], 
  schema = [], 
  rowCount = 0, 
  title = 'Data Grid Preview', 
  activeDataset = null, 
  onSelectColumn,
  onRefresh,
  pageSize: initialPageSize = 25
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(initialPageSize);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

  // Extract columns dynamically if schema is empty or incomplete
  const columns = useMemo(() => {
    if (schema && schema.length > 0) {
      return schema.map(s => {
        if (typeof s === 'string') {
          return { name: s, type: 'string', nullCount: 0, nullPct: '0.0', uniqueCount: 0 };
        }
        return {
          name: s.column_name || s.name || s.column || 'Column',
          type: s.dtype || s.type || 'object',
          nullCount: s.null_count ?? 0,
          nullPct: s.null_pct ? Number(s.null_pct).toFixed(1) : '0.0',
          uniqueCount: s.unique_count ?? 0
        };
      });
    }
    if (tableData && tableData.length > 0) {
      const firstRow = tableData[0];
      if (Array.isArray(firstRow)) {
        return firstRow.map((_, i) => ({
          name: `Col_${i + 1}`,
          type: 'string',
          nullCount: 0,
          nullPct: '0.0',
          uniqueCount: 0
        }));
      }
      if (typeof firstRow === 'object' && firstRow !== null) {
        return Object.keys(firstRow).map(col => ({
          name: col,
          type: typeof firstRow[col] === 'number' ? 'number' : typeof firstRow[col] === 'boolean' ? 'boolean' : 'string',
          nullCount: 0,
          nullPct: '0.0',
          uniqueCount: 0
        }));
      }
    }
    return [];
  }, [schema, tableData]);

  // Filter rows based on search term
  const filteredData = useMemo(() => {
    if (!Array.isArray(tableData)) return [];
    if (!searchTerm.trim()) return tableData;
    const term = searchTerm.toLowerCase();

    return tableData.filter((row) => {
      if (Array.isArray(row)) {
        return row.some(val => val !== null && val !== undefined && String(val).toLowerCase().includes(term));
      }
      if (typeof row === 'object' && row !== null) {
        return Object.values(row).some(val => val !== null && val !== undefined && String(val).toLowerCase().includes(term));
      }
      return String(row).toLowerCase().includes(term);
    });
  }, [tableData, searchTerm]);

  // Sort rows if sortConfig is active
  const sortedData = useMemo(() => {
    if (!sortConfig.key) return filteredData;
    return [...filteredData].sort((a, b) => {
      const valA = Array.isArray(a) ? a[columns.findIndex(c => c.name === sortConfig.key)] : a?.[sortConfig.key];
      const valB = Array.isArray(b) ? b[columns.findIndex(c => c.name === sortConfig.key)] : b?.[sortConfig.key];

      if (valA === valB) return 0;
      if (valA === null || valA === undefined) return 1;
      if (valB === null || valB === undefined) return -1;

      if (typeof valA === 'number' && typeof valB === 'number') {
        return sortConfig.direction === 'asc' ? valA - valB : valB - valA;
      }
      return sortConfig.direction === 'asc' 
        ? String(valA).localeCompare(String(valB)) 
        : String(valB).localeCompare(String(valA));
    });
  }, [filteredData, sortConfig, columns]);

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(sortedData.length / pageSize));
  const pageData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return sortedData.slice(start, start + pageSize);
  }, [sortedData, currentPage, pageSize]);

  const handleSort = (colName) => {
    setSortConfig(prev => ({
      key: colName,
      direction: prev.key === colName && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const handleExportCSV = () => {
    if (!columns.length || !tableData.length) return;
    const headerRow = columns.map(c => `"${c.name}"`).join(',');
    const bodyRows = tableData.map(row => {
      return columns.map(c => {
        const val = Array.isArray(row) ? row[columns.indexOf(c)] : row[c.name];
        if (val === null || val === undefined) return '""';
        return `"${String(val).replace(/"/g, '""')}"`;
      }).join(',');
    }).join('\n');

    const blob = new Blob([`${headerRow}\n${bodyRows}`], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `datapilot_export_${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getTypeBadge = (typeStr) => {
    const t = String(typeStr).toLowerCase();
    if (t.includes('int') || t.includes('float') || t.includes('num')) {
      return <span style={{ background: 'rgba(14, 165, 233, 0.15)', color: '#38bdf8', padding: '2px 5px', borderRadius: 4, fontSize: 9, fontWeight: 700 }}>NUM</span>;
    }
    if (t.includes('date') || t.includes('time')) {
      return <span style={{ background: 'rgba(168, 85, 247, 0.15)', color: '#c084fc', padding: '2px 5px', borderRadius: 4, fontSize: 9, fontWeight: 700 }}>DATE</span>;
    }
    if (t.includes('bool')) {
      return <span style={{ background: 'rgba(234, 179, 8, 0.15)', color: '#facc15', padding: '2px 5px', borderRadius: 4, fontSize: 9, fontWeight: 700 }}>BOOL</span>;
    }
    return <span style={{ background: 'rgba(148, 163, 184, 0.15)', color: '#cbd5e1', padding: '2px 5px', borderRadius: 4, fontSize: 9, fontWeight: 700 }}>STR</span>;
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      width: '100%',
      background: '#090d16',
      color: '#f8fafc',
      fontFamily: 'Inter, system-ui, sans-serif',
      overflow: 'hidden'
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
        gap: 12,
        flexShrink: 0
      }}>
        {/* Title & Stats */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ background: '#10b981', padding: 6, borderRadius: 6, display: 'flex', alignItems: 'center' }}>
            <FileSpreadsheet size={16} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 800, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>{activeDataset?.name || title}</span>
              <span style={{ fontSize: 11, background: '#1e293b', color: '#38bdf8', padding: '2px 8px', borderRadius: 12, fontWeight: 600 }}>
                {(rowCount || sortedData.length).toLocaleString()} Rows × {columns.length} Cols
              </span>
            </div>
          </div>
        </div>

        {/* Controls: Search, Page Size & CSV Export */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
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
              placeholder="Search table rows..."
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

          {/* Rows Per Page */}
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

          {/* CSV Export Button */}
          {tableData.length > 0 && (
            <button
              onClick={handleExportCSV}
              title="Download Data as CSV"
              style={{
                background: '#1e293b',
                color: '#4ade80',
                border: '1px solid #334155',
                padding: '4px 10px',
                borderRadius: 4,
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4
              }}
            >
              <Download size={12} />
              CSV
            </button>
          )}
        </div>
      </div>

      {/* Main Grid View Container */}
      <div style={{ flex: 1, overflow: 'auto', background: '#090d16', width: '100%' }}>
        {columns.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b', gap: 10, padding: 30 }}>
            <Database size={32} color="#334155" />
            <span style={{ fontSize: 13 }}>No dataset or table loaded.</span>
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
            <thead>
              <tr style={{ background: '#0f172a', borderBottom: '2px solid #1e293b', position: 'sticky', top: 0, zIndex: 10 }}>
                <th style={{ padding: '8px 12px', width: 50, color: '#64748b', borderRight: '1px solid #1e293b', textAlign: 'center' }}>#</th>
                {columns.map((col, idx) => (
                  <th
                    key={idx}
                    style={{ padding: '8px 14px', borderRight: '1px solid #1e293b', whiteSpace: 'nowrap', cursor: 'pointer', userSelect: 'none' }}
                    onClick={() => {
                      handleSort(col.name);
                      if (onSelectColumn) onSelectColumn(col.name);
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                      <div>
                        <div style={{ fontWeight: 700, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: 4 }}>
                          {col.name}
                          {sortConfig.key === col.name && (
                            <span style={{ color: '#38bdf8', fontSize: 10 }}>{sortConfig.direction === 'asc' ? '▲' : '▼'}</span>
                          )}
                        </div>
                        {col.nullCount > 0 && (
                          <div style={{ fontSize: 9, color: '#f87171', fontWeight: 600 }}>
                            {col.nullCount} nulls ({col.nullPct}%)
                          </div>
                        )}
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
                  <td colSpan={columns.length + 1} style={{ textAlign: 'center', padding: 28, color: '#64748b' }}>
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
                        transition: 'background 0.1s ease'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.background = '#1e293b'}
                      onMouseLeave={(e) => e.currentTarget.style.background = rIdx % 2 === 0 ? '#090d16' : '#0f172a'}
                    >
                      <td style={{ padding: '6px 12px', color: '#64748b', borderRight: '1px solid #1e293b', textAlign: 'center', fontWeight: 600 }}>
                        {absoluteIndex}
                      </td>
                      {columns.map((col, cIdx) => {
                        const val = Array.isArray(row) ? row[cIdx] : row[col.name];
                        const isNull = val === null || val === undefined;
                        const displayVal = isNull ? 'NaN' : String(val);

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
        color: '#94a3b8',
        flexShrink: 0
      }}>
        <div>
          Showing <span style={{ color: '#38bdf8', fontWeight: 700 }}>{sortedData.length > 0 ? (currentPage - 1) * pageSize + 1 : 0}</span> to{' '}
          <span style={{ color: '#38bdf8', fontWeight: 700 }}>{Math.min(currentPage * pageSize, sortedData.length)}</span> of{' '}
          <span style={{ color: '#f8fafc', fontWeight: 700 }}>{sortedData.length.toLocaleString()}</span> entries
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
