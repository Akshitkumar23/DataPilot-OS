import React, { useRef, useState } from 'react';
import { Database, Plus, X, Link, RefreshCw, FileSpreadsheet, ChevronDown, ChevronRight, CheckCircle, FileText, Table2 } from 'lucide-react';

export default function DatasetManager({
  datasets = [],
  activeDatasetId,
  onConnect,
  onDisconnect,
  onSetActive,
  onRefresh,
  engineStatus
}) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [expandedDatasetIds, setExpandedDatasetIds] = useState({});
  const isEngineReady = engineStatus === 'connected';

  const toggleExpand = (id, e) => {
    e.stopPropagation();
    setExpandedDatasetIds(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const handleConnectClick = () => {
    if (!isEngineReady) return;
    if (window.electron?.openFile) {
      window.electron.openFile().then(filePaths => {
        if (filePaths) {
          const target = Array.isArray(filePaths) ? filePaths[0] : filePaths;
          if (target) onConnect(target);
        }
      }).catch(err => console.error('Failed to open file via Electron dialog:', err));
    } else {
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
        fileInputRef.current.click();
      }
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && isEngineReady) {
      onConnect(file);
    }
    e.target.value = '';
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer?.files?.[0];
    if (file && isEngineReady) {
      onConnect(file);
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#0b0f19',
        color: '#f8fafc',
        height: '100%',
        boxSizing: 'border-box',
        position: 'relative'
      }}
    >
      {/* Header */}
      <div style={{
        padding: '12px 14px',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: '#0f172a',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={15} color="#10b981" />
          <h2 style={{ fontSize: '12px', fontWeight: 800, margin: 0, letterSpacing: '0.8px', color: '#f1f5f9' }}>
            DATASETS
          </h2>
          <span style={{
            fontSize: '10px',
            background: '#1e293b',
            color: '#94a3b8',
            padding: '1px 6px',
            borderRadius: '10px',
            fontWeight: 700
          }}>
            {datasets.length}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          {onRefresh && (
            <button
              onClick={onRefresh}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#94a3b8',
                cursor: 'pointer',
                padding: '4px',
                borderRadius: '4px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              title="Refresh Datasets"
            >
              <RefreshCw size={13} />
            </button>
          )}

          <button
            onClick={handleConnectClick}
            disabled={!isEngineReady}
            style={{
              background: isEngineReady ? '#10b981' : '#334155',
              border: 'none',
              color: isEngineReady ? '#fff' : '#94a3b8',
              borderRadius: '4px',
              padding: '3px 8px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              cursor: isEngineReady ? 'pointer' : 'not-allowed',
              fontSize: '11px',
              fontWeight: 700
            }}
            title={isEngineReady ? 'Connect CSV / Excel / Parquet' : 'Local service is still starting'}
          >
            <Plus size={12} /> Connect
          </button>
        </div>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        accept=".csv,.xlsx,.xls,.json,.parquet,.sqlite,.db"
        onChange={handleFileChange}
      />

      {/* Drag Over Overlay */}
      {isDragging && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(16, 185, 129, 0.15)',
          border: '2px dashed #10b981',
          zIndex: 10,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          pointerEvents: 'none',
          backdropFilter: 'blur(2px)'
        }}>
          <FileSpreadsheet size={32} color="#10b981" />
          <span style={{ fontSize: '13px', fontWeight: 700, color: '#10b981', marginTop: '8px' }}>
            Drop CSV or Dataset to Load
          </span>
        </div>
      )}

      {/* Dataset List Container */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px'
      }}>
        {datasets.length === 0 ? (
          <div style={{
            textAlign: 'center',
            color: '#94a3b8',
            marginTop: '28px',
            padding: '0 12px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '8px'
          }}>
            <FileSpreadsheet size={36} style={{ color: '#475569', margin: '0 auto 4px' }} />
            <p style={{ margin: 0, fontSize: '13px', fontWeight: 600, color: '#cbd5e1' }}>No datasets connected</p>
            <p style={{ margin: 0, fontSize: '11px', color: '#64748b', lineHeight: 1.4 }}>
              Drag & drop a CSV file here, or connect a data source.
            </p>
          </div>
        ) : (
          datasets.map((dataset, idx) => {
            const id = dataset.id || dataset.dataset_id || `ds_${idx}`;
            const isActive = id === activeDatasetId || dataset.is_active;
            const name = dataset.name || 'Dataset';
            const rowCount = dataset.rowCount ?? dataset.row_count ?? dataset.rows ?? dataset.total_rows ?? 0;
            const columns = Array.isArray(dataset.columns)
              ? dataset.columns
              : (Array.isArray(dataset.schema) ? dataset.schema : (dataset.columns ? Object.keys(dataset.columns) : []));
            const colCount = dataset.colCount ?? dataset.column_count ?? columns.length ?? 0;
            const isExpanded = !!expandedDatasetIds[id];
            const format = dataset.format || dataset.source_type || 'CSV';

            return (
              <div
                key={id}
                onClick={() => onSetActive && onSetActive(id)}
                style={{
                  background: isActive ? '#141d2f' : '#0f172a',
                  border: `1px solid ${isActive ? '#10b981' : '#1e293b'}`,
                  borderRadius: '6px',
                  padding: '10px 12px',
                  cursor: 'pointer',
                  position: 'relative',
                  transition: 'all 0.15s ease',
                  boxShadow: isActive ? '0 0 12px rgba(16, 185, 129, 0.12)' : 'none'
                }}
              >
                {/* Dataset Item Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 1, minWidth: 0 }}>
                    {isActive ? (
                      <div
                        style={{
                          width: '7px',
                          height: '7px',
                          borderRadius: '50%',
                          backgroundColor: '#10b981',
                          boxShadow: '0 0 8px #10b981',
                          flexShrink: 0
                        }}
                        title="Active Dataset"
                      />
                    ) : (
                      <div
                        style={{
                          width: '7px',
                          height: '7px',
                          borderRadius: '50%',
                          backgroundColor: '#475569',
                          flexShrink: 0
                        }}
                      />
                    )}

                    <span
                      style={{
                        margin: 0,
                        fontSize: '13px',
                        fontWeight: isActive ? 700 : 600,
                        color: isActive ? '#f8fafc' : '#cbd5e1',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        flex: 1
                      }}
                      title={name}
                    >
                      {name}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '2px', flexShrink: 0 }}>
                    <button
                      onClick={(e) => toggleExpand(id, e)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#64748b',
                        cursor: 'pointer',
                        padding: '2px 4px',
                        display: 'flex',
                        alignItems: 'center'
                      }}
                      title={isExpanded ? "Collapse schema" : "View columns schema"}
                    >
                      {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                    </button>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (onDisconnect) onDisconnect(id);
                      }}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#64748b',
                        cursor: 'pointer',
                        padding: '2px 4px',
                        display: 'flex',
                        alignItems: 'center'
                      }}
                      title="Disconnect dataset"
                    >
                      <X size={13} />
                    </button>
                  </div>
                </div>

                {/* Metadata row */}
                <div style={{
                  marginTop: '6px',
                  fontSize: '11px',
                  color: '#94a3b8',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  flexWrap: 'wrap'
                }}>
                  <span style={{ fontWeight: 600, color: '#cbd5e1' }}>
                    {rowCount.toLocaleString()} rows &times; {colCount} cols
                  </span>

                  <span style={{
                    textTransform: 'uppercase',
                    fontSize: '9px',
                    fontWeight: 700,
                    background: '#1e293b',
                    color: '#38bdf8',
                    padding: '1px 5px',
                    borderRadius: '3px'
                  }}>
                    {format}
                  </span>

                  {dataset.domain && (
                    <span style={{
                      fontSize: '9px',
                      background: '#334155',
                      color: '#e2e8f0',
                      padding: '1px 5px',
                      borderRadius: '3px'
                    }}>
                      {dataset.domain}
                    </span>
                  )}
                </div>

                {/* Expanded Column Schema Preview */}
                {isExpanded && columns.length > 0 && (
                  <div
                    onClick={(e) => e.stopPropagation()}
                    style={{
                      marginTop: '8px',
                      paddingTop: '8px',
                      borderTop: '1px solid #1e293b',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px',
                      maxHeight: '160px',
                      overflowY: 'auto'
                    }}
                  >
                    <div style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      Schema ({columns.length} columns)
                    </div>
                    {columns.map((col, cIdx) => {
                      const colName = typeof col === 'string' ? col : (col.name || `col_${cIdx}`);
                      const colType = typeof col === 'object' ? (col.physical_type || col.semantic_type || col.type || col.dtype || '') : '';
                      return (
                        <div
                          key={colName + cIdx}
                          style={{
                            fontSize: '11px',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            color: '#94a3b8',
                            padding: '2px 4px',
                            borderRadius: '3px',
                            background: 'rgba(255, 255, 255, 0.02)'
                          }}
                        >
                          <span style={{ display: 'flex', alignItems: 'center', gap: '4px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            <Table2 size={10} color="#64748b" />
                            <span style={{ color: '#e2e8f0' }}>{colName}</span>
                          </span>
                          {colType && (
                            <span style={{ fontSize: '9px', color: '#64748b', fontFamily: 'monospace' }}>
                              {colType}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Footer relationships indicator */}
      <div style={{
        padding: '9px 12px',
        borderTop: '1px solid #1e293b',
        fontSize: '11px',
        color: '#64748b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: '#090d16',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Link size={12} color="#64748b" />
          <span>{datasets.length > 1 ? `${datasets.length - 1} relationship candidates` : 'No relationships'}</span>
        </div>
        {engineStatus && (
          <span style={{
            fontSize: '9px',
            textTransform: 'uppercase',
            fontWeight: 700,
            color: engineStatus === 'connected' ? '#10b981' : '#f59e0b'
          }}>
            {engineStatus}
          </span>
        )}
      </div>
    </div>
  );
}
