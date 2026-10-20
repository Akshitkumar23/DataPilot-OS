import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Database, Play, CheckCircle2, AlertCircle, Copy, Download, Eraser,
  RefreshCw, Search, Clock, ChevronLeft, ChevronRight, Terminal, Sparkles,
  Table as TableIcon, History, ArrowUpDown, ChevronDown, Check, X, ShieldAlert
} from 'lucide-react';
import { API, getAuthHeaders } from '../apiConfig';

export default function SqlStudio({ activeDataset }) {
  // SQL Editor state
  const [sql, setSql] = useState('SELECT * FROM active_dataset LIMIT 25;');
  const [editorHistory, setEditorHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Execution & Validation state
  const [isRunning, setIsRunning] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [queryResult, setQueryResult] = useState(null);
  const [validationResult, setValidationResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [activeTab, setActiveTab] = useState('results'); // 'results' | 'history'

  // Schema state
  const [schemaInfo, setSchemaInfo] = useState(null);
  const [schemaSearch, setSchemaSearch] = useState('');
  const [isLoadingSchema, setIsLoadingSchema] = useState(false);

  // Natural Language to SQL state
  const [nlQuery, setNlQuery] = useState('');
  const [nlLoading, setNlLoading] = useState(false);
  const [nlMessage, setNlMessage] = useState(null);

  // Pagination & Sorting state
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [sortCol, setSortCol] = useState(null);
  const [sortDir, setSortDir] = useState('asc'); // 'asc' | 'desc'
  const [copied, setCopied] = useState(false);

  // History state
  const [queryHistory, setQueryHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Multi-table catalog state (FW-04)
  const [catalogTables, setCatalogTables] = useState([]);

  const textareaRef = useRef(null);

  // 1. Fetch Schema for Active Dataset
  const fetchSchema = useCallback(async () => {
    if (!activeDataset?.id) return;
    setIsLoadingSchema(true);
    try {
      const res = await fetch(`${API}/api/sql/schema?dataset_id=${encodeURIComponent(activeDataset.id)}`, {
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        setSchemaInfo(data);
      }
    } catch (err) {
      console.error('Failed to load SQL schema:', err);
    } finally {
      setIsLoadingSchema(false);
    }
  }, [activeDataset]);

  // 1b. Fetch Multi-Table Catalog Relations (FW-04)
  const fetchCatalogTables = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/sql/tables`, {
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        setCatalogTables(data.tables || []);
      }
    } catch (err) {
      console.error('Failed to load catalog tables:', err);
    }
  }, []);

  // 2. Fetch Query History
  const fetchHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const res = await fetch(`${API}/api/sql/history?limit=25`, {
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        setQueryHistory(data.history || []);
      }
    } catch (err) {
      console.error('Failed to load SQL history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  useEffect(() => {
    fetchSchema();
    fetchCatalogTables();
    fetchHistory();
  }, [fetchSchema, fetchCatalogTables, fetchHistory]);

  // 3. Execute SQL Query
  const handleExecute = async () => {
    if (!sql.trim()) return;
    setIsRunning(true);
    setErrorMessage(null);
    setValidationResult(null);

    try {
      const res = await fetch(`${API}/api/sql/execute`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          sql: sql.trim(),
          dataset_id: activeDataset?.id || 'active_dataset',
          page: 1,
          page_size: pageSize,
        }),
      });

      const data = await res.json();
      if (res.ok && data.status === 'SUCCEEDED') {
        setQueryResult(data);
        setPage(1);
        setActiveTab('results');
        fetchHistory();
      } else {
        const errMsg = data.error_message || data.detail || (data.message ? data.message : 'Execution failed.');
        setErrorMessage(errMsg);
        setQueryResult(null);
      }
    } catch (err) {
      setErrorMessage(`Network or system error: ${err.message}`);
      setQueryResult(null);
    } finally {
      setIsRunning(false);
    }
  };

  // 4. Validate SQL Query
  const handleValidate = async () => {
    if (!sql.trim()) return;
    setIsValidating(true);
    setErrorMessage(null);

    try {
      const res = await fetch(`${API}/api/sql/validate`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          sql: sql.trim(),
          dataset_id: activeDataset?.id || 'active_dataset',
        }),
      });

      const data = await res.json();
      setValidationResult(data);
      if (data.status === 'INVALID') {
        setErrorMessage(data.message || 'SQL query validation failed.');
      }
    } catch (err) {
      setErrorMessage(`Validation error: ${err.message}`);
    } finally {
      setIsValidating(false);
    }
  };

  // 5. NL to SQL Generation
  const handleNlGenerate = async () => {
    if (!nlQuery.trim()) return;
    setNlLoading(true);
    setNlMessage(null);

    try {
      // If query is a multi-table relational join request, route to /api/sql/relational-intent
      const isRelational = /\b(join|merge|combine|lookup|enrich)\b/i.test(nlQuery.trim());
      if (isRelational) {
        const relRes = await fetch(`${API}/api/sql/relational-intent`, {
          method: 'POST',
          headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ query: nlQuery.trim() })
        });
        if (!relRes.ok) {
          const errData = await relRes.json().catch(() => ({}));
          const detail = errData.detail || {};
          if (detail.error_code === 'OLLAMA_UNAVAILABLE') {
            setNlMessage({ type: 'error', text: 'Ollama semantic service unavailable. Natural-language relational requests cannot be interpreted until Ollama is available.' });
            return;
          }
          if (detail.error_code === 'JOIN_REQUIRES_CLARIFICATION' || detail.error_code === 'OLLAMA_SCHEMA_VALIDATION_ERROR') {
            setNlMessage({ type: 'warning', text: `Natural-language relational request requires clarification: ${detail.message || 'Please specify join tables and keys.'}` });
            return;
          }
          setNlMessage({ type: 'error', text: detail.message || 'Relational semantic interpretation failed.' });
          return;
        }
        const relData = await relRes.json();
        if (relData.intent?.needs_clarification) {
          setNlMessage({ type: 'warning', text: `Natural-language relational request requires clarification: ${relData.intent.clarification_reason || 'Please specify the exact join keys.'}` });
          return;
        }
      }

      // Send intent generation request to semantic planner / SQL compiler
      const res = await fetch(`${API}/api/intent/plan`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          prompt: nlQuery.trim(),
          dataset_id: activeDataset?.id || 'active_dataset',
        }),
      });

      if (res.ok) {
        const plan = await res.json();
        // Check if plan returned a sql_query command or query
        const sqlCmd = plan.commands?.find(c => c.tool === 'sql_query') || plan.commands?.[0];
        if (sqlCmd && (sqlCmd.arguments?.query || sqlCmd.arguments?.sql)) {
          const generatedSql = sqlCmd.arguments.query || sqlCmd.arguments.sql;
          setSql(generatedSql);
          setNlMessage({ type: 'success', text: 'SQL query generated from schema.' });
        } else if (plan.decision === 'clarify' || plan.clarification) {
          setNlMessage({ type: 'warning', text: plan.clarification || 'Clarification needed.' });
        } else {
          setNlMessage({ type: 'error', text: 'Could not resolve query into schema-grounded SQL. Please refine prompt.' });
        }
      } else {
        const err = await res.json().catch(() => ({}));
        setNlMessage({ type: 'error', text: err.detail || 'NL-to-SQL planning failed.' });
      }
    } catch (err) {
      setNlMessage({ type: 'error', text: `NL-to-SQL error: ${err.message}` });
    } finally {
      setNlLoading(false);
    }
  };

  // 6. Handle CSV Export
  const handleExportCsv = async () => {
    if (!sql.trim()) return;
    try {
      const res = await fetch(`${API}/api/sql/export?dataset_id=${encodeURIComponent(activeDataset?.id || 'active_dataset')}`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          sql: sql.trim(),
          dataset_id: activeDataset?.id || 'active_dataset',
          page: 1,
          page_size: 10000,
        }),
      });

      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `duckdb_query_${Date.now()}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (err) {
      alert(`Export failed: ${err.message}`);
    }
  };

  // Insert column name at cursor
  const handleInsertColumn = (colName) => {
    const formatted = `"${colName}"`;
    if (textareaRef.current) {
      const start = textareaRef.current.selectionStart;
      const end = textareaRef.current.selectionEnd;
      const nextSql = sql.substring(0, start) + formatted + sql.substring(end);
      setSql(nextSql);
      setTimeout(() => {
        textareaRef.current.focus();
        textareaRef.current.setSelectionRange(start + formatted.length, start + formatted.length);
      }, 0);
    } else {
      setSql(prev => `${prev} ${formatted}`);
    }
  };

  // Keyboard shortcut Ctrl+Enter to Run
  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleExecute();
    }
  };

  // Copy SQL
  const handleCopySql = () => {
    navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Client-side table sorting
  const sortedRows = React.useMemo(() => {
    if (!queryResult?.rows || !sortCol) return queryResult?.rows || [];
    const copy = [...queryResult.rows];
    copy.sort((a, b) => {
      const va = a[sortCol];
      const vb = b[sortCol];
      if (va === vb) return 0;
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      if (typeof va === 'number' && typeof vb === 'number') {
        return sortDir === 'asc' ? va - vb : vb - va;
      }
      return sortDir === 'asc' ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });
    return copy;
  }, [queryResult, sortCol, sortDir]);

  // Client-side pagination slice
  const paginatedRows = React.useMemo(() => {
    const start = (page - 1) * pageSize;
    return sortedRows.slice(start, start + pageSize);
  }, [sortedRows, page, pageSize]);

  const totalPages = Math.ceil((sortedRows.length || 0) / pageSize) || 1;

  // Filtered schema columns
  const filteredColumns = React.useMemo(() => {
    if (!schemaInfo?.columns) return [];
    if (!schemaSearch.trim()) return schemaInfo.columns;
    const term = schemaSearch.toLowerCase();
    return schemaInfo.columns.filter(c => c.name.toLowerCase().includes(term) || c.type.toLowerCase().includes(term));
  }, [schemaInfo, schemaSearch]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'row',
      height: '100%',
      width: '100%',
      background: '#090d16',
      color: '#f8fafc',
      fontFamily: 'Inter, system-ui, sans-serif',
      overflow: 'hidden'
    }}>
      {/* 1. LEFT SIDEBAR: SCHEMA BROWSER */}
      <div style={{
        width: 300,
        background: '#0d1322',
        borderRight: '1px solid #1e293b',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0
      }}>
        {/* Dataset Header Card */}
        <div style={{ padding: '14px 16px', borderBottom: '1px solid #1e293b', background: '#0a0f1c' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ background: '#3b82f6', padding: 6, borderRadius: 6, display: 'flex' }}>
                <Database size={16} color="#ffffff" />
              </div>
              <span style={{ fontSize: 13, fontWeight: 800, color: '#f8fafc', letterSpacing: '0.02em' }}>
                DUCKDB STUDIO
              </span>
            </div>
            <button
              onClick={fetchSchema}
              title="Refresh Schema"
              style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', padding: 4 }}
            >
              <RefreshCw size={14} className={isLoadingSchema ? 'animate-spin' : ''} />
            </button>
          </div>

          <div style={{ fontSize: 12, fontWeight: 700, color: '#e2e8f0', wordBreak: 'break-all' }}>
            {activeDataset?.name || 'No Active Dataset'}
          </div>
          <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: 11, color: '#94a3b8' }}>
            <span>Relation: <code style={{ color: '#38bdf8' }}>active_dataset</code></span>
            <span>•</span>
            <span>{schemaInfo?.row_count ?? activeDataset?.row_count ?? 0} rows</span>
            <span>•</span>
            <span>{schemaInfo?.columns?.length ?? activeDataset?.column_count ?? 0} cols</span>
          </div>

          {/* Catalog Tables (FW-04 Multi-Table Display) */}
          {catalogTables.length > 1 && (
            <div style={{ marginTop: 10, paddingTop: 8, borderTop: '1px dashed #1e293b' }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', marginBottom: 4 }}>
                Registered Relations ({catalogTables.length})
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {catalogTables.map((t, idx) => (
                  <span
                    key={idx}
                    onClick={() => handleInsertColumn(t.table_name)}
                    title={`Click to insert table name "${t.table_name}" into SQL editor`}
                    style={{
                      fontSize: 10,
                      fontFamily: 'monospace',
                      background: '#1e293b',
                      color: '#e2e8f0',
                      padding: '2px 6px',
                      borderRadius: 4,
                      cursor: 'pointer',
                      border: '1px solid #334155'
                    }}
                  >
                    {t.table_name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Schema Search */}
        <div style={{ padding: '10px 14px', borderBottom: '1px solid #1e293b' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            background: '#151d30',
            borderRadius: 6,
            padding: '6px 10px',
            border: '1px solid #222f49'
          }}>
            <Search size={14} color="#64748b" style={{ marginRight: 8 }} />
            <input
              type="text"
              placeholder="Search columns..."
              value={schemaSearch}
              onChange={(e) => setSchemaSearch(e.target.value)}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#f8fafc',
                fontSize: 12,
                outline: 'none',
                width: '100%'
              }}
            />
            {schemaSearch && (
              <X size={12} color="#94a3b8" style={{ cursor: 'pointer' }} onClick={() => setSchemaSearch('')} />
            )}
          </div>
        </div>

        {/* Column List */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', padding: '6px 8px' }}>
            Schema Columns ({filteredColumns.length})
          </div>
          {filteredColumns.length === 0 ? (
            <div style={{ padding: 16, textAlign: 'center', fontSize: 12, color: '#64748b' }}>
              No columns match '{schemaSearch}'
            </div>
          ) : (
            filteredColumns.map((col, idx) => (
              <div
                key={idx}
                onClick={() => handleInsertColumn(col.name)}
                title={`Click to insert "${col.name}" into editor`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '6px 10px',
                  borderRadius: 6,
                  cursor: 'pointer',
                  fontSize: 12,
                  color: '#e2e8f0',
                  transition: 'background 0.15s ease',
                  marginBottom: 2
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = '#1e293b'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
              >
                <span style={{ fontFamily: 'monospace', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {col.name}
                </span>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <span style={{
                    fontSize: 9,
                    fontWeight: 700,
                    padding: '2px 5px',
                    borderRadius: 4,
                    background: '#1e293b',
                    color: '#38bdf8',
                    fontFamily: 'monospace'
                  }}>
                    {col.type}
                  </span>
                  {col.nullable && (
                    <span style={{ fontSize: 8, color: '#94a3b8' }}>NULL</span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Quick SQL Templates */}
        <div style={{ padding: '12px 14px', borderTop: '1px solid #1e293b', background: '#0a0f1c' }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>
            Quick Queries
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <button
              onClick={() => setSql('SELECT * FROM active_dataset LIMIT 25;')}
              style={{
                background: '#151d30', border: '1px solid #222f49', color: '#94a3b8',
                borderRadius: 4, padding: '4px 8px', fontSize: 11, textAlign: 'left', cursor: 'pointer'
              }}
            >
              SELECT * LIMIT 25
            </button>
            <button
              onClick={() => setSql('SELECT COUNT(*) AS total_rows FROM active_dataset;')}
              style={{
                background: '#151d30', border: '1px solid #222f49', color: '#94a3b8',
                borderRadius: 4, padding: '4px 8px', fontSize: 11, textAlign: 'left', cursor: 'pointer'
              }}
            >
              COUNT(*) Summary
            </button>
          </div>
        </div>
      </div>

      {/* 2. MAIN AREA: NL BAR, SQL EDITOR & RESULTS GRID */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        
        {/* Natural Language to SQL Assistant Bar */}
        <div style={{
          padding: '10px 18px',
          background: '#0d1322',
          borderBottom: '1px solid #1e293b',
          display: 'flex',
          alignItems: 'center',
          gap: 12
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#a855f7' }}>
            <Sparkles size={16} />
            <span style={{ fontSize: 12, fontWeight: 700, whiteSpace: 'nowrap' }}>AI SQL Assistant:</span>
          </div>
          <input
            type="text"
            placeholder="Ask a question in plain English (e.g. 'Show average values by category', 'Top 10 highest records')..."
            value={nlQuery}
            onChange={(e) => setNlQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleNlGenerate()}
            style={{
              flex: 1,
              background: '#151d30',
              border: '1px solid #222f49',
              borderRadius: 6,
              padding: '7px 12px',
              color: '#f8fafc',
              fontSize: 12,
              outline: 'none'
            }}
          />
          <button
            onClick={handleNlGenerate}
            disabled={nlLoading || !nlQuery.trim()}
            style={{
              background: '#7c3aed',
              color: '#ffffff',
              border: 'none',
              borderRadius: 6,
              padding: '7px 14px',
              fontSize: 12,
              fontWeight: 700,
              cursor: nlLoading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6
            }}
          >
            {nlLoading ? <RefreshCw size={14} className="animate-spin" /> : <Sparkles size={14} />}
            Generate SQL
          </button>
        </div>

        {/* NL Message banner */}
        {nlMessage && (
          <div style={{
            padding: '6px 18px',
            fontSize: 11,
            fontWeight: 600,
            background: nlMessage.type === 'error' ? '#450a0a' : nlMessage.type === 'warning' ? '#422006' : '#052e16',
            color: nlMessage.type === 'error' ? '#fca5a5' : nlMessage.type === 'warning' ? '#fde047' : '#86efac',
            borderBottom: '1px solid #1e293b',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <span>{nlMessage.text}</span>
            <X size={12} style={{ cursor: 'pointer' }} onClick={() => setNlMessage(null)} />
          </div>
        )}

        {/* SQL Editor Area */}
        <div style={{
          height: 190,
          background: '#090d16',
          borderBottom: '1px solid #1e293b',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative'
        }}>
          {/* Action Toolbar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '8px 16px',
            background: '#0f172a',
            borderBottom: '1px solid #1e293b'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button
                onClick={handleExecute}
                disabled={isRunning}
                style={{
                  background: '#10b981',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '6px 14px',
                  fontSize: 12,
                  fontWeight: 700,
                  cursor: isRunning ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6
                }}
              >
                {isRunning ? <RefreshCw size={14} className="animate-spin" /> : <Play size={14} />}
                Run (Ctrl+Enter)
              </button>

              <button
                onClick={handleValidate}
                disabled={isValidating}
                style={{
                  background: '#1e293b',
                  color: '#94a3b8',
                  border: '1px solid #334155',
                  borderRadius: 6,
                  padding: '6px 12px',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6
                }}
              >
                <CheckCircle2 size={14} /> Validate
              </button>

              <button
                onClick={() => setSql('')}
                style={{
                  background: 'transparent',
                  color: '#64748b',
                  border: 'none',
                  borderRadius: 6,
                  padding: '6px 10px',
                  fontSize: 12,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4
                }}
              >
                <Eraser size={14} /> Clear
              </button>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {queryResult?.execution_time_ms !== undefined && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#94a3b8' }}>
                  <Clock size={12} />
                  <span>{queryResult.execution_time_ms.toFixed(2)} ms</span>
                </div>
              )}
              <button
                onClick={handleCopySql}
                title="Copy SQL"
                style={{
                  background: '#1e293b',
                  color: copied ? '#10b981' : '#94a3b8',
                  border: '1px solid #334155',
                  borderRadius: 6,
                  padding: '6px 10px',
                  fontSize: 12,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4
                }}
              >
                {copied ? <Check size={14} /> : <Copy size={14} />} {copied ? 'Copied' : 'Copy'}
              </button>
              <button
                onClick={handleExportCsv}
                title="Export Result CSV"
                style={{
                  background: '#1e293b',
                  color: '#94a3b8',
                  border: '1px solid #334155',
                  borderRadius: 6,
                  padding: '6px 10px',
                  fontSize: 12,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4
                }}
              >
                <Download size={14} /> Export CSV
              </button>
            </div>
          </div>

          {/* Multiline Editor Textarea */}
          <div style={{ flex: 1, position: 'relative', display: 'flex' }}>
            <textarea
              ref={textareaRef}
              value={sql}
              onChange={(e) => setSql(e.target.value)}
              onKeyDown={handleKeyDown}
              spellCheck={false}
              placeholder="Write DuckDB SQL against 'active_dataset' (e.g. SELECT * FROM active_dataset LIMIT 20)..."
              style={{
                flex: 1,
                width: '100%',
                height: '100%',
                background: '#090d16',
                color: '#38bdf8',
                fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                fontSize: 13,
                lineHeight: 1.5,
                padding: '12px 16px',
                border: 'none',
                outline: 'none',
                resize: 'none'
              }}
            />
          </div>
        </div>

        {/* Structured Validation / Error Card */}
        {errorMessage && (
          <div style={{
            padding: '10px 16px',
            background: '#450a0a',
            borderBottom: '1px solid #7f1d1d',
            color: '#fecaca',
            fontSize: 12,
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10
          }}>
            <ShieldAlert size={18} color="#ef4444" style={{ flexShrink: 0, marginTop: 1 }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, marginBottom: 2 }}>Execution / Validation Interdict:</div>
              <div style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', lineHeight: 1.4 }}>
                {errorMessage}
              </div>
            </div>
            <X size={14} style={{ cursor: 'pointer' }} onClick={() => setErrorMessage(null)} />
          </div>
        )}

        {validationResult && validationResult.status === 'VALID' && (
          <div style={{
            padding: '8px 16px',
            background: '#052e16',
            borderBottom: '1px solid #14532d',
            color: '#86efac',
            fontSize: 11,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <CheckCircle2 size={14} color="#22c55e" />
              <span>Query is valid. Estimated operation: <strong>{validationResult.estimated_operation}</strong></span>
              {validationResult.referenced_columns?.length > 0 && (
                <span>• Columns: {validationResult.referenced_columns.join(', ')}</span>
              )}
            </div>
            <X size={12} style={{ cursor: 'pointer' }} onClick={() => setValidationResult(null)} />
          </div>
        )}

        {/* 3. BOTTOM AREA: RESULTS GRID OR QUERY HISTORY */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Tabs bar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: '#0f172a',
            borderBottom: '1px solid #1e293b',
            padding: '0 16px'
          }}>
            <div style={{ display: 'flex', gap: 4 }}>
              <button
                onClick={() => setActiveTab('results')}
                style={{
                  background: activeTab === 'results' ? '#1e293b' : 'transparent',
                  color: activeTab === 'results' ? '#f8fafc' : '#94a3b8',
                  border: 'none',
                  borderBottom: activeTab === 'results' ? '2px solid #3b82f6' : '2px solid transparent',
                  padding: '10px 14px',
                  fontSize: 12,
                  fontWeight: 700,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6
                }}
              >
                <TableIcon size={14} /> Results Grid
                {queryResult && (
                  <span style={{ background: '#3b82f6', color: '#fff', fontSize: 10, padding: '1px 6px', borderRadius: 10 }}>
                    {queryResult.row_count}
                  </span>
                )}
              </button>

              <button
                onClick={() => { setActiveTab('history'); fetchHistory(); }}
                style={{
                  background: activeTab === 'history' ? '#1e293b' : 'transparent',
                  color: activeTab === 'history' ? '#f8fafc' : '#94a3b8',
                  border: 'none',
                  borderBottom: activeTab === 'history' ? '2px solid #3b82f6' : '2px solid transparent',
                  padding: '10px 14px',
                  fontSize: 12,
                  fontWeight: 700,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6
                }}
              >
                <History size={14} /> Query History ({queryHistory.length})
              </button>
            </div>

            {/* Pagination Controls */}
            {activeTab === 'results' && queryResult && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#94a3b8' }}>
                <span>
                  Showing {Math.min((page - 1) * pageSize + 1, queryResult.row_count || 0)} - {Math.min(page * pageSize, queryResult.row_count || 0)} of {queryResult.row_count} rows
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 8 }}>
                  <button
                    disabled={page <= 1}
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    style={{
                      background: '#1e293b', border: '1px solid #334155', color: '#f8fafc',
                      borderRadius: 4, padding: '3px 8px', cursor: page <= 1 ? 'not-allowed' : 'pointer'
                    }}
                  >
                    <ChevronLeft size={14} />
                  </button>
                  <span style={{ fontWeight: 600 }}>{page} / {totalPages}</span>
                  <button
                    disabled={page >= totalPages}
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    style={{
                      background: '#1e293b', border: '1px solid #334155', color: '#f8fafc',
                      borderRadius: 4, padding: '3px 8px', cursor: page >= totalPages ? 'not-allowed' : 'pointer'
                    }}
                  >
                    <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Results Grid Content */}
          {activeTab === 'results' && (
            <div style={{ flex: 1, overflow: 'auto', background: '#090d16' }}>
              {queryResult?.columns && queryResult.columns.length > 0 ? (
                <table style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: 12,
                  textAlign: 'left'
                }}>
                  <thead>
                    <tr style={{ background: '#0d1322', position: 'sticky', top: 0, zIndex: 1, borderBottom: '1px solid #1e293b' }}>
                      <th style={{ padding: '8px 12px', width: 45, color: '#64748b', fontWeight: 600, borderRight: '1px solid #1e293b' }}>
                        #
                      </th>
                      {queryResult.columns.map((col, idx) => (
                        <th
                          key={idx}
                          onClick={() => {
                            if (sortCol === col) {
                              setSortDir(d => d === 'asc' ? 'desc' : 'asc');
                            } else {
                              setSortCol(col);
                              setSortDir('asc');
                            }
                          }}
                          style={{
                            padding: '8px 12px',
                            color: '#e2e8f0',
                            fontWeight: 700,
                            fontFamily: 'monospace',
                            cursor: 'pointer',
                            userSelect: 'none',
                            borderRight: '1px solid #1e293b',
                            whiteSpace: 'nowrap'
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                            <span>{col}</span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                              <span style={{ fontSize: 9, color: '#38bdf8', fontWeight: 600 }}>
                                {queryResult.column_types?.[col] || ''}
                              </span>
                              <ArrowUpDown size={12} color={sortCol === col ? '#3b82f6' : '#64748b'} />
                            </div>
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedRows.length === 0 ? (
                      <tr>
                        <td colSpan={queryResult.columns.length + 1} style={{ padding: 24, textAlign: 'center', color: '#64748b' }}>
                          Query returned zero rows.
                        </td>
                      </tr>
                    ) : (
                      paginatedRows.map((row, rIdx) => (
                        <tr
                          key={rIdx}
                          style={{
                            borderBottom: '1px solid #1e293b',
                            background: rIdx % 2 === 0 ? '#090d16' : '#0d1322',
                            transition: 'background 0.1s ease'
                          }}
                          onMouseEnter={(e) => e.currentTarget.style.background = '#1e293b'}
                          onMouseLeave={(e) => e.currentTarget.style.background = rIdx % 2 === 0 ? '#090d16' : '#0d1322'}
                        >
                          <td style={{ padding: '6px 12px', color: '#64748b', fontSize: 11, borderRight: '1px solid #1e293b' }}>
                            {(page - 1) * pageSize + rIdx + 1}
                          </td>
                          {queryResult.columns.map((col, cIdx) => {
                            const val = row[col];
                            const isNull = val === null || val === undefined;
                            return (
                              <td
                                key={cIdx}
                                style={{
                                  padding: '6px 12px',
                                  color: isNull ? '#64748b' : '#f8fafc',
                                  fontFamily: typeof val === 'number' ? 'monospace' : 'inherit',
                                  borderRight: '1px solid #1e293b',
                                  whiteSpace: 'nowrap'
                                }}
                              >
                                {isNull ? <em style={{ color: '#64748b' }}>NULL</em> : String(val)}
                              </td>
                            );
                          })}
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              ) : (
                <div style={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#64748b',
                  gap: 12,
                  padding: 32
                }}>
                  <Terminal size={32} color="#334155" />
                  <div style={{ fontSize: 13, fontWeight: 600 }}>No Query Executed Yet</div>
                  <div style={{ fontSize: 11, maxWidth: 360, textAlign: 'center' }}>
                    Type a DuckDB SQL statement above and click <strong>Run</strong> or use the AI Assistant to generate queries.
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Query History Content */}
          {activeTab === 'history' && (
            <div style={{ flex: 1, overflow: 'auto', background: '#090d16', padding: 12 }}>
              {queryHistory.length === 0 ? (
                <div style={{ padding: 24, textAlign: 'center', color: '#64748b', fontSize: 12 }}>
                  No historical queries recorded yet.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {queryHistory.map((item, idx) => (
                    <div
                      key={idx}
                      onClick={() => setSql(item.sql)}
                      title="Click to load SQL into editor"
                      style={{
                        padding: '10px 14px',
                        background: '#0d1322',
                        border: '1px solid #1e293b',
                        borderRadius: 6,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        transition: 'background 0.15s ease'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.background = '#151d30'}
                      onMouseLeave={(e) => e.currentTarget.style.background = '#0d1322'}
                    >
                      <div style={{ flex: 1, marginRight: 16 }}>
                        <div style={{ fontFamily: 'monospace', fontSize: 12, color: '#38bdf8', marginBottom: 4 }}>
                          {item.sql}
                        </div>
                        <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#94a3b8' }}>
                          <span>{item.timestamp}</span>
                          <span>•</span>
                          <span>{item.row_count ?? 0} rows</span>
                          <span>•</span>
                          <span>{item.execution_time_ms ? `${item.execution_time_ms.toFixed(1)}ms` : ''}</span>
                        </div>
                      </div>
                      <div style={{
                        padding: '3px 8px',
                        borderRadius: 4,
                        fontSize: 10,
                        fontWeight: 700,
                        background: item.status === 'SUCCEEDED' ? '#052e16' : '#450a0a',
                        color: item.status === 'SUCCEEDED' ? '#4ade80' : '#f87171'
                      }}>
                        {item.status}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
