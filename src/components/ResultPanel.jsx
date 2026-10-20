import React, { useState, useMemo } from 'react';
import { 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  BarChart2, 
  Table as TableIcon, 
  FileText, 
  CheckSquare, 
  RefreshCw, 
  Maximize2, 
  Download, 
  Terminal, 
  ZoomIn, 
  Sparkles,
  Layers,
  Activity,
  Copy,
  Check,
  Search,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  HelpCircle,
  Wrench,
  BookOpen,
  ArrowDownUp,
  Flame,
  FileSpreadsheet,
  Clock,
  ArrowUp,
  ArrowDown,
  Columns,
  Split,
  Code,
  Database,
  Plus,
  TrendingUp,
  Eye,
  Sliders,
  X
} from 'lucide-react';
import { copyImageToClipboard } from '../utils/imageClipboard';
import InteractiveChart from './InteractiveChart';
import { API, getAuthHeaders } from '../apiConfig';

// Helper to normalize image sources (handling raw base64 without data URI scheme)
const normalizeImageSrc = (src) => {
  if (!src || typeof src !== 'string') return null;
  if (src.startsWith('data:image') || src.startsWith('http://') || src.startsWith('https://') || src.startsWith('blob:')) {
    return src;
  }
  return `data:image/png;base64,${src}`;
};

// Helper to render basic markdown text (bold, inline code, bullets)
const renderMarkdown = (text) => {
  if (!text) return null;
  const lines = text.split('\n');
  return lines.map((line, lIdx) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return <div key={lIdx} style={{ height: 8 }} />;
    }
    const isBullet = trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ');
    const isNumbered = /^\d+\.\s/.test(trimmed);
    const content = isBullet ? trimmed.slice(2) : isNumbered ? trimmed.replace(/^\d+\.\s/, '') : line;

    const parts = content.split(/(\*\*.*?\*\*|`.*?`)/g);
    const renderedParts = parts.map((part, pIdx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={pIdx} style={{ color: '#38bdf8', fontWeight: 700 }}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={pIdx}
            style={{
              background: 'rgba(56, 189, 248, 0.12)',
              color: '#7dd3fc',
              padding: '1px 5px',
              borderRadius: 4,
              fontSize: '0.9em',
              fontFamily: 'var(--font-mono, monospace)'
            }}
          >
            {part.slice(1, -1)}
          </code>
        );
      }
      return part;
    });

    if (isBullet || isNumbered) {
      return (
        <div key={lIdx} style={{ display: 'flex', gap: 8, marginLeft: 8, marginBottom: 4, lineHeight: 1.5 }}>
          <span style={{ color: '#38bdf8', fontWeight: 700 }}>{isBullet ? '•' : trimmed.match(/^\d+\./)[0]}</span>
          <div style={{ color: '#cbd5e1' }}>{renderedParts}</div>
        </div>
      );
    }

    return (
      <div key={lIdx} style={{ marginBottom: 4, lineHeight: 1.5, color: '#cbd5e1' }}>
        {renderedParts}
      </div>
    );
  });
};

// Canonical Diagnostic Hint Map for DataPilot Error States
const ERROR_DIAGNOSTICS = {
  INTENT_PLAN_INVALID: {
    title: "Invalid or Ambiguous Analysis Plan",
    stage: "Intent Planning Engine",
    hint: "The generated workflow plan could not be mapped to existing dataset features or valid statistical methods.",
    action: "Specify exact column names in your query or verify dataset schema in the Data Grid tab.",
    icon: AlertTriangle
  },
  AMBIGUOUS_INTENT: {
    title: "Ambiguous Query Intent",
    stage: "Query Disambiguation",
    hint: "The query could refer to multiple columns or competing analytical operations.",
    action: "Clarify your requested target variable, date range, or aggregation function.",
    icon: HelpCircle
  },
  INTENT_AMBIGUOUS: {
    title: "Ambiguous Query Intent",
    stage: "Query Disambiguation",
    hint: "Multiple plausible candidate features match the user query.",
    action: "Select from the suggested candidate columns in the Intent Panel.",
    icon: HelpCircle
  },
  OLLAMA_UNAVAILABLE: {
    title: "Local Ollama LLM Service Offline",
    stage: "Neural Planner / LLM Service",
    hint: "DataPilot could not establish a connection with the local Ollama daemon at http://localhost:11434.",
    action: "Start Ollama by running 'ollama serve' in your terminal and verify 'ollama pull qwen2.5:3b-instruct'.",
    icon: AlertTriangle
  },
  INTENT_PLANNER_UNAVAILABLE: {
    title: "Planning Engine Unreachable",
    stage: "Intent Planning Engine",
    hint: "The planning microservice or configured local model is offline.",
    action: "Ensure your Ollama daemon is active and configured model is pulled.",
    icon: AlertTriangle
  },
  SCHEMA_MISMATCH: {
    title: "Dataset Schema Incompatibility",
    stage: "Code Validation / Schema Check",
    hint: "The Python script attempted to access dataframe columns that do not exist in the active dataset.",
    action: "Check the Data Grid preview tab for the exact column headers and data types.",
    icon: Wrench
  },
  CAPABILITY_INCOMPATIBLE: {
    title: "Incompatible Analysis Capability",
    stage: "Capability Matcher",
    hint: "The requested data science operation requires features (e.g. numeric columns, datetime index) not present in this dataset.",
    action: "Convert data types or connect a dataset containing the requisite features.",
    icon: Wrench
  },
  CODE_UNSAFE: {
    title: "Security Policy Violation (AST Check)",
    stage: "Sandbox AST Security Gate",
    hint: "Generated Python code contained blocked operations (system calls, network sockets, or unauthorized file IO).",
    action: "DataPilot sandbox only permits in-memory pandas/numpy/scipy/scikit-learn/matplotlib/seaborn execution.",
    icon: XCircle
  },
  EXECUTION_TIMEOUT: {
    title: "Python Sandbox Execution Timeout",
    stage: "Runtime Execution Sandbox",
    hint: "Execution exceeded the safety limit (30 seconds). The code may have entered an infinite loop or excessive computation.",
    action: "Sample the dataset to fewer rows or optimize the algorithm complexity.",
    icon: Clock
  },
  EXECUTION_FAILED: {
    title: "Python Sandbox Runtime Exception",
    stage: "Python Runtime Sandbox",
    hint: "An unhandled exception occurred during Python execution. Inspect the Traceback / Stderr tab below for line numbers.",
    action: "Review code in the Code Panel or ask AI Copilot to fix the syntax/logic error.",
    icon: XCircle
  }
};

export default function ResultPanel({ 
  executionRecord, 
  artifacts, 
  validationReport, 
  onRequestInsights, 
  onZoomImage 
}) {
  // 1. Extract all plots from artifacts and executionRecord safely with normalization
  const plotsList = useMemo(() => {
    const plots = [];

    // Plots from artifacts
    if (Array.isArray(artifacts)) {
      artifacts.forEach((a, idx) => {
        const rawImg = a.data || a.plot_image || a.image || a.url;
        if (a.type === 'plot' || a.plot_image || a.image || (typeof a.data === 'string' && (a.data.startsWith('data:image') || a.data.length > 200))) {
          const normalized = normalizeImageSrc(rawImg);
          if (normalized && !plots.some(p => p.data === normalized)) {
            plots.push({
              id: a.id || `artifact_plot_${idx}`,
              data: normalized,
              title: a.title || a.name || `Visualization Plot ${plots.length + 1}`
            });
          }
        }
      });
    }

    // Single plot image from executionRecord
    if (executionRecord?.plot_image) {
      const normalized = normalizeImageSrc(executionRecord.plot_image);
      if (normalized && !plots.some(p => p.data === normalized)) {
        plots.push({
          id: 'exec_plot_primary',
          data: normalized,
          title: executionRecord.plot_title || 'Generated Data Plot'
        });
      }
    }

    // Multiple plot images array from sandbox execution
    if (Array.isArray(executionRecord?.plot_images)) {
      executionRecord.plot_images.forEach((img, idx) => {
        const normalized = normalizeImageSrc(img);
        if (normalized && !plots.some(p => p.data === normalized)) {
          plots.push({
            id: `exec_plot_${idx}`,
            data: normalized,
            title: `Visualization Plot ${plots.length + 1}`
          });
        }
      });
    }

    if (executionRecord?.plotly_spec) {
      if (plots.length > 0) {
        plots[0].spec = executionRecord.plotly_spec;
      } else {
        plots.push({
          id: 'exec_plot_vector',
          data: null,
          spec: executionRecord.plotly_spec,
          title: executionRecord.plot_title || executionRecord.plotly_spec?.layout?.title?.text || executionRecord.plotly_spec?.layout?.title || 'Interactive Visual Analysis'
        });
      }
    }

    return plots;
  }, [artifacts, executionRecord]);

  // 2. Extract structured output tables safely (handling arrays, matrices, comparison tables)
  const tablesList = useMemo(() => {
    const tables = [];
    const rawTables = [
      ...(Array.isArray(executionRecord?.output_tables) ? executionRecord.output_tables : []),
      ...(Array.isArray(executionRecord?.tables) ? executionRecord.tables : []),
      ...(executionRecord?.table && typeof executionRecord.table === 'object' ? [executionRecord.table] : []),
      ...(executionRecord?.comparison_table && typeof executionRecord.comparison_table === 'object' ? [executionRecord.comparison_table] : []),
      ...(Array.isArray(executionRecord?.comparison_tables) ? executionRecord.comparison_tables : []),
      ...(executionRecord?.correlation_matrix && typeof executionRecord.correlation_matrix === 'object' ? [
        { name: 'Correlation Matrix', ...(executionRecord.correlation_matrix.columns ? executionRecord.correlation_matrix : { rows: executionRecord.correlation_matrix }), isCorrelation: true }
      ] : [])
    ];

    if (Array.isArray(rawTables)) {
      rawTables.forEach((tbl, idx) => {
        if (Array.isArray(tbl)) {
          // tbl is a raw array of record dicts: [{ colA: valA, colB: valB }]
          const columns = tbl.length > 0 && typeof tbl[0] === 'object' && tbl[0] !== null 
            ? Object.keys(tbl[0]) 
            : [];
          tables.push({
            name: `Output Table ${idx + 1}`,
            columns,
            rows: tbl,
            isCorrelation: false
          });
        } else if (tbl && typeof tbl === 'object') {
          const rows = tbl.rows || tbl.data || tbl.records || [];
          let columns = tbl.columns || tbl.headers || [];
          if ((!columns || columns.length === 0) && rows.length > 0 && typeof rows[0] === 'object' && rows[0] !== null) {
            columns = Array.isArray(rows[0]) 
              ? rows[0].map((_, i) => `Column ${i + 1}`) 
              : Object.keys(rows[0]);
          }

          // Check if table is a correlation matrix
          const isCorrelation = tbl.isCorrelation || (
            columns.length > 1 &&
            rows.length === columns.length &&
            columns.every(c => rows.some(r => typeof (r[c] ?? (Array.isArray(r) ? r[0] : null)) === 'number'))
          );

          tables.push({
            name: tbl.name || tbl.title || (isCorrelation ? 'Correlation Matrix' : `Output Table ${idx + 1}`),
            columns,
            rows,
            isCorrelation
          });
        }
      });
    }

    // Also check artifacts for tables
    if (Array.isArray(artifacts)) {
      artifacts.forEach((a, idx) => {
        if (a.type === 'table' && a.data) {
          const rows = Array.isArray(a.data) ? a.data : (a.data.rows || []);
          const columns = a.data.columns || (rows.length > 0 && typeof rows[0] === 'object' && rows[0] !== null ? Object.keys(rows[0]) : []);
          tables.push({
            name: a.title || a.name || `Artifact Table ${idx + 1}`,
            columns,
            rows,
            isCorrelation: false
          });
        }
      });
    }

    return tables;
  }, [executionRecord, artifacts]);

  // 3. Extract Executive Summary / Narrative Markdown
  const executiveSummary = useMemo(() => {
    if (executionRecord?.summary) return executionRecord.summary;
    if (executionRecord?.executive_summary) return executionRecord.executive_summary;
    if (executionRecord?.response) return executionRecord.response;
    if (executionRecord?.output && typeof executionRecord.output === 'string' && !executionRecord.output.startsWith('Traceback') && executionRecord.output.length > 30) {
      return executionRecord.output;
    }
    if (Array.isArray(artifacts)) {
      const rep = artifacts.find(a => a.type === 'report' || a.type === 'summary' || a.type === 'insight');
      if (rep) return rep.data || rep.summary || rep.text;
    }
    return null;
  }, [executionRecord, artifacts]);

  const [copiedImage, setCopiedImage] = useState(null);
  const [copiedText, setCopiedText] = useState(false);
  const [tableSearchTerms, setTableSearchTerms] = useState({});
  const [tablePages, setTablePages] = useState({});
  const [tablePageSizes, setTablePageSizes] = useState({});
  const [errorSubTab, setErrorSubTab] = useState('diagnostic'); // 'diagnostic' | 'stderr' | 'stdout'
  const [tableSortConfigs, setTableSortConfigs] = useState({}); // { [searchKey]: { column: string, direction: 'asc'|'desc' } }
  const [columnVisibility, setColumnVisibility] = useState({}); // { [searchKey]: Set of hidden column names }
  const [activeColumnMenu, setActiveColumnMenu] = useState(null); // searchKey of table with column menu open
  const [selectedRowInspection, setSelectedRowInspection] = useState(null); // { row, columns, tableName }
  const [showExplainPlan, setShowExplainPlan] = useState(false);
  const [splitView, setSplitView] = useState(false);
  const [savingDatasetIdx, setSavingDatasetIdx] = useState(null);
  const [savedDatasetFeedback, setSavedDatasetFeedback] = useState({}); // { [idx]: { name: string, id: string } }
  const [thresholdTau, setThresholdTau] = useState(0.50);
  const [costFP, setCostFP] = useState(1);
  const [costFN, setCostFN] = useState(5);
  const [thresholdSplit, setThresholdSplit] = useState('eval'); // 'tuning' | 'eval'
  const [copiedSQL, setCopiedSQL] = useState(false);

  const handleCopySQL = async (sqlText) => {
    try {
      await navigator.clipboard.writeText(sqlText);
      setCopiedSQL(true);
      setTimeout(() => setCopiedSQL(false), 2000);
    } catch (err) {
      console.error('Failed to copy SQL', err);
    }
  };

  const handleCopyImage = async (imageSrc) => {
    try {
      await copyImageToClipboard(imageSrc);
      setCopiedImage(imageSrc);
      setTimeout(() => setCopiedImage(null), 2000);
    } catch (err) {
      console.error('Failed to copy image', err);
    }
  };

  const handleCopyStderr = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedText(true);
      setTimeout(() => setCopiedText(false), 2000);
    } catch (err) {
      console.error('Failed to copy text', err);
    }
  };

  // Helper: Export Table to CSV (Supports both full table and visible page exports)
  const handleExportTableCSV = (table, idx, exportMode = 'all', pageRows = null) => {
    try {
      if (!table || !Array.isArray(table.rows) || !Array.isArray(table.columns)) return;
      const targetRows = (exportMode === 'page' && Array.isArray(pageRows)) ? pageRows : table.rows;
      const headers = table.columns.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',');
      const rows = targetRows.map(row => {
        return table.columns.map(col => {
          const val = Array.isArray(row) ? row[table.columns.indexOf(col)] : row[col];
          if (val === null || val === undefined) return '""';
          return `"${String(val).replace(/"/g, '""')}"`;
        }).join(',');
      }).join('\n');
      const csvContent = `${headers}\n${rows}`;
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      const safeName = (table.name || `table_${idx + 1}`).toLowerCase().replace(/[^a-z0-9_]/g, '_');
      const suffix = exportMode === 'page' ? '_page' : '_full';
      link.setAttribute('download', `${safeName}${suffix}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to export CSV', err);
    }
  };

  // Helper: Export Reproducible Python Script
  const handleExportPythonCode = () => {
    try {
      const code = executionRecord?.reproducible_code || executionRecord?.code;
      if (!code) return;
      const blob = new Blob([code], { type: 'text/x-python;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', 'datapilot_analysis.py');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to export Python script', err);
    }
  };

  // Helper: Save Result Table as New Dataset
  const handleSaveResultAsDataset = async (table, idx) => {
    try {
      const defaultName = (table.name || `result_dataset_${idx + 1}`).replace(/[^a-zA-Z0-9_]/g, '_');
      const datasetName = window.prompt('Enter name for the promoted dataset:', defaultName);
      if (!datasetName || !datasetName.trim()) return;

      setSavingDatasetIdx(idx);
      // Format rows as array of records
      const formattedRows = table.rows.map(row => {
        if (Array.isArray(row)) {
          const obj = {};
          table.columns.forEach((col, cIdx) => {
            obj[col] = row[cIdx];
          });
          return obj;
        }
        return row;
      });

      const response = await fetch(`${API}/api/datasets/save-result`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          name: datasetName.trim(),
          columns: table.columns,
          rows: formattedRows,
          parent_dataset_id: executionRecord?.manifest?.dataset_id || executionRecord?.dataset_id || null,
          metadata: {
            operator: executionRecord?.manifest?.operator || 'analysis_result',
            row_count: formattedRows.length
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const resData = await response.json();
      setSavedDatasetFeedback(prev => ({
        ...prev,
        [idx]: {
          name: resData.dataset?.name || datasetName.trim(),
          id: resData.dataset_id
        }
      }));
    } catch (err) {
      console.error('Failed to promote result as dataset:', err);
      alert(`Could not save dataset: ${err.message}`);
    } finally {
      setSavingDatasetIdx(null);
    }
  };

  // Helper: Toggle Column Sorting
  const handleSortColumn = (searchKey, colName) => {
    const current = tableSortConfigs[searchKey];
    if (current?.column === colName) {
      if (current.direction === 'asc') {
        setTableSortConfigs({ ...tableSortConfigs, [searchKey]: { column: colName, direction: 'desc' } });
      } else {
        // clear sorting
        const updated = { ...tableSortConfigs };
        delete updated[searchKey];
        setTableSortConfigs(updated);
      }
    } else {
      setTableSortConfigs({ ...tableSortConfigs, [searchKey]: { column: colName, direction: 'asc' } });
    }
    setTablePages({ ...tablePages, [searchKey]: 1 });
  };

  // Helper: Compute Column Summaries for Table Footer
  const computeColumnSummary = (rows, col) => {
    let count = 0;
    let nullCount = 0;
    let sum = 0;
    let numCount = 0;
    let min = Infinity;
    let max = -Infinity;
    const uniqueValues = new Set();

    rows.forEach(r => {
      const val = Array.isArray(r) ? r[col] : r[col];
      if (val === null || val === undefined || val === '') {
        nullCount++;
      } else {
        count++;
        uniqueValues.add(val);
        if (typeof val === 'number' && !isNaN(val)) {
          numCount++;
          sum += val;
          if (val < min) min = val;
          if (val > max) max = val;
        }
      }
    });

    if (numCount > 0 && numCount === count) {
      const mean = (sum / numCount).toFixed(2);
      return { type: 'numeric', count, nullCount, mean, min: min.toFixed(2), max: max.toFixed(2) };
    }
    return { type: 'categorical', count, nullCount, unique: uniqueValues.size };
  };

  // Extract Error State
  const isRunning = executionRecord?.status === 'running' || executionRecord?.status === 'RUNNING';
  const hasError = executionRecord?.status === 'FAILED' || executionRecord?.status === 'failed' || executionRecord?.status === 'EXECUTION_ERROR' || !!executionRecord?.error || !!executionRecord?.stderr;
  const isSucceeded = (executionRecord?.status === 'SUCCEEDED' || executionRecord?.status === 'success') && !hasError;
  const stdoutContent = executionRecord?.stdout || (typeof executionRecord?.output === 'string' && !executionRecord.output.startsWith('Traceback') ? executionRecord.output : '');
  
  let stderrContent = executionRecord?.stderr || '';
  if (!stderrContent && executionRecord?.error) {
    stderrContent = typeof executionRecord.error === 'object' 
      ? (executionRecord.error.message || JSON.stringify(executionRecord.error, null, 2))
      : String(executionRecord.error);
  }

  const rawErrorCode = executionRecord?.error?.code || executionRecord?.error_code || (hasError ? 'EXECUTION_FAILED' : null);
  const executionManifest = executionRecord?.manifest && typeof executionRecord.manifest === 'object'
    ? executionRecord.manifest
    : null;
  const hasExecutionEvidence = !!executionManifest;
  const hasConsoleOutput = executionRecord?.execution_mode !== 'typed_operator' && !!(stdoutContent || stderrContent);
  const matchedDiagnostic = rawErrorCode ? ERROR_DIAGNOSTICS[rawErrorCode] || {
    title: `Execution Alert: ${rawErrorCode}`,
    stage: executionRecord?.error?.stage || "Runtime Sandbox",
    hint: executionRecord?.error?.message || stderrContent || "An execution error occurred during pipeline execution.",
    action: executionRecord?.error?.required_action || "Inspect stdout/stderr logs below or rephrase your analysis prompt.",
    icon: XCircle
  } : null;

  // The result panel is keyed by execution in App. Pick the initial useful view,
  // then preserve a user's tab choice instead of forcing them back to Charts.
  const [activeTab, setActiveTab] = useState(() => {
    if (hasError) return 'error';
    if (executiveSummary) return 'summary';
    if (tablesList.length > 0) return 'tables';
    if (plotsList.length > 0) return 'charts';
    if (hasExecutionEvidence) return 'evidence';
    return 'tables';
  });

  if (!executionRecord && !validationReport) {
    return (
      <div style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#090d16',
        color: '#64748b',
        gap: 8,
        padding: 24,
        textAlign: 'center'
      }}>
        <Activity size={28} color="#334155" />
        <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: '#94a3b8' }}>No Execution Results Yet</p>
        <p style={{ margin: 0, fontSize: 11, color: '#475569' }}>Run code or execute a workflow node to inspect tables, charts, metrics, summaries, and logs here.</p>
      </div>
    );
  }

  const tabs = [
    ...(hasError ? [{ id: 'error', label: 'Error Diagnostics', icon: AlertTriangle, isError: true }] : []),
    ...(executiveSummary ? [{ id: 'summary', label: 'Executive Summary', icon: BookOpen }] : []),
    { id: 'tables', label: 'Tables', icon: TableIcon, count: tablesList.length },
    { id: 'charts', label: 'Charts', icon: BarChart2, count: plotsList.length },
    ...(hasExecutionEvidence ? [{ id: 'evidence', label: 'Execution Evidence', icon: Layers }] : []),
    ...(hasConsoleOutput ? [{ id: 'stdout', label: 'Console & Stdout', icon: Terminal, hasContent: true }] : []),
    { id: 'metrics', label: 'Metrics', icon: FileText, count: executionRecord?.metrics ? Object.keys(executionRecord.metrics).length : 0 },
    { id: 'validation', label: 'Validation', icon: CheckSquare, hasContent: !!validationReport }
  ];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: '#090d16',
      color: '#f8fafc',
      fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
    }}>
      {/* 1. Result Status Header Bar */}
      <div style={{
        padding: '8px 16px',
        borderBottom: '1px solid #1e293b',
        background: '#0f172a',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexShrink: 0,
        gap: 8
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, fontWeight: 800, color: '#f8fafc', letterSpacing: '0.05em' }}>RESULTS HUD</span>
          {isRunning ? (
            <span style={{ fontSize: 11, color: '#fbbf24', display: 'flex', alignItems: 'center', gap: 5, background: 'rgba(251, 191, 36, 0.1)', padding: '2px 8px', borderRadius: 4, fontWeight: 600 }}>
              <RefreshCw size={12} className="spinner" /> Executing in Python Sandbox...
            </span>
          ) : hasError ? (
            <span style={{ fontSize: 11, color: '#f87171', display: 'flex', alignItems: 'center', gap: 5, background: 'rgba(248, 113, 113, 0.1)', padding: '2px 8px', borderRadius: 4, fontWeight: 600, border: '1px solid rgba(248, 113, 113, 0.2)' }}>
              <XCircle size={12} /> Execution Failed {rawErrorCode ? `[${rawErrorCode}]` : ''}
            </span>
          ) : (
            <span style={{ fontSize: 11, color: '#4ade80', display: 'flex', alignItems: 'center', gap: 5, background: 'rgba(74, 222, 128, 0.1)', padding: '2px 8px', borderRadius: 4, fontWeight: 600, border: '1px solid rgba(74, 222, 128, 0.2)' }}>
              <CheckCircle size={12} /> Succeeded {executionRecord?.timing || executionRecord?.execution_time ? `(${executionRecord.timing || executionRecord.execution_time + 's'})` : ''}
            </span>
          )}
          {/* 5-Tier Source Storage Status Badge */}
          {(executionRecord?.source_storage === 'source_native' || executionRecord?.is_source_native) ? (
            <span style={{ fontSize: 10, color: '#10b981', display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(16, 185, 129, 0.12)', padding: '2px 8px', borderRadius: 4, fontWeight: 700, border: '1px solid rgba(16, 185, 129, 0.3)' }} title="Original file — accessed in place without disk duplication">
              Original file — accessed in place
            </span>
          ) : (executionRecord?.source_storage === 'memory_only' || executionRecord?.source_storage === 'dataframe_memory') ? (
            <span style={{ fontSize: 10, color: '#f59e0b', display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(245, 158, 11, 0.12)', padding: '2px 8px', borderRadius: 4, fontWeight: 700, border: '1px solid rgba(245, 158, 11, 0.3)' }} title="Original file loaded into memory for analytical execution">
              Original file — loaded into memory
            </span>
          ) : (executionRecord?.source_storage === 'managed_upload' || executionRecord?.source_storage === 'internal_copy') ? (
            <span style={{ fontSize: 10, color: '#60a5fa', display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(96, 165, 250, 0.12)', padding: '2px 8px', borderRadius: 4, fontWeight: 700, border: '1px solid rgba(96, 165, 250, 0.3)' }} title="Uploaded dataset stored in managed application storage">
              Managed upload — stored internally
            </span>
          ) : (executionRecord?.source_storage === 'database' || executionRecord?.source_storage === 'remote_sql') ? (
            <span style={{ fontSize: 10, color: '#c084fc', display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(192, 132, 252, 0.12)', padding: '2px 8px', borderRadius: 4, fontWeight: 700, border: '1px solid rgba(192, 132, 252, 0.3)' }} title="Direct live database connector with query pushdown">
              Database source — connected
            </span>
          ) : (executionRecord?.source_storage === 'promoted_result' || executionRecord?.source_storage === 'derived') ? (
            <span style={{ fontSize: 10, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(56, 189, 248, 0.12)', padding: '2px 8px', borderRadius: 4, fontWeight: 700, border: '1px solid rgba(56, 189, 248, 0.3)' }} title="Derived result table persisted separately as a registered dataset">
              Derived result — persisted separately
            </span>
          ) : null}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Dual-Pane Split View Toggle */}
          <button
            onClick={() => setSplitView(!splitView)}
            title={splitView ? "Switch to Standard Tabbed View" : "Switch to Dual-Pane Split View"}
            style={{
              background: splitView ? 'rgba(56, 189, 248, 0.2)' : '#1e293b',
              color: splitView ? '#38bdf8' : '#94a3b8',
              border: splitView ? '1px solid #0284c7' : '1px solid #334155',
              borderRadius: 4,
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              transition: 'all 0.15s ease'
            }}
          >
            <Split size={12} />
            {splitView ? "Standard View" : "Dual-Pane View"}
          </button>

          {/* Export Reproducible Python Script */}
          {(executionRecord?.reproducible_code || executionRecord?.code) && (
            <button
              onClick={handleExportPythonCode}
              title="Download standalone reproducible Python script"
              style={{
                background: '#1e293b',
                color: '#38bdf8',
                border: '1px solid #334155',
                borderRadius: 4,
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                transition: 'all 0.15s ease'
              }}
            >
              <Code size={12} />
              Export Python
            </button>
          )}

          {onRequestInsights && (
            <button
              onClick={onRequestInsights}
              style={{
                background: 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
                color: '#ffffff',
                border: 'none',
                borderRadius: 4,
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                flexShrink: 0
              }}
            >
              <Sparkles size={12} />
              Interpret Insights
            </button>
          )}
        </div>
      </div>

      {/* 2. Non-overlapping, Horizontally Scrollable Tab Headers */}
      {!splitView && (
        <div style={{
          display: 'flex',
          borderBottom: '1px solid #1e293b',
          background: '#0a0e1a',
          padding: '0 8px',
          overflowX: 'auto',
          whiteSpace: 'nowrap',
          scrollbarWidth: 'none',
          flexShrink: 0,
          alignItems: 'center',
          gap: 4
        }}>
          {tabs.map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            const isErrorTab = tab.isError;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: '8px 14px',
                  background: isActive ? (isErrorTab ? 'rgba(239, 68, 68, 0.15)' : '#1e293b') : 'transparent',
                  border: 'none',
                  borderBottom: isActive ? (isErrorTab ? '2px solid #ef4444' : '2px solid #38bdf8') : '2px solid transparent',
                  borderRadius: '4px 4px 0 0',
                  cursor: 'pointer',
                  color: isErrorTab ? '#f87171' : isActive ? '#38bdf8' : '#94a3b8',
                  fontWeight: isActive ? 700 : 500,
                  fontSize: 12,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  transition: 'all 0.15s ease',
                  flexShrink: 0
                }}
              >
                <Icon size={13} />
                <span>{tab.label}</span>
                {tab.count !== undefined && tab.count > 0 && (
                  <span style={{
                    background: isActive ? '#0284c7' : '#334155',
                    color: '#ffffff',
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '1px 6px',
                    borderRadius: 10
                  }}>
                    {tab.count}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* 3. Panel Body Content Area */}
      {splitView ? (
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'minmax(360px, 1fr) minmax(360px, 1fr)', gap: 12, padding: 14, overflow: 'hidden', background: '#090d16' }}>
          {/* Left Pane: Visuals & Narrative (Charts / Executive Summary) */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto', paddingRight: 6, borderRight: '1px solid #1e293b' }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: 6 }}>
              <BarChart2 size={13} /> Visualizations & Findings
            </div>
            {executiveSummary && (
              <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 12 }}>
                <div style={{ color: '#38bdf8', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Executive Findings</div>
                <div style={{ fontSize: 12, lineHeight: 1.5, color: '#f1f5f9' }}>{renderMarkdown(executiveSummary)}</div>
              </div>
            )}
            {plotsList.length > 0 ? (
              plotsList.map((plot, idx) => (
                <InteractiveChart
                  key={idx}
                  spec={plot.spec}
                  fallbackImage={plot.data}
                  title={plot.title}
                  height={240}
                  onZoomImage={onZoomImage}
                />
              ))
            ) : (
              <div style={{ padding: 24, textAlign: 'center', color: '#64748b', fontSize: 12, background: '#0c101d', borderRadius: 8, border: '1px solid #1e293b' }}>
                No charts generated for this run.
              </div>
            )}
          </div>

          {/* Right Pane: Data & Metrics (Tables / ML Extended Metrics / CV) */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto', paddingLeft: 6 }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: '#a7f3d0', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: 6 }}>
              <TableIcon size={13} /> Data Tables & Verification
            </div>

            {/* ML Baseline & Extended Metrics in Split View */}
            {executionRecord?.extended_metrics && (
              <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#38bdf8', marginBottom: 6, display: 'flex', justifyContent: 'space-between' }}>
                  <span>ML Metrics & Baseline Comparison</span>
                  {executionRecord.extended_metrics.improvement_vs_baseline && (
                    <span style={{ color: '#4ade80' }}>vs Baseline: {executionRecord.extended_metrics.improvement_vs_baseline}</span>
                  )}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 6 }}>
                  {Object.entries(executionRecord.extended_metrics).map(([k, v]) => (
                    <div key={k} style={{ background: '#13192b', padding: '6px 8px', borderRadius: 4 }}>
                      <div style={{ color: '#64748b', fontSize: 9, fontWeight: 700, textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}</div>
                      <div style={{ color: '#f8fafc', fontSize: 11, fontWeight: 700, fontFamily: 'monospace' }}>{typeof v === 'number' ? v.toFixed(4) : String(v)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Tables in Split View */}
            {tablesList.length > 0 ? (
              tablesList.map((table, idx) => (
                <div key={idx} style={{ background: '#0c101d', borderRadius: 8, border: '1px solid #1e293b', overflow: 'hidden' }}>
                  <div style={{ padding: '6px 10px', background: '#13192b', borderBottom: '1px solid #1e293b', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: '#38bdf8' }}>{table.name} ({table.rows.length} rows)</span>
                    <button onClick={() => handleExportTableCSV(table, idx)} style={{ background: '#1e293b', color: '#38bdf8', border: '1px solid #334155', borderRadius: 3, padding: '2px 6px', fontSize: 10, cursor: 'pointer' }}>CSV</button>
                  </div>
                  <div style={{ overflowX: 'auto', maxHeight: 220, overflowY: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, textAlign: 'left' }}>
                      <thead>
                        <tr style={{ background: '#0f172a', borderBottom: '1px solid #1e293b' }}>
                          {table.columns.map((c, ci) => (
                            <th key={ci} style={{ padding: '4px 8px', color: '#cbd5e1', fontWeight: 700, whiteSpace: 'nowrap' }}>{String(c)}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {table.rows.slice(0, 15).map((r, ri) => (
                          <tr key={ri} style={{ borderBottom: '1px solid #1e293b' }}>
                            {table.columns.map((c, ci) => (
                              <td key={ci} style={{ padding: '4px 8px', whiteSpace: 'nowrap', color: '#94a3b8' }}>
                                {String(Array.isArray(r) ? r[ci] : r[c])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))
            ) : (
              <div style={{ padding: 24, textAlign: 'center', color: '#64748b', fontSize: 12, background: '#0c101d', borderRadius: 8, border: '1px solid #1e293b' }}>
                No structured tables generated.
              </div>
            )}
          </div>
        </div>
      ) : (
        <div style={{ flex: 1, overflow: 'auto', padding: 14, background: '#090d16' }}>
        {isRunning && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px 0', gap: 12 }}>
            <RefreshCw size={28} className="spinner" color="#38bdf8" />
            <span style={{ fontSize: 13, color: '#94a3b8' }}>Executing script in isolated runtime environment...</span>
          </div>
        )}

        {/* TAB: ERROR DIAGNOSTICS */}
        {!isRunning && activeTab === 'error' && matchedDiagnostic && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {/* Rich Error Diagnostic Card */}
            <div style={{
              background: '#180d10',
              border: '1px solid #dc2626',
              borderRadius: 8,
              padding: 16,
              boxShadow: '0 4px 16px rgba(220, 38, 38, 0.15)'
            }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ background: 'rgba(239, 68, 68, 0.2)', padding: 8, borderRadius: 6, color: '#f87171' }}>
                    <AlertTriangle size={20} />
                  </div>
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 800, color: '#fecaca' }}>
                      {matchedDiagnostic.title}
                    </div>
                    <div style={{ fontSize: 11, color: '#f87171', display: 'flex', alignItems: 'center', gap: 8, marginTop: 2 }}>
                      <span>Stage: <strong>{matchedDiagnostic.stage}</strong></span>
                      <span>•</span>
                      <span style={{ background: '#7f1d1d', color: '#fee2e2', padding: '1px 6px', borderRadius: 4, fontWeight: 700 }}>
                        {rawErrorCode || 'EXEC_ERROR'}
                      </span>
                    </div>
                  </div>
                </div>

                {stderrContent && (
                  <button
                    onClick={() => handleCopyStderr(stderrContent)}
                    style={{
                      background: '#2d1216',
                      color: copiedText ? '#4ade80' : '#fca5a5',
                      border: '1px solid #7f1d1d',
                      borderRadius: 4,
                      padding: '4px 10px',
                      fontSize: 11,
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4
                    }}
                  >
                    {copiedText ? <Check size={12} /> : <Copy size={12} />}
                    {copiedText ? 'Copied Traceback' : 'Copy Traceback'}
                  </button>
                )}
              </div>

              {/* Diagnostic Description & Required Action */}
              <div style={{ background: '#241014', borderRadius: 6, padding: '10px 12px', border: '1px solid #7f1d1d', marginBottom: 12 }}>
                <div style={{ fontSize: 12, color: '#fecaca', lineHeight: 1.5, marginBottom: 6 }}>
                  {matchedDiagnostic.hint}
                </div>
                <div style={{ fontSize: 11, color: '#fbbf24', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Wrench size={13} />
                  <span>Suggested Recovery: {matchedDiagnostic.action}</span>
                </div>
              </div>

              {/* Sub-tabs for Traceback vs Stdout */}
              <div style={{ display: 'flex', gap: 6, borderBottom: '1px solid #7f1d1d', paddingBottom: 6, marginBottom: 8 }}>
                <button
                  onClick={() => setErrorSubTab('diagnostic')}
                  style={{
                    background: errorSubTab === 'diagnostic' ? '#7f1d1d' : 'transparent',
                    color: errorSubTab === 'diagnostic' ? '#ffffff' : '#fca5a5',
                    border: 'none',
                    borderRadius: 4,
                    padding: '3px 8px',
                    fontSize: 11,
                    fontWeight: 700,
                    cursor: 'pointer'
                  }}
                >
                  Diagnostic Summary
                </button>
                {stderrContent && (
                  <button
                    onClick={() => setErrorSubTab('stderr')}
                    style={{
                      background: errorSubTab === 'stderr' ? '#7f1d1d' : 'transparent',
                      color: errorSubTab === 'stderr' ? '#ffffff' : '#fca5a5',
                      border: 'none',
                      borderRadius: 4,
                      padding: '3px 8px',
                      fontSize: 11,
                      fontWeight: 700,
                      cursor: 'pointer'
                    }}
                  >
                    Raw Stderr Traceback
                  </button>
                )}
                {stdoutContent && (
                  <button
                    onClick={() => setErrorSubTab('stdout')}
                    style={{
                      background: errorSubTab === 'stdout' ? '#7f1d1d' : 'transparent',
                      color: errorSubTab === 'stdout' ? '#ffffff' : '#fca5a5',
                      border: 'none',
                      borderRadius: 4,
                      padding: '3px 8px',
                      fontSize: 11,
                      fontWeight: 700,
                      cursor: 'pointer'
                    }}
                  >
                    Captured Stdout
                  </button>
                )}
              </div>

              {errorSubTab === 'stderr' && stderrContent && (
                <pre style={{
                  background: '#0a0506',
                  border: '1px solid #7f1d1d',
                  borderRadius: 6,
                  padding: 10,
                  fontFamily: 'var(--font-mono, monospace)',
                  fontSize: 11,
                  color: '#f87171',
                  overflowX: 'auto',
                  whiteSpace: 'pre-wrap',
                  maxHeight: 220,
                  margin: 0
                }}>
                  {stderrContent}
                </pre>
              )}

              {errorSubTab === 'stdout' && stdoutContent && (
                <pre style={{
                  background: '#07080a',
                  border: '1px solid #1e293b',
                  borderRadius: 6,
                  padding: 10,
                  fontFamily: 'var(--font-mono, monospace)',
                  fontSize: 11,
                  color: '#4ade80',
                  overflowX: 'auto',
                  whiteSpace: 'pre-wrap',
                  maxHeight: 220,
                  margin: 0
                }}>
                  {stdoutContent}
                </pre>
              )}
            </div>
          </div>
        )}

        {/* TAB: EXECUTIVE SUMMARY & INSIGHTS */}
        {!isRunning && activeTab === 'summary' && executiveSummary && (
          <div style={{
            background: '#0c101d',
            border: '1px solid #1e293b',
            borderRadius: 8,
            padding: 16,
            display: 'flex',
            flexDirection: 'column',
            gap: 12
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', paddingBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#38bdf8', fontSize: 13, fontWeight: 800 }}>
                <BookOpen size={16} />
                <span>Executive Data Science Findings</span>
              </div>
              <span style={{ fontSize: 11, color: '#94a3b8', background: '#1e293b', padding: '2px 8px', borderRadius: 10 }}>
                Synthesized Narrative
              </span>
            </div>

            <div style={{ fontSize: 13, lineHeight: 1.6, color: '#f1f5f9' }}>
              {renderMarkdown(executiveSummary)}
            </div>
          </div>
        )}

        {/* TAB 1: Structured Tables & Correlation Matrices */}
        {!isRunning && activeTab === 'tables' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {tablesList.length > 0 ? (
              tablesList.map((table, idx) => {
                const searchKey = `tbl_${idx}`;
                const searchTerm = tableSearchTerms[searchKey] || '';
                const pageSize = tablePageSizes[searchKey] || 25;
                const currentPage = tablePages[searchKey] || 1;
                const sortConfig = tableSortConfigs[searchKey];
                const savedDatasetInfo = savedDatasetFeedback[idx];
                const hiddenCols = columnVisibility[searchKey] || new Set();
                const visibleColumns = table.columns.filter(c => !hiddenCols.has(c));

                // Filter rows if search is active
                let filteredRows = searchTerm.trim() 
                  ? table.rows.filter(row => {
                      if (Array.isArray(row)) {
                        return row.some(val => val !== null && val !== undefined && String(val).toLowerCase().includes(searchTerm.toLowerCase()));
                      }
                      if (typeof row === 'object' && row !== null) {
                        return Object.values(row).some(val => val !== null && val !== undefined && String(val).toLowerCase().includes(searchTerm.toLowerCase()));
                      }
                      return String(row).toLowerCase().includes(searchTerm.toLowerCase());
                    })
                  : [...table.rows];

                // Deterministic Client-side Sort
                if (sortConfig && sortConfig.column) {
                  const { column, direction } = sortConfig;
                  filteredRows = [...filteredRows].sort((a, b) => {
                    const valA = Array.isArray(a) ? a[table.columns.indexOf(column)] : a[column];
                    const valB = Array.isArray(b) ? b[table.columns.indexOf(column)] : b[column];

                    if ((valA === null || valA === undefined || valA === '') && (valB === null || valB === undefined || valB === '')) return 0;
                    if (valA === null || valA === undefined || valA === '') return 1; // nulls always last
                    if (valB === null || valB === undefined || valB === '') return -1;

                    if (typeof valA === 'number' && typeof valB === 'number') {
                      return direction === 'asc' ? valA - valB : valB - valA;
                    }

                    const numA = Number(valA);
                    const numB = Number(valB);
                    if (!isNaN(numA) && !isNaN(numB) && typeof valA !== 'boolean' && typeof valB !== 'boolean') {
                      return direction === 'asc' ? numA - numB : numB - numA;
                    }

                    return direction === 'asc'
                      ? String(valA).localeCompare(String(valB))
                      : String(valB).localeCompare(String(valA));
                  });
                }

                const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
                const pageRows = filteredRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);

                return (
                  <div key={idx} style={{ background: '#0c101d', borderRadius: 8, border: '1px solid #1e293b', overflow: 'hidden' }}>
                    {/* Table Header Bar */}
                    <div style={{
                      padding: '8px 14px',
                      background: '#13192b',
                      borderBottom: '1px solid #1e293b',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      flexWrap: 'wrap',
                      gap: 8
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        {table.isCorrelation ? <Flame size={15} color="#f59e0b" /> : <TableIcon size={15} color="#38bdf8" />}
                        <span style={{ fontSize: 12, fontWeight: 700, color: table.isCorrelation ? '#f59e0b' : '#38bdf8' }}>
                          {table.name}
                        </span>
                        {table.isCorrelation && (
                          <span style={{ fontSize: 10, background: '#78350f', color: '#fef3c7', padding: '1px 6px', borderRadius: 4, fontWeight: 700 }}>
                            HEATMAP
                          </span>
                        )}
                        <span style={{ fontSize: 11, color: '#94a3b8', background: '#1e293b', padding: '2px 8px', borderRadius: 10 }}>
                          {table.rows.length.toLocaleString()} rows × {visibleColumns.length} cols {hiddenCols.size > 0 ? `(${hiddenCols.size} hidden)` : ''}
                        </span>
                        {sortConfig && (
                          <span style={{ fontSize: 10, background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', padding: '1px 6px', borderRadius: 4, fontWeight: 600 }}>
                            Sorted by {sortConfig.column} ({sortConfig.direction.toUpperCase()})
                          </span>
                        )}
                      </div>

                      {/* In-table Search, Column Visibility, Export & Promote Controls */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        {table.rows.length > 5 && (
                          <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            background: '#090d16',
                            border: '1px solid #1e293b',
                            borderRadius: 4,
                            padding: '2px 8px',
                            gap: 4
                          }}>
                            <Search size={12} color="#64748b" />
                            <input
                              type="text"
                              placeholder="Filter rows..."
                              value={searchTerm}
                              onChange={e => {
                                setTableSearchTerms({ ...tableSearchTerms, [searchKey]: e.target.value });
                                setTablePages({ ...tablePages, [searchKey]: 1 });
                              }}
                              style={{
                                background: 'transparent',
                                border: 'none',
                                outline: 'none',
                                color: '#f8fafc',
                                fontSize: 11,
                                width: 110
                              }}
                            />
                          </div>
                        )}

                        {/* Column Visibility Menu */}
                        <div style={{ position: 'relative' }}>
                          <button
                            onClick={() => setActiveColumnMenu(activeColumnMenu === searchKey ? null : searchKey)}
                            title="Toggle column visibility"
                            style={{
                              background: activeColumnMenu === searchKey ? 'rgba(56, 189, 248, 0.2)' : '#1e293b',
                              color: activeColumnMenu === searchKey ? '#38bdf8' : '#94a3b8',
                              border: '1px solid #334155',
                              borderRadius: 4,
                              padding: '3px 8px',
                              fontSize: 11,
                              fontWeight: 600,
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 4
                            }}
                          >
                            <Columns size={12} />
                            Columns {hiddenCols.size > 0 ? `(${visibleColumns.length}/${table.columns.length})` : ''}
                          </button>
                          {activeColumnMenu === searchKey && (
                            <div style={{
                              position: 'absolute',
                              right: 0,
                              top: '100%',
                              marginTop: 4,
                              zIndex: 30,
                              background: '#0f172a',
                              border: '1px solid #334155',
                              borderRadius: 6,
                              padding: 8,
                              boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                              maxHeight: 220,
                              overflowY: 'auto',
                              minWidth: 170
                            }}>
                              <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>
                                Visible Columns
                              </div>
                              {table.columns.map(col => {
                                const isHidden = hiddenCols.has(col);
                                return (
                                  <label key={col} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#cbd5e1', padding: '3px 0', cursor: 'pointer' }}>
                                    <input
                                      type="checkbox"
                                      checked={!isHidden}
                                      onChange={() => {
                                        const nextHidden = new Set(hiddenCols);
                                        if (isHidden) {
                                          nextHidden.delete(col);
                                        } else {
                                          if (visibleColumns.length > 1) {
                                            nextHidden.add(col);
                                          }
                                        }
                                        setColumnVisibility({ ...columnVisibility, [searchKey]: nextHidden });
                                      }}
                                    />
                                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{String(col)}</span>
                                  </label>
                                );
                              })}
                            </div>
                          )}
                        </div>

                        {/* Export to CSV (Full vs Page) */}
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                          <button
                            onClick={() => handleExportTableCSV(table, idx, 'all')}
                            title="Export all rows to CSV file"
                            style={{
                              background: '#1e293b',
                              color: '#38bdf8',
                              border: '1px solid #334155',
                              borderRadius: '4px 0 0 4px',
                              padding: '3px 8px',
                              fontSize: 11,
                              fontWeight: 600,
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 4
                            }}
                          >
                            <FileSpreadsheet size={12} />
                            CSV
                          </button>
                          <button
                            onClick={() => handleExportTableCSV(table, idx, 'page', pageRows)}
                            title="Export current page rows to CSV file"
                            style={{
                              background: '#1e293b',
                              color: '#94a3b8',
                              border: '1px solid #334155',
                              borderLeft: 'none',
                              borderRadius: '0 4px 4px 0',
                              padding: '3px 6px',
                              fontSize: 10,
                              fontWeight: 600,
                              cursor: 'pointer'
                            }}
                          >
                            Page
                          </button>
                        </div>

                        {/* Promote Result as Dataset */}
                        {savedDatasetInfo ? (
                          <span style={{
                            background: 'rgba(74, 222, 128, 0.15)',
                            color: '#4ade80',
                            border: '1px solid #16a34a',
                            borderRadius: 4,
                            padding: '3px 8px',
                            fontSize: 11,
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 4
                          }}>
                            <CheckCircle size={12} />
                            Saved: {savedDatasetInfo.name}
                          </span>
                        ) : (
                          <button
                            onClick={() => handleSaveResultAsDataset(table, idx)}
                            disabled={savingDatasetIdx === idx}
                            title="Promote and save this result table as a new dataset in DataPilot OS"
                            style={{
                              background: '#1e293b',
                              color: '#a7f3d0',
                              border: '1px solid #059669',
                              borderRadius: 4,
                              padding: '3px 8px',
                              fontSize: 11,
                              fontWeight: 600,
                              cursor: savingDatasetIdx === idx ? 'not-allowed' : 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 4,
                              opacity: savingDatasetIdx === idx ? 0.6 : 1,
                              transition: 'all 0.15s ease'
                            }}
                          >
                            {savingDatasetIdx === idx ? <RefreshCw size={12} className="spinner" /> : <Database size={12} />}
                            {savingDatasetIdx === idx ? "Promoting..." : "Save as Dataset"}
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Horizontally and Vertically Scrollable Table Container */}
                    <div style={{ overflowX: 'auto', width: '100%', maxHeight: 360, overflowY: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
                        <thead>
                          <tr style={{ background: '#0f172a', borderBottom: '2px solid #1e293b', position: 'sticky', top: 0, zIndex: 5 }}>
                            <th style={{ padding: '6px 10px', width: 44, color: '#64748b', borderRight: '1px solid #1e293b', textAlign: 'center' }}>#</th>
                            {visibleColumns.map((col, cIdx) => {
                              const isSorted = sortConfig?.column === col;
                              return (
                                <th
                                  key={cIdx}
                                  onClick={() => handleSortColumn(searchKey, col)}
                                  title="Click to sort ascending, descending, or reset"
                                  style={{
                                    padding: '6px 12px',
                                    color: isSorted ? '#38bdf8' : '#cbd5e1',
                                    fontWeight: 700,
                                    borderRight: '1px solid #1e293b',
                                    whiteSpace: 'nowrap',
                                    cursor: 'pointer',
                                    userSelect: 'none',
                                    background: isSorted ? 'rgba(56, 189, 248, 0.08)' : 'transparent'
                                  }}
                                >
                                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                                    <span>{String(col)}</span>
                                    {isSorted ? (
                                      sortConfig.direction === 'asc' ? <ArrowUp size={12} color="#38bdf8" /> : <ArrowDown size={12} color="#38bdf8" />
                                    ) : (
                                      <ArrowDownUp size={11} color="#475569" style={{ opacity: 0.6 }} />
                                    )}
                                  </div>
                                </th>
                              );
                            })}
                          </tr>
                        </thead>
                        <tbody>
                          {pageRows.map((row, rIdx) => {
                            const absIdx = (currentPage - 1) * pageSize + rIdx + 1;
                            return (
                              <tr
                                key={rIdx}
                                style={{
                                  background: rIdx % 2 === 0 ? '#090d16' : '#0e1424',
                                  borderBottom: '1px solid #1e293b',
                                  transition: 'background 0.1s'
                                }}
                                onMouseEnter={e => e.currentTarget.style.background = '#1e293b'}
                                onMouseLeave={e => e.currentTarget.style.background = rIdx % 2 === 0 ? '#090d16' : '#0e1424'}
                              >
                                <td
                                  onClick={() => setSelectedRowInspection({ row, columns: table.columns, tableName: table.name, rowIdx: absIdx })}
                                  title="Click to inspect this row's full schema and values"
                                  style={{ padding: '6px 10px', color: '#64748b', borderRight: '1px solid #1e293b', textAlign: 'center', fontSize: 11, cursor: 'pointer', userSelect: 'none' }}
                                >
                                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3 }}>
                                    <span>{absIdx}</span>
                                    <Eye size={10} style={{ opacity: 0.6 }} />
                                  </div>
                                </td>
                                {visibleColumns.map((col, cIdx) => {
                                  const colIdx = table.columns.indexOf(col);
                                  const val = Array.isArray(row) ? row[colIdx] : row[col];
                                  const isNull = val === null || val === undefined;
                                  const isNum = typeof val === 'number';

                                  // Correlation Matrix Heatmap Color
                                  let cellBg = 'transparent';
                                  let textColor = isNull ? '#ef4444' : '#cbd5e1';
                                  if (table.isCorrelation && isNum) {
                                    if (val >= 0.8) {
                                      cellBg = 'rgba(16, 185, 129, 0.35)';
                                      textColor = '#6ee7b7';
                                    } else if (val >= 0.5) {
                                      cellBg = 'rgba(16, 185, 129, 0.2)';
                                      textColor = '#a7f3d0';
                                    } else if (val <= -0.5) {
                                      cellBg = 'rgba(239, 68, 68, 0.25)';
                                      textColor = '#fca5a5';
                                    } else if (val <= -0.8) {
                                      cellBg = 'rgba(239, 68, 68, 0.4)';
                                      textColor = '#fecaca';
                                    }
                                  }

                                  // Imputation resolution highlight
                                  const colNameLower = String(col).toLowerCase();
                                  const isImputedZero = (colNameLower.includes('after') || colNameLower.includes('resolved')) && (val === 0 || val === '0' || val === '0.0');

                                  return (
                                    <td
                                      key={cIdx}
                                      style={{
                                        padding: '6px 12px',
                                        borderRight: '1px solid #1e293b',
                                        whiteSpace: 'nowrap',
                                        maxWidth: 280,
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        background: cellBg,
                                        color: isImputedZero ? '#4ade80' : textColor,
                                        fontWeight: table.isCorrelation || isImputedZero ? 700 : 400
                                      }}
                                    >
                                      {isNull ? (
                                        <span style={{ color: '#ef4444', fontStyle: 'italic', fontSize: 11 }}>NaN</span>
                                      ) : typeof val === 'boolean' ? (
                                        <span style={{ color: val ? '#4ade80' : '#f87171', fontWeight: 700 }}>{String(val)}</span>
                                      ) : isNum && table.isCorrelation ? (
                                        val.toFixed(3)
                                      ) : isImputedZero ? (
                                        `✓ ${String(val)} nulls`
                                      ) : (
                                        String(val)
                                      )}
                                    </td>
                                  );
                                })}
                              </tr>
                            );
                          })}
                        </tbody>
                        {/* Table Summary Footer Row */}
                        <tfoot>
                          <tr style={{ background: '#0a0f1d', borderTop: '2px solid #334155', fontSize: 11, color: '#94a3b8' }}>
                            <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 700, color: '#64748b', borderRight: '1px solid #1e293b' }}>
                              Σ
                            </td>
                            {visibleColumns.map((col, cIdx) => {
                              const summary = computeColumnSummary(table.rows, col);
                              return (
                                <td key={cIdx} style={{ padding: '6px 12px', borderRight: '1px solid #1e293b', whiteSpace: 'nowrap' }}>
                                  {summary.type === 'numeric' ? (
                                    <div>
                                      <span style={{ color: '#38bdf8', fontWeight: 700 }}>μ={summary.mean}</span>{' '}
                                      <span style={{ color: '#64748b' }}>[{summary.min}..{summary.max}]</span>
                                      {summary.nullCount > 0 && <span style={{ color: '#f87171', marginLeft: 4 }}>({summary.nullCount} nulls)</span>}
                                    </div>
                                  ) : (
                                    <div>
                                      <span style={{ color: '#a7f3d0' }}>{summary.unique} unique</span>
                                      {summary.nullCount > 0 && <span style={{ color: '#f87171', marginLeft: 4 }}>({summary.nullCount} nulls)</span>}
                                    </div>
                                  )}
                                </td>
                              );
                            })}
                          </tr>
                        </tfoot>
                      </table>
                    </div>

                    {/* Pagination Footer */}
                    {table.rows.length > 25 && (
                      <div style={{
                        padding: '6px 14px',
                        fontSize: 11,
                        color: '#94a3b8',
                        background: '#0f172a',
                        borderTop: '1px solid #1e293b',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between'
                      }}>
                        <span>
                          Showing {(currentPage - 1) * pageSize + 1} to {Math.min(currentPage * pageSize, filteredRows.length)} of {filteredRows.length.toLocaleString()} rows
                        </span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <button
                            onClick={() => setTablePages({ ...tablePages, [searchKey]: 1 })}
                            disabled={currentPage === 1}
                            style={{ background: '#1e293b', border: 'none', color: '#f8fafc', padding: 3, borderRadius: 3, cursor: currentPage === 1 ? 'not-allowed' : 'pointer', opacity: currentPage === 1 ? 0.4 : 1 }}
                          >
                            <ChevronsLeft size={13} />
                          </button>
                          <button
                            onClick={() => setTablePages({ ...tablePages, [searchKey]: Math.max(1, currentPage - 1) })}
                            disabled={currentPage === 1}
                            style={{ background: '#1e293b', border: 'none', color: '#f8fafc', padding: 3, borderRadius: 3, cursor: currentPage === 1 ? 'not-allowed' : 'pointer', opacity: currentPage === 1 ? 0.4 : 1 }}
                          >
                            <ChevronLeft size={13} />
                          </button>
                          <span style={{ padding: '0 6px', fontWeight: 700, color: '#38bdf8' }}>
                            {currentPage} / {totalPages}
                          </span>
                          <button
                            onClick={() => setTablePages({ ...tablePages, [searchKey]: Math.min(totalPages, currentPage + 1) })}
                            disabled={currentPage === totalPages}
                            style={{ background: '#1e293b', border: 'none', color: '#f8fafc', padding: 3, borderRadius: 3, cursor: currentPage === totalPages ? 'not-allowed' : 'pointer', opacity: currentPage === totalPages ? 0.4 : 1 }}
                          >
                            <ChevronRight size={13} />
                          </button>
                          <button
                            onClick={() => setTablePages({ ...tablePages, [searchKey]: totalPages })}
                            disabled={currentPage === totalPages}
                            style={{ background: '#1e293b', border: 'none', color: '#f8fafc', padding: 3, borderRadius: 3, cursor: currentPage === totalPages ? 'not-allowed' : 'pointer', opacity: currentPage === totalPages ? 0.4 : 1 }}
                          >
                            <ChevronsRight size={13} />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <div style={{ textAlign: 'center', padding: '36px 0', color: '#64748b' }}>
                <TableIcon size={28} color="#334155" style={{ marginBottom: 8 }} />
                <p style={{ margin: 0, fontSize: 13 }}>No structured output tables returned.</p>
                <p style={{ margin: '4px 0 0', fontSize: 11 }}>Assign your output DataFrame to `result_df` or `output_df` in Python to see tables here.</p>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: Generated Charts with Zoom Trigger */}
        {!isRunning && activeTab === 'charts' && (
          <div style={{ display: 'grid', gridTemplateColumns: plotsList.length > 1 ? 'repeat(auto-fit, minmax(340px, 1fr))' : '1fr', gap: 16 }}>
            {plotsList.length > 0 ? (
              plotsList.map((plot, idx) => (
                <InteractiveChart
                  key={idx}
                  spec={plot.spec}
                  fallbackImage={plot.data}
                  title={plot.title || `Plot Visualization ${idx + 1}`}
                  height={340}
                  onZoomImage={onZoomImage}
                />
              ))
            ) : (
              <div style={{ textAlign: 'center', padding: '36px 0', color: '#64748b' }}>
                <BarChart2 size={28} color="#334155" style={{ marginBottom: 8 }} />
                <p style={{ margin: 0, fontSize: 13 }}>No plot visualizations generated.</p>
                <p style={{ margin: '4px 0 0', fontSize: 11 }}>Generate charts via matplotlib or seaborn using `plt.show()` in your code.</p>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: Typed-engine execution evidence */}
        {!isRunning && activeTab === 'evidence' && executionManifest && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#38bdf8', fontSize: 13, fontWeight: 800, marginBottom: 12 }}>
                <Layers size={16} />
                <span>Typed Operator Evidence</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 8 }}>
                {Object.entries(executionManifest)
                  .filter(([key, value]) => key !== 'inferred_bindings' && value !== null && value !== undefined && value !== '' && (!Array.isArray(value) || value.length > 0))
                  .map(([key, value]) => (
                    <div key={key} style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                      <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
                        {key.replace(/_/g, ' ')}
                      </div>
                      <div style={{ color: '#e2e8f0', fontSize: 12, overflowWrap: 'anywhere' }}>
                        {Array.isArray(value) ? value.join(', ') : typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {Array.isArray(executionManifest.inferred_bindings) && executionManifest.inferred_bindings.length > 0 && (
              <div style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, padding: 12 }}>
                <div style={{ color: '#93c5fd', fontSize: 12, fontWeight: 800, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span>Binding &amp; Policy Provenance</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {executionManifest.inferred_bindings.map((binding, idx) => (
                    <div key={idx} style={{ background: '#13192b', borderRadius: 6, padding: '8px 12px', borderLeft: binding.source === 'policy_default' ? '3px solid #a855f7' : '3px solid #38bdf8' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <span style={{ fontWeight: 700, fontSize: 12, color: '#f8fafc' }}>
                          Slot: <code style={{ color: '#38bdf8' }}>{binding.slot}</code> → <code style={{ color: '#a7f3d0' }}>{Array.isArray(binding.column) ? binding.column.join(', ') : String(binding.column)}</code>
                        </span>
                        <span style={{
                          fontSize: 10,
                          fontWeight: 700,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: binding.source === 'policy_default' ? 'rgba(168, 85, 247, 0.2)' : 'rgba(56, 189, 248, 0.2)',
                          color: binding.source === 'policy_default' ? '#d8b4fe' : '#7dd3fc',
                          textTransform: 'uppercase'
                        }}>
                          {binding.source ? binding.source.replace(/_/g, ' ') : 'inferred'}
                        </span>
                      </div>
                      {Array.isArray(binding.evidence) && binding.evidence.map((ev, evIdx) => (
                        <div key={evIdx} style={{ fontSize: 11, color: '#94a3b8' }}>• {ev}</div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {Array.isArray(executionManifest.assumptions) && executionManifest.assumptions.length > 0 && (
              <div style={{ background: '#102033', border: '1px solid #1d4ed8', borderRadius: 8, padding: 12 }}>
                <div style={{ color: '#7dd3fc', fontSize: 12, fontWeight: 800, marginBottom: 6 }}>Assumptions</div>
                {executionManifest.assumptions.map((item, index) => <div key={index} style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>• {item}</div>)}
              </div>
            )}

            {Array.isArray(executionManifest.warnings) && executionManifest.warnings.length > 0 && (
              <div style={{ background: '#241b08', border: '1px solid #a16207', borderRadius: 8, padding: 12 }}>
                <div style={{ color: '#fbbf24', fontSize: 12, fontWeight: 800, marginBottom: 6 }}>Scientific Warnings</div>
                {executionManifest.warnings.map((item, index) => <div key={index} style={{ color: '#fde68a', fontSize: 12, lineHeight: 1.5 }}>• {item}</div>)}
              </div>
            )}
          </div>
        )}

        {/* TAB 4: Console Output for manually executed Python only */}
        {!isRunning && activeTab === 'stdout' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* Stderr Error Block if Execution Failed */}
            {stderrContent && (
              <div style={{
                background: '#1a0d10',
                border: '1px solid #7f1d1d',
                borderRadius: 6,
                padding: 12,
                fontFamily: 'var(--font-mono, monospace)',
                fontSize: 12,
                color: '#f87171',
                whiteSpace: 'pre-wrap',
                lineHeight: 1.5
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700, marginBottom: 6, color: '#fca5a5' }}>
                  <XCircle size={14} />
                  Traceback / Stderr Output:
                </div>
                {stderrContent}
              </div>
            )}

            {/* Stdout Output Stream Block */}
            <div style={{
              background: '#07080a',
              border: '1px solid #1e293b',
              borderRadius: 6,
              padding: 12,
              fontFamily: 'var(--font-mono, monospace)',
              fontSize: 12,
              color: '#4ade80',
              whiteSpace: 'pre-wrap',
              lineHeight: 1.6,
              minHeight: 120,
              maxHeight: 400,
              overflowY: 'auto'
            }}>
              <div style={{
                fontSize: 10,
                fontWeight: 700,
                color: '#64748b',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                marginBottom: 8,
                display: 'flex',
                justifyContent: 'space-between',
                borderBottom: '1px solid #1e293b',
                paddingBottom: 4
              }}>
                <span>Standard Output Stream (stdout):</span>
                {executionRecord?.timing && <span>Elapsed: {executionRecord.timing}</span>}
              </div>
              {stdoutContent ? (
                <div>{stdoutContent}</div>
              ) : (
                <div style={{ color: '#475569', fontStyle: 'italic' }}>
                  Code executed with 0 stdout prints. No console text captured.
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 5: Extracted Metrics & ML Diagnostics */}
        {!isRunning && activeTab === 'metrics' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {/* Extended ML Evaluation Diagnostics Card */}
            {executionRecord?.extended_metrics && (
              <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#38bdf8', fontSize: 13, fontWeight: 800 }}>
                    <TrendingUp size={16} />
                    <span>Machine Learning Evaluation & Baseline Comparison</span>
                  </div>
                  {executionRecord.extended_metrics.improvement_vs_baseline && (
                    <span style={{
                      background: executionRecord.extended_metrics.improvement_vs_baseline.includes('+') ? 'rgba(74, 222, 128, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                      color: executionRecord.extended_metrics.improvement_vs_baseline.includes('+') ? '#4ade80' : '#f87171',
                      border: `1px solid ${executionRecord.extended_metrics.improvement_vs_baseline.includes('+') ? '#16a34a' : '#dc2626'}`,
                      borderRadius: 4,
                      padding: '2px 8px',
                      fontSize: 11,
                      fontWeight: 700
                    }}>
                      vs Baseline: {executionRecord.extended_metrics.improvement_vs_baseline}
                    </span>
                  )}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8 }}>
                  {Object.entries(executionRecord.extended_metrics).map(([mKey, mVal]) => (
                    <div key={mKey} style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                      <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
                        {mKey.replace(/_/g, ' ')}
                      </div>
                      <div style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 700, fontFamily: 'monospace' }}>
                        {typeof mVal === 'number' ? mVal.toFixed(4) : String(mVal)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Cross-Validation Folds Breakdown */}
            {Array.isArray(executionRecord?.cv_folds) && executionRecord.cv_folds.length > 0 && (
              <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                  <div style={{ color: '#a7f3d0', fontSize: 13, fontWeight: 800, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Layers size={16} />
                    <span>K-Fold Cross-Validation Breakdown ({executionRecord.cv_folds.length} Folds)</span>
                  </div>
                  {executionRecord.cv_mean !== null && executionRecord.cv_mean !== undefined && (
                    <span style={{ fontSize: 11, color: '#38bdf8', background: '#1e293b', padding: '2px 8px', borderRadius: 4, fontWeight: 700 }}>
                      CV Mean: {Number(executionRecord.cv_mean).toFixed(4)} ± {Number(executionRecord.cv_std || 0).toFixed(4)}
                    </span>
                  )}
                </div>

                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {executionRecord.cv_folds.map((foldScore, fIdx) => (
                    <div key={fIdx} style={{
                      flex: 1,
                      minWidth: 90,
                      background: '#13192b',
                      border: '1px solid #1e293b',
                      borderRadius: 6,
                      padding: '8px 10px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: 10, color: '#64748b', fontWeight: 700, textTransform: 'uppercase' }}>Fold {fIdx + 1}</div>
                      <div style={{ fontSize: 13, color: '#38bdf8', fontWeight: 700, fontFamily: 'monospace', marginTop: 4 }}>
                        {Number(foldScore).toFixed(4)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Interactive Decision Threshold Explorer (FW-02) Card */}
            {executionRecord?.threshold_tuning_payload && (() => {
              const payload = executionRecord.threshold_tuning_payload;
              const hasCVSplit = Array.isArray(payload.tuning_split?.probs) && payload.tuning_split.probs.length > 0;
              const activeSplit = (thresholdSplit === 'tuning' && hasCVSplit) ? payload.tuning_split : (payload.eval_split || {
                split_name: 'Held-Out Test Split',
                sample_size: Array.isArray(payload.y_test) ? payload.y_test.length : 0,
                y_true: payload.y_test || [],
                probs: payload.test_probs || [],
                description: 'Pristine held-out test split'
              });

              const yActive = Array.isArray(activeSplit.y_true) ? activeSplit.y_true : [];
              const pActive = Array.isArray(activeSplit.probs) ? activeSplit.probs : [];

              let curTP = 0, curFP = 0, curTN = 0, curFN = 0;
              let baseTP = 0, baseFP = 0, baseTN = 0, baseFN = 0;
              for (let i = 0; i < yActive.length; i++) {
                const act = yActive[i];
                const prob = pActive[i];
                const pred = prob >= thresholdTau ? 1 : 0;
                if (act === 1 && pred === 1) curTP++;
                else if (act === 0 && pred === 0) curTN++;
                else if (act === 0 && pred === 1) curFP++;
                else if (act === 1 && pred === 0) curFN++;

                const bPred = prob >= 0.50 ? 1 : 0;
                if (act === 1 && bPred === 1) baseTP++;
                else if (act === 0 && bPred === 0) baseTN++;
                else if (act === 0 && bPred === 1) baseFP++;
                else if (act === 1 && bPred === 0) baseFN++;
              }

              const curPrec = (curTP + curFP) > 0 ? (curTP / (curTP + curFP)) : 0;
              const curRec = (curTP + curFN) > 0 ? (curTP / (curTP + curFN)) : 0;
              const curF1 = (curPrec + curRec) > 0 ? ((2 * curPrec * curRec) / (curPrec + curRec)) : 0;
              const curSpec = (curTN + curFP) > 0 ? (curTN / (curTN + curFP)) : 0;
              const curBalAcc = (curRec + curSpec) / 2;

              const validCosts = (costFP > 0 || costFN > 0);
              const curCost = validCosts ? (curFP * costFP + curFN * costFN) : 0;
              const baseCost = validCosts ? (baseFP * costFP + baseFN * costFN) : 0;
              const costDiff = baseCost - curCost;
              const pctSavings = (validCosts && baseCost > 0) ? ((costDiff / baseCost) * 100) : 0;
              const bayesTau = (costFP + costFN) > 0 ? (costFP / (costFP + costFN)) : 0.5;

              return (
                <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#38bdf8', fontSize: 13, fontWeight: 800 }}>
                      <Sliders size={16} />
                      <span>Cost-Sensitive Decision Threshold Explorer (FW-02)</span>
                    </div>

                    {/* Split Isolation & Actions Bar */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                      {/* Split Switcher */}
                      {hasCVSplit && (
                        <div style={{ display: 'flex', background: '#13192b', borderRadius: 4, padding: 2, border: '1px solid #1e293b' }}>
                          <button
                            onClick={() => setThresholdSplit('tuning')}
                            style={{
                              background: thresholdSplit === 'tuning' ? '#0284c7' : 'transparent',
                              color: thresholdSplit === 'tuning' ? '#ffffff' : '#94a3b8',
                              border: 'none',
                              borderRadius: 3,
                              padding: '2px 8px',
                              fontSize: 10,
                              fontWeight: 700,
                              cursor: 'pointer'
                            }}
                            title="Tune threshold on out-of-fold cross-validation predictions without test-set leakage"
                          >
                            Tuning Split (5-Fold CV)
                          </button>
                          <button
                            onClick={() => setThresholdSplit('eval')}
                            style={{
                              background: thresholdSplit === 'eval' ? '#0284c7' : 'transparent',
                              color: thresholdSplit === 'eval' ? '#ffffff' : '#94a3b8',
                              border: 'none',
                              borderRadius: 3,
                              padding: '2px 8px',
                              fontSize: 10,
                              fontWeight: 700,
                              cursor: 'pointer'
                            }}
                            title="Evaluate chosen threshold on pristine untouched held-out test split"
                          >
                            Evaluation Split (Held-Out Test)
                          </button>
                        </div>
                      )}

                      <button
                        onClick={() => setThresholdTau(Number(bayesTau.toFixed(2)))}
                        style={{ background: '#1e293b', color: '#7dd3fc', border: '1px solid #0284c7', borderRadius: 4, padding: '2px 8px', fontSize: 10, fontWeight: 700, cursor: 'pointer' }}
                        title="Set threshold to theoretical Bayes optimal: C_FP / (C_FP + C_FN) (assumes calibrated probabilities)"
                      >
                        Set Bayes Optimal (τ* = {bayesTau.toFixed(2)})
                      </button>
                      <button
                        onClick={() => { setThresholdTau(0.50); setCostFP(1); setCostFN(5); }}
                        style={{ background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 10, fontWeight: 600, cursor: 'pointer' }}
                        title="Reset threshold to default 0.50"
                      >
                        Reset (0.50)
                      </button>
                    </div>
                  </div>

                  {/* Active Split Identification Banner */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: thresholdSplit === 'tuning' ? 'rgba(56, 189, 248, 0.08)' : 'rgba(16, 185, 129, 0.08)',
                    border: `1px solid ${thresholdSplit === 'tuning' ? 'rgba(56, 189, 248, 0.25)' : 'rgba(16, 185, 129, 0.25)'}`,
                    borderRadius: 6,
                    padding: '6px 10px',
                    marginBottom: 12,
                    fontSize: 11
                  }}>
                    <span style={{ color: thresholdSplit === 'tuning' ? '#38bdf8' : '#34d399', fontWeight: 700 }}>
                      Active Split: {activeSplit.split_name} (N = {activeSplit.sample_size})
                    </span>
                    <span style={{ color: '#94a3b8', fontSize: 10, fontStyle: 'italic' }}>
                      {thresholdSplit === 'tuning'
                        ? 'Methodological guard: Select threshold here using out-of-fold CV to prevent test-set leakage.'
                        : 'Final evaluation: Evaluating generalization on held-out test split without retraining.'}
                    </span>
                  </div>

                  {/* Threshold Slider and Cost Parameter Controls */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 14, marginBottom: 14 }}>
                    <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8' }}>Decision Threshold (τ):</span>
                        <span style={{ fontSize: 13, fontWeight: 800, color: '#38bdf8', fontFamily: 'monospace' }}>{thresholdTau.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.01"
                        max="0.99"
                        step="0.01"
                        value={thresholdTau}
                        onChange={(e) => setThresholdTau(parseFloat(e.target.value))}
                        style={{ width: '100%', accentColor: '#38bdf8', cursor: 'pointer' }}
                      />
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#64748b', marginTop: 4 }}>
                        <span>0.01 (High Recall / Aggressive)</span>
                        <span>0.50 (Default)</span>
                        <span>0.99 (High Precision / Conservative)</span>
                      </div>
                    </div>

                    <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: 12, display: 'flex', gap: 10 }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8', marginBottom: 4 }}>Cost FP (C_FP)</div>
                        <input
                          type="number"
                          min="0"
                          step="1"
                          value={costFP}
                          onChange={(e) => setCostFP(Math.max(0, parseFloat(e.target.value) || 0))}
                          style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', color: '#f8fafc', fontSize: 12, fontFamily: 'monospace' }}
                        />
                        <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>False alarm penalty</div>
                      </div>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8', marginBottom: 4 }}>Cost FN (C_FN)</div>
                        <input
                          type="number"
                          min="0"
                          step="1"
                          value={costFN}
                          onChange={(e) => setCostFN(Math.max(0, parseFloat(e.target.value) || 0))}
                          style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', color: '#f8fafc', fontSize: 12, fontFamily: 'monospace' }}
                        />
                        <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>Missed positive penalty</div>
                      </div>
                    </div>
                  </div>

                  {/* 2x2 Interactive Confusion Matrix & Metrics Grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: 12, marginBottom: 12 }}>
                    {/* 2x2 Matrix */}
                    <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: 10 }}>
                      <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', marginBottom: 8 }}>
                        {activeSplit.split_name} Confusion Matrix (τ = {thresholdTau.toFixed(2)})
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, textAlign: 'center' }}>
                        <div style={{ background: 'rgba(56, 189, 248, 0.12)', border: '1px solid rgba(56, 189, 248, 0.25)', borderRadius: 4, padding: 8 }}>
                          <div style={{ fontSize: 9, color: '#7dd3fc', fontWeight: 600 }}>True Positives (TP)</div>
                          <div style={{ fontSize: 15, fontWeight: 800, color: '#38bdf8', fontFamily: 'monospace', marginTop: 2 }}>{curTP}</div>
                        </div>
                        <div style={{ background: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.25)', borderRadius: 4, padding: 8 }}>
                          <div style={{ fontSize: 9, color: '#fca5a5', fontWeight: 600 }}>False Positives (FP)</div>
                          <div style={{ fontSize: 15, fontWeight: 800, color: '#f87171', fontFamily: 'monospace', marginTop: 2 }}>{curFP}</div>
                        </div>
                        <div style={{ background: 'rgba(245, 158, 11, 0.12)', border: '1px solid rgba(245, 158, 11, 0.25)', borderRadius: 4, padding: 8 }}>
                          <div style={{ fontSize: 9, color: '#fcd34d', fontWeight: 600 }}>False Negatives (FN)</div>
                          <div style={{ fontSize: 15, fontWeight: 800, color: '#fbbf24', fontFamily: 'monospace', marginTop: 2 }}>{curFN}</div>
                        </div>
                        <div style={{ background: 'rgba(16, 185, 129, 0.12)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: 4, padding: 8 }}>
                          <div style={{ fontSize: 9, color: '#6ee7b7', fontWeight: 600 }}>True Negatives (TN)</div>
                          <div style={{ fontSize: 15, fontWeight: 800, color: '#34d399', fontFamily: 'monospace', marginTop: 2 }}>{curTN}</div>
                        </div>
                      </div>
                    </div>

                    {/* Operational Metrics */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                      <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 4, padding: '6px 8px' }}>
                        <div style={{ fontSize: 9, color: '#64748b', fontWeight: 700 }}>Precision</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc', fontFamily: 'monospace' }}>{(curPrec * 100).toFixed(1)}%</div>
                      </div>
                      <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 4, padding: '6px 8px' }}>
                        <div style={{ fontSize: 9, color: '#64748b', fontWeight: 700 }}>Recall (Sensitivity)</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc', fontFamily: 'monospace' }}>{(curRec * 100).toFixed(1)}%</div>
                      </div>
                      <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 4, padding: '6px 8px' }}>
                        <div style={{ fontSize: 9, color: '#64748b', fontWeight: 700 }}>F1 Score</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc', fontFamily: 'monospace' }}>{curF1.toFixed(4)}</div>
                      </div>
                      <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 4, padding: '6px 8px' }}>
                        <div style={{ fontSize: 9, color: '#64748b', fontWeight: 700 }}>Balanced Accuracy</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc', fontFamily: 'monospace' }}>{(curBalAcc * 100).toFixed(1)}%</div>
                      </div>
                      {payload.brier_score !== undefined && (
                        <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 4, padding: '6px 8px' }}>
                          <div style={{ fontSize: 9, color: '#64748b', fontWeight: 700 }}>Brier Score (MSE Error)</div>
                          <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', fontFamily: 'monospace' }}>{Number(payload.brier_score).toFixed(4)}</div>
                        </div>
                      )}
                      {payload.roc_auc !== undefined && payload.roc_auc !== null && (
                        <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 4, padding: '6px 8px' }}>
                          <div style={{ fontSize: 9, color: '#64748b', fontWeight: 700 }}>ROC-AUC / PR-AUC</div>
                          <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', fontFamily: 'monospace' }}>{Number(payload.roc_auc).toFixed(3)} / {Number(payload.pr_auc || 0).toFixed(3)}</div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Asymmetric Cost Reduction Banner */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: costDiff > 0 ? 'rgba(16, 185, 129, 0.12)' : costDiff < 0 ? 'rgba(239, 68, 68, 0.12)' : 'rgba(30, 41, 59, 0.5)',
                    border: `1px solid ${costDiff > 0 ? '#10b981' : costDiff < 0 ? '#ef4444' : '#334155'}`,
                    borderRadius: 6,
                    padding: '8px 12px'
                  }}>
                    <div>
                      <span style={{ fontSize: 11, fontWeight: 700, color: costDiff > 0 ? '#34d399' : costDiff < 0 ? '#f87171' : '#94a3b8' }}>
                        Decision Cost on {activeSplit.split_name}: <strong>{curCost}</strong> (vs Baseline τ=0.50 Cost: <strong>{baseCost}</strong>)
                      </span>
                      <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>
                        {costDiff > 0 ? `Cost Reduction: ${costDiff} (${pctSavings.toFixed(1)}% savings vs default 0.50 threshold)` : costDiff < 0 ? `Cost Increase: +${Math.abs(costDiff)} vs default threshold` : 'Equal cost to baseline threshold'}
                      </div>
                    </div>
                    <span style={{ fontSize: 10, color: '#64748b', fontStyle: 'italic' }}>
                      {thresholdSplit === 'tuning' ? 'Tuned on CV split without test leakage' : 'Pristine test split evaluation'}
                    </span>
                  </div>
                </div>
              );
            })()}

            {/* Regression Residual Diagnostics Card */}
            {executionRecord?.residual_diagnostics && (
              <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#f59e0b', fontSize: 13, fontWeight: 800 }}>
                    <Activity size={16} />
                    <span>Regression Residual Diagnostics</span>
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {executionRecord.residual_diagnostics.durbin_watson !== undefined && (
                      <span style={{
                        background: executionRecord.residual_diagnostics.durbin_watson >= 1.5 && executionRecord.residual_diagnostics.durbin_watson <= 2.5
                          ? 'rgba(74, 222, 128, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                        color: executionRecord.residual_diagnostics.durbin_watson >= 1.5 && executionRecord.residual_diagnostics.durbin_watson <= 2.5
                          ? '#4ade80' : '#f87171',
                        border: `1px solid ${executionRecord.residual_diagnostics.durbin_watson >= 1.5 && executionRecord.residual_diagnostics.durbin_watson <= 2.5 ? '#16a34a' : '#dc2626'}`,
                        borderRadius: 4,
                        padding: '2px 8px',
                        fontSize: 10,
                        fontWeight: 700
                      }}>
                        DW: {Number(executionRecord.residual_diagnostics.durbin_watson).toFixed(3)} ({
                          executionRecord.residual_diagnostics.durbin_watson < 1.5 ? 'Positive Autocorr' :
                          executionRecord.residual_diagnostics.durbin_watson > 2.5 ? 'Negative Autocorr' : 'No Autocorr'
                        })
                      </span>
                    )}
                    {executionRecord.residual_diagnostics.spearman_pvalue !== undefined && (
                      <span style={{
                        background: executionRecord.residual_diagnostics.spearman_pvalue >= 0.05
                          ? 'rgba(74, 222, 128, 0.15)' : 'rgba(251, 191, 36, 0.15)',
                        color: executionRecord.residual_diagnostics.spearman_pvalue >= 0.05 ? '#4ade80' : '#fbbf24',
                        border: `1px solid ${executionRecord.residual_diagnostics.spearman_pvalue >= 0.05 ? '#16a34a' : '#d97706'}`,
                        borderRadius: 4,
                        padding: '2px 8px',
                        fontSize: 10,
                        fontWeight: 700
                      }}>
                        Heteroscedasticity p: {Number(executionRecord.residual_diagnostics.spearman_pvalue).toFixed(4)} ({
                          executionRecord.residual_diagnostics.spearman_pvalue >= 0.05 ? 'Homoscedastic' : 'Heteroscedastic'
                        })
                      </span>
                    )}
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 8 }}>
                  {Object.entries(executionRecord.residual_diagnostics)
                    .filter(([k]) => !['durbin_watson', 'spearman_pvalue'].includes(k))
                    .map(([k, v]) => (
                      <div key={k} style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                        <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
                          {k.replace(/_/g, ' ')}
                        </div>
                        <div style={{ color: '#f8fafc', fontSize: 13, fontWeight: 700, fontFamily: 'monospace' }}>
                          {typeof v === 'number' ? v.toFixed(4) : String(v)}
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Inferential Hypothesis Testing Diagnostics Card */}
            {((executionRecord?.metrics && ('p_value' in executionRecord.metrics || 'test_statistic' in executionRecord.metrics)) || executionRecord?.statistical_test) && (
              <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#a7f3d0', fontSize: 13, fontWeight: 800 }}>
                    <CheckSquare size={16} />
                    <span>Inferential Hypothesis Testing Diagnostics</span>
                  </div>
                  {executionRecord?.metrics?.p_value !== undefined && (
                    <span style={{
                      background: Number(executionRecord.metrics.p_value) < 0.05 ? 'rgba(74, 222, 128, 0.15)' : 'rgba(148, 163, 184, 0.15)',
                      color: Number(executionRecord.metrics.p_value) < 0.05 ? '#4ade80' : '#94a3b8',
                      border: `1px solid ${Number(executionRecord.metrics.p_value) < 0.05 ? '#16a34a' : '#475569'}`,
                      borderRadius: 4,
                      padding: '2px 8px',
                      fontSize: 11,
                      fontWeight: 700
                    }}>
                      {Number(executionRecord.metrics.p_value) < 0.05 ? 'p < 0.05 (Statistically Significant)' : 'p ≥ 0.05 (Fail to Reject H₀)'}
                    </span>
                  )}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 8 }}>
                  {executionRecord?.metrics?.test_statistic !== undefined && (
                    <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                      <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>Test Statistic</div>
                      <div style={{ color: '#38bdf8', fontSize: 13, fontWeight: 700, fontFamily: 'monospace' }}>
                        {Number(executionRecord.metrics.test_statistic).toFixed(4)}
                      </div>
                    </div>
                  )}
                  {executionRecord?.metrics?.p_value !== undefined && (
                    <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                      <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>p-value</div>
                      <div style={{ color: Number(executionRecord.metrics.p_value) < 0.05 ? '#4ade80' : '#f8fafc', fontSize: 13, fontWeight: 700, fontFamily: 'monospace' }}>
                        {Number(executionRecord.metrics.p_value).toFixed(6)}
                      </div>
                    </div>
                  )}
                  {executionRecord?.metrics?.effect_size !== undefined && (
                    <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                      <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>Effect Size</div>
                      <div style={{ color: '#f8fafc', fontSize: 13, fontWeight: 700, fontFamily: 'monospace' }}>
                        {typeof executionRecord.metrics.effect_size === 'number' ? executionRecord.metrics.effect_size.toFixed(4) : String(executionRecord.metrics.effect_size)}
                      </div>
                    </div>
                  )}
                  {executionRecord?.metrics?.confidence_interval !== undefined && (
                    <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                      <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>95% Confidence Interval</div>
                      <div style={{ color: '#f8fafc', fontSize: 13, fontWeight: 700, fontFamily: 'monospace' }}>
                        {Array.isArray(executionRecord.metrics.confidence_interval)
                          ? `[${executionRecord.metrics.confidence_interval.map(v => typeof v === 'number' ? v.toFixed(3) : v).join(', ')}]`
                          : String(executionRecord.metrics.confidence_interval)}
                      </div>
                    </div>
                  )}
                  {(executionRecord?.metrics?.degrees_of_freedom !== undefined || executionRecord?.metrics?.df !== undefined) && (
                    <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                      <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>Degrees of Freedom (df)</div>
                      <div style={{ color: '#f8fafc', fontSize: 13, fontWeight: 700, fontFamily: 'monospace' }}>
                        {Number(executionRecord?.metrics?.degrees_of_freedom ?? executionRecord?.metrics?.df)}
                      </div>
                    </div>
                  )}
                </div>

                {/* H0 / H1 Hypothesis Statements */}
                <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                    <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 2 }}>Null Hypothesis (H₀)</div>
                    <div style={{ color: '#cbd5e1', fontSize: 11, fontStyle: 'italic' }}>
                      {executionRecord?.metrics?.null_hypothesis || 'No statistically significant difference or effect (μ₁ = μ₂ / effect = 0)'}
                    </div>
                  </div>
                  <div style={{ background: '#13192b', border: '1px solid #1e293b', borderRadius: 6, padding: '8px 10px' }}>
                    <div style={{ color: '#64748b', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', marginBottom: 2 }}>Alternative Hypothesis (H₁)</div>
                    <div style={{ color: '#cbd5e1', fontSize: 11, fontStyle: 'italic' }}>
                      {executionRecord?.metrics?.alt_hypothesis || 'Statistically significant difference or effect present (μ₁ ≠ μ₂ / effect ≠ 0)'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Executed SQL Query Card */}
            {executionRecord?.executed_sql && (
              <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, flexWrap: 'wrap', gap: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#38bdf8', fontSize: 13, fontWeight: 800 }}>
                    <Terminal size={16} />
                    <span>Executed SQL Query</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {executionRecord?.sql_metadata?.execution_time_ms !== undefined && (
                      <span style={{ fontSize: 10, color: '#94a3b8', background: '#13192b', border: '1px solid #1e293b', padding: '2px 7px', borderRadius: 4, fontFamily: 'monospace' }}>
                        {Number(executionRecord.sql_metadata.execution_time_ms).toFixed(1)} ms
                      </span>
                    )}
                    {executionRecord?.sql_metadata?.row_count !== undefined && (
                      <span style={{ fontSize: 10, color: '#94a3b8', background: '#13192b', border: '1px solid #1e293b', padding: '2px 7px', borderRadius: 4, fontFamily: 'monospace' }}>
                        {Number(executionRecord.sql_metadata.row_count).toLocaleString()} rows
                      </span>
                    )}
                    <button
                      onClick={() => handleCopySQL(executionRecord.executed_sql)}
                      style={{
                        background: '#1e293b',
                        color: copiedSQL ? '#4ade80' : '#e2e8f0',
                        border: '1px solid #334155',
                        borderRadius: 4,
                        padding: '2px 8px',
                        fontSize: 10,
                        fontWeight: 600,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 4
                      }}
                      title="Copy SQL Query to Clipboard"
                    >
                      {copiedSQL ? <Check size={11} color="#4ade80" /> : <Copy size={11} />}
                      {copiedSQL ? 'Copied!' : 'Copy SQL'}
                    </button>
                  </div>
                </div>

                <pre style={{
                  background: '#070a13',
                  border: '1px solid #1e293b',
                  borderRadius: 6,
                  padding: 12,
                  fontFamily: 'var(--font-mono, monospace)',
                  fontSize: 11,
                  color: '#7dd3fc',
                  overflowX: 'auto',
                  whiteSpace: 'pre-wrap',
                  lineHeight: 1.5,
                  margin: 0
                }}>
                  {executionRecord.executed_sql}
                </pre>
              </div>
            )}

            {/* DuckDB Query Execution Plan Card */}
            {executionRecord?.explain_plan && (
              <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                <div
                  onClick={() => setShowExplainPlan(!showExplainPlan)}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', userSelect: 'none' }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#38bdf8', fontSize: 13, fontWeight: 800 }}>
                    <Terminal size={16} />
                    <span>DuckDB Physical Query Plan (EXPLAIN)</span>
                  </div>
                  <span style={{ fontSize: 11, color: '#94a3b8', background: '#1e293b', padding: '2px 8px', borderRadius: 4 }}>
                    {showExplainPlan ? 'Hide Execution Tree ▲' : 'Show Execution Tree ▼'}
                  </span>
                </div>

                {showExplainPlan && (
                  <pre style={{
                    marginTop: 12,
                    background: '#070a13',
                    border: '1px solid #1e293b',
                    borderRadius: 6,
                    padding: 12,
                    fontFamily: 'var(--font-mono, monospace)',
                    fontSize: 11,
                    color: '#7dd3fc',
                    overflowX: 'auto',
                    whiteSpace: 'pre',
                    lineHeight: 1.5,
                    maxHeight: 280,
                    margin: '12px 0 0'
                  }}>
                    {executionRecord.explain_plan}
                  </pre>
                )}
              </div>
            )}

            {/* Standard Key-Value Metrics */}
            {executionRecord?.metrics && Object.keys(executionRecord.metrics).length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Pipeline Evidence Metrics
                </div>
                {Object.entries(executionRecord.metrics).map(([key, value]) => (
                  <div key={key} style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    background: '#0c101d',
                    border: '1px solid #1e293b',
                    borderRadius: 6
                  }}>
                    <span style={{ fontSize: 13, color: '#94a3b8' }}>{key}</span>
                    <span style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', fontFamily: 'monospace' }}>
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (!executionRecord?.extended_metrics && !executionRecord?.cv_folds?.length && !executionRecord?.residual_diagnostics) ? (
              <div style={{ textAlign: 'center', padding: '36px 0', color: '#64748b' }}>
                <FileText size={28} color="#334155" style={{ marginBottom: 8 }} />
                <p style={{ margin: 0, fontSize: 13 }}>No key-value metrics extracted from execution.</p>
              </div>
            ) : null}
          </div>
        )}

        {/* TAB 6: Validation Report */}
        {!isRunning && activeTab === 'validation' && (
          <div>
            {validationReport ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#0c101d', padding: '10px 14px', borderRadius: 6, border: '1px solid #1e293b' }}>
                  <span style={{ fontWeight: 700, fontSize: 13 }}>Evidence Sufficiency:</span>
                  <span style={{
                    background: validationReport.evidence_sufficient ? 'rgba(74, 222, 128, 0.15)' : 'rgba(251, 191, 36, 0.15)',
                    color: validationReport.evidence_sufficient ? '#4ade80' : '#fbbf24',
                    padding: '2px 8px',
                    borderRadius: 4,
                    fontSize: 12,
                    fontWeight: 700,
                    border: `1px solid ${validationReport.evidence_sufficient ? '#16a34a' : '#d97706'}`
                  }}>
                    {validationReport.evidence_sufficient ? 'Sufficient ✔' : 'Partial Evidence ⚠'}
                  </span>
                </div>

                {validationReport.checks && validationReport.checks.length > 0 && (
                  <div style={{ background: '#0c101d', padding: 12, borderRadius: 6, border: '1px solid #1e293b' }}>
                    <h4 style={{ margin: '0 0 10px', fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Validation Checks ({validationReport.checks.length})
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {validationReport.checks.map((check, idx) => (
                        <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                          {check.passed ? <CheckCircle size={14} color="#4ade80" /> : <XCircle size={14} color="#f87171" />}
                          <span style={{ color: check.passed ? '#cbd5e1' : '#f87171' }}>{check.description || check.name || JSON.stringify(check)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {!validationReport.evidence_sufficient && validationReport.missing_evidence?.length > 0 && (
                  <div style={{ background: '#1c1507', padding: 12, borderRadius: 6, border: '1px solid #78350f' }}>
                    <h4 style={{ margin: '0 0 8px', fontSize: 12, color: '#fbbf24', textTransform: 'uppercase' }}>Missing Evidence</h4>
                    <ul style={{ paddingLeft: 18, fontSize: 12, color: '#fde68a', margin: 0 }}>
                      {validationReport.missing_evidence.map((item, idx) => (
                        <li key={idx} style={{ marginBottom: 4 }}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '36px 0', color: '#64748b' }}>
                <CheckSquare size={28} color="#334155" style={{ marginBottom: 8 }} />
                <p style={{ margin: 0, fontSize: 13 }}>No validation report available for this execution.</p>
              </div>
            )}
          </div>
        )}
      </div>
      )}

      {/* Row Inspection Modal Drawer */}
      {selectedRowInspection && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.75)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backdropFilter: 'blur(3px)'
          }}
          onClick={() => setSelectedRowInspection(null)}
        >
          <div
            style={{
              background: '#0c101d',
              border: '1px solid #334155',
              borderRadius: 8,
              width: '90%',
              maxWidth: 680,
              maxHeight: '80vh',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 20px 40px rgba(0,0,0,0.6)',
              overflow: 'hidden'
            }}
            onClick={e => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{
              padding: '12px 16px',
              background: '#13192b',
              borderBottom: '1px solid #1e293b',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Eye size={16} color="#38bdf8" />
                <span style={{ fontSize: 13, fontWeight: 800, color: '#f8fafc' }}>
                  Row #{selectedRowInspection.rowIdx} Inspection — {selectedRowInspection.tableName}
                </span>
              </div>
              <button
                onClick={() => setSelectedRowInspection(null)}
                style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', padding: 4 }}
              >
                <X size={16} />
              </button>
            </div>

            {/* Modal Body */}
            <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#0f172a', borderBottom: '1px solid #1e293b' }}>
                    <th style={{ padding: '6px 12px', textAlign: 'left', color: '#64748b', width: '35%' }}>Field Name</th>
                    <th style={{ padding: '6px 12px', textAlign: 'left', color: '#64748b' }}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedRowInspection.columns.map((col, idx) => {
                    const val = Array.isArray(selectedRowInspection.row) ? selectedRowInspection.row[idx] : selectedRowInspection.row[col];
                    const isNull = val === null || val === undefined;
                    return (
                      <tr key={idx} style={{ borderBottom: '1px solid #1e293b', background: idx % 2 === 0 ? '#090d16' : '#0e1424' }}>
                        <td style={{ padding: '6px 12px', fontWeight: 600, color: '#38bdf8', fontFamily: 'monospace' }}>
                          {String(col)}
                        </td>
                        <td style={{ padding: '6px 12px', color: isNull ? '#ef4444' : '#cbd5e1', fontFamily: 'monospace' }}>
                          {isNull ? (
                            <span style={{ fontStyle: 'italic' }}>null</span>
                          ) : typeof val === 'boolean' ? (
                            <span style={{ color: val ? '#4ade80' : '#f87171', fontWeight: 700 }}>{String(val)}</span>
                          ) : (
                            String(val)
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Modal Footer */}
            <div style={{
              padding: '8px 16px',
              background: '#0f172a',
              borderTop: '1px solid #1e293b',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span style={{ fontSize: 11, color: '#64748b' }}>
                {selectedRowInspection.columns.length} columns inspected
              </span>
              <button
                onClick={() => {
                  const dataObj = {};
                  selectedRowInspection.columns.forEach((c, i) => {
                    dataObj[c] = Array.isArray(selectedRowInspection.row) ? selectedRowInspection.row[i] : selectedRowInspection.row[c];
                  });
                  navigator.clipboard.writeText(JSON.stringify(dataObj, null, 2));
                  alert('Copied row data as JSON to clipboard');
                }}
                style={{
                  background: '#1e293b',
                  color: '#38bdf8',
                  border: '1px solid #334155',
                  borderRadius: 4,
                  padding: '4px 10px',
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4
                }}
              >
                <Copy size={12} />
                Copy JSON
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
