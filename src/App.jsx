import React, { useState, useEffect, useCallback } from 'react';
import {
  FolderOpen, ChevronUp, ChevronDown, Terminal, Database, Brain, Lock,
  Code as CodeIcon, BarChart2, Layers, GitBranch, History, Play, Sparkles, Activity,
  PlugZap, Eraser, Shuffle, LineChart, BrainCircuit, BarChart3, Code2, Table,
  FileSpreadsheet, UploadCloud, CheckCircle2, ArrowRight, RefreshCw, Plus
} from 'lucide-react';

import './index.css';

// Components
import TitleBar from './components/TitleBar';
import StatusBar from './components/StatusBar';
import TerminalHUD from './components/TerminalHUD';
import AICoPilot from './components/AICoPilot';
import Ribbon from './components/Ribbon';
import Sidebar from './components/Sidebar';
import DatasetManager from './components/DatasetManager';
import IntentPanel from './components/IntentPanel';
import AnalysisGraph from './components/AnalysisGraph';
import CodePanel from './components/CodePanel';
import ResultPanel from './components/ResultPanel';
import InsightPanel from './components/InsightPanel';
import ProductionPanel from './components/ProductionPanel';
import WorkflowHistory from './components/WorkflowHistory';
import DBConnectionModal from './components/DBConnectionModal';
import PlotZoomModal from './components/PlotZoomModal';
import DataPreview from './components/DataPreview';
import OperationPanel from './components/OperationPanel';
import DataScienceStudio from './components/DataScienceStudio';
import ReportView from './components/ReportView';
import SqlStudio from './components/SqlStudio';
import ReportingStudio from './components/ReportingStudio';
import AgentStudio from './components/AgentStudio';
import { AppProvider, useWorkflow, useDatasets } from './state/AppContext';
import { API, getAuthHeaders, fetchApiToken, formatApiError } from './apiConfig';
import { logUserAction } from './utils/telemetry';

function AppContent() {
  const { state: workflowState, dispatch: workflowDispatch } = useWorkflow();
  const { state: datasetState, dispatch: datasetDispatch } = useDatasets();

  // Top navigation active module: 'home' | 'preview' | 'connect' | 'clean' | 'transform' | 'stats' | 'ml' | 'reports' | 'code'
  const [activeModule, setActiveModule] = useState('home');

  // Legacy state bridges
  const [activeDataset, setActiveDataset] = useState(null);
  const [schema, setSchema] = useState(null);
  const [rowCount, setRowCount] = useState(0);
  const [tableData, setTableData] = useState([]);
  const [datasetProfile, setDatasetProfile] = useState(null);
  // The desktop shell starts the local Python service after the renderer.  Keep
  // the initial state distinct from a real outage so the UI does not report a
  // false "Ollama Offline" error while that service is booting.
  const [engineStatus, setEngineStatus] = useState('starting');
  const [backendLifecycle, setBackendLifecycle] = useState(() => (
    typeof window !== 'undefined' && window.electron?.getBackendStatus ? 'starting' : 'unknown'
  ));
  const [isSaving, setIsSaving] = useState(false);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [pipelineSteps, setPipelineSteps] = useState([]);
  const [theme, setTheme] = useState(() => localStorage.getItem('datapilot_theme') || 'dark');
  const [isRibbonCollapsed, setIsRibbonCollapsed] = useState(false);

  // Tab & Panel States
  const [leftTab, setLeftTab] = useState('datasets'); // 'datasets' | 'graph' | 'history'
  const [centerBottomTab, setCenterBottomTab] = useState('results'); // 'results' | 'terminal' | 'production'
  const [externalCodeToRun, setExternalCodeToRun] = useState(null);
  const [zoomedImage, setZoomedImage] = useState(null);
  const [isDbModalOpen, setIsDbModalOpen] = useState(false);
  const [workflows, setWorkflows] = useState([]);
  const [productionCheck, setProductionCheck] = useState(null);
  const [isProductionChecking, setIsProductionChecking] = useState(false);
  const [isProductionExecuting, setIsProductionExecuting] = useState(false);
  const [isRightPanelCollapsed, setIsRightPanelCollapsed] = useState(() => {
    return localStorage.getItem('datapilot_right_panel_collapsed') === 'true';
  });

  const toggleRightPanel = () => {
    setIsRightPanelCollapsed(prev => {
      const next = !prev;
      localStorage.setItem('datapilot_right_panel_collapsed', String(next));
      return next;
    });
  };

  // Theme Sync
  useEffect(() => {
    localStorage.setItem('datapilot_theme', theme);
  }, [theme]);
  const toggleTheme = () => setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Allow Esc to close modals even if focused on an input
      if (e.key === 'Escape') {
        setIsDbModalOpen(false);
        setZoomedImage(null);
      }

      // Do not interfere with typing in input/textarea elements
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
        return;
      }

      if (e.ctrlKey && e.shiftKey) {
        switch (e.key.toLowerCase()) {
          case 'h':
            e.preventDefault();
            setActiveModule('home');
            break;
          case 'g':
            e.preventDefault();
            setActiveModule('preview');
            break;
          case 'c':
            e.preventDefault();
            setActiveModule('connect');
            break;
          case 'k':
            e.preventDefault();
            const chatInput = document.querySelector('input[placeholder*="Ask AI Data Scientist"]');
            if (chatInput) {
              chatInput.focus();
            }
            break;
          default:
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Electron, rather than a guessed wall-clock timeout, owns the local
  // backend lifecycle. This keeps a real cold start distinct from a failure.
  useEffect(() => {
    const bridge = window.electron;
    if (!bridge?.getBackendStatus || !bridge?.onBackendStatusChange) return undefined;
    let active = true;
    bridge.getBackendStatus()
      .then((status) => { if (active && status) setBackendLifecycle(status); })
      .catch(() => { if (active) setBackendLifecycle('failed'); });
    const unsubscribe = bridge.onBackendStatusChange((status) => {
      if (active && status) setBackendLifecycle(status);
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);

  // 1. Health Check Polling (Adaptive interval for fast startup detection)
  useEffect(() => {
    let isMounted = true;
    let timerId = null;

    const reportUnavailable = (statusCode = 'offline') => {
      const isStarting = backendLifecycle === 'starting';
      setEngineStatus(isStarting ? 'starting' : statusCode);
      workflowDispatch({
        type: 'SET_OLLAMA_STATUS',
        payload: {
          available: false,
          status: {
            online: false,
            phase: isStarting ? 'starting' : 'unavailable',
            model: isStarting ? '' : 'unavailable'
          }
        }
      });
    };

    const checkHealth = async () => {
      try {
        const headers = await getAuthHeaders();
        const r = await fetch(`${API}/health`, { headers });
        if (r.ok) {
          const data = await r.json();
          if (!isMounted) return;
          setEngineStatus('connected');
          const isPlannerReady = Boolean(data.intent_planner_ready && data.ollama_ready && data.model_ready);
          workflowDispatch({
            type: 'SET_OLLAMA_STATUS',
            payload: {
              available: isPlannerReady,
              status: {
                online: isPlannerReady,
                phase: isPlannerReady ? 'ready' : 'unavailable',
                model: data.configured_model || data.ollama_status?.model || 'qwen2.5:3b-instruct'
              }
            }
          });
          timerId = setTimeout(checkHealth, 8000);
        } else {
          if (!isMounted) return;
          reportUnavailable('disconnected');
          timerId = setTimeout(checkHealth, 800);
        }
      } catch {
        if (!isMounted) return;
        reportUnavailable('offline');
        timerId = setTimeout(checkHealth, 800);
      }
    };

    checkHealth();
    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [workflowDispatch, backendLifecycle]);

  // Fetch Table Preview Data
  const fetchTableData = useCallback(async (datasetId = null) => {
    try {
      const query = datasetId ? `?limit=100&dataset_id=${encodeURIComponent(datasetId)}` : '?limit=100';
      const res = await fetch(`${API}/api/datasets/data${query}`, { headers: getAuthHeaders() });
      if (res.ok) {
        const data = await res.json();
        const rows = data.rows || data.data || [];
        setTableData(rows);
        if (data.profile) {
          setDatasetProfile(data.profile);
        }
      }
    } catch (err) {
      console.warn('Could not fetch table preview data:', err);
    }
  }, []);

  // Fetch Dataset Profile
  const fetchDatasetProfile = useCallback(async (datasetId = null) => {
    try {
      const query = datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : '';
      const res = await fetch(`${API}/api/datasets/profile${query}`, { headers: getAuthHeaders() });
      if (res.ok) {
        const prof = await res.json();
        setDatasetProfile(prof);
        return prof;
      }
    } catch (err) {
      console.warn('Could not fetch dataset profile:', err);
    }
    return null;
  }, []);

  // 2. Initial Dataset Hydration & Refresh
  const fetchDatasets = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/datasets`, { headers: getAuthHeaders() });
      if (res.ok) {
        const list = await res.json();
        if (Array.isArray(list) && list.length > 0) {
          datasetDispatch({ type: 'SET_DATASETS', payload: list });
          const activeEntry = list.find(d => d.is_active) || list[0];
          if (activeEntry) {
            const id = activeEntry.dataset_id || activeEntry.id;
            let fullCols = activeEntry.columns || [];
            let fullRowCount = activeEntry.row_count ?? activeEntry.rowCount ?? 0;
            try {
              const fullDsRes = await fetch(`${API}/api/datasets/${id}`, { headers: getAuthHeaders() });
              if (fullDsRes.ok) {
                const fullDs = await fullDsRes.json();
                fullCols = fullDs.columns || fullCols;
                fullRowCount = fullDs.row_count ?? fullRowCount;
              }
            } catch (fetchErr) {}

            const ds = {
              id,
              name: activeEntry.name,
              rowCount: fullRowCount,
              colCount: Array.isArray(fullCols) ? fullCols.length : Object.keys(fullCols || {}).length,
              columns: fullCols,
              loadedAt: activeEntry.loaded_at
            };
            setActiveDataset(ds);
            setSchema(fullCols);
            setRowCount(fullRowCount);
            fetchTableData(id);
            fetchDatasetProfile(id);
          }
        }
      }
    } catch (e) {
      console.warn('Dataset hydration skipped/failed:', e);
    }
  }, [datasetDispatch, fetchTableData, fetchDatasetProfile]);

  const fetchWorkflows = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/workflows`, { headers: getAuthHeaders() });
      if (!res.ok) return;
      const data = await res.json();
      setWorkflows(Array.isArray(data) ? data : []);
    } catch (err) {
      console.warn('Workflow history could not be loaded:', err);
    }
  }, []);

  useEffect(() => {
    if (engineStatus === 'connected') {
      fetchDatasets();
      fetchWorkflows();
    }
  }, [engineStatus, fetchDatasets, fetchWorkflows]);

  // 3. Dataset Connection Handler
  const handleConnect = async (rawInput) => {
    if (engineStatus !== 'connected') {
      const stateMessage = engineStatus === 'starting'
        ? 'DataPilot local service is still starting. Please wait until the AI Brain shows Online, then connect the dataset.'
        : 'DataPilot local service is unavailable. Restart the desktop app and wait for the AI Brain to show Online before connecting a dataset.';
      alert(stateMessage);
      return;
    }
    try {
      await fetchApiToken();

      // Defensive unwrapping if rawInput is array or string
      const fileInput = Array.isArray(rawInput) ? rawInput[0] : rawInput;
      if (!fileInput) return;

      let data = null;
      let lastErrMsg = '';
      const isString = typeof fileInput === 'string';
      const filePath = isString ? fileInput : (fileInput?.path || null);
      const isFakePath = filePath && filePath.includes('fakepath');

      // Dynamically infer source_type from extension
      const pathOrName = filePath || fileInput?.name || '';
      const ext = pathOrName.split('.').pop()?.toLowerCase();
      const sourceTypeMap = {
        csv: 'csv', xlsx: 'excel', xls: 'excel', json: 'json',
        parquet: 'parquet', pq: 'parquet', sqlite: 'sqlite', db: 'sqlite'
      };
      const sourceType = sourceTypeMap[ext] || 'csv';

      // 1. Try file_path connection if it's a real desktop string path and not fakepath
      if (filePath && !isFakePath) {
        try {
          const res = await fetch(`${API}/api/datasets/connect`, {
            method: 'POST',
            headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ file_path: filePath, source_type: sourceType }),
          });
          if (res.ok) {
            data = await res.json();
          } else {
            const errJson = await res.json().catch(() => null);
            lastErrMsg = formatApiError(errJson, res.status) || lastErrMsg;
          }
        } catch (pathErr) {
          console.warn('File path connection skipped/failed, trying binary upload:', pathErr);
        }
      }

      // 2. Fallback: upload binary File via FormData if path connection didn't return data
      if (!data && !isString && fileInput) {
        try {
          const formData = new FormData();
          formData.append('file', fileInput);
          const res = await fetch(`${API}/api/datasets/upload`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: formData,
          });
          if (res.ok) {
            data = await res.json();
          } else {
            const errJson = await res.json().catch(() => null);
            lastErrMsg = formatApiError(errJson, res.status) || lastErrMsg;
          }
        } catch (uploadErr) {
          lastErrMsg = uploadErr?.message || 'Network error during file upload';
        }
      }

      if (!data) {
        throw new Error(lastErrMsg || 'Dataset file could not be read or found on backend. Please try selecting a valid CSV/Excel file.');
      }

      const rowCnt = data.total_rows ?? data.row_count ?? data.rows ?? data.rowCount ?? 0;
      const colList = data.schema || data.columns || [];
      const colCount = data.column_count ?? data.col_count ?? data.colCount ?? (Array.isArray(colList) ? colList.length : Object.keys(colList || {}).length);
      const fileName = typeof fileInput === 'string'
        ? fileInput.split(/[/\\]/).pop()
        : (fileInput?.name || data.name || 'Dataset');
      const dsId = data.dataset_id || data.id || `ds_${Date.now()}`;

      const ds = {
        id: dsId,
        dataset_id: dsId,
        name: data.name || fileName,
        rowCount: rowCnt,
        colCount: colCount,
        columns: colList,
        schema: colList,
        domain: data.estimated_domain || data.domain || null,
        format: data.format || sourceType,
        loadedAt: Date.now()
      };

      logUserAction('DATASET_LOADED', { name: ds.name, rows: rowCnt, columns: colCount, format: ds.format });

      datasetDispatch({ type: 'ADD_DATASET', payload: ds });
      datasetDispatch({ type: 'SET_ACTIVE', payload: ds.id });
      setActiveDataset(ds);
      setSchema(colList);
      setRowCount(rowCnt);
      if (data.profile_summary) {
        setDatasetProfile(data.profile_summary);
      } else {
        fetchDatasetProfile(dsId);
      }
      fetchTableData(dsId);
    } catch (err) {
      logUserAction('DATASET_LOAD_FAILED', { error: err.message });
      alert(`Dataset Connection Failed: ${err.message}`);
    }
  };

  const handleDisconnect = async (id) => {
    logUserAction('DATASET_DISCONNECTED', { id });
    datasetDispatch({ type: 'REMOVE_DATASET', payload: id });
    try {
      await fetch(`${API}/api/datasets/${id}`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      });
    } catch (e) {}

    const remaining = datasetState.datasets.filter(d => d.id !== id && d.dataset_id !== id);
    if (datasetState.activeDatasetId === id) {
      if (remaining.length > 0) {
        const nextDs = remaining[0];
        const nextId = nextDs.id || nextDs.dataset_id;
        handleSetActive(nextId);
      } else {
        datasetDispatch({ type: 'SET_ACTIVE', payload: null });
        setActiveDataset(null);
        setSchema(null);
        setRowCount(0);
        setTableData([]);
        setDatasetProfile(null);
      }
    }
  };

  const handleSetActive = async (id) => {
    datasetDispatch({ type: 'SET_ACTIVE', payload: id });
    const ds = datasetState.datasets.find(d => d.id === id || d.dataset_id === id);
    if (ds) {
      logUserAction('DATASET_ACTIVATED', { id, name: ds.name, rows: ds.rowCount ?? ds.row_count });
      let cols = ds.columns || ds.schema || [];
      if ((!cols || cols.length === 0) && id) {
        try {
          const res = await fetch(`${API}/api/datasets/${id}`, { headers: getAuthHeaders() });
          if (res.ok) {
            const data = await res.json();
            cols = data.columns || [];
            ds.columns = cols;
          }
        } catch (err) {}
      }
      setActiveDataset(ds);
      setSchema(cols);
      setRowCount(ds.rowCount ?? ds.row_count ?? 0);
      setDatasetProfile(null);
      fetchTableData(id);
      fetchDatasetProfile(id);
    }

    try {
      await fetch(`${API}/api/datasets/${id}/activate`, {
        method: 'POST',
        headers: getAuthHeaders()
      });
    } catch (e) {}
  };

  // Operations Lifecycle Handlers
  const handleOperationApplied = async (result) => {
    if (result) {
      if (result.row_count !== undefined) setRowCount(result.row_count);
      if (result.schema) setSchema(result.schema);
      if (result.columns) setSchema(result.columns);
      if (result.pipeline_steps) setPipelineSteps(result.pipeline_steps);
      setCanUndo(true);
      setCanRedo(false);
      const activeId = result.dataset_id || datasetState.activeDatasetId;
      await fetchTableData(activeId);
      if (result.profile_summary) {
        setDatasetProfile(result.profile_summary);
      } else {
        await fetchDatasetProfile(activeId);
      }
      try {
        await fetchDatasets();
      } catch (e) {}
    }
  };

  const handleUndo = () => {
    console.info('Undo is quarantined until Phase 8 workflow state rollback.');
  };

  const handleRedo = () => {
    console.info('Redo is quarantined until Phase 8 workflow state rollback.');
  };

  const handleSaveFile = async () => {
    if (!activeDataset) return;
    logUserAction('EXPORT_CSV_CLICKED', { dataset: activeDataset.name });
    setIsSaving(true);
    try {
      const activeId = datasetState.activeDatasetId;
      const datasetQuery = activeId ? `?limit=10000&dataset_id=${encodeURIComponent(activeId)}` : '?limit=10000';
      const expRes = await fetch(`${API}/api/datasets/data${datasetQuery}`, { headers: getAuthHeaders() });
      if (expRes.ok) {
        const d = await expRes.json();
        const rows = d.rows || d.data || [];
        if (rows.length > 0) {
          const cols = Object.keys(rows[0]).filter(c => c !== '_row_idx');
          const csvContent = [
            cols.join(','),
            ...rows.map(r => cols.map(c => `"${String(r[c] ?? '').replace(/"/g, '""')}"`).join(','))
          ].join('\n');
          const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `datapilot_${activeDataset.name || 'export.csv'}`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }
      }
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setIsSaving(false);
    }
  };

  const handleRunCodeSnippet = (code) => {
    if (!code) return;
    logUserAction('RUN_CODE_SNIPPET', { code_length: code.length });
    setExternalCodeToRun({ code, timestamp: Date.now() });
    setCenterBottomTab('terminal');
  };

  // 4. Workflow Graph Polling & Execution Loop
  const startGraphPolling = useCallback((workflowId) => {
    const poll = async () => {
      try {
        const res = await fetch(`${API}/api/graph/${workflowId}/state`, { headers: getAuthHeaders() });
        if (res.ok) {
          const data = await res.json();
          workflowDispatch({ type: 'SET_GRAPH', payload: data });
          if (data.phase === 'SUCCEEDED' || data.phase === 'CODE_GENERATION') {
            workflowDispatch({ type: 'SET_PHASE', payload: 'coding' });
            handleGenerateCode(workflowId);
          } else if (data.phase === 'EXECUTING' || data.phase === 'PENDING') {
            setTimeout(poll, 2000);
          }
        }
      } catch (err) {
        console.error('Graph polling error:', err);
      }
    };
    poll();
  }, [workflowDispatch]);

  const handleAddMessage = useCallback((msg) => {
    workflowDispatch({ type: 'ADD_MESSAGE', payload: msg });
  }, [workflowDispatch]);

  const handlePatchMessage = useCallback((idOrIndex, patch) => {
    const payload = typeof idOrIndex === 'number' ? { index: idOrIndex, patch } : { id: idOrIndex, patch };
    workflowDispatch({ type: 'PATCH_MESSAGE', payload });
  }, [workflowDispatch]);

  const handleClearMessages = useCallback(() => {
    workflowDispatch({ type: 'CLEAR_MESSAGES' });
  }, [workflowDispatch]);

  const handleSubmitQuery = async (query) => {
    logUserAction('USER_QUERY_SUBMITTED', { query, active_dataset: activeDataset?.name, dataset_id: datasetState?.activeDatasetId, active_module: activeModule });
    workflowDispatch({ type: 'SET_QUERY', payload: query });
    workflowDispatch({ type: 'SET_PHASE', payload: 'PLANNING' });

    const recentHistory = (workflowState?.messages || [])
      .slice(-6)
      .map(m => ({ role: m.sender === 'user' ? 'user' : 'assistant', content: m.text || m.content || '' }));

    const contextPayload = {
      active_module: activeModule || 'home',
      dataset_id: datasetState?.activeDatasetId,
      selected_columns: datasetState?.selectedColumns || [],
      filters: datasetState?.activeFilters || []
    };

    try {
      const res = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          query,
          dataset_id: datasetState.activeDatasetId,
          conversation_history: recentHistory,
          context: contextPayload
        })
      });

      const rawBody = await res.json().catch(() => null);
      const payload = (rawBody && typeof rawBody.detail === 'object' && rawBody.detail !== null)
        ? rawBody.detail
        : (rawBody || {});

      const normalized = normalizeResponseContract(payload);
      const statusVal = normalized.status || (res.ok ? STATUS_TYPES.SUCCEEDED : STATUS_TYPES.FAILED);
      const decisionVal = normalized.decision || (statusVal === STATUS_TYPES.SUCCEEDED ? DECISION_TYPES.EXECUTE : DECISION_TYPES.FAILED);

      if (statusVal === STATUS_TYPES.NEEDS_CLARIFICATION || decisionVal === DECISION_TYPES.CLARIFY) {
        const clarificationError = normalized.error || {
          code: 'CLARIFICATION_REQUIRED',
          message: normalized.clarification || 'More information is required before this analysis can run.',
          phase: payload.phase || 'PLAN_VALIDATION',
          stage: 'PLAN_VALIDATION'
        };
        const clarificationMessage = normalized.clarification || clarificationError.message || 'More information is required.';
        const candidateCols = normalized.candidateColumns || clarificationError.candidate_columns || [];

        workflowDispatch({ type: 'SET_CLARIFICATION', payload: {
          message: clarificationMessage,
          candidate_columns: candidateCols,
          options: clarificationError.suggested_options || []
        }});
        workflowDispatch({ type: 'SET_ERROR', payload: clarificationMessage });
        workflowDispatch({ type: 'SET_PHASE', payload: payload.phase || 'PLAN_VALIDATION' });
        workflowDispatch({
          type: 'SET_EXECUTION',
          payload: {
            status: STATUS_TYPES.NEEDS_CLARIFICATION,
            decision: DECISION_TYPES.CLARIFY,
            output: null,
            error: clarificationError,
            artifacts: { clarification: clarificationMessage }
          }
        });
        return payload;
      }

      if (statusVal === STATUS_TYPES.UNSUPPORTED || decisionVal === DECISION_TYPES.UNSUPPORTED || statusVal === STATUS_TYPES.UNSAFE || decisionVal === DECISION_TYPES.UNSAFE) {
        const nonExecError = normalized.error || {
          code: statusVal === STATUS_TYPES.UNSUPPORTED ? 'UNSUPPORTED_OPERATION' : 'UNSAFE_PLAN',
          message: normalized.error?.message || (statusVal === STATUS_TYPES.UNSUPPORTED ? 'This analytical request is currently unsupported.' : 'This analytical request is scientifically invalid or unsafe.'),
          phase: payload.phase || 'PLAN_VALIDATION',
          required_action: normalized.error?.required_action || 'Modify the request or verify dataset assumptions.'
        };
        workflowDispatch({ type: 'SET_ERROR', payload: nonExecError.message });
        workflowDispatch({ type: 'SET_PHASE', payload: payload.phase || 'PLAN_VALIDATION' });
        workflowDispatch({
          type: 'SET_EXECUTION',
          payload: {
            status: statusVal,
            decision: decisionVal,
            output: null,
            error: nonExecError,
            artifacts: { error: nonExecError }
          }
        });
        return payload;
      }

      if (!res.ok || statusVal === STATUS_TYPES.FAILED || decisionVal === DECISION_TYPES.FAILED) {
        const errorDetail = normalized.error || {
          code: payload?.error?.code || 'QUERY_FAILED',
          message: payload?.error?.message || `HTTP ${res.status}: Query processing failed`,
          phase: payload?.phase || 'FAILED',
          required_action: payload?.error?.required_action || 'Check parameters and retry.'
        };

        const failedPhase = errorDetail.phase || 'FAILED';
        workflowDispatch({ type: 'SET_ERROR', payload: errorDetail.message });
        workflowDispatch({ type: 'SET_PHASE', payload: failedPhase });
        workflowDispatch({
          type: 'SET_EXECUTION',
          payload: {
            status: 'FAILED',
            decision: 'failed',
            output: null,
            error: errorDetail,
            plot_image: null,
            artifacts: { error: errorDetail }
          }
        });

        if (errorDetail.code === 'AMBIGUOUS_INTENT' || errorDetail.code === 'AMBIGUOUS_COLUMN') {
          const candidateCols = errorDetail.candidate_columns || payload?.candidate_columns || [];
          workflowDispatch({
            type: 'SET_CLARIFICATION',
            payload: {
              message: errorDetail.message,
              candidate_columns: candidateCols
            }
          });
          workflowDispatch({ type: 'SET_PHASE', payload: 'PLAN_VALIDATION' });
        }

        throw new Error(`[${errorDetail.code}] ${errorDetail.message}`);
      }

      // Normalize the authoritative typed DS-OS result once. Every result tab
      // now reads the same tables, metrics, manifest and rendered artifacts.
      const presentation = createQueryPresentation(payload);
      if (presentation.reproducible_code) {
        workflowDispatch({
          type: 'SET_CODE',
          payload: { code: presentation.reproducible_code, expected_outputs: presentation.artifacts?.expected_outputs }
        });
      }

      workflowDispatch({
        type: 'SET_EXECUTION',
        payload: presentation
      });
      workflowDispatch({ type: 'SET_VALIDATION', payload: presentation.validationReport });
      workflowDispatch({ type: 'SET_ARTIFACTS', payload: [] });

      if (presentation.plot_image) {
        setCenterBottomTab('results');
      }

      if (payload.workflow_id) {
        workflowDispatch({ type: 'SET_WORKFLOW_ID', payload: payload.workflow_id });
        fetchWorkflows();
      }

      workflowDispatch({ type: 'SET_PHASE', payload: 'SUCCEEDED' });
      return payload;
    } catch (err) {
      console.error('Canonical Query Pipeline Failure:', err);
      let code = 'QUERY_ERROR';
      let msg = err.message;
      const match = err.message.match(/^\[(.*?)\] (.*)/);
      if (match) {
        code = match[1];
        msg = match[2];
      }
      workflowDispatch({ type: 'SET_ERROR', payload: msg });
      workflowDispatch({ type: 'SET_PHASE', payload: 'FAILED' });
      workflowDispatch({
        type: 'SET_EXECUTION',
        payload: {
          status: 'FAILED',
          output: null,
          error: { code, message: msg }
        }
      });
      throw err;
    }
  };
  const handleGenerateCode = async (workflowId) => {
    try {
      const res = await fetch(`${API}/api/code/generate`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          workflow_id: workflowId,
          node_id: 'node_1',
          capability: 'data_analysis',
          intent: workflowState.intent || {},
          dataset_id: datasetState.activeDatasetId || ''
        })
      });
      if (res.ok) {
        const data = await res.json();
        workflowDispatch({ type: 'SET_CODE', payload: data });
      }
    } catch (err) {
      console.error('Code generation error:', err);
    }
  };

  const handleValidateCode = async (codeToValidate = workflowState.codeVersion?.code) => {
    if (!codeToValidate) return;
    try {
      const res = await fetch(`${API}/api/code/validate`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          code: codeToValidate,
          dataset_id: datasetState.activeDatasetId || ''
        })
      });
      if (res.ok) {
        const data = await res.json();
        workflowDispatch({ type: 'SET_VALIDATION', payload: data });
      }
    } catch (err) {
      console.error('Validation error:', err);
    }
  };

  const handleRunCode = async (codeToRun = workflowState.codeVersion?.code) => {
    if (!codeToRun?.trim()) return;
    workflowDispatch({ type: 'SET_PHASE', payload: 'executing' });
    workflowDispatch({ type: 'SET_EXECUTION', payload: { status: 'RUNNING', execution_mode: 'manual_python' } });
    setCenterBottomTab('terminal');
    // One owner, one run: TerminalHUD invokes the sandbox exactly once.
    setExternalCodeToRun({ code: codeToRun, timestamp: Date.now() });
  };

  const handleRequestInsights = async (workflowId = workflowState.graphState?.workflow_id) => {
    if (!workflowId) return;
    try {
      const res = await fetch(`${API}/api/results/interpret`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          execution_id: workflowId,
          user_query: workflowState.userQuery,
          intent: workflowState.intent || {}
        })
      });
      if (res.ok) {
        const data = await res.json();
        workflowDispatch({ type: 'SET_INSIGHT', payload: data });
      }
    } catch (err) {
      console.error('Insights error:', err);
    }
  };

  const handleOpenFilePrompt = () => {
    if (window.electron?.openFile) {
      window.electron.openFile().then(filePaths => {
        if (filePaths && filePaths.length > 0) {
          handleConnect(filePaths[0]);
        }
      });
    } else {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.csv,.xlsx,.xls,.json,.parquet,.sqlite,.db';
      input.onchange = (e) => {
        const file = e.target.files?.[0];
        if (file) handleConnect(file);
      };
      input.click();
    }
  };

  const handleLoadWorkflow = async (workflowId) => {
    try {
      const res = await fetch(`${API}/api/workflows/${workflowId}`, { headers: getAuthHeaders() });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const workflow = await res.json();
      workflowDispatch({ type: 'SET_WORKFLOW_ID', payload: workflow.workflow_id });
      workflowDispatch({ type: 'SET_QUERY', payload: workflow.user_query || '' });
      workflowDispatch({ type: 'SET_PHASE', payload: workflow.status || 'idle' });
      if (workflow.intent) workflowDispatch({ type: 'SET_INTENT', payload: workflow.intent });
      if (workflow.code) workflowDispatch({ type: 'SET_CODE', payload: { code: workflow.code } });
      setProductionCheck(null);
      setLeftTab('graph');
    } catch (err) {
      alert(`Workflow could not be loaded: ${err.message}`);
    }
  };

  const handleDeleteWorkflow = async (workflowId) => {
    try {
      const res = await fetch(`${API}/api/workflows/${workflowId}`, {
        method: 'DELETE', headers: getAuthHeaders()
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await fetchWorkflows();
      if (workflowState.workflowId === workflowId) {
        workflowDispatch({ type: 'SET_WORKFLOW_ID', payload: null });
      }
    } catch (err) {
      alert(`Workflow could not be deleted: ${err.message}`);
    }
  };

  const handleProductionValidate = async (productionDatasetId) => {
    if (!workflowState.workflowId) return;
    setIsProductionChecking(true);
    setProductionCheck(null);
    try {
      const res = await fetch(`${API}/api/production/validate`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ workflow_id: workflowState.workflowId, production_dataset_id: productionDatasetId })
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(formatApiError(data, res.status));
      setProductionCheck(data);
    } catch (err) {
      setProductionCheck({ is_compatible: false, checks: [], warnings: [err.message] });
    } finally {
      setIsProductionChecking(false);
    }
  };

  const handleProductionExecute = async (productionDatasetId) => {
    if (!workflowState.workflowId) return;
    setIsProductionExecuting(true);
    try {
      const res = await fetch(`${API}/api/production/execute`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          workflow_id: workflowState.workflowId,
          production_dataset_id: productionDatasetId,
          user_approved: true
        })
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(formatApiError(data, res.status));
      workflowDispatch({ type: 'SET_EXECUTION', payload: data });
      workflowDispatch({ type: 'SET_PHASE', payload: 'SUCCEEDED' });
      await fetchWorkflows();
    } catch (err) {
      workflowDispatch({ type: 'SET_ERROR', payload: err.message });
      workflowDispatch({ type: 'SET_PHASE', payload: 'FAILED' });
    } finally {
      setIsProductionExecuting(false);
    }
  };


  return (
    <div className={`app-shell ${theme === 'dark' ? 'dark-theme' : ''}`} style={{
      display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw',
      overflow: 'hidden', background: '#050811', color: '#f1f5f9'
    }}>
      {/* 1. Header & Navigation Ribbon */}
      <TitleBar
        datasetName={activeDataset?.name}
        rowCount={rowCount}
        colCount={schema ? (Array.isArray(schema) ? schema.length : Object.keys(schema).length) : 0}
        onOpenFile={handleOpenFilePrompt}
        onSaveFile={handleSaveFile}
        isSaving={isSaving}
        theme={theme}
        onToggleTheme={toggleTheme}
        onNavigateHome={() => setActiveModule('home')}
      />

      <Ribbon
        activeModule={activeModule}
        onModuleChange={(mod) => {
          logUserAction('TAB_SWITCHED', { from: activeModule, to: mod });
          setActiveModule(mod);
          if (mod === 'preview') {
            fetchTableData();
          }
        }}
        datasetLoaded={!!activeDataset}
        onOpenFile={handleOpenFilePrompt}
        onOpenDbModal={() => setIsDbModalOpen(true)}
        onSaveFile={handleSaveFile}
        onUndo={handleUndo}
        onRedo={handleRedo}
        canUndo={canUndo}
        canRedo={canRedo}
        isSaving={isSaving}
        onOpenCodeInspector={() => setActiveModule('code')}
        isCollapsed={isRibbonCollapsed}
        onToggleCollapse={() => setIsRibbonCollapsed(!isRibbonCollapsed)}
      />

      {/* 2. Main Dynamic Workspace Body */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', background: '#050811' }}>

        {/* VIEW 1: HOME (3-Column AI Data Science Workspace) */}
        {activeModule === 'home' && (
          <div style={{ display: 'flex', flex: 1, overflow: 'hidden', width: '100%' }}>
            {/* LEFT SIDEBAR (Width 280px) */}
            <Sidebar
              activeTab={leftTab}
              onTabChange={setLeftTab}
              datasets={datasetState.datasets}
              activeDatasetId={datasetState.activeDatasetId}
              onConnect={handleConnect}
              onDisconnect={handleDisconnect}
              onSetActive={handleSetActive}
              onRefresh={fetchDatasets}
              engineStatus={engineStatus}
              workflowState={workflowState}
              onRefreshGraph={() => startGraphPolling(workflowState.graphState?.workflow_id)}
              workflows={workflows}
              onLoadWorkflow={handleLoadWorkflow}
              onDeleteWorkflow={handleDeleteWorkflow}
            />

            {/* CENTER COLUMN (Flex 1) */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#070a12' }}>
              {/* Top Half: Code Panel */}
              <div style={{ flex: 1, minHeight: 180, borderBottom: '1px solid #1e293b', overflow: 'hidden' }}>
                <CodePanel
                  codeVersion={workflowState.codeVersion}
                  validationResult={workflowState.validationReport}
                  onValidate={handleValidateCode}
                  onRun={handleRunCode}
                  executionEnabled={false}
                  onSaveUserEdit={(newCode) => workflowDispatch({
                    type: 'SET_CODE', payload: { ...workflowState.codeVersion, code: newCode }
                  })}
                  isRunning={workflowState.executionRecord?.status === 'RUNNING'}
                  isValidating={false}
                />
              </div>

              {/* Bottom Half: Results, Terminal, Production Tabs */}
              <div style={{ height: 'clamp(280px, 46vh, 480px)', minHeight: 240, display: 'flex', flexDirection: 'column', background: '#090d16', flexShrink: 0 }}>
                <div style={{
                  display: 'flex', borderBottom: '1px solid #1e293b', background: '#0f172a',
                  padding: '0 12px', flexShrink: 0, alignItems: 'center', justifyContent: 'space-between'
                }}>
                  <div style={{ display: 'flex', gap: 2 }}>
                    {[
                      { id: 'results', label: 'Results & Output', icon: BarChart2 }
                    ].map(tab => {
                      const Icon = tab.icon;
                      const isActive = centerBottomTab === tab.id;
                      return (
                        <button
                          key={tab.id}
                          onClick={() => setCenterBottomTab(tab.id)}
                          style={{
                            padding: '8px 14px', background: isActive ? '#090d16' : 'transparent',
                            border: 'none', borderBottom: isActive ? '2px solid #8b5cf6' : '2px solid transparent',
                            color: isActive ? '#a78bfa' : '#64748b', fontSize: 12, fontWeight: 700,
                            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6
                          }}
                        >
                          <Icon size={14} /> {tab.label}
                        </button>
                      );
                    })}
                  </div>

                  {workflowState.phase && (
                    <div style={{ fontSize: 11, color: '#64748b', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Activity size={12} style={{ color: '#38bdf8' }} />
                      <span>Phase: <strong style={{ color: '#38bdf8' }}>{workflowState.phase}</strong></span>
                    </div>
                  )}
                </div>

                <div style={{ flex: 1, overflow: 'auto' }}>
                  {centerBottomTab === 'results' && (
                    <ResultPanel
                      key={workflowState.executionRecord?.request_id || workflowState.executionRecord?.timing || workflowState.executionRecord?.status || 'empty'}
                      executionRecord={workflowState.executionRecord}
                      artifacts={workflowState.artifacts}
                      validationReport={workflowState.validationReport}
                      onRequestInsights={() => handleRequestInsights()}
                      onZoomImage={setZoomedImage}
                    />
                  )}

                  {centerBottomTab === 'terminal' && (
                    <TerminalHUD
                      schema={schema}
                      rowCount={rowCount}
                      pipelineSteps={pipelineSteps}
                      onRefresh={fetchTableData}
                      externalCodeToRun={externalCodeToRun}
                      onPlotGenerated={(plotB64) => {
                        workflowDispatch({
                          type: 'SET_EXECUTION',
                          payload: {
                            ...(workflowState.executionRecord || {}),
                            status: 'SUCCEEDED',
                            plot_image: plotB64
                          }
                        });
                      }}
                      onExecutionComplete={(terminalResult, code) => {
                        const presentation = createManualExecutionPresentation(terminalResult, code);
                        workflowDispatch({ type: 'SET_EXECUTION', payload: presentation });
                        workflowDispatch({ type: 'SET_VALIDATION', payload: null });
                        workflowDispatch({ type: 'SET_PHASE', payload: presentation.status === 'SUCCEEDED' ? 'SUCCEEDED' : 'FAILED' });
                      }}
                      onZoomImage={setZoomedImage}
                    />
                  )}

                  {centerBottomTab === 'production' && (
                    <ProductionPanel
                      workflowId={workflowState.workflowId}
                      datasets={datasetState.datasets}
                      activeDatasetId={datasetState.activeDatasetId}
                      onValidate={handleProductionValidate}
                      onExecuteProduction={handleProductionExecute}
                      onDatasetChange={() => setProductionCheck(null)}
                      validationResult={productionCheck}
                      isChecking={isProductionChecking}
                      isExecuting={isProductionExecuting}
                    />
                  )}
                </div>
              </div>
            </div>

            {/* RIGHT SIDEBAR (Width 380px) */}
            <div style={{
              width: 380, borderLeft: '1px solid #1e293b', background: '#0e1424',
              display: 'flex', flexDirection: 'column', flexShrink: 0, overflow: 'hidden', height: '100%'
            }}>
              <div style={{
                flex: isRightPanelCollapsed ? 1 : '1 1 54%',
                minHeight: 0,
                borderBottom: isRightPanelCollapsed ? 'none' : '1px solid #1e293b',
                overflow: 'hidden',
                transition: 'flex 0.2s ease-in-out'
              }}>
                <AICoPilot
                  activeDataset={activeDataset}
                  schema={schema}
                  datasetLoaded={!!activeDataset}
                  onSaveFile={handleSaveFile}
                  isSaving={isSaving}
                  onRunCodeInTerminal={(code) => {
                    setCenterBottomTab('terminal');
                    setExternalCodeToRun({ code, timestamp: Date.now() });
                  }}
                  onRefresh={fetchTableData}
                  onZoomImage={setZoomedImage}
                  ollamaStatus={workflowState.ollamaStatus}
                  onSubmitQuery={handleSubmitQuery}
                  onAddMessage={handleAddMessage}
                  appendMessage={handleAddMessage}
                  patchMessage={handlePatchMessage}
                  clearMessages={handleClearMessages}
                  messages={workflowState.messages || []}
                  workflowPhase={workflowState.phase}
                  onSelectTab={(tab) => {
                    setCenterBottomTab(tab);
                    if (tab === 'code') {
                      // Scroll to top of center column to focus code editor
                      window.scrollTo({ top: 0, behavior: 'smooth' });
                    }
                  }}
                  onOpenCodePanel={() => {
                    // Switch to code editor view
                    setActiveModule('code');
                  }}
                />
              </div>

              <div
                onClick={toggleRightPanel}
                title={isRightPanelCollapsed ? "Expand Intent & Insights Panel" : "Minimize Intent & Insights Panel"}
                style={{
                  height: 34,
                  background: '#0d131f',
                  borderTop: '1px solid #1e293b',
                  borderBottom: isRightPanelCollapsed ? 'none' : '1px solid #1e293b',
                  padding: '0 12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  userSelect: 'none',
                  fontSize: 11,
                  fontWeight: 700,
                  color: '#94a3b8',
                  flexShrink: 0
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Sparkles size={13} color="#38bdf8" />
                  <span>Intent & Evidence Insights</span>
                  {isRightPanelCollapsed && (
                    <span style={{ fontSize: 10, color: '#38bdf8', background: '#0369a122', padding: '2px 6px', borderRadius: 4 }}>
                      Collapsed
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#38bdf8' }}>
                  <span style={{ fontSize: 10 }}>{isRightPanelCollapsed ? 'Expand' : 'Minimize'}</span>
                  {isRightPanelCollapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </div>
              </div>

              {!isRightPanelCollapsed && (
                <div style={{ flex: '1 1 44%', minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#090d16' }}>
                  <div style={{ flex: 1, minHeight: 0, borderBottom: '1px solid #1e293b', overflow: 'auto' }}>
                    <IntentPanel
                      intent={workflowState.intent}
                      clarification={workflowState.clarification}
                      isAnalyzing={workflowState.phase === 'analyzing'}
                      ollamaStatus={workflowState.ollamaStatus}
                    />
                  </div>

                  <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
                    <InsightPanel
                      insight={workflowState.insight}
                      isGenerating={false}
                      onRegenerate={() => handleRequestInsights()}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* VIEW 2: DATA GRID PREVIEW */}
        {activeModule === 'preview' && (
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', height: '100%' }}>
            {activeDataset ? (
              <DataPreview
                tableData={tableData}
                schema={schema}
                rowCount={rowCount}
                activeDataset={activeDataset}
                profile={datasetProfile}
                onRefresh={() => {
                  const dsId = activeDataset?.id || activeDataset?.dataset_id;
                  fetchTableData(dsId);
                  fetchDatasetProfile(dsId);
                }}
              />
            ) : (
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                gap: 16, background: '#090d16', color: '#94a3b8'
              }}>
                <Table size={48} color="#334155" />
                <h3 style={{ color: '#f8fafc', fontSize: 18, margin: 0 }}>No Active Dataset Loaded</h3>
                <p style={{ fontSize: 13, maxWidth: 400, textAlign: 'center', margin: 0 }}>
                  Connect a CSV, Excel, or Parquet dataset to explore columns, inspect data types, and review data quality.
                </p>
                <div style={{ display: 'flex', gap: 12 }}>
                  <button
                    className="btn btn--primary"
                    onClick={handleOpenFilePrompt}
                    style={{ padding: '8px 16px', fontWeight: 700 }}
                  >
                    <FolderOpen size={16} /> Open File
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* VIEW 3: CONNECT HUB */}
        {activeModule === 'connect' && (
          <div style={{
            flex: 1, overflowY: 'auto', padding: 28, background: '#090d16', display: 'flex', flexDirection: 'column', gap: 24
          }}>
            {/* Header Banner */}
            <div style={{
              background: '#0f172a', border: '1px solid #1e293b', borderRadius: 12, padding: '20px 24px',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{
                  background: 'linear-gradient(135deg, #10b981, #0ea5e9)', padding: 12, borderRadius: 10, display: 'flex'
                }}>
                  <PlugZap size={24} color="#ffffff" />
                </div>
                <div>
                  <h2 style={{ margin: 0, fontSize: 18, fontWeight: 800, color: '#f8fafc' }}>
                    Dataset & Data Source Connection Hub
                  </h2>
                  <p style={{ margin: '4px 0 0 0', fontSize: 12, color: '#94a3b8' }}>
                    Connect local files or import databases via SQL.
                  </p>
                </div>
              </div>

              {activeDataset && (
                <div style={{
                  background: '#1e293b', border: '1px solid #334155', borderRadius: 20, padding: '6px 14px',
                  display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#34d399', fontWeight: 700
                }}>
                  <CheckCircle2 size={15} />
                  <span>Active: {activeDataset.name}</span>
                </div>
              )}
            </div>

            {/* Quick Connection Action Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
              {/* Card 1: Local Files */}
              <div style={{
                background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 20,
                display: 'flex', flexDirection: 'column', justifyContent: 'space-between'
              }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#38bdf8', marginBottom: 10 }}>
                    <FolderOpen size={20} />
                    <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>Local Files</h3>
                  </div>
                  <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, margin: '0 0 16px 0' }}>
                    Load CSV, Excel (.xlsx, .xls), JSON, Parquet, or SQLite database files directly from your computer.
                  </p>
                </div>
                <button
                  onClick={handleOpenFilePrompt}
                  className="btn btn--primary"
                  style={{ width: '100%', justifyContent: 'center', padding: '10px 14px', fontWeight: 700 }}
                >
                  <FolderOpen size={15} /> Browse & Open Local File
                </button>
              </div>

              {/* Card 2: Database Connection */}
              <div style={{
                background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 20,
                display: 'flex', flexDirection: 'column', justifyContent: 'space-between'
              }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#a78bfa', marginBottom: 10 }}>
                    <Database size={20} />
                    <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>Database & SQL</h3>
                  </div>
                  <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, margin: '0 0 16px 0' }}>
                    Connect to PostgreSQL, MySQL, SQLite, Snowflake, or SQL Server to query and import tables.
                  </p>
                </div>
                <button
                  onClick={() => setIsDbModalOpen(true)}
                  style={{
                    width: '100%', background: '#8b5cf6', color: '#fff', border: 'none', borderRadius: 6,
                    padding: '10px 14px', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center',
                    justifyContent: 'center', gap: 8, fontSize: 13
                  }}
                >
                  <Database size={15} /> Connect Database (SQL Modal)
                </button>
              </div>

            </div>

            {/* Currently Connected Datasets Table */}
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Layers size={16} color="#38bdf8" /> Connected Datasets Registry ({datasetState.datasets.length})
                </h3>
                <button
                  onClick={fetchDatasets}
                  className="btn btn--ghost btn--sm"
                  style={{ color: '#94a3b8' }}
                >
                  <RefreshCw size={13} /> Refresh List
                </button>
              </div>

              {datasetState.datasets.length === 0 ? (
                <p style={{ color: '#64748b', fontSize: 13, textAlign: 'center', padding: '24px 0' }}>
                  No datasets connected. Open a file or connect a database above!
                </p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {datasetState.datasets.map(ds => {
                    const dsId = ds.dataset_id || ds.id;
                    const isActive = datasetState.activeDatasetId === dsId || activeDataset?.id === dsId;
                    return (
                      <div
                        key={dsId}
                        style={{
                          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                          background: isActive ? '#1e293b' : '#0a0f1d', border: isActive ? '1px solid #0ea5e9' : '1px solid #1e293b',
                          borderRadius: 8, padding: '10px 16px'
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <FileSpreadsheet size={18} color={isActive ? '#38bdf8' : '#64748b'} />
                          <div>
                            <strong style={{ fontSize: 13, color: '#f8fafc' }}>{ds.name}</strong>
                            <div style={{ fontSize: 11, color: '#94a3b8' }}>
                              {(ds.rowCount ?? ds.row_count ?? 0).toLocaleString()} rows • {ds.colCount ?? (ds.columns?.length || 0)} cols
                            </div>
                          </div>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          {isActive ? (
                            <span style={{ fontSize: 11, background: '#0369a1', color: '#fff', padding: '3px 10px', borderRadius: 12, fontWeight: 700 }}>
                              Active
                            </span>
                          ) : (
                            <button
                              onClick={() => handleSetActive(dsId)}
                              className="btn btn--ghost btn--sm"
                              style={{ color: '#38bdf8', border: '1px solid #334155' }}
                            >
                              Set Active
                            </button>
                          )}
                          <button
                            onClick={() => handleDisconnect(dsId)}
                            className="btn btn--danger btn--sm"
                            style={{ padding: '3px 8px', fontSize: 11 }}
                          >
                            Disconnect
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* VIEW 4: CLEAN */}
        {activeModule === 'clean' && (
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', height: '100%' }}>
            {activeDataset ? (
              <OperationPanel mode="clean" schema={schema} datasetId={activeDataset?.id || activeDataset?.dataset_id} onApplied={handleOperationApplied} />
            ) : (
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                gap: 16, background: '#090d16', color: '#94a3b8'
              }}>
                <Eraser size={48} color="#334155" />
                <h3 style={{ color: '#f8fafc', fontSize: 18, margin: 0 }}>No Dataset Loaded to Clean</h3>
                <p style={{ fontSize: 13, maxWidth: 400, textAlign: 'center', margin: 0 }}>
                  Load a dataset first to impute missing values, cap outliers using IQR, drop nulls, and deduplicate rows.
                </p>
                <button
                  className="btn btn--primary"
                  onClick={handleOpenFilePrompt}
                  style={{ padding: '8px 16px', fontWeight: 700 }}
                >
                  <FolderOpen size={16} /> Open Dataset File
                </button>
              </div>
            )}
          </div>
        )}

        {/* VIEW 5: TRANSFORM */}
        {activeModule === 'transform' && (
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', height: '100%' }}>
            {activeDataset ? (
              <OperationPanel mode="transform" schema={schema} datasetId={activeDataset?.id || activeDataset?.dataset_id} onApplied={handleOperationApplied} />
            ) : (
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                gap: 16, background: '#090d16', color: '#94a3b8'
              }}>
                <Shuffle size={48} color="#334155" />
                <h3 style={{ color: '#f8fafc', fontSize: 18, margin: 0 }}>No Dataset Loaded to Transform</h3>
                <p style={{ fontSize: 13, maxWidth: 400, textAlign: 'center', margin: 0 }}>
                  Load a dataset first to apply One-Hot encoding, Yeo-Johnson power transforms, Min-Max scaling, and row filters.
                </p>
                <button
                  className="btn btn--primary"
                  onClick={handleOpenFilePrompt}
                  style={{ padding: '8px 16px', fontWeight: 700 }}
                >
                  <FolderOpen size={16} /> Open Dataset File
                </button>
              </div>
            )}
          </div>
        )}

        {/* VIEW 6: EDA & STATS */}
        {activeModule === 'stats' && (
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', height: '100%' }}>
            {activeDataset ? (
              <DataScienceStudio
                schema={schema}
                rowCount={rowCount}
                activeDataset={activeDataset}
                datasetId={activeDataset?.id || activeDataset?.dataset_id}
                pipelineSteps={pipelineSteps}
                onRunCode={handleRunCodeSnippet}
                onRefresh={fetchTableData}
                defaultTab="eda"
              />
            ) : (
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                gap: 16, background: '#090d16', color: '#94a3b8'
              }}>
                <LineChart size={48} color="#334155" />
                <h3 style={{ color: '#f8fafc', fontSize: 18, margin: 0 }}>No Dataset Loaded for EDA</h3>
                <p style={{ fontSize: 13, maxWidth: 400, textAlign: 'center', margin: 0 }}>
                  Load a dataset first to generate Seaborn correlation heatmaps, compute VIF multicollinearity, and test skewness.
                </p>
                <button
                  className="btn btn--primary"
                  onClick={handleOpenFilePrompt}
                  style={{ padding: '8px 16px', fontWeight: 700 }}
                >
                  <FolderOpen size={16} /> Open Dataset File
                </button>
              </div>
            )}
          </div>
        )}

        {/* VIEW 7: ML STUDIO */}
        {activeModule === 'ml' && (
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', height: '100%' }}>
            {activeDataset ? (
              <DataScienceStudio
                schema={schema}
                rowCount={rowCount}
                activeDataset={activeDataset}
                datasetId={activeDataset?.id || activeDataset?.dataset_id}
                pipelineSteps={pipelineSteps}
                onRunCode={handleRunCodeSnippet}
                onRefresh={fetchTableData}
                defaultTab="ml"
              />
            ) : (
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                gap: 16, background: '#090d16', color: '#94a3b8'
              }}>
                <BrainCircuit size={48} color="#334155" />
                <h3 style={{ color: '#f8fafc', fontSize: 18, margin: 0 }}>No Dataset Loaded for ML Studio</h3>
                <p style={{ fontSize: 13, maxWidth: 400, textAlign: 'center', margin: 0 }}>
                  Load a dataset to train supervised machine learning models (Random Forest, Gradient Boosting, Extra Trees) with AutoML.
                </p>
                <button
                  className="btn btn--primary"
                  onClick={handleOpenFilePrompt}
                  style={{ padding: '8px 16px', fontWeight: 700 }}
                >
                  <FolderOpen size={16} /> Open Dataset File
                </button>
              </div>
            )}
          </div>
        )}

        {/* VIEW 8: REPORTS */}
        {(activeModule === 'reports' || activeModule === 'report') && (
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', height: '100%' }}>
            <ReportingStudio activeDataset={activeDataset} />
          </div>
        )}

        {/* VIEW 9: CODE & SQL INSPECTOR */}
        {activeModule === 'code' && (
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', height: '100%' }}>
            <SqlStudio activeDataset={activeDataset} />
          </div>
        )}

        {/* VIEW 10: WORKSPACE AGENT */}
        {activeModule === 'agent' && (
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', height: '100%' }}>
            <AgentStudio
              activeDataset={activeDataset}
              onDatasetConnected={fetchDatasets}
            />
          </div>
        )}

      </div>

      {/* 3. Footer Status Bar */}
      <StatusBar
        datasetName={activeDataset?.name}
        rowCount={rowCount}
        colCount={schema ? (Array.isArray(schema) ? schema.length : Object.keys(schema).length) : 0}
        dirtyCount={0}
        activeModule={activeModule}
      />

      {/* Modals */}
      <PlotZoomModal imageSrc={zoomedImage} onClose={() => setZoomedImage(null)} />
      <DBConnectionModal
        isOpen={isDbModalOpen}
        onClose={() => setIsDbModalOpen(false)}
        onImportSuccess={(d) => {
          handleConnect({ path: d.name });
          setIsDbModalOpen(false);
        }}
      />
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

