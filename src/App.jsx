import { useState, useEffect, useCallback, useRef } from 'react';
import { FolderOpen, ChevronUp, ChevronDown, Terminal } from 'lucide-react';

import './index.css';

// Components
import TitleBar from './components/TitleBar';
import StatusBar from './components/StatusBar';
import TerminalHUD from './components/TerminalHUD';
import AICoPilot from './components/AICoPilot';
import Ribbon from './components/Ribbon';
import OperationPanel from './components/OperationPanel';
import ObjectExplorer from './components/ObjectExplorer';
import PropertiesPanel from './components/PropertiesPanel';
import DBConnectionModal from './components/DBConnectionModal';
import DataPreview from './components/DataPreview';
import ReportView from './components/ReportView';

const API = import.meta.env.VITE_API_URL || window.ENV_API_URL || 'http://localhost:8000';
const isElectron = typeof window !== 'undefined' && window.electron;

export default function App() {
  // ── State ──────────────────────────────────
  const [activeDataset, setActiveDataset] = useState(null);
  const [schema, setSchema] = useState(null);
  const [tableData, setTableData] = useState([]);
  const [rowCount, setRowCount] = useState(0);
  const [pipelineSteps, setPipelineSteps] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState(null);
  const [engineStatus, setEngineStatus] = useState('offline');
  const [isSaving, setIsSaving] = useState(false);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);
  const [externalCodeToRun, setExternalCodeToRun] = useState(null);
  const [activePlotImage, setActivePlotImage] = useState(null);
  const [activeModule, setActiveModule] = useState('home');
  const [isDbModalOpen, setIsDbModalOpen] = useState(false);

  const plotCanvasRef = useRef(null);

  // Auto-scroll to Visualization Canvas when a new chart plot is generated
  useEffect(() => {
    if (activePlotImage && plotCanvasRef.current) {
      plotCanvasRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [activePlotImage]);

  // Fetch initial data
  useEffect(() => {
    refreshData();
  }, []);

  // ── Fetch data ─────────────────────────────
  const refreshData = useCallback(async () => {
    try {
      const [schemaRes, dataRes, pipeRes] = await Promise.all([
        fetch(`${API}/api/dataset/schema`),
        fetch(`${API}/api/dataset/data?limit=-1`),
        fetch(`${API}/api/pipeline`),
      ]);

      if (schemaRes.ok) {
        const s = await schemaRes.json();
        setSchema(s.columns ?? []);
        setRowCount(s.total_rows ?? s.row_count ?? 0);
        setActiveDataset(prev => {
          if (!prev && !s.name) return prev;
          return {
            name: s.name || prev?.name || 'Dataset',
            rowCount: s.total_rows ?? s.row_count ?? prev?.rowCount ?? 0,
            columns: s.columns ?? prev?.columns ?? [],
            loadedAt: prev?.loadedAt || Date.now()
          };
        });
      }
      if (dataRes.ok) {
        const d = await dataRes.json();
        setTableData(d.data ?? []);
      }
      if (pipeRes.ok) {
        const p = await pipeRes.json();
        setPipelineSteps(p.steps ?? []);
      }
    } catch {
      // offline / not connected yet
    }
  }, []);

  // ── Health Check Loop ──────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/health`);
        setEngineStatus(r.ok ? 'connected' : 'error');
      } catch {
        setEngineStatus('error');
      }
    };
    check();
    const t = setInterval(check, 5000);
    return () => clearInterval(t);
  }, []);

  const handleConnected = useCallback((datasetInfo) => {
    const rowCnt = datasetInfo.total_rows ?? datasetInfo.row_count ?? datasetInfo.rows ?? datasetInfo.rowCount ?? 0;
    const colList = datasetInfo.schema || datasetInfo.columns || [];
    setActiveDataset({
      name: datasetInfo.name || 'Dataset',
      rowCount: rowCnt,
      rows: rowCnt,
      columns: colList,
      loadedAt: Date.now()
    });
    setRowCount(rowCnt);
    if (datasetInfo.schema) setSchema(datasetInfo.schema);
    refreshData();
  }, [refreshData]);

  const fileInputRef = useRef(null);

  const handleHTMLFileChange = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      if (file.path) {
        // Desktop / Electron mode: path string is available
        const resp = await fetch(`${API}/api/connect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source_type: 'csv', file_path: file.path }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Connection failed');
        handleConnected(data);
      } else {
        // Web browser mode: file.path is undefined -> upload via FormData
        const formData = new FormData();
        formData.append('file', file);
        const resp = await fetch(`${API}/api/upload`, {
          method: 'POST',
          body: formData,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'File upload failed');
        handleConnected(data);
      }
    } catch (err) {
      if (err.name === 'TypeError' || (err.message && err.message.includes('fetch'))) {
        alert('Backend Offline: Python server (server.py) is not running. Please start it by running `python server.py` or double-clicking DataPilot-OS.bat');
      } else {
        alert(`File Open Error: ${err.message}`);
      }
    }
  }, [handleConnected]);

  const handleOpenFile = useCallback(async () => {
    try {
      if (isElectron && window.electron?.openFileDialog) {
        const selected = await window.electron.openFileDialog();
        if (selected) {
          const resp = await fetch(`${API}/api/connect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_type: 'csv', file_path: selected }),
          });
          const data = await resp.json();
          if (!resp.ok) throw new Error(data.detail || 'Connection failed');
          handleConnected(data);
          return;
        }
      } else {
        fileInputRef.current?.click();
      }
    } catch (err) {
      if (err.name === 'TypeError' || (err.message && err.message.includes('fetch'))) {
        alert('Backend Offline: Python server (server.py) is not running. Please start it by running `python server.py` or double-clicking DataPilot-OS.bat');
      } else {
        alert(`Connection failed: ${err.message}`);
      }
    }
  }, [handleConnected]);

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    try {
      const res = await fetch(`${API}/api/dataset/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: 'csv' }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.message || 'Save failed');
      alert(data.message || 'File saved successfully in-place!');
    } catch (err) {
      alert(`Save Result: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  }, []);

  const handleUndo = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/operations/undo`, { method: 'POST' });
      if (res.ok) refreshData();
    } catch (err) {
      console.error('Undo error:', err);
    }
  }, [refreshData]);

  const handleRedo = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/operations/redo`, { method: 'POST' });
      if (res.ok) refreshData();
    } catch (err) {
      console.error('Redo error:', err);
    }
  }, [refreshData]);

  return (
    <div className="app-shell" style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', background: 'var(--bg-main)' }}>
      {/* Title Bar with Open/Save Controls */}
      <TitleBar
        datasetName={activeDataset?.name}
        onOpenFile={handleOpenFile}
        onSaveFile={handleSave}
        isSaving={isSaving}
      />

      {/* Ribbon Navigation Header */}
      <Ribbon
        activeModule={activeModule === 'eda' ? 'stats' : activeModule}
        onModuleChange={(mod) => {
          if (mod === 'connect') {
            setIsDbModalOpen(true);
          } else if (mod === 'stats') {
            setActiveModule('eda');
          } else {
            setActiveModule(mod);
          }
        }}
        datasetLoaded={!!activeDataset}
        onOpenFile={handleOpenFile}
        onOpenDbModal={() => setIsDbModalOpen(true)}
        onSaveFile={handleSave}
        onUndo={handleUndo}
        onRedo={handleRedo}
        canUndo={pipelineSteps.length > 1}
        canRedo={false}
        isSaving={isSaving}
        onOpenCodeInspector={() => setIsTerminalOpen(prev => !prev)}
      />

      {/* Dynamic View Component Container according to activeModule */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', position: 'relative' }}>
        {activeModule === 'home' && (
          <AICoPilot
            activeDataset={activeDataset}
            schema={schema}
            datasetLoaded={!!activeDataset}
            onSaveFile={handleSave}
            isSaving={isSaving}
            onRunCodeInTerminal={(code) => {
              setIsTerminalOpen(true);
              setExternalCodeToRun({ code, timestamp: Date.now() });
            }}
            onRefresh={refreshData}
          />
        )}

        {(activeModule === 'clean' || activeModule === 'transform') && (
          <OperationPanel
            mode={activeModule}
            schema={schema}
            onApplied={() => {
              refreshData();
            }}
          />
        )}

        {(activeModule === 'preview' || activeModule === 'data' || activeModule === 'table' || activeModule === 'grid') && (
          <DataPreview
            tableData={tableData}
            schema={schema}
            rowCount={rowCount}
            activeDataset={activeDataset}
            onRefresh={refreshData}
          />
        )}

        {activeModule === 'explorer' && (
          <ObjectExplorer
            connections={activeDataset ? [activeDataset] : []}
            activeDataset={activeDataset}
            onSelectDataset={(ds) => {
              setActiveDataset(ds);
              refreshData();
            }}
            onModuleChange={(mod) => {
              if (mod === 'stats') setActiveModule('eda');
              else setActiveModule(mod);
            }}
            pipelineSteps={pipelineSteps}
          />
        )}

        {(activeModule === 'eda' || activeModule === 'stats') && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', overflow: 'auto', padding: 16, background: '#090d16' }}>
            <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
              <label style={{ fontSize: 12, fontWeight: 700, color: '#38bdf8' }}>Select Column to Inspect:</label>
              <select
                value={selectedColumn || ''}
                onChange={(e) => setSelectedColumn(e.target.value || null)}
                style={{ background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', padding: '5px 10px', borderRadius: 6, fontSize: 12 }}
              >
                <option value="">-- Choose Column --</option>
                {schema?.map(c => <option key={c.name} value={c.name}>{c.name} ({c.dtype || c.type || 'unknown'})</option>)}
              </select>
            </div>
            <PropertiesPanel selectedColumn={selectedColumn} schema={schema} />
          </div>
        )}

        {activeModule === 'ml' && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', background: '#090d16', padding: 16 }}>
            <div style={{ marginBottom: 12, padding: '10px 14px', background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800, color: '#38bdf8' }}>🧠 ML Studio & Intelligent AI Assistant</h3>
              <p style={{ margin: '4px 0 0 0', fontSize: 11, color: '#94a3b8' }}>
                Ask AI Co-Pilot to train ML models (Classification, Regression, AutoML), calculate feature importance, run VIF multicollinearity, or perform clustering.
              </p>
            </div>
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <AICoPilot
                activeDataset={activeDataset}
                schema={schema}
                onRunCodeInTerminal={(code) => {
                  setIsTerminalOpen(true);
                  setExternalCodeToRun({ code, timestamp: Date.now() });
                }}
                onRefresh={refreshData}
              />
            </div>
          </div>
        )}

        {activeModule === 'report' && (
          <ReportView activeDataset={activeDataset} pipelineSteps={pipelineSteps} />
        )}
      </div>

      {/* Collapsible Bottom Terminal HUD Drawer */}
      <div style={{
        flexShrink: 0,
        background: '#090d16',
        borderTop: '1px solid #1e293b',
        display: 'flex',
        flexDirection: 'column',
        height: !isTerminalOpen ? '34px' : (activePlotImage ? '520px' : '320px'),
        transition: 'height 0.25s cubic-bezier(0.16, 1, 0.3, 1)'
      }}>
        {/* Toggle Bar */}
        <div
          onClick={() => setIsTerminalOpen(!isTerminalOpen)}
          style={{
            height: 34,
            padding: '0 14px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            background: '#0f172a',
            borderBottom: isTerminalOpen ? '1px solid #1e293b' : 'none',
            userSelect: 'none'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, fontWeight: 700, color: '#38bdf8' }}>
            <Terminal size={14} color="#38bdf8" />
            <span>AI Terminal & Execution Engine</span>
            <span style={{ fontSize: 10, color: '#64748b', fontWeight: 500 }}>
              (Click to {isTerminalOpen ? 'collapse' : 'expand'})
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#94a3b8', fontSize: 11 }}>
            <span>{isTerminalOpen ? 'Collapse' : 'Expand Terminal'}</span>
            {isTerminalOpen ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
          </div>
        </div>

        {/* Terminal HUD & Visual Insight Canvas Container (Collapses TOGETHER!) */}
        {isTerminalOpen && (
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', background: '#ffffff' }}>
            <div style={{ minHeight: 280, flexShrink: 0 }}>
              <TerminalHUD
                schema={schema}
                rowCount={rowCount}
                pipelineSteps={pipelineSteps}
                onRefresh={refreshData}
                externalCodeToRun={externalCodeToRun}
                onPlotGenerated={(b64Plot) => {
                  setActivePlotImage(b64Plot);
                }}
              />
            </div>

            {/* Generated Pro Visualization Canvas (Inside Drawer - Collapses with Terminal!) */}
            {activePlotImage && (
              <div
                ref={plotCanvasRef}
                style={{
                  flexShrink: 0,
                  background: '#ffffff',
                  borderTop: '3px solid #0ea5e9',
                  padding: '16px 20px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 -4px 15px rgba(0,0,0,0.08)'
                }}
              >
                <div style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: 10,
                  borderBottom: '1px solid #e2e8f0',
                  paddingBottom: 8
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }} />
                    <span style={{ fontSize: 13, fontWeight: 800, color: '#0f172a', letterSpacing: '0.01em' }}>
                      📊 Data Science Analytics & Visual Insight Canvas
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <a
                      href={activePlotImage}
                      download="datapilot_visualization.png"
                      style={{
                        background: '#0284c7',
                        color: '#ffffff',
                        textDecoration: 'none',
                        borderRadius: 5,
                        padding: '4px 12px',
                        fontSize: 11,
                        fontWeight: 700,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 4
                      }}
                    >
                      📥 Download PNG
                    </a>
                    <button
                      onClick={() => setActivePlotImage(null)}
                      style={{
                        background: '#f1f5f9',
                        border: '1px solid #cbd5e1',
                        borderRadius: 5,
                        padding: '4px 12px',
                        cursor: 'pointer',
                        color: '#475569',
                        fontSize: 11,
                        fontWeight: 700
                      }}
                    >
                      ✕ Close Chart
                    </button>
                  </div>
                </div>

                <div style={{
                  background: '#ffffff',
                  borderRadius: 8,
                  padding: 12,
                  border: '1px solid #cbd5e1',
                  display: 'flex',
                  justifyContent: 'center',
                  width: '100%',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.06)'
                }}>
                  <img
                    src={activePlotImage}
                    alt="Generated Visualization Chart"
                    style={{ width: '100%', maxHeight: 400, objectFit: 'contain', borderRadius: 4 }}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Clean Status Bar */}
      <StatusBar
        connectionStatus={engineStatus}
        datasetName={activeDataset?.name}
        rowCount={rowCount}
        colCount={schema?.length ?? 0}
        dirtyCount={schema ? schema.filter(c => (c.null_count ?? 0) > 0).length : 0}
        activeModule={activeModule}
      />

      {/* DB Connection Modal */}
      <DBConnectionModal
        isOpen={isDbModalOpen}
        onClose={() => setIsDbModalOpen(false)}
        onImportSuccess={(data) => {
          handleConnected(data);
          setIsDbModalOpen(false);
        }}
      />

      {/* Fallback Native File Picker */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleHTMLFileChange}
        style={{ display: 'none' }}
        accept=".csv,.xlsx,.xls,.parquet,.json"
      />
    </div>
  );
}
