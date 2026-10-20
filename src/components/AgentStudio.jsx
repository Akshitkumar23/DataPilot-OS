import React, { useState, useEffect, useCallback } from 'react';
import {
  FolderOpen, Folder, File, FileText, Search, Play, CheckCircle2,
  XCircle, AlertTriangle, Shield, ShieldAlert, ShieldCheck, Terminal,
  RefreshCw, RotateCcw, Ban, Check, X, Eye, FileSpreadsheet,
  ChevronRight, ChevronDown, Layers, Sparkles, Database, Trash2
} from 'lucide-react';
import { API, getAuthHeaders } from '../apiConfig';

export default function AgentStudio({ activeDataset, onDatasetConnected }) {
  const [workspaces, setWorkspaces] = useState([]);
  const [activeWorkspace, setActiveWorkspace] = useState('');
  const [policy, setPolicy] = useState('review');
  const [fileTree, setFileTree] = useState([]);
  const [disclosedExclusions, setDisclosedExclusions] = useState([]);
  const [isLoadingTree, setIsLoadingTree] = useState(false);

  // Search & Inspection
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileDetails, setFileDetails] = useState(null);
  const [isLoadingFile, setIsLoadingFile] = useState(false);

  // Task planning & execution
  const [taskPrompt, setTaskPrompt] = useState('');
  const [isPlanning, setIsPlanning] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [currentPlan, setCurrentPlan] = useState(null);
  const [planError, setPlanError] = useState(null);
  const [selectedDiff, setSelectedDiff] = useState(null);
  const [activeSubTab, setActiveSubTab] = useState('tasks'); // 'tasks' | 'terminal' | 'audit'

  // Audit records
  const [auditRecords, setAuditRecords] = useState([]);

  // Fetch workspaces & policy on mount
  const fetchWorkspaces = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/agent/workspaces`, { headers: getAuthHeaders() });
      if (res.ok) {
        const data = await res.json();
        setWorkspaces(data.authorized_workspaces || []);
        setActiveWorkspace(data.active_workspace || '');
        setPolicy(data.policy || 'review');
      }
    } catch (err) {
      console.warn('Failed to fetch workspaces:', err);
    }
  }, []);

  const fetchDirectory = useCallback(async (dirPath) => {
    if (!dirPath) return;
    setIsLoadingTree(true);
    try {
      const res = await fetch(`${API}/api/agent/tools/invoke`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          tool_name: 'list_directory',
          arguments: { path: dirPath, recursive: false },
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setFileTree(data.output || []);
        setDisclosedExclusions(data.metadata?.disclosed_exclusions || []);
      }
    } catch (err) {
      console.error('Failed to list directory:', err);
    } finally {
      setIsLoadingTree(false);
    }
  }, []);

  const fetchAuditLog = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/agent/audit?limit=50`, { headers: getAuthHeaders() });
      if (res.ok) {
        const data = await res.json();
        setAuditRecords(data.records || []);
      }
    } catch (err) {
      console.warn('Failed to fetch audit log:', err);
    }
  }, []);

  useEffect(() => {
    fetchWorkspaces();
    fetchAuditLog();
  }, [fetchWorkspaces, fetchAuditLog]);

  useEffect(() => {
    if (activeWorkspace) {
      fetchDirectory(activeWorkspace);
    }
  }, [activeWorkspace, fetchDirectory]);

  // Open native directory dialog
  const handleOpenFolderDialog = async () => {
    let chosenPath = null;
    if (window.electron && typeof window.electron.openDirectoryDialog === 'function') {
      chosenPath = await window.electron.openDirectoryDialog({
        title: 'Select Workspace Directory for DataPilot Agent',
      });
    } else {
      chosenPath = window.prompt('Enter absolute local directory path:');
    }

    if (!chosenPath) return;

    try {
      const res = await fetch(`${API}/api/agent/workspaces/authorize`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ path: chosenPath }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        setActiveWorkspace(data.authorized_path);
        setWorkspaces(data.workspaces || []);
        fetchDirectory(data.authorized_path);
      } else {
        alert(data.message || 'Could not authorize directory.');
      }
    } catch (err) {
      alert(`Authorization error: ${err.message}`);
    }
  };

  const handleRevokeWorkspace = async (target) => {
    try {
      const res = await fetch(`${API}/api/agent/workspaces/revoke`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ path: target }),
      });
      const data = await res.json();
      if (res.ok) {
        setWorkspaces(data.workspaces || []);
        setActiveWorkspace(data.active_workspace || '');
        if (data.active_workspace) fetchDirectory(data.active_workspace);
        else setFileTree([]);
      }
    } catch (err) {
      console.error('Revocation failed:', err);
    }
  };

  const handlePolicyChange = async (newPolicy) => {
    setPolicy(newPolicy);
    try {
      await fetch(`${API}/api/agent/policy`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ policy: newPolicy }),
      });
    } catch (err) {
      console.error('Failed to set policy:', err);
    }
  };

  // Inspect / Read File
  const handleSelectFile = async (entry) => {
    if (entry.is_dir) {
      fetchDirectory(entry.path);
      return;
    }
    setSelectedFile(entry);
    setIsLoadingFile(true);
    setFileDetails(null);
    try {
      const res = await fetch(`${API}/api/agent/tools/invoke`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          tool_name: 'read_file',
          arguments: { path: entry.path, max_bytes: 524288 },
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setFileDetails({
          content: data.output,
          encoding: data.metadata?.encoding,
          sha256: data.metadata?.sha256,
          total_lines: data.metadata?.total_lines,
          is_binary: data.metadata?.is_binary,
        });
      } else {
        setFileDetails({ error: data.message || 'Could not read file.' });
      }
    } catch (err) {
      setFileDetails({ error: err.message });
    } finally {
      setIsLoadingFile(false);
    }
  };

  // Connect local file as Dataset In-Place
  const handleConnectDatasetInPlace = async (filePath) => {
    try {
      const res = await fetch(`${API}/api/datasets/connect`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          source_type: 'local',
          file_path: filePath,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        alert(`Dataset '${data.name}' connected in-place as source-native!`);
        if (onDatasetConnected) onDatasetConnected();
      } else {
        alert(`Failed to connect dataset: ${data.detail || data.message}`);
      }
    } catch (err) {
      alert(`Connect error: ${err.message}`);
    }
  };

  // Search files & contents
  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!searchQuery.trim() || !activeWorkspace) return;
    setIsSearching(true);
    try {
      const res = await fetch(`${API}/api/agent/tools/invoke`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          tool_name: 'search_files',
          arguments: { workspace_path: activeWorkspace, pattern: searchQuery },
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setSearchResults(data.output || []);
      }
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setIsSearching(false);
    }
  };

  // Plan natural language task
  const handlePlanTask = async (customPrompt) => {
    const promptToUse = customPrompt || taskPrompt;
    if (!promptToUse.trim()) return;

    setIsPlanning(true);
    setPlanError(null);
    setCurrentPlan(null);
    setSelectedDiff(null);

    try {
      const res = await fetch(`${API}/api/agent/plan`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          prompt: promptToUse,
          workspace_path: activeWorkspace,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setCurrentPlan(data);
      } else {
        setPlanError(data.message || data.required_action || 'Failed to synthesize task plan.');
      }
    } catch (err) {
      setPlanError(`Planner communication error: ${err.message}`);
    } finally {
      setIsPlanning(false);
    }
  };

  // Execute current plan
  const handleExecuteTask = async () => {
    if (!currentPlan) return;
    setIsExecuting(true);
    try {
      const res = await fetch(`${API}/api/agent/execute`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ task_id: currentPlan.task_id }),
      });
      const data = await res.json();
      if (res.ok) {
        setCurrentPlan(data);
        fetchDirectory(activeWorkspace);
        fetchAuditLog();
      } else {
        alert(data.message || 'Execution halted.');
      }
    } catch (err) {
      alert(`Execution error: ${err.message}`);
    } finally {
      setIsExecuting(false);
    }
  };

  // Approve a step
  const handleApproveStep = async (stepId) => {
    if (!currentPlan) return;
    setIsExecuting(true);
    try {
      const res = await fetch(`${API}/api/agent/tasks/${currentPlan.task_id}/approve`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ step_id: stepId }),
      });
      const data = await res.json();
      if (res.ok) {
        setCurrentPlan(data);
        fetchDirectory(activeWorkspace);
        fetchAuditLog();
      } else {
        alert(data.message || 'Step approval failed.');
      }
    } catch (err) {
      alert(`Approval error: ${err.message}`);
    } finally {
      setIsExecuting(false);
    }
  };

  // Reject a step
  const handleRejectStep = async (stepId) => {
    if (!currentPlan) return;
    try {
      const res = await fetch(`${API}/api/agent/tasks/${currentPlan.task_id}/reject`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ step_id: stepId, reason: 'Rejected by user via UI' }),
      });
      const data = await res.json();
      if (res.ok) {
        setCurrentPlan(data);
        fetchAuditLog();
      }
    } catch (err) {
      alert(`Rejection error: ${err.message}`);
    }
  };

  // Cancel task
  const handleCancelTask = async () => {
    if (!currentPlan) return;
    try {
      const res = await fetch(`${API}/api/agent/tasks/${currentPlan.task_id}/cancel`, {
        method: 'POST',
        headers: getAuthHeaders(),
      });
      const data = await res.json();
      if (res.ok) {
        setCurrentPlan(data);
      }
    } catch (err) {
      console.error('Cancel error:', err);
    }
  };

  // Rollback task
  const handleRollbackTask = async () => {
    if (!currentPlan) return;
    if (!window.confirm('Roll back all file changes made by this task?')) return;
    try {
      const res = await fetch(`${API}/api/agent/tasks/${currentPlan.task_id}/rollback`, {
        method: 'POST',
        headers: getAuthHeaders(),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        alert('File changes rolled back successfully.');
        fetchDirectory(activeWorkspace);
        fetchAuditLog();
      } else {
        alert('Rollback failed or no backups available.');
      }
    } catch (err) {
      alert(`Rollback error: ${err.message}`);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#050811', color: '#f1f5f9', overflow: 'hidden' }}>
      
      {/* 1. TOP WORKSPACE & GOVERNANCE BAR */}
      <div style={{
        background: '#0a0f1d', borderBottom: '1px solid #1e293b', padding: '12px 20px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexShrink: 0
      }}>
        {/* Workspace Root Badge & Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
          <div style={{ background: '#0284c7', padding: '8px 10px', borderRadius: 8, display: 'flex', alignItems: 'center' }}>
            <FolderOpen size={18} color="#fff" />
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>AUTHORIZED WORKSPACE ROOT</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {activeWorkspace ? (
                <span style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 420 }} title={activeWorkspace}>
                  {activeWorkspace}
                </span>
              ) : (
                <span style={{ fontSize: 13, color: '#ef4444', fontWeight: 600 }}>No workspace selected</span>
              )}
              {activeWorkspace && (
                <button
                  onClick={() => handleRevokeWorkspace(activeWorkspace)}
                  className="btn btn--ghost btn--sm"
                  style={{ color: '#ef4444', padding: '2px 6px', fontSize: 11 }}
                  title="Revoke workspace access"
                >
                  <Trash2 size={12} /> Revoke
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Center: Quick Action Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            onClick={handleOpenFolderDialog}
            className="btn btn--primary"
            style={{ padding: '8px 14px', fontSize: 12, fontWeight: 700 }}
          >
            <FolderOpen size={14} /> Open Workspace Folder
          </button>
        </div>

        {/* Right: Policy Engine Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: '4px 10px' }}>
          <Shield size={16} color={policy === 'review' ? '#38bdf8' : policy === 'auto_edit' ? '#f59e0b' : '#10b981'} />
          <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Policy:</span>
          <select
            value={policy}
            onChange={(e) => handlePolicyChange(e.target.value)}
            style={{
              background: 'transparent', color: '#f8fafc', border: 'none', fontSize: 12,
              fontWeight: 700, outline: 'none', cursor: 'pointer'
            }}
          >
            <option value="review" style={{ background: '#0f172a' }}>Review (Requires Approvals)</option>
            <option value="auto_edit" style={{ background: '#0f172a' }}>Auto-edit (Writes Authorized)</option>
            <option value="autonomous" style={{ background: '#0f172a' }}>Controlled Autonomous</option>
          </select>
        </div>
      </div>

      {/* 2. MAIN TWO-PANEL WORKSPACE BODY */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* LEFT PANEL: Workspace Explorer & Search (Width 360px) */}
        <div style={{
          width: 360, borderRight: '1px solid #1e293b', background: '#080c18',
          display: 'flex', flexDirection: 'column', overflow: 'hidden'
        }}>
          {/* Search Header */}
          <div style={{ padding: '12px 14px', borderBottom: '1px solid #1e293b' }}>
            <form onSubmit={handleSearch} style={{ display: 'flex', gap: 6 }}>
              <div style={{ flex: 1, position: 'relative' }}>
                <Search size={14} style={{ position: 'absolute', left: 10, top: 10, color: '#64748b' }} />
                <input
                  type="text"
                  placeholder="Search files in workspace..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{
                    width: '100%', background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6,
                    padding: '6px 10px 6px 30px', fontSize: 12, color: '#f8fafc', outline: 'none'
                  }}
                />
              </div>
              <button type="submit" className="btn btn--ghost btn--sm" style={{ padding: '6px 10px' }} disabled={isSearching}>
                {isSearching ? <RefreshCw size={12} className="spin" /> : 'Find'}
              </button>
            </form>
          </div>

          {/* Directory Tree Toolbar */}
          <div style={{
            padding: '8px 14px', background: '#0a0f1d', borderBottom: '1px solid #1e293b',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8'
          }}>
            <span>WORKSPACE FILES ({fileTree.length})</span>
            <button
              onClick={() => fetchDirectory(activeWorkspace)}
              className="btn btn--ghost btn--sm"
              style={{ color: '#94a3b8', padding: '2px 4px' }}
              title="Refresh Directory"
            >
              <RefreshCw size={12} className={isLoadingTree ? 'spin' : ''} />
            </button>
          </div>

          {/* File Tree List */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px 10px' }}>
            {isLoadingTree ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#64748b', fontSize: 12 }}>
                <RefreshCw size={18} className="spin" style={{ margin: '0 auto 8px' }} />
                Scanning workspace files...
              </div>
            ) : fileTree.length === 0 ? (
              <div style={{ padding: 24, textAlign: 'center', color: '#64748b', fontSize: 12 }}>
                {activeWorkspace ? 'Workspace directory is empty.' : 'Authorize a workspace folder to view files.'}
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {fileTree.map((item) => {
                  const isSelected = selectedFile?.path === item.path;
                  const isCsvOrExcel = /\.(csv|tsv|parquet|xlsx|xls)$/i.test(item.name);
                  return (
                    <div
                      key={item.path}
                      onClick={() => handleSelectFile(item)}
                      style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '6px 10px', borderRadius: 6, fontSize: 12, cursor: 'pointer',
                        background: isSelected ? '#1e293b' : 'transparent',
                        color: isSelected ? '#38bdf8' : '#cbd5e1',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, overflow: 'hidden' }}>
                        {item.is_dir ? <Folder size={14} color="#f59e0b" /> : isCsvOrExcel ? <FileSpreadsheet size={14} color="#10b981" /> : <FileText size={14} color="#94a3b8" />}
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.name}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        {isCsvOrExcel && (
                          <button
                            onClick={(e) => { e.stopPropagation(); handleConnectDatasetInPlace(item.path); }}
                            className="btn btn--ghost btn--sm"
                            style={{ padding: '2px 6px', fontSize: 10, color: '#10b981', border: '1px solid #059669' }}
                            title="Connect directly without copying to uploads"
                          >
                            +Dataset
                          </button>
                        )}
                        {!item.is_dir && item.size !== null && (
                          <span style={{ fontSize: 10, color: '#64748b' }}>
                            {item.size > 1024 ? `${Math.round(item.size / 1024)} KB` : `${item.size} B`}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Disclosed Exclusions Notice */}
            {disclosedExclusions.length > 0 && (
              <div style={{
                marginTop: 14, padding: '8px 10px', background: '#0a0f1d', border: '1px solid #1e293b',
                borderRadius: 6, fontSize: 11, color: '#64748b'
              }}>
                <div style={{ fontWeight: 600, color: '#94a3b8', marginBottom: 2 }}>Excluded by default:</div>
                <div>{disclosedExclusions.join(', ')}</div>
              </div>
            )}
          </div>

          {/* Quick File Inspection Previewer (Bottom of Left Panel) */}
          {selectedFile && (
            <div style={{
              height: 220, borderTop: '1px solid #1e293b', background: '#0a0f1d',
              display: 'flex', flexDirection: 'column', overflow: 'hidden'
            }}>
              <div style={{
                padding: '6px 12px', background: '#0f172a', borderBottom: '1px solid #1e293b',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11
              }}>
                <span style={{ fontWeight: 700, color: '#f8fafc', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {selectedFile.name}
                </span>
                <span style={{ color: '#64748b' }}>
                  {fileDetails?.encoding || 'utf-8'}
                </span>
              </div>
              <div style={{ flex: 1, overflow: 'auto', padding: 10, fontFamily: 'monospace', fontSize: 11, background: '#050811' }}>
                {isLoadingFile ? (
                  <div style={{ color: '#64748b' }}>Reading file...</div>
                ) : fileDetails?.error ? (
                  <div style={{ color: '#ef4444' }}>{fileDetails.error}</div>
                ) : (
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                    {fileDetails?.content || 'No content.'}
                  </pre>
                )}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT PANEL: Task Console, Execution HUD, Diff Viewer */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* Prompt Input Header */}
          <div style={{ padding: '16px 24px', background: '#090d16', borderBottom: '1px solid #1e293b' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <Sparkles size={16} color="#38bdf8" />
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#f8fafc' }}>
                Workspace Agent Task Console
              </h3>
            </div>
            
            <form onSubmit={(e) => { e.preventDefault(); handlePlanTask(); }} style={{ display: 'flex', gap: 10 }}>
              <input
                type="text"
                placeholder='e.g., "Inspect this folder and explain architecture", "Create docs/architecture.md", "Run tests and verify"'
                value={taskPrompt}
                onChange={(e) => setTaskPrompt(e.target.value)}
                style={{
                  flex: 1, background: '#0f172a', border: '1px solid #334155', borderRadius: 8,
                  padding: '10px 16px', color: '#f8fafc', fontSize: 13, outline: 'none'
                }}
              />
              <button
                type="submit"
                className="btn btn--primary"
                disabled={isPlanning || !activeWorkspace || !taskPrompt.trim()}
                style={{ padding: '10px 20px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}
              >
                {isPlanning ? <RefreshCw size={15} className="spin" /> : <Play size={15} />}
                {isPlanning ? 'Planning...' : 'Plan Task'}
              </button>
            </form>

            {/* Quick Suggestion Chips */}
            <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
              {[
                'Inspect folder architecture',
                'Read requirements.txt and identify dependencies',
                'Find test files and execute them',
                'Create config.json with sample settings',
              ].map((s) => (
                <button
                  key={s}
                  onClick={() => { setTaskPrompt(s); handlePlanTask(s); }}
                  className="btn btn--ghost btn--sm"
                  style={{ fontSize: 11, background: '#0f172a', border: '1px solid #1e293b', color: '#94a3b8' }}
                >
                  {s}
                </button>
              ))}
            </div>

            {planError && (
              <div style={{
                marginTop: 12, padding: '10px 14px', background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid #ef4444', borderRadius: 6, fontSize: 12, color: '#fca5a5',
                display: 'flex', alignItems: 'center', gap: 8
              }}>
                <AlertTriangle size={15} color="#ef4444" />
                <span>{planError}</span>
              </div>
            )}
          </div>

          {/* Subtabs for Right Panel: Tasks & Diffs | Terminal HUD | Audit Trail */}
          <div style={{
            display: 'flex', borderBottom: '1px solid #1e293b', background: '#0a0f1d', padding: '0 24px'
          }}>
            {[
              { id: 'tasks', label: 'Task Execution & Diffs', icon: Layers },
              { id: 'terminal', label: 'Command Runner HUD', icon: Terminal },
              { id: 'audit', label: 'Operation Audit Log', icon: ShieldCheck },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeSubTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveSubTab(tab.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8, padding: '10px 16px',
                    border: 'none', background: 'transparent',
                    borderBottom: isActive ? '2px solid #38bdf8' : '2px solid transparent',
                    color: isActive ? '#38bdf8' : '#94a3b8', fontWeight: isActive ? 700 : 500,
                    cursor: 'pointer', fontSize: 12
                  }}
                >
                  <Icon size={14} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Subtab 1: Tasks & Diffs */}
          {activeSubTab === 'tasks' && (
            <div style={{ flex: 1, overflowY: 'auto', padding: 24, display: 'flex', flexDirection: 'column', gap: 18 }}>
              {currentPlan ? (
                <>
                  {/* Task Objective Header */}
                  <div style={{
                    background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 16,
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between'
                  }}>
                    <div>
                      <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>TASK OBJECTIVE</div>
                      <div style={{ fontSize: 15, fontWeight: 700, color: '#f8fafc', marginTop: 2 }}>
                        {currentPlan.objective}
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{
                        padding: '4px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
                        background: currentPlan.status === 'COMPLETED' ? '#065f46' : currentPlan.status === 'FAILED' ? '#991b1b' : '#0369a1',
                        color: '#fff'
                      }}>
                        {currentPlan.status}
                      </span>
                      {currentPlan.status !== 'COMPLETED' && currentPlan.status !== 'FAILED' && (
                        <button
                          onClick={handleExecuteTask}
                          className="btn btn--primary btn--sm"
                          disabled={isExecuting}
                          style={{ padding: '6px 14px', fontWeight: 700 }}
                        >
                          <Play size={12} /> {isExecuting ? 'Running...' : 'Execute Plan'}
                        </button>
                      )}
                      <button
                        onClick={handleCancelTask}
                        className="btn btn--danger btn--sm"
                        style={{ padding: '6px 10px' }}
                        title="Cancel task execution"
                      >
                        <Ban size={12} /> Cancel
                      </button>
                      <button
                        onClick={handleRollbackTask}
                        className="btn btn--ghost btn--sm"
                        style={{ color: '#f59e0b', border: '1px solid #f59e0b', padding: '6px 10px' }}
                        title="Roll back modified files"
                      >
                        <RotateCcw size={12} /> Rollback
                      </button>
                    </div>
                  </div>

                  {/* Planned Steps List */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>PLANNED STEPS ({currentPlan.steps.length})</div>
                    {currentPlan.steps.map((step, idx) => {
                      const isPendingApproval = step.status === 'AWAITING_APPROVAL';
                      return (
                        <div
                          key={step.step_id}
                          style={{
                            background: isPendingApproval ? 'rgba(245, 158, 11, 0.08)' : '#0f172a',
                            border: isPendingApproval ? '1px solid #f59e0b' : '1px solid #1e293b',
                            borderRadius: 8, padding: 14, display: 'flex', flexDirection: 'column', gap: 10
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                              <span style={{ fontSize: 11, fontWeight: 700, color: '#64748b' }}>#{idx + 1}</span>
                              <span style={{
                                padding: '2px 8px', borderRadius: 4, background: '#1e293b',
                                color: '#38bdf8', fontSize: 11, fontFamily: 'monospace', fontWeight: 700
                              }}>
                                {step.tool_name}
                              </span>
                              <span style={{ fontSize: 13, color: '#cbd5e1' }}>{step.explanation}</span>
                            </div>

                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{
                                fontSize: 11, fontWeight: 700,
                                color: step.status === 'SUCCEEDED' ? '#34d399' : step.status === 'FAILED' ? '#ef4444' : isPendingApproval ? '#f59e0b' : '#94a3b8'
                              }}>
                                {step.status}
                              </span>
                            </div>
                          </div>

                          {/* Approval Banner if Step Requires Approval */}
                          {isPendingApproval && (
                            <div style={{
                              background: '#1c1917', border: '1px solid #78350f', borderRadius: 6,
                              padding: '10px 14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between'
                            }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#fde68a' }}>
                                <AlertTriangle size={15} color="#f59e0b" />
                                <span>Sensitive operation requires explicit user approval under current policy ({policy}).</span>
                              </div>
                              <div style={{ display: 'flex', gap: 8 }}>
                                <button
                                  onClick={() => handleApproveStep(step.step_id)}
                                  className="btn btn--primary btn--sm"
                                  style={{ background: '#10b981', border: 'none', padding: '4px 12px', fontSize: 11, fontWeight: 700 }}
                                >
                                  <Check size={12} /> Approve Step
                                </button>
                                <button
                                  onClick={() => handleRejectStep(step.step_id)}
                                  className="btn btn--danger btn--sm"
                                  style={{ padding: '4px 10px', fontSize: 11 }}
                                >
                                  <X size={12} /> Reject
                                </button>
                              </div>
                            </div>
                          )}

                          {/* Step Output / Diff preview */}
                          {step.result && typeof step.result === 'string' && step.result.includes('@@') && (
                            <div style={{ background: '#050811', border: '1px solid #1e293b', borderRadius: 6, padding: 10, fontFamily: 'monospace', fontSize: 11 }}>
                              <pre style={{ margin: 0, color: '#34d399', whiteSpace: 'pre-wrap' }}>{step.result}</pre>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  {/* Task Summary Card */}
                  {currentPlan.summary && (
                    <div style={{
                      background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 16,
                      display: 'flex', flexDirection: 'column', gap: 8
                    }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#f8fafc' }}>EXECUTION SUMMARY</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, fontSize: 12 }}>
                        <div>
                          <span style={{ color: '#94a3b8' }}>Files Inspected: </span>
                          <strong style={{ color: '#38bdf8' }}>{(currentPlan.summary.files_inspected || []).length}</strong>
                        </div>
                        <div>
                          <span style={{ color: '#94a3b8' }}>Files Changed: </span>
                          <strong style={{ color: '#10b981' }}>{(currentPlan.summary.files_changed || []).length}</strong>
                        </div>
                        <div>
                          <span style={{ color: '#94a3b8' }}>Commands Executed: </span>
                          <strong style={{ color: '#a78bfa' }}>{(currentPlan.summary.commands_executed || []).length}</strong>
                        </div>
                        <div>
                          <span style={{ color: '#94a3b8' }}>Checks Passed: </span>
                          <strong style={{ color: '#34d399' }}>{(currentPlan.summary.checks_passed || []).length}</strong>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div style={{
                  padding: 40, textAlign: 'center', color: '#64748b', display: 'flex',
                  flexDirection: 'column', alignItems: 'center', gap: 12
                }}>
                  <Sparkles size={36} color="#334155" />
                  <h4 style={{ margin: 0, color: '#94a3b8', fontSize: 16 }}>No Active Task Plan</h4>
                  <p style={{ margin: 0, maxWidth: 440, fontSize: 13, lineHeight: 1.5 }}>
                    Enter a natural language request above to plan architecture inspections, multi-file code modifications, or targeted command executions.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Subtab 2: Command Runner Terminal HUD */}
          {activeSubTab === 'terminal' && (
            <div style={{ flex: 1, background: '#050811', padding: 20, overflowY: 'auto', fontFamily: 'monospace', fontSize: 12 }}>
              <div style={{ color: '#38bdf8', marginBottom: 12 }}>
                DataPilot OS Controlled Subprocess Terminal
              </div>
              {currentPlan?.summary?.commands_executed?.length > 0 ? (
                currentPlan.summary.commands_executed.map((cmd, i) => (
                  <div key={i} style={{ marginBottom: 16, borderBottom: '1px solid #1e293b', paddingBottom: 12 }}>
                    <div style={{ color: '#f59e0b' }}>$ {cmd.command}</div>
                    <div style={{ color: cmd.exit_code === 0 ? '#34d399' : '#ef4444', fontSize: 11, marginTop: 4 }}>
                      Exit Code: {cmd.exit_code} • Duration: {cmd.duration_ms}ms
                    </div>
                  </div>
                ))
              ) : (
                <div style={{ color: '#64748b' }}>No commands executed yet in current task session.</div>
              )}
            </div>
          )}

          {/* Subtab 3: Audit Log */}
          {activeSubTab === 'audit' && (
            <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>RECENT AGENT OPERATIONS ({auditRecords.length})</span>
                <button onClick={fetchAuditLog} className="btn btn--ghost btn--sm" style={{ color: '#38bdf8' }}>
                  <RefreshCw size={12} /> Refresh Audit
                </button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {auditRecords.map((r, i) => (
                  <div
                    key={i}
                    style={{
                      background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6,
                      padding: '8px 12px', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'space-between'
                    }}
                  >
                    <div>
                      <span style={{ color: '#38bdf8', fontWeight: 700, marginRight: 8 }}>{r.operation}</span>
                      <span style={{ color: '#cbd5e1' }}>{r.path || r.command || 'N/A'}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ color: '#64748b' }}>{r.timestamp?.slice(11, 19)}</span>
                      <span style={{ color: r.status === 'SUCCESS' ? '#34d399' : '#ef4444', fontWeight: 700 }}>
                        {r.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
