import React, { useState, useEffect, useCallback } from 'react';
import {
  Database, HardDrive, Server, Link, AlertCircle, CheckCircle2,
  Loader2, X, Play, ArrowRight, Table, Terminal, RefreshCw
} from 'lucide-react';

import { API, getAuthHeaders } from '../apiConfig';

const ENGINES = [
  { id: 'sqlite', name: 'SQLite', icon: HardDrive, desc: 'Local .db or .sqlite file' },
  { id: 'postgresql', name: 'PostgreSQL', icon: Server, desc: 'Enterprise Postgres database' },
  { id: 'mysql', name: 'MySQL', icon: Database, desc: 'MySQL / MariaDB database' },
  { id: 'duckdb', name: 'DuckDB', icon: HardDrive, desc: 'Fast analytical .duckdb database' },
  { id: 'custom', name: 'Custom URI', icon: Link, desc: 'Direct connection URI string' },
];

export default function DBConnectionModal({ isOpen, onClose, onImportSuccess }) {
  const [engine, setEngine] = useState('sqlite');
  
  // Connection Form Fields
  const [filePath, setFilePath] = useState('');
  const [host, setHost] = useState('localhost');
  const [port, setPort] = useState(5432);
  const [user, setUser] = useState('');
  const [password, setPassword] = useState('');
  const [database, setDatabase] = useState('');
  const [uri, setUri] = useState('');

  // Table & Query Selection States
  const [fetchedTables, setFetchedTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState('');
  const [sqlQuery, setSqlQuery] = useState('');

  // UI Status States
  const [isTesting, setIsTesting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [testSuccess, setTestSuccess] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // Update default port on engine change
  useEffect(() => {
    if (engine === 'postgresql') setPort(5432);
    else if (engine === 'mysql') setPort(3306);
    setErrorMsg(null);
    setTestSuccess(null);
  }, [engine]);

  // Escape key handler to close modal
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose?.();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // When table is selected, update default SQL query
  const handleTableChange = (tableName) => {
    setSelectedTable(tableName);
    if (tableName) {
      setSqlQuery(`SELECT * FROM "${tableName}"`);
    } else {
      setSqlQuery('');
    }
  };

  const getPayload = () => ({
    engine,
    file_path: filePath.trim(),
    host: host.trim(),
    port: port ? Number(port) : null,
    user: user.trim(),
    username: user.trim(),
    password,
    database: database.trim(),
    uri: uri.trim(),
  });

  // ── Test Connection & Fetch Tables ─────────────────────────────
  const handleTestConnection = async () => {
    setIsTesting(true);
    setErrorMsg(null);
    setTestSuccess(null);

    try {
      const resp = await fetch(`${API}/api/db/test`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(getPayload()),
      });

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.detail || data.message || 'Failed to connect to database');
      }

      const tables = data.tables || [];
      setFetchedTables(tables);
      setTestSuccess(`Successfully connected! Found ${tables.length} table${tables.length === 1 ? '' : 's'}.`);
      
      if (tables.length > 0) {
        setSelectedTable(tables[0]);
        setSqlQuery(`SELECT * FROM "${tables[0]}"`);
      }
    } catch (err) {
      setErrorMsg(err.message || 'Connection test failed. Check server status or credentials.');
    } finally {
      setIsTesting(false);
    }
  };

  // ── Import Table into DataPilot OS ────────────────────────────
  const handleImportTable = async () => {
    setIsImporting(true);
    setErrorMsg(null);

    try {
      const payload = {
        ...getPayload(),
        table_name: selectedTable,
        query: sqlQuery.trim() || (selectedTable ? `SELECT * FROM "${selectedTable}"` : ''),
      };

      const resp = await fetch(`${API}/api/db/import`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload),
      });

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.detail || data.message || 'Database table import failed');
      }

      if (onImportSuccess) {
        onImportSuccess(data);
      }
      if (onClose) {
        onClose();
      }
    } catch (err) {
      setErrorMsg(err.message || 'Import failed. Check SQL query syntax or connection.');
    } finally {
      setIsImporting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(9, 13, 22, 0.82)',
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
        padding: '16px',
        animation: 'fadeIn 0.2s ease-out'
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
    >
      <style>{`
        @keyframes modalSlideUp {
          from { opacity: 0; transform: translateY(12px) scale(0.98); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .db-modal-input:focus {
          outline: none;
          border-color: #10b981 !important;
          box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.25) !important;
        }
        .db-engine-card:hover {
          border-color: #10b981 !important;
          background: #1e293b !important;
        }
        .db-spin {
          animation: spin 1s linear infinite;
        }
      `}</style>

      <div
        style={{
          width: '100%',
          maxWidth: '680px',
          maxHeight: '90vh',
          backgroundColor: '#0f172a',
          border: '1px solid #1e293b',
          borderRadius: '14px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.75), 0 0 0 1px rgba(255, 255, 255, 0.05)',
          color: '#f8fafc',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          animation: 'modalSlideUp 0.2s ease-out'
        }}
      >
        {/* Modal Header */}
        <div
          style={{
            padding: '18px 24px',
            borderBottom: '1px solid #1e293b',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'linear-gradient(180deg, #1e293b 0%, #0f172a 100%)'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '10px',
                backgroundColor: 'rgba(2, 132, 199, 0.15)',
                border: '1px solid rgba(2, 132, 199, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#38bdf8'
              }}
            >
              <Database size={22} />
            </div>
            <div>
              <h2 style={{ fontSize: '17px', fontWeight: 700, color: '#f8fafc', margin: 0, letterSpacing: '-0.01em' }}>
                Direct SQL Database Connector
              </h2>
              <p style={{ fontSize: '12px', color: '#94a3b8', margin: '2px 0 0 0' }}>
                Connect SQLite, Postgres, MySQL, DuckDB or Custom URI into DataPilot OS
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.15s ease'
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = '#f8fafc')}
            onMouseLeave={(e) => (e.currentTarget.style.color = '#94a3b8')}
          >
            <X size={20} />
          </button>
        </div>

        {/* Modal Content Body */}
        <div style={{ padding: '20px 24px', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Error Alert Banner */}
          {errorMsg && (
            <div
              style={{
                padding: '12px 16px',
                borderRadius: '8px',
                backgroundColor: 'rgba(127, 29, 29, 0.4)',
                border: '1px solid #991b1b',
                color: '#fca5a5',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '10px'
              }}
            >
              <AlertCircle size={18} style={{ flexShrink: 0, marginTop: '2px', color: '#f87171' }} />
              <div style={{ flex: 1, wordBreak: 'break-word' }}>
                <strong>Connection Error:</strong> {errorMsg}
              </div>
            </div>
          )}

          {/* Success Alert Banner */}
          {testSuccess && (
            <div
              style={{
                padding: '12px 16px',
                borderRadius: '8px',
                backgroundColor: 'rgba(6, 78, 59, 0.4)',
                border: '1px solid #065f46',
                color: '#6ee7b7',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}
            >
              <CheckCircle2 size={18} style={{ flexShrink: 0, color: '#34d399' }} />
              <div>{testSuccess}</div>
            </div>
          )}

          {/* 1. Database Engine Selector */}
          <div>
            <label style={{ fontSize: '12px', fontWeight: 600, color: '#cbd5e1', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px', display: 'block' }}>
              Select Database Engine
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
              {ENGINES.map((eng) => {
                const IconComponent = eng.icon;
                const isSelected = engine === eng.id;
                return (
                  <button
                    key={eng.id}
                    type="button"
                    className="db-engine-card"
                    onClick={() => setEngine(eng.id)}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      padding: '12px 8px',
                      borderRadius: '8px',
                      border: `1px solid ${isSelected ? '#10b981' : '#334155'}`,
                      backgroundColor: isSelected ? '#1e293b' : '#0f172a',
                      color: isSelected ? '#38bdf8' : '#94a3b8',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    <IconComponent size={20} style={{ marginBottom: '6px', color: isSelected ? '#38bdf8' : '#64748b' }} />
                    <span style={{ fontSize: '12px', fontWeight: isSelected ? 700 : 500 }}>{eng.name}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 2. Dynamic Connection Form Inputs */}
          <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '10px', border: '1px solid #334155' }}>
            <h3 style={{ fontSize: '13px', fontWeight: 600, color: '#e2e8f0', margin: '0 0 14px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Server size={14} style={{ color: '#10b981' }} />
              Connection Parameters
            </h3>

            {(engine === 'sqlite' || engine === 'duckdb') && (
              <div>
                <label style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '6px', display: 'block' }}>
                  Database File Path ({engine === 'sqlite' ? '.db / .sqlite' : '.duckdb'})
                </label>
                <input
                  type="text"
                  className="db-modal-input"
                  placeholder={engine === 'sqlite' ? 'e.g. C:/datasets/sqlite_sample.db' : 'e.g. C:/datasets/analytics.duckdb'}
                  value={filePath}
                  onChange={(e) => setFilePath(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: '6px',
                    color: '#f8fafc',
                    fontSize: '13px',
                    fontFamily: 'monospace'
                  }}
                />
              </div>
            )}

            {(engine === 'postgresql' || engine === 'mysql') && (
              <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '12px' }}>
                <div>
                  <label style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '6px', display: 'block' }}>Host / Server</label>
                  <input
                    type="text"
                    className="db-modal-input"
                    placeholder="localhost or 127.0.0.1"
                    value={host}
                    onChange={(e) => setHost(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      backgroundColor: '#0f172a',
                      border: '1px solid #334155',
                      borderRadius: '6px',
                      color: '#f8fafc',
                      fontSize: '13px'
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '6px', display: 'block' }}>Port</label>
                  <input
                    type="number"
                    className="db-modal-input"
                    placeholder={engine === 'postgresql' ? '5432' : '3306'}
                    value={port}
                    onChange={(e) => setPort(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      backgroundColor: '#0f172a',
                      border: '1px solid #334155',
                      borderRadius: '6px',
                      color: '#f8fafc',
                      fontSize: '13px'
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '6px', display: 'block' }}>Username</label>
                  <input
                    type="text"
                    className="db-modal-input"
                    placeholder={engine === 'postgresql' ? 'postgres' : 'root'}
                    value={user}
                    onChange={(e) => setUser(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      backgroundColor: '#0f172a',
                      border: '1px solid #334155',
                      borderRadius: '6px',
                      color: '#f8fafc',
                      fontSize: '13px'
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '6px', display: 'block' }}>Password</label>
                  <input
                    type="password"
                    className="db-modal-input"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      backgroundColor: '#0f172a',
                      border: '1px solid #334155',
                      borderRadius: '6px',
                      color: '#f8fafc',
                      fontSize: '13px'
                    }}
                  />
                </div>

                <div style={{ gridColumn: 'span 2' }}>
                  <label style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '6px', display: 'block' }}>Database Name</label>
                  <input
                    type="text"
                    className="db-modal-input"
                    placeholder="e.g. production_db"
                    value={database}
                    onChange={(e) => setDatabase(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      backgroundColor: '#0f172a',
                      border: '1px solid #334155',
                      borderRadius: '6px',
                      color: '#f8fafc',
                      fontSize: '13px'
                    }}
                  />
                </div>
              </div>
            )}

            {engine === 'custom' && (
              <div>
                <label style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '6px', display: 'block' }}>
                  Connection URI String
                </label>
                <input
                  type="text"
                  className="db-modal-input"
                  placeholder="postgresql://user:password@localhost:5432/dbname or sqlite:///path/to/db.sqlite"
                  value={uri}
                  onChange={(e) => setUri(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: '6px',
                    color: '#f8fafc',
                    fontSize: '13px',
                    fontFamily: 'monospace'
                  }}
                />
              </div>
            )}

            {/* Test Connection Button */}
            <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'flex-end' }}>
              <button
                type="button"
                onClick={handleTestConnection}
                disabled={isTesting}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 16px',
                  backgroundColor: '#10b981',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: isTesting ? 'not-allowed' : 'pointer',
                  opacity: isTesting ? 0.7 : 1,
                  transition: 'background-color 0.15s ease'
                }}
              >
                {isTesting ? <Loader2 size={15} className="db-spin" /> : <RefreshCw size={15} />}
                {isTesting ? 'Testing Connection...' : 'Test Connection & Fetch Tables'}
              </button>
            </div>
          </div>

          {/* 3. Table Selection & Custom SQL Query */}
          <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '10px', border: '1px solid #334155' }}>
            <h3 style={{ fontSize: '13px', fontWeight: 600, color: '#e2e8f0', margin: '0 0 14px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Table size={14} style={{ color: '#38bdf8' }} />
              Select Table or Custom Query
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <label style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '6px', display: 'block' }}>
                  Database Table ({fetchedTables.length} available)
                </label>
                <select
                  value={selectedTable}
                  onChange={(e) => handleTableChange(e.target.value)}
                  className="db-modal-input"
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: '6px',
                    color: selectedTable ? '#f8fafc' : '#64748b',
                    fontSize: '13px'
                  }}
                >
                  <option value="">-- Choose a Table --</option>
                  {fetchedTables.map((t) => (
                    <option key={t} value={t} style={{ color: '#f8fafc', backgroundColor: '#0f172a' }}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Terminal size={12} />
                  Custom SQL Query (Optional)
                </label>
                <textarea
                  rows={3}
                  className="db-modal-input"
                  placeholder="SELECT * FROM table_name WHERE status = 'active'"
                  value={sqlQuery}
                  onChange={(e) => setSqlQuery(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: '6px',
                    color: '#f8fafc',
                    fontSize: '13px',
                    fontFamily: 'monospace',
                    resize: 'vertical'
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div
          style={{
            padding: '14px 24px',
            borderTop: '1px solid #1e293b',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            backgroundColor: '#0f172a'
          }}
        >
          <span style={{ fontSize: '12px', color: '#64748b' }}>
            DataPilot OS SQL Direct Connector
          </span>

          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: '8px 16px',
                backgroundColor: 'transparent',
                border: '1px solid #334155',
                borderRadius: '6px',
                color: '#94a3b8',
                fontSize: '13px',
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              Cancel
            </button>

            <button
              type="button"
              onClick={handleImportTable}
              disabled={isImporting || (!selectedTable && !sqlQuery.trim())}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 20px',
                backgroundColor: '#16a34a',
                color: '#ffffff',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: 600,
                cursor: isImporting || (!selectedTable && !sqlQuery.trim()) ? 'not-allowed' : 'pointer',
                opacity: isImporting || (!selectedTable && !sqlQuery.trim()) ? 0.5 : 1,
                transition: 'background-color 0.15s ease'
              }}
            >
              {isImporting ? <Loader2 size={16} className="db-spin" /> : <Play size={15} />}
              {isImporting ? 'Importing Dataset...' : 'Import Table into DataPilot OS'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
