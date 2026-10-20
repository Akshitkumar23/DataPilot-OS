import { useState, useEffect, useRef } from 'react';
import { RefreshCw, TerminalSquare, Trash2, PlusCircle, Download, Activity, CheckCircle, XCircle, Clock, Maximize2 } from 'lucide-react';

import { API, getAuthHeaders } from '../apiConfig';

function TerminalBlock({ item, onUpdateCode, onRunCode, onDelete, onZoomImage }) {
  const gutterRef = useRef(null);
  const textareaRef = useRef(null);
  const codeLines = item.code ? item.code.split('\n') : [''];

  return (
    <div style={{
      borderRadius: 8,
      overflow: 'hidden',
      boxShadow: '0 4px 14px rgba(0,0,0,0.12)',
      border: '1px solid #334155',
      background: '#0c0d12',
      display: 'flex',
      flexDirection: 'column',
      minHeight: 180,
    }}>
      {/* Execution State Header */}
      <div style={{
        padding: '6px 16px',
        background: '#1e293b',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        fontSize: 11,
        fontWeight: 700,
        borderBottom: '1px solid #0f172a'
      }}>
        {item.executionState === 'QUEUED' && <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 4 }}><Clock size={12} /> QUEUED</span>}
        {item.executionState === 'RUNNING' && <span style={{ color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 4 }} className="pulse"><Activity size={12} /> RUNNING</span>}
        {item.executionState === 'SUCCEEDED' && <span style={{ color: '#4ade80', display: 'flex', alignItems: 'center', gap: 4 }}><CheckCircle size={12} /> SUCCEEDED</span>}
        {item.executionState === 'FAILED' && <span style={{ color: '#f87171', display: 'flex', alignItems: 'center', gap: 4 }}><XCircle size={12} /> FAILED</span>}
        {item.executionState === 'TIMED_OUT' && <span style={{ color: '#fbbf24', display: 'flex', alignItems: 'center', gap: 4 }}><Clock size={12} /> TIMED_OUT</span>}
        {!item.executionState && <span style={{ color: '#94a3b8' }}>READY</span>}
      </div>

      {/* Terminal Header */}
      <div style={{
        background: '#2d323e',
        padding: '8px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        color: '#f8fafc',
        fontFamily: 'var(--font-mono)',
        fontSize: 13,
        fontWeight: 700,
        borderBottom: '1px solid #1e293b',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <TerminalSquare size={16} color="#38bdf8" />
          <span>[{item.id}] &nbsp; python &nbsp; // &nbsp; {item.title}</span>
        </div>

        <button
          onClick={onDelete}
          title="Remove this terminal block"
          style={{
            background: 'transparent',
            border: 'none',
            color: '#94a3b8',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            padding: 2,
          }}
        >
          <Trash2 size={13} />
        </button>
      </div>

      {/* Terminal Code Editor Body (Interactive Textarea + Line Numbers) */}
      <div style={{
        display: 'flex',
        background: '#0c0d12',
        padding: '10px 0',
        fontFamily: 'var(--font-mono)',
        fontSize: 13,
        lineHeight: 1.6,
        minHeight: 90,
        position: 'relative',
      }}>
        {/* Line numbers gutter */}
        <div
          ref={gutterRef}
          style={{
            width: 42,
            color: '#64748b',
            textAlign: 'right',
            paddingRight: 12,
            userSelect: 'none',
            borderRight: '1px solid #1e293b',
            overflowY: 'hidden',
          }}
        >
          {codeLines.map((_, i) => (
            <div key={i}>{i + 1}</div>
          ))}
        </div>

        {/* Interactive Code Editor Area */}
        <textarea
          ref={textareaRef}
          value={item.code}
          onChange={e => onUpdateCode(e.target.value)}
          onScroll={e => {
            if (gutterRef.current) {
              gutterRef.current.scrollTop = e.target.scrollTop;
            }
          }}
          placeholder="# Executable Python code snippet will appear here or type custom script..."
          style={{
            flex: 1,
            paddingLeft: 14,
            background: 'transparent',
            color: '#38bdf8',
            border: 'none',
            outline: 'none',
            fontFamily: 'var(--font-mono)',
            fontSize: 13,
            lineHeight: 1.6,
            resize: 'vertical',
            minHeight: 80,
          }}
        />
      </div>

      {/* Terminal Action Bar */}
      <div style={{
        background: '#161922',
        padding: '6px 14px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        borderTop: '1px solid #1e293b',
      }}>
        <button
          onClick={onRunCode}
          disabled={item.isRunning}
          style={{
            background: 'linear-gradient(135deg, #22c55e, #16a34a)',
            color: '#000000',
            border: 'none',
            borderRadius: 16,
            padding: '6px 22px',
            fontSize: 13,
            fontWeight: 800,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            boxShadow: '0 3px 14px rgba(34, 197, 94, 0.55)',
            transition: 'all 0.15s ease',
          }}
        >
          {item.isRunning ? <RefreshCw size={14} className="spinner" /> : <span>► RUN CODE</span>}
        </button>
      </div>

      {/* Output Console Result Display */}
      {item.output && (
        <div style={{
          background: '#07080a',
          borderTop: '1px solid #1e293b',
          padding: '10px 14px',
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          color: item.output.includes('[✗ ERROR]') ? '#f87171' : '#4ade80',
          lineHeight: 1.5,
          maxHeight: 140,
          overflowY: 'auto',
          whiteSpace: 'pre-wrap',
          userSelect: 'text'
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4, display: 'flex', justifyContent: 'space-between' }}>
            <span>Terminal Output & Execution Result:</span>
            {item.executionTime && <span>Execution time: {item.executionTime}s</span>}
          </div>
          <div>{item.output}</div>
        </div>
      )}

      {/* Inline Visualization Canvas (Fits compactly right inside the terminal block!) */}
      {item.plotImage && (
        <div style={{
          background: '#0f172a',
          borderTop: '1px solid #1e293b',
          padding: '12px 16px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 8,
        }}>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 800, color: '#e2e8f0', letterSpacing: '0.01em' }}>
              📊 Generated Plot Visualization
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button
                onClick={() => onZoomImage && onZoomImage(item.plotImage)}
                title="Click to Zoom & Inspect Fullscreen"
                style={{
                  background: '#1e293b',
                  color: '#38bdf8',
                  border: '1px solid #334155',
                  borderRadius: 4,
                  padding: '4px 10px',
                  fontSize: 11,
                  fontWeight: 700,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  transition: 'all 0.15s ease'
                }}
              >
                <Maximize2 size={12} />
                Zoom Image
              </button>
              <a
                href={item.plotImage}
                download={`plot_session_${item.id}.png`}
                style={{
                  background: 'linear-gradient(135deg, #10b981, #059669)',
                  color: '#ffffff',
                  textDecoration: 'none',
                  borderRadius: 4,
                  padding: '4px 10px',
                  fontSize: 11,
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4
                }}
              >
                <Download size={12} />
                Download PNG
              </a>
            </div>
          </div>

          <div style={{
            background: '#090d16',
            borderRadius: 6,
            padding: 8,
            border: '1px solid #1e293b',
            display: 'flex',
            justifyContent: 'center',
            width: '100%',
          }}>
            <img
              src={item.plotImage}
              alt={`Visualization Plot Session ${item.id}`}
              title="Click image to zoom & inspect full screen 🔍"
              onClick={() => onZoomImage && onZoomImage(item.plotImage)}
              style={{ maxHeight: 240, width: 'auto', maxWidth: '100%', borderRadius: 4, objectFit: 'contain', cursor: 'zoom-in', transition: 'transform 0.15s ease' }}
              onMouseOver={e => e.currentTarget.style.transform = 'scale(1.02)'}
              onMouseOut={e => e.currentTarget.style.transform = 'scale(1)'}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default function TerminalHUD({ onRefresh, externalCodeToRun, onPlotGenerated, onExecutionComplete, onZoomImage }) {
  const [executions, setExecutions] = useState([
    {
      id: 1,
      title: "Python Execution Terminal",
      code: "",
      output: "",
      plotImage: null,
      isRunning: false,
      executionState: null,
      executionTime: null
    }
  ]);

  const lastProcessedRef = useRef(null);
  const nextExecutionIdRef = useRef(2);
  const containerRef = useRef(null);

  const runCodeApi = async (blockId, codeToRun) => {
    if (!codeToRun.trim()) return;

    setExecutions(prev => prev.map(b => b.id === blockId ? { ...b, isRunning: true, output: "Executing Python code snippet...", executionState: 'RUNNING', executionTime: null } : b));

    const startTime = Date.now();
    const execTime = ((Date.now() - startTime) / 1000).toFixed(2);
    setExecutions(prev => prev.map(b => b.id === blockId ? {
      ...b,
      isRunning: false,
      output: "[QUARANTINED] Direct terminal code execution is disabled in the production zero-arbitrary-code runtime. Submit analytical requests via AI CoPilot to execute deterministically through typed operators.",
      executionState: 'DISABLED',
      executionTime: execTime
    } : b));
  };

  // Auto execute code passed from AI Co-Pilot into a NEW terminal block
  useEffect(() => {
    if (!externalCodeToRun) return;
    const rawCode = typeof externalCodeToRun === 'object' ? externalCodeToRun?.code : externalCodeToRun;
    const timestamp = typeof externalCodeToRun === 'object' ? externalCodeToRun?.timestamp : rawCode;

    if (!rawCode || lastProcessedRef.current === timestamp) return;
    lastProcessedRef.current = timestamp;

    const targetBlockId = nextExecutionIdRef.current++;
    setExecutions(prev => [...prev, {
        id: targetBlockId,
        title: `AI Co-Pilot Code Snippet [${targetBlockId}]`,
        code: rawCode,
        output: "Executing Python code snippet...",
        plotImage: null,
        isRunning: true
    }]);

    // Reliably trigger execution outside the state updater
    setTimeout(() => {
      runCodeApi(targetBlockId, rawCode);
    }, 60);

    // Auto-scroll to the bottom to focus on the new terminal block
    setTimeout(() => {
      if (containerRef.current) {
        containerRef.current.scrollTop = containerRef.current.scrollHeight;
      }
    }, 150);
  }, [externalCodeToRun]);

  return (
    <div ref={containerRef} style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: '#090d16',
      color: '#e2e8f0',
      overflowY: 'auto',
      padding: '16px 20px',
      gap: 16,
      fontFamily: "system-ui, -apple-system, sans-serif",
    }} className="fade-in">

      {/* This is a manual code runner. Typed AI queries execute in the Results
          panel and expose their evidence there; no fabricated resource values. */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '10px 14px',
        background: '#0f172a',
        border: '1px solid #1e293b',
        borderRadius: 8,
        fontSize: 12,
        color: '#94a3b8'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <TerminalSquare size={15} color="#38bdf8" />
            <span style={{ color: '#e2e8f0', fontWeight: 700 }}>Manual Python sandbox</span>
            <span style={{ color: '#64748b' }}>• Each run shows its actual outcome and elapsed time below.</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#172554', color: '#bfdbfe', padding: '3px 8px', borderRadius: 6, fontSize: 11, fontWeight: 700 }}>
          <span>AI query evidence is in Results & Output</span>
        </div>
      </div>

      {/* Control Header for Multi-Terminal Management */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingBottom: 6,
        borderBottom: '1px solid #1e293b',
        flexShrink: 0
      }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>Active Terminal Sessions ({executions.length})</span>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={() => {
              const newId = nextExecutionIdRef.current++;
              setExecutions(prev => [...prev, {
                id: newId,
                title: `Terminal Session [${newId}]`,
                code: "",
                output: "",
                plotImage: null,
                isRunning: false
              }]);
              setTimeout(() => {
                if (containerRef.current) {
                  containerRef.current.scrollTop = containerRef.current.scrollHeight;
                }
              }, 100);
            }}
            style={{
              background: '#10b981',
              color: '#ffffff',
              border: 'none',
              borderRadius: 5,
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4
            }}
          >
            <PlusCircle size={12} />
            + New Terminal
          </button>

          <button
            onClick={() => {
              setExecutions([{
                id: 1,
                title: "Python Execution Terminal",
                code: "",
                output: "",
                plotImage: null,
                isRunning: false
              }]);
            }}
            style={{
              background: '#f1f5f9',
              color: '#64748b',
              border: '1px solid #cbd5e1',
              borderRadius: 5,
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4
            }}
          >
            <Trash2 size={12} />
            Clear History
          </button>
        </div>
      </div>

      {/* Render all execution terminal blocks sequentially */}
      {executions.map((item) => (
        <TerminalBlock
          key={item.id}
          item={item}
          onUpdateCode={(newCode) => {
            setExecutions(prev => prev.map(b => b.id === item.id ? { ...b, code: newCode } : b));
          }}
          onRunCode={() => runCodeApi(item.id, item.code)}
          onDelete={() => {
            if (executions.length > 1) {
              setExecutions(prev => prev.filter(b => b.id !== item.id));
            } else {
              setExecutions([{ id: 1, title: "Python Execution Terminal", code: "", output: "", plotImage: null, isRunning: false }]);
            }
          }}
          onZoomImage={onZoomImage}
        />
      ))}
    </div>
  );
}
