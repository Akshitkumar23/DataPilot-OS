import { useState, useEffect, useRef } from 'react';
import { RefreshCw, TerminalSquare, Trash2, PlusCircle, Download } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || window.ENV_API_URL || 'http://localhost:8000';

function TerminalBlock({ item, onUpdateCode, onRunCode, onDelete }) {
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
            background: '#22c55e',
            color: '#000000',
            border: 'none',
            borderRadius: 16,
            padding: '6px 20px',
            fontSize: 13,
            fontWeight: 800,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            boxShadow: '0 2px 10px rgba(34,197,94,0.4)',
            transition: 'transform 0.1s',
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
          <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>
            Terminal Output & Execution Result:
          </div>
          <div>{item.output}</div>
        </div>
      )}

      {/* Inline Visualization Canvas (Fits compactly right inside the terminal block!) */}
      {item.plotImage && (
        <div style={{
          background: '#ffffff',
          borderTop: '1px solid #334155',
          padding: '12px 16px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 8,
        }}>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 12, fontWeight: 800, color: '#0f172a', letterSpacing: '0.01em' }}>
              📊 Generated Plot Visualization
            </span>
            <a
              href={item.plotImage}
              download={`plot_session_${item.id}.png`}
              style={{
                background: '#0284c7',
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

          <div style={{
            background: '#ffffff',
            borderRadius: 6,
            padding: 8,
            border: '1px solid #e2e8f0',
            display: 'flex',
            justifyContent: 'center',
            width: '100%',
          }}>
            <img
              src={item.plotImage}
              alt={`Visualization Plot Session ${item.id}`}
              style={{ maxHeight: 240, width: 'auto', maxWidth: '100%', borderRadius: 4, objectFit: 'contain' }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default function TerminalHUD({ schema, rowCount, pipelineSteps, onRefresh, externalCodeToRun, onPlotGenerated }) {
  const [executions, setExecutions] = useState([
    {
      id: 1,
      title: "Python Execution Terminal",
      code: "",
      output: "",
      plotImage: null,
      isRunning: false
    }
  ]);

  const lastProcessedRef = useRef(null);
  const containerRef = useRef(null);

  const runCodeApi = async (blockId, codeToRun) => {
    if (!codeToRun.trim()) return;

    setExecutions(prev => prev.map(b => b.id === blockId ? { ...b, isRunning: true, output: "Executing Python code snippet..." } : b));

    try {
      const res = await fetch(`${API}/api/terminal/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: codeToRun }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Execution failed');

      const outputText = data.output || "Code executed cleanly with 0 errors.";
      const plotB64 = data.plot_image || null;

      setExecutions(prev => prev.map(b => b.id === blockId ? {
        ...b,
        isRunning: false,
        output: outputText,
        plotImage: plotB64
      } : b));

      if (plotB64) {
        onPlotGenerated?.(plotB64);
      }
      onRefresh?.();
    } catch (err) {
      setExecutions(prev => prev.map(b => b.id === blockId ? { ...b, isRunning: false, output: `[✗ ERROR] Execution failed: ${err.message}` } : b));
    }
  };

  // Auto execute code passed from AI Co-Pilot into a NEW terminal block
  useEffect(() => {
    if (!externalCodeToRun) return;
    const rawCode = typeof externalCodeToRun === 'object' ? externalCodeToRun?.code : externalCodeToRun;
    const timestamp = typeof externalCodeToRun === 'object' ? externalCodeToRun?.timestamp : rawCode;

    if (!rawCode || lastProcessedRef.current === timestamp) return;
    lastProcessedRef.current = timestamp;

    setExecutions(prev => {
      const lastBlock = prev[prev.length - 1];
      // Reuse first block if it's completely blank and hasn't been run
      if (prev.length === 1 && !lastBlock.code.trim() && !lastBlock.output) {
        const updatedBlock = {
          ...lastBlock,
          title: "AI Co-Pilot Code Snippet [1]",
          code: rawCode,
          output: "Executing Python code snippet...",
          plotImage: null,
          isRunning: true
        };
        runCodeApi(lastBlock.id, rawCode);
        return [updatedBlock];
      }

      // Otherwise create a NEW separate terminal execution block!
      const newId = prev.length + 1;
      const newBlock = {
        id: newId,
        title: `AI Co-Pilot Code Snippet [${newId}]`,
        code: rawCode,
        output: "Executing Python code snippet...",
        plotImage: null,
        isRunning: true
      };
      runCodeApi(newId, rawCode);
      return [...prev, newBlock];
    });

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
      background: '#ffffff',
      color: '#1e293b',
      overflowY: 'auto',
      padding: '16px 20px',
      gap: 16,
      fontFamily: "system-ui, -apple-system, sans-serif",
    }} className="fade-in">

      {/* Control Header for Multi-Terminal Management */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingBottom: 6,
        borderBottom: '1px solid #e2e8f0',
        flexShrink: 0
      }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: '#475569', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>Active Terminal Sessions ({executions.length})</span>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={() => {
              const newId = executions.length + 1;
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
              background: '#0284c7',
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
        />
      ))}
    </div>
  );
}
