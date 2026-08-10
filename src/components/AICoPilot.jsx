import { useState, useEffect, useRef } from 'react';
import { Camera, Sparkles, Play, Send, Bot, User, CheckCircle2, Cpu, Save, Copy } from 'lucide-react';

import { API, getAuthHeaders } from '../apiConfig';

export default function AICoPilot({
  activeDataset,
  schema,
  onRunCodeInTerminal,
  onRefresh,
  onSaveFile,
  isSaving,
  datasetLoaded,
  onZoomImage
}) {
  const [messages, setMessages] = useState([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isReadingScreen, setIsReadingScreen] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [recommendations, setRecommendations] = useState([]);
  const [showQuickPills, setShowQuickPills] = useState(true);
  const chatFeedRef = useRef(null);
  const abortControllerRef = useRef(null);

  const quickPills = [
    { label: '✨ Quality Audit', prompt: 'Audit data quality score and missing values' },
    { label: '📊 Auto EDA', prompt: 'Generate full automated EDA dashboard' },
    { label: '🔮 Forecast 30D', prompt: 'Forecast next 30 days time series trend' },
    { label: '🤖 AutoML Benchmark', prompt: 'Compare AutoML leaderboard models' },
    { label: '🔍 Root Cause', prompt: 'Why did metric drop? Run root cause analysis' }
  ];
  
  // ── Notify when dataset is connected ──────────────────
  useEffect(() => {
    if (activeDataset?.name) {
      const colCnt = activeDataset.columns?.length || schema?.length || 0;
      setMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: `📁 Connected to dataset '${activeDataset.name}' (${activeDataset.rowCount?.toLocaleString() || 0} rows × ${colCnt} columns). Ask me to clean nulls, find outliers, plot heatmaps, or train ML models!`
        }
      ]);
    }
  }, [activeDataset?.name, activeDataset?.loadedAt]);

  // Auto-scroll chat feed whenever messages update or thinking state changes
  const scrollToBottom = () => {
    setTimeout(() => {
      if (chatFeedRef.current) {
        chatFeedRef.current.scrollTop = chatFeedRef.current.scrollHeight;
      }
    }, 50);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages.length, isThinking]);

  // ── 📸 Read Screen Context Action ──────────────────────
  const handleReadScreen = async () => {
    setIsReadingScreen(true);
    try {
      const res = await fetch(`${API}/api/screen/analyze`, { method: 'POST', headers: getAuthHeaders() });
      if (!res.ok) {
        throw new Error(`Screen analyze failed (${res.status})`);
      }
      const data = await res.json();
      
      if ((data.auto_connected || data.auto_connect || data.connected) && onRefresh) {
        onRefresh();
      }

      setMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: data.summary || `📸 Read Complete: ${data.active_title || 'Active Window'}`,
          recommendations: data.recommendations || []
        }
      ]);

    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: `⚠️ Read status: ${err.message}`,
          recommendations: []
        }
      ]);
    } finally {
      setIsReadingScreen(false);
    }
  };

  // Helper to parse markdown bold **text**
  const renderFormattedText = (text) => {
    if (!text) return null;
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={idx} style={{ color: '#34d399', fontWeight: 700 }}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  // ── Send Custom Data Science Query ─────────────────────
  const handleSendQuery = async (e) => {
    e.preventDefault();
    if (!inputPrompt.trim()) return;

    // Cancel any previous in-flight request if user hits send again
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const userText = inputPrompt;
    setInputPrompt('');
    setMessages(prev => [...prev, { sender: 'user', text: userText }]);
    setIsThinking(true);

    try {
      // 1. Validation Engine check
      let validationBadge = null;
      try {
        const valRes = await fetch(`${API}/api/validate`, {
          method: 'POST',
          headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ query: userText })
        });
        if (valRes.ok) {
          const vData = await valRes.json();
          if (vData.ambiguities_found && vData.ambiguities_found.length > 0) {
            validationBadge = vData.ambiguities_found.join(" | ");
          }
        }
      } catch (vErr) {
        console.warn("Validation API check skipped:", vErr);
      }

      // 2. Main query execution
      const res = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ query: userText }),
        signal: abortControllerRef.current.signal
      });
      if (!res.ok) {
        throw new Error(`Chat query failed (${res.status})`);
      }
      const data = await res.json();
      
      const responseMsg = data.response || data.answer || data.reply || (data.artifacts?.reports ? data.artifacts.reports[0] : null) || 'Query processed successfully.';
      const codeSnippet = data.code || (data.artifacts?.code && data.artifacts.code.length > 0 ? data.artifacts.code.join('\n\n') : null);
      const plotImg = data.plot_image || data.plot || (data.artifacts?.plots && data.artifacts.plots.length > 0 ? data.artifacts.plots[0] : null);
      const multiResults = data.multi_results || null;

      setMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: responseMsg,
          code: codeSnippet,
          actionCode: codeSnippet,
          plotImage: plotImg,
          validationBadge: validationBadge,
          multiResults: multiResults
        }
      ]);
      if (data.recommendations) setRecommendations(data.recommendations);
    } catch (err) {
      if (err.name === 'AbortError') return;
      setMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: `Reasoning Engine Error: ${err.message}`
        }
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#060907', color: '#f8fafc', fontFamily: 'var(--font-ui)' }}>
      {/* Top Controls: Save File & Read Screen Header */}
      <div style={{
        padding: '6px 12px',
        background: '#0a0e0c',
        borderBottom: '1px solid #1a2921',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        gap: 8,
        flexShrink: 0
      }}>
        {datasetLoaded && onSaveFile && (
          <button
            onClick={onSaveFile}
            disabled={isSaving}
            style={{
              background: '#16a34a',
              color: '#ffffff',
              border: 'none',
              borderRadius: 6,
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 700,
              cursor: isSaving ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              boxShadow: '0 2px 6px rgba(22,163,74,0.3)',
              transition: 'all 0.15s ease',
              opacity: isSaving ? 0.6 : 1
            }}
            title="Save changes back to file"
          >
            <Save size={12} />
            {isSaving ? 'Saving…' : '💾 Save File'}
          </button>
        )}

        <button
          onClick={handleReadScreen}
          disabled={isReadingScreen}
          style={{
            background: 'linear-gradient(135deg, #10b981, #059669)',
            color: '#ffffff',
            border: 'none',
            borderRadius: 6,
            padding: '4px 10px',
            fontSize: 11,
            fontWeight: 700,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            boxShadow: '0 2px 8px rgba(16,185,129,0.35)',
            transition: 'all 0.15s ease'
          }}
        >
          <Camera size={12} />
          {isReadingScreen ? 'Reading...' : '📸 Read Screen'}
        </button>
      </div>

      {/* Chat Messages Feed */}
      <div ref={chatFeedRef} style={{ flex: 1, overflowY: 'auto', padding: '14px', display: 'flex', flexDirection: 'column', gap: 14 }}>
        {messages.map((msg, idx) => {
          const isUser = msg.sender === 'user';
          return (
            <div
              key={idx}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignSelf: isUser ? 'flex-end' : 'flex-start',
                maxWidth: '88%',
                background: isUser ? 'linear-gradient(135deg, #132219, #192b20)' : 'linear-gradient(135deg, #0d1310, #111a14)',
                border: isUser ? '1px solid #233e2f' : '1px solid #1a2921',
                borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                padding: '12px 14px',
                fontSize: 13,
                lineHeight: 1.55,
                boxShadow: isUser ? '0 4px 12px rgba(0,0,0,0.35)' : '0 4px 14px rgba(0,0,0,0.45)',
                transition: 'all 0.15s ease'
              }}
            >
              {/* Header Avatar Badge */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, fontSize: 11, fontWeight: 700, color: '#34d399' }}>
                <div style={{
                  width: 20,
                  height: 20,
                  borderRadius: '50%',
                  background: isUser ? 'rgba(52, 211, 153, 0.2)' : 'rgba(16, 185, 129, 0.25)',
                  border: '1px solid rgba(52, 211, 153, 0.4)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#34d399'
                }}>
                  {isUser ? <User size={11} /> : <Bot size={11} />}
                </div>
                <span style={{ letterSpacing: '0.02em' }}>{isUser ? 'You' : 'Reasoning Engine'}</span>
              </div>

              {/* Validation Guard Badge */}
              {msg.validationBadge && (
                <div style={{
                  margin: '4px 0 8px 0',
                  padding: '5px 10px',
                  background: 'rgba(16, 185, 129, 0.12)',
                  border: '1px solid rgba(52, 211, 153, 0.3)',
                  borderRadius: 8,
                  fontSize: 11,
                  color: '#6ee7b7',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6
                }}>
                  <CheckCircle2 size={13} color="#34d399" />
                  <span>Validation Guard: {msg.validationBadge}</span>
                </div>
              )}

              {/* Text Body */}
              <div style={{ color: '#f1f5f9', whiteSpace: 'pre-wrap', fontFamily: 'var(--font-ui)', fontSize: 12.5, lineHeight: 1.55 }}>
                {renderFormattedText(msg.text)}
              </div>

              {/* Multi-Intent Sub-Results Sequential Card List */}
              {msg.multiResults && msg.multiResults.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 10 }}>
                  {msg.multiResults.map((sub, sIdx) => (
                    <div key={sIdx} style={{ background: '#070b09', borderRadius: 8, border: '1px solid #1a2921', padding: 10 }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: '#34d399', marginBottom: 4 }}>
                        Part {sIdx + 1}: "{sub.sub_query}" ({sub.label})
                      </div>
                      <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.5, marginBottom: 8 }}>
                        {renderFormattedText(sub.response)}
                      </div>
                      {sub.code && (
                        <div style={{ marginTop: 6, marginBottom: 8 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                            <span style={{ fontSize: 10, color: '#64748b', fontWeight: 600 }}>Python Script</span>
                            <button
                              onClick={() => onRunCodeInTerminal(sub.code)}
                              style={{
                                background: 'linear-gradient(135deg, #10b981, #059669)',
                                color: '#ffffff',
                                border: 'none',
                                borderRadius: 4,
                                padding: '2px 8px',
                                fontSize: 10,
                                fontWeight: 700,
                                cursor: 'pointer'
                              }}
                            >
                              Run in Terminal
                            </button>
                          </div>
                          <pre style={{ background: '#040605', border: '1px solid #1a2921', borderRadius: 6, padding: 8, fontSize: 11, color: '#a7f3d0', overflowX: 'auto' }}>
                            <code>{sub.code}</code>
                          </pre>
                        </div>
                      )}
                      {sub.plot_image && (
                        <div style={{ marginTop: 8, background: '#040605', borderRadius: 8, overflow: 'hidden', border: '1px solid #1a2921', padding: 6 }}>
                          <div style={{ fontSize: 10, fontWeight: 700, color: '#34d399', marginBottom: 4, display: 'flex', justifyContent: 'space-between' }}>
                            <span>📊 Plot: {sub.label}</span>
                            <a href={sub.plot_image} download={`plot_part_${sIdx+1}.png`} style={{ color: '#34d399', fontSize: 10, fontWeight: 700, textDecoration: 'none' }}>Download PNG</a>
                          </div>
                          <img
                            src={sub.plot_image}
                            alt={sub.label}
                            onClick={() => onZoomImage && onZoomImage(sub.plot_image)}
                            style={{ width: '100%', borderRadius: 6, cursor: 'zoom-in' }}
                          />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Single Generated Plot Image Card (when not multiResults) */}
              {!msg.multiResults && msg.plotImage && (
                <div style={{ marginTop: 10, background: '#050806', borderRadius: 8, overflow: 'hidden', border: '1px solid #1a2921', padding: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: '#34d399', marginBottom: 6, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>📊 Generated Data Plot Visualization (Click image to zoom 🔍)</span>
                    <a
                      href={msg.plotImage}
                      download="plot_result.png"
                      style={{
                        background: 'linear-gradient(135deg, #10b981, #059669)',
                        color: '#ffffff',
                        padding: '3px 8px',
                        borderRadius: 4,
                        fontSize: 10,
                        textDecoration: 'none',
                        fontWeight: 700
                      }}
                    >
                      Download PNG
                    </a>
                  </div>
                  <img
                    src={msg.plotImage}
                    alt="Generated Plot"
                    title="Click to zoom, rotate & inspect full screen"
                    onClick={() => onZoomImage && onZoomImage(msg.plotImage)}
                    style={{ width: '100%', borderRadius: 6, border: '1px solid #1a2921', cursor: 'zoom-in', transition: 'transform 0.15s ease' }}
                    onMouseOver={e => e.currentTarget.style.transform = 'scale(1.01)'}
                    onMouseOut={e => e.currentTarget.style.transform = 'scale(1)'}
                  />
                </div>
              )}

              {/* Base64 Screenshot Thumbnail Badge */}
              {msg.thumbnail && (
                <div style={{ marginTop: 8 }}>
                  <img
                    src={msg.thumbnail}
                    alt="Screen context"
                    title="Click to zoom screenshot"
                    onClick={() => onZoomImage && onZoomImage(msg.thumbnail)}
                    style={{
                      maxWidth: '100%',
                      maxHeight: 140,
                      borderRadius: 8,
                      border: '1px solid #10b981',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                      cursor: 'zoom-in'
                    }}
                  />
                </div>
              )}

              {/* Action Code Card */}
              {msg.actionCode && (
                <div style={{ marginTop: 10, background: '#050806', borderRadius: 8, overflow: 'hidden', border: '1px solid #1a2921' }}>
                  <div style={{
                    background: '#0a0e0c',
                    padding: '6px 10px',
                    borderBottom: '1px solid #1a2921',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: 11,
                    color: '#a7f3d0',
                    fontWeight: 700
                  }}>
                    <span>🐍 Python Script Generated</span>
                    <button
                      onClick={() => {
                        onRunCodeInTerminal(msg.actionCode);
                        if (onRefresh) onRefresh();
                      }}
                      style={{
                        background: 'linear-gradient(135deg, #10b981, #059669)',
                        color: '#ffffff',
                        border: 'none',
                        borderRadius: 6,
                        padding: '4px 10px',
                        fontSize: 11,
                        fontWeight: 700,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 4,
                        boxShadow: '0 2px 6px rgba(16,185,129,0.3)'
                      }}
                    >
                      <Play size={10} fill="#ffffff" />
                      Run in Terminal
                    </button>
                  </div>
                  <pre style={{
                    margin: 0,
                    padding: 10,
                    fontSize: 11,
                    fontFamily: 'var(--font-mono)',
                    color: '#34d399',
                    overflowX: 'auto',
                    whiteSpace: 'pre-wrap'
                  }}>
                    {msg.actionCode}
                  </pre>
                  <div style={{ padding: 6, background: '#0a0e0c', borderTop: '1px solid #1a2921', display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(msg.actionCode);
                      }}
                      style={{
                        background: '#121c17',
                        color: '#34d399',
                        border: '1px solid #1a2921',
                        borderRadius: 6,
                        padding: '4px 10px',
                        fontSize: 10,
                        fontWeight: 600,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 4
                      }}
                    >
                      <Copy size={11} />
                      <span>Copy Code</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {isThinking && (
          <div style={{ alignSelf: 'flex-start', background: '#0a0e0c', border: '1px solid #1a2921', padding: '8px 14px', borderRadius: 16, fontSize: 11, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 8 }}>
            <Cpu size={14} className="spin" color="#34d399" />
            <span>AI Brain thinking & generating solution...</span>
          </div>
        )}
      </div>

      {/* Suggested Follow-up Action Chips */}
      {recommendations.length > 0 && (
        <div style={{ padding: '6px 12px', background: '#060907', display: 'flex', gap: 6, overflowX: 'auto', flexShrink: 0, borderTop: '1px solid #1a2921' }}>
          {recommendations.map((rec, idx) => (
            <button
              key={idx}
              onClick={() => {
                setInputPrompt(rec);
              }}
              style={{
                background: '#0d1310',
                color: '#34d399',
                border: '1px solid #1a2921',
                borderRadius: 14,
                padding: '4px 12px',
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease'
              }}
              onMouseOver={e => { e.currentTarget.style.background = '#16221c'; e.currentTarget.style.borderColor = '#34d399'; }}
              onMouseOut={e => { e.currentTarget.style.background = '#0d1310'; e.currentTarget.style.borderColor = '#1a2921'; }}
            >
              ⚡ {rec}
            </button>
          ))}
        </div>
      )}

      {/* Suggested Quick Question Pills */}
      {showQuickPills && (
        <div style={{ padding: '6px 12px', background: '#060907', display: 'flex', gap: 6, overflowX: 'auto', flexShrink: 0, borderTop: '1px solid #1a2921' }}>
          {quickPills.map((pill, idx) => (
            <button
              key={idx}
              onClick={() => {
                setInputPrompt(pill.prompt);
                setShowQuickPills(false);
              }}
              style={{
                background: '#0d1310',
                color: '#34d399',
                border: '1px solid #1a2921',
                borderRadius: 14,
                padding: '4px 12px',
                fontSize: 11,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease'
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = '#10b981'; e.currentTarget.style.color = '#ffffff'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = '#0d1310'; e.currentTarget.style.color = '#34d399'; }}
            >
              {pill.label}
            </button>
          ))}
        </div>
      )}

      {/* Chat Prompt Input Form */}
      <form onSubmit={handleSendQuery} style={{ padding: '10px 14px', background: '#0a0e0c', borderTop: '1px solid #1a2921', display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <input
          type="text"
          value={inputPrompt}
          onChange={e => setInputPrompt(e.target.value)}
          placeholder="Ask AI Data Scientist or type Python/SQL command..."
          style={{
            flex: 1,
            background: '#121c17',
            border: '1px solid #23352b',
            borderRadius: 20,
            padding: '9px 16px',
            fontSize: 12,
            fontFamily: 'var(--font-ui)',
            color: '#f8fafc',
            outline: 'none',
            boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.4)',
            transition: 'border 0.15s ease'
          }}
          onFocus={e => e.currentTarget.style.borderColor = '#10b981'}
          onBlur={e => e.currentTarget.style.borderColor = '#23352b'}
        />
        <button
          type="submit"
          style={{
            width: 36,
            height: 36,
            borderRadius: 18,
            background: 'linear-gradient(135deg, #10b981, #059669)',
            color: '#ffffff',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            boxShadow: '0 3px 10px rgba(16, 185, 129, 0.45)',
            transition: 'transform 0.15s ease'
          }}
          onMouseDown={e => e.currentTarget.style.transform = 'scale(0.95)'}
          onMouseUp={e => e.currentTarget.style.transform = 'scale(1)'}
        >
          <Send size={14} />
        </button>
      </form>
    </div>
  );
}
