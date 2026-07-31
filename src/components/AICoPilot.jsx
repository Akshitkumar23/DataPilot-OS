import { useState, useEffect, useRef } from 'react';
import { Camera, Sparkles, Play, Send, Bot, User, CheckCircle2, Cpu, Save, Copy } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || window.ENV_API_URL || 'http://localhost:8000';

export default function AICoPilot({
  activeDataset,
  schema,
  onRunCodeInTerminal,
  onRefresh,
  onSaveFile,
  isSaving,
  datasetLoaded
}) {
  const [messages, setMessages] = useState([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isReadingScreen, setIsReadingScreen] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const chatFeedRef = useRef(null);
  
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
      const res = await fetch(`${API}/api/screen/analyze`, { method: 'POST' });
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
          recommendations: []
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

  // ── Send Custom Data Science Query ─────────────────────
  const handleSendQuery = async (e) => {
    e.preventDefault();
    if (!inputPrompt.trim()) return;

    const userText = inputPrompt;
    setInputPrompt('');
    setMessages(prev => [...prev, { sender: 'user', text: userText }]);
    setIsThinking(true);

    try {
      const res = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userText }),
      });
      if (!res.ok) {
        throw new Error(`Chat query failed (${res.status})`);
      }
      const data = await res.json();
      
      setMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: data.response || data.answer || 'Data Science AI response complete.',
          code: data.code || null
        }
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: `AI Assistant Error: ${err.message}`
        }
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#090d16', color: '#f8fafc' }}>
      {/* Top Controls: Save File & Read Screen Header */}
      <div style={{
        padding: '4px 8px',
        background: '#0f172a',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        gap: 6,
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
              borderRadius: 4,
              padding: '3px 8px',
              fontSize: 10,
              fontWeight: 700,
              cursor: isSaving ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              boxShadow: '0 2px 6px rgba(22,163,74,0.3)',
              transition: 'all 0.15s ease',
              opacity: isSaving ? 0.6 : 1
            }}
            title="Save changes back to file"
          >
            <Save size={11} />
            {isSaving ? 'Saving…' : '💾 Save File'}
          </button>
        )}

        <button
          onClick={handleReadScreen}
          disabled={isReadingScreen}
          style={{
            background: 'linear-gradient(135deg, #0284c7, #2563eb)',
            color: '#ffffff',
            border: 'none',
            borderRadius: 4,
            padding: '3px 8px',
            fontSize: 10,
            fontWeight: 700,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            boxShadow: '0 2px 6px rgba(2,132,199,0.3)',
            transition: 'all 0.15s ease'
          }}
        >
          <Camera size={11} />
          {isReadingScreen ? 'Reading...' : '📸 Read Screen'}
        </button>
      </div>




      {/* Chat Messages Feed */}
      <div ref={chatFeedRef} style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '92%',
              background: msg.sender === 'user' ? '#1e293b' : '#0f172a',
              border: msg.sender === 'user' ? '1px solid #334155' : '1px solid #1e293b',
              borderRadius: 8,
              padding: '10px 12px',
              fontSize: 12,
              lineHeight: 1.5
            }}
          >
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4, fontSize: 11, fontWeight: 700, color: msg.sender === 'user' ? '#38bdf8' : '#34d399' }}>
              {msg.sender === 'user' ? <User size={13} /> : <Bot size={13} />}
              <span>{msg.sender === 'user' ? 'You' : 'DataPilot AI Engine'}</span>
            </div>

            {/* Text */}
            <div style={{ color: '#e2e8f0', whiteSpace: 'pre-wrap' }}>
              {msg.text}
            </div>

            {/* Clean Code Snippet Block */}
            {msg.code && (
              <div style={{ marginTop: 8 }}>
                <div style={{
                  background: '#020617',
                  padding: '8px 10px',
                  borderRadius: 6,
                  border: '1px solid #1e293b',
                  marginBottom: 6,
                  fontFamily: 'monospace',
                  fontSize: 11,
                  color: '#a7f3d0',
                  overflowX: 'auto',
                  whiteSpace: 'pre'
                }}>
                  {msg.code}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <button
                    onClick={() => {
                      if (onRunCodeInTerminal) onRunCodeInTerminal(msg.code);
                      if (onRefresh) onRefresh();
                    }}
                    style={{
                      background: '#16a34a',
                      color: '#ffffff',
                      border: 'none',
                      borderRadius: 4,
                      padding: '4px 10px',
                      fontSize: 10,
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 5
                    }}
                  >
                    <Play size={10} fill="#ffffff" />
                    <span>▶ Run in Live Terminal</span>
                  </button>

                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(msg.code);
                      alert('Code copied to clipboard!');
                    }}
                    style={{
                      background: '#1e293b',
                      color: '#38bdf8',
                      border: '1px solid #334155',
                      borderRadius: 4,
                      padding: '4px 10px',
                      fontSize: 10,
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 5
                    }}
                  >
                    <Copy size={10} />
                    <span>Copy Code</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}

        {isThinking && (
          <div style={{ alignSelf: 'flex-start', background: '#0f172a', padding: '8px 12px', borderRadius: 6, fontSize: 11, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Cpu size={13} className="spin" color="#38bdf8" />
            <span>AI Brain thinking...</span>
          </div>
        )}
      </div>

      {/* Chat Prompt Input Form */}
      <form onSubmit={handleSendQuery} style={{ padding: '8px', background: '#0f172a', borderTop: '1px solid #1e293b', display: 'flex', gap: 6, flexShrink: 0 }}>
        <input
          type="text"
          value={inputPrompt}
          onChange={e => setInputPrompt(e.target.value)}
          placeholder="Ask AI or type Python/SQL command..."
          style={{
            flex: 1,
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: 6,
            padding: '6px 10px',
            fontSize: 11,
            color: '#f8fafc',
            outline: 'none'
          }}
        />
        <button
          type="submit"
          style={{
            background: '#0284c7',
            color: '#ffffff',
            border: 'none',
            borderRadius: 6,
            padding: '6px 10px',
            fontSize: 11,
            fontWeight: 700,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 4
          }}
        >
          <Send size={12} />
        </button>
      </form>
    </div>
  );
}
