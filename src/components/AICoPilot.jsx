import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera,
  Sparkles,
  Play,
  Send,
  Bot,
  User,
  CheckCircle2,
  Cpu,
  Save,
  Copy,
  Activity,
  ServerCrash,
  AlertTriangle,
  ListTree,
  ChevronDown,
  ChevronRight,
  Check,
  BrainCircuit,
  Maximize2,
  X,
  RefreshCw,
  Terminal,
  Zap
} from 'lucide-react';

import { API, getAuthHeaders } from '../apiConfig';
import PlotZoomModal from './PlotZoomModal';
import ClarificationPrompt from './ClarificationPrompt';
import { logUserAction } from '../utils/telemetry';

const normalizeImageSrc = (src) => {
  if (!src || typeof src !== 'string') return null;
  if (src.startsWith('data:image') || src.startsWith('http://') || src.startsWith('https://') || src.startsWith('blob:')) {
    return src;
  }
  return `data:image/png;base64,${src}`;
};

export default function AICoPilot({
  activeDataset,
  schema,
  onRunCodeInTerminal,
  onRefresh,
  onSaveFile,
  isSaving,
  datasetLoaded,
  onZoomImage,
  onSubmitQuery,
  onAddMessage,
  appendMessage,
  patchMessage,
  clearMessages,
  messages = [],
  ollamaStatus = { online: false, model: 'unknown' },
  workflowPhase = 'idle', // idle, analyzing, planning, executing, validated, complete, error
  onSelectTab,
  onOpenCodePanel
}) {
  const appendMessageRef = useRef(appendMessage);
  const onAddMessageRef = useRef(onAddMessage);
  appendMessageRef.current = appendMessage;
  onAddMessageRef.current = onAddMessage;

  const addMessage = useCallback((msg) => {
    if (typeof appendMessageRef.current === 'function') {
      appendMessageRef.current(msg);
    } else if (typeof onAddMessageRef.current === 'function') {
      onAddMessageRef.current(msg);
    }
  }, []);

  const [inputPrompt, setInputPrompt] = useState('');
  const [isReadingScreen, setIsReadingScreen] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [recommendations, setRecommendations] = useState([]);
  const [showQuickPills, setShowQuickPills] = useState(true);
  const [openThoughtDrawers, setOpenThoughtDrawers] = useState({});
  const [openCodeDrawers, setOpenCodeDrawers] = useState({});
  const [openPlotDrawers, setOpenPlotDrawers] = useState({});
  const [copiedKey, setCopiedKey] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [activeScreenContext, setActiveScreenContext] = useState(null);

  const toggleCodeDrawer = (key) => {
    setOpenCodeDrawers(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const togglePlotDrawer = (key) => {
    setOpenPlotDrawers(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const chatFeedRef = useRef(null);
  const abortControllerRef = useRef(null);
  const streamingTimerRef = useRef(null);
  const lastNotifiedDatasetRef = useRef(null);

  const quickPills = [
    { label: '✨ Quality Audit', prompt: 'Audit data quality score and missing values' },
    { label: '📊 Auto EDA', prompt: 'Generate full automated EDA dashboard' },
    { label: '🔮 Forecast 30D', prompt: 'Forecast next 30 days time series trend' },
    { label: '🤖 AutoML Benchmark', prompt: 'Compare AutoML leaderboard models' },
    { label: '🔍 Root Cause', prompt: 'Why did metric drop? Run root cause analysis' }
  ];

  // ── Notify when dataset is connected (Triggered strictly ONCE per dataset switch) ──
  useEffect(() => {
    if (activeDataset?.name) {
      const datasetKey = `${activeDataset.name}-${activeDataset.rowCount || 0}-${activeDataset.loadedAt || ''}`;
      if (lastNotifiedDatasetRef.current !== datasetKey) {
        lastNotifiedDatasetRef.current = datasetKey;
        const colCnt = activeDataset.columns?.length || schema?.length || 0;
        addMessage({
          id: `ds-${Date.now()}`,
          sender: 'ai',
          text: `📁 Connected to dataset '${activeDataset.name}' (${activeDataset.rowCount?.toLocaleString() || 0} rows × ${colCnt} columns). Ask me to clean nulls, find outliers, plot heatmaps, or train ML models!`
        });
      }
    }
  }, [activeDataset?.name, activeDataset?.rowCount, activeDataset?.loadedAt, schema?.length, addMessage]);

  // Auto-scroll chat feed whenever messages update or thinking state changes
  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      if (chatFeedRef.current) {
        chatFeedRef.current.scrollTop = chatFeedRef.current.scrollHeight;
      }
    }, 50);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking, scrollToBottom]);

  // Cleanup streaming timer on unmount
  useEffect(() => {
    return () => {
      if (streamingTimerRef.current) clearInterval(streamingTimerRef.current);
    };
  }, []);

  // ── Copy Code to Clipboard Handler with Visual State ──
  const handleCopyCode = useCallback(async (code, key) => {
    if (!code) return;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(code);
      } else {
        // Fallback for non-secure contexts
        const textarea = document.createElement('textarea');
        textarea.value = code;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      }
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 2000);
    } catch (err) {
      console.error('Failed to copy code to clipboard:', err);
    }
  }, []);

  // ── Run Code in Terminal Handler ───────────────────────
  const handleRunCode = useCallback((code) => {
    if (!code) return;
    if (typeof onRunCodeInTerminal === 'function') {
      onRunCodeInTerminal(code);
    }
    if (typeof onRefresh === 'function') {
      onRefresh();
    }
  }, [onRunCodeInTerminal, onRefresh]);

  // ── Image Zoom Handler (Parent callback or internal fallback) ─
  const handleZoomImage = useCallback((imageSrc) => {
    if (!imageSrc) return;
    if (typeof onZoomImage === 'function') {
      onZoomImage(imageSrc);
    } else {
      setPreviewImage(imageSrc);
    }
  }, [onZoomImage]);

  // ── Thought Process Drawer Toggle ──────────────────────
  const toggleThoughtDrawer = useCallback((msgId) => {
    setOpenThoughtDrawers(prev => ({
      ...prev,
      [msgId]: !prev[msgId]
    }));
  }, []);

  // ── Detach Active Screen Context ───────────────────────
  const handleDetachContext = () => {
    setActiveScreenContext(null);
  };

  // ── Read Screen Context Action (No-op in production; vision backend unmounted)
  const handleReadScreen = () => {
    setIsReadingScreen(false);
  };

  // ── Helper to parse markdown bold **text** and code ───
  const renderFormattedText = (text) => {
    if (!text) return null;
    const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g);
    return parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={idx} style={{ color: '#34d399', fontWeight: 700 }}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={idx}
            style={{
              background: 'rgba(52, 211, 153, 0.12)',
              color: '#6ee7b7',
              padding: '1px 5px',
              borderRadius: 4,
              fontSize: '0.92em',
              fontFamily: 'var(--font-mono)'
            }}
          >
            {part.slice(1, -1)}
          </code>
        );
      }
      return part;
    });
  };

  // ── Smooth Message Streamer ───────────────────────────
  const streamAiResponse = (fullMessageObj) => {
    const fullText = fullMessageObj.text || '';
    const messageId = fullMessageObj.id;

    if (!fullText || fullText.length < 20) {
      // Very short text: append immediately
      addMessage(fullMessageObj);
      setIsThinking(false);
      return;
    }

    // Initial placeholder with streaming cursor
    const initialObj = {
      ...fullMessageObj,
      text: '',
      isStreaming: true
    };

    addMessage(initialObj);
    setIsThinking(false);

    let currentIndex = 0;
    const chunkSize = Math.max(2, Math.floor(fullText.length / 40));

    if (streamingTimerRef.current) clearInterval(streamingTimerRef.current);

    streamingTimerRef.current = setInterval(() => {
      currentIndex += chunkSize;
      if (currentIndex >= fullText.length) {
        currentIndex = fullText.length;
        clearInterval(streamingTimerRef.current);
        streamingTimerRef.current = null;

        const finalizedMsg = { ...fullMessageObj, text: fullText, isStreaming: false };
        if (typeof patchMessage === 'function') {
          patchMessage(messageId, finalizedMsg);
        }
      } else {
        const streamedSlice = fullText.slice(0, currentIndex);
        if (typeof patchMessage === 'function') {
          patchMessage(messageId, { text: streamedSlice, isStreaming: true });
        }
      }
    }, 18);
  };

  // ── Send Custom Data Science Query ─────────────────────
  const handleSendQuery = async (e, directQuery = null) => {
    if (e && e.preventDefault) e.preventDefault();

    const rawQuery = typeof directQuery === 'string' ? directQuery : inputPrompt;
    if (!rawQuery || !rawQuery.trim()) return;

    const userText = rawQuery.trim();
    if (!directQuery) setInputPrompt('');

    // Cancel any previous in-flight request if user hits send again
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const userMsgId = `user-${Date.now()}`;
    addMessage({ id: userMsgId, sender: 'user', text: userText });
    setIsThinking(true);

    // Build context-augmented payload if screen context is attached
    let queryPayload = userText;
    if (activeScreenContext && activeScreenContext.extractedText) {
      queryPayload = `[Scanned Active Window: "${activeScreenContext.activeTitle}"]:\n${activeScreenContext.extractedText.slice(0, 8000)}\n\n[User Request]:\n${userText}`;
    }

    try {
      // 1. Query Execution via onSubmitQuery prop or direct /api/chat fetch
      let data = null;
      if (onSubmitQuery) {
        data = await onSubmitQuery(queryPayload);
      } else {
        const res = await fetch(`${API}/api/chat`, {
          method: 'POST',
          headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ query: queryPayload }),
          signal: abortControllerRef.current.signal
        });
        if (!res.ok) {
          throw new Error(`Chat query failed (${res.status})`);
        }
        data = await res.json();
      }

      if (!data) {
        throw new Error('Empty response from reasoning engine');
      }

      const resObj = data.result || data;
      const responseMsg =
        resObj.response ||
        resObj.answer ||
        resObj.reply ||
        data.clarification ||
        (resObj.artifacts?.reports ? resObj.artifacts.reports[0] : null) ||
        'Query processed successfully.';
      const codeSnippet =
        resObj.code ||
        (resObj.artifacts?.code && resObj.artifacts.code.length > 0
          ? resObj.artifacts.code.join('\n\n')
          : null);
      const rawPlotImg =
        resObj.plot_image ||
        resObj.plot ||
        (resObj.artifacts?.plots && resObj.artifacts.plots.length > 0 ? resObj.artifacts.plots[0] : null);
      const plotImg = normalizeImageSrc(rawPlotImg);
      const multiResults = resObj.multi_results || resObj.artifacts?.multi_results || null;
      if (Array.isArray(multiResults)) {
        multiResults.forEach(sub => {
          if (sub.plot_image) {
            sub.plot_image = normalizeImageSrc(sub.plot_image);
          }
        });
      }
      const thoughtProcess =
        resObj.thought ||
        resObj.thought_process ||
        resObj.reasoning ||
        resObj.artifacts?.thought ||
        resObj.artifacts?.reasoning ||
        resObj.artifacts?.plan ||
        data.plan?.prompt_summary ||
        null;
      const executionTime = resObj.artifacts?.execution_time_sec || resObj.execution_time || null;
      const intentData = data.plan || data.intentData || data.intent || null;
      let errorState = data.error || data.errorState || (data.success === false && data.artifacts?.error ? { code: 'EXEC_ERROR', message: data.artifacts.error } : null);
      if (errorState && !errorState.action) {
        if (errorState.code === 'CLARIFICATION_REQUIRED') {
          errorState.message = data.clarification || errorState.message || 'More information is required before this analysis can run.';
          errorState.action = 'Reply with the requested parameter or choose one of the listed columns.';
        } else if (errorState.code === 'TARGET_LEAKAGE_RISK') {
          errorState.message = data.clarification || errorState.message || 'A selected predictor may reveal the target after the outcome occurs.';
          errorState.action = 'Rerun with only features available before the outcome, or explicitly request a retrospective explanation.';
        } else if (errorState.code === 'INTENT_PLAN_INVALID') {
          errorState.message = errorState.message || 'The generated analysis plan was invalid or ambiguous.';
          errorState.action = 'Please rephrase your query with more specific metrics or column names.';
        } else if (errorState.code === 'INTENT_PLANNER_UNAVAILABLE') {
          errorState.message = errorState.message || 'The planning engine is currently offline or unreachable.';
          errorState.action = 'Ensure your local Ollama service is running and the required model is pulled.';
        }
      }

      if (data.recommendations && Array.isArray(data.recommendations)) {
        setRecommendations(data.recommendations);
      }

      const validationBadge = data.validation_badge || data.validationBadge || resObj.validation_badge || null;

      const aiMsgObj = {
        id: `ai-${Date.now()}`,
        sender: 'ai',
        text: responseMsg,
        code: codeSnippet,
        actionCode: codeSnippet,
        plotImage: plotImg,
        validationBadge: validationBadge,
        multiResults: multiResults,
        thoughtProcess: thoughtProcess,
        executionTime: executionTime,
        intentData: intentData,
        errorState: errorState,
        candidateColumns: data.candidate_columns || data.candidateColumns || data.error?.candidate_columns || [],
        clarification: data.clarification || null,
        slotFilling: data.slot_filling || data.slotFilling || data.error?.slot_filling || null,
        inferredBindings: data.inferred_bindings || data.inferredBindings || [],
        isDatasetWide: Boolean(data.is_dataset_wide || data.isDatasetWide),
        decision: data.decision || null,
        status: data.status || null
      };

      streamAiResponse(aiMsgObj);
    } catch (err) {
      if (err.name === 'AbortError') return;
      
      let errorCode = 'UNKNOWN_ERROR';
      let errorMsg = err.message;
      let errorAction = 'Please try again or check the system logs.';
      const canonicalCode = err.message.match(/\[([A-Z][A-Z0-9_]+)\]/)?.[1];
      if (canonicalCode) {
        errorCode = canonicalCode;
      }

      if (errorCode === 'TARGET_LEAKAGE_RISK') {
        errorMsg = 'The proposed predictive model contains a feature that appears to reveal the target after the outcome.';
        errorAction = 'Rerun with features available before the outcome, or request a retrospective explanation.';
      } else if (err.message.includes('INTENT_PLAN_INVALID')) {
        errorCode = 'INTENT_PLAN_INVALID';
        errorMsg = 'The generated analysis plan was invalid or ambiguous.';
        errorAction = 'Please rephrase your query with more specific metrics or column names.';
      } else if (err.message.includes('INTENT_PLANNER_UNAVAILABLE')) {
        errorCode = 'INTENT_PLANNER_UNAVAILABLE';
        errorMsg = 'The planning engine is currently offline or unreachable.';
        errorAction = 'Ensure your local Ollama service is running and the required model is pulled.';
      }

      addMessage({
        id: `err-${Date.now()}`,
        sender: 'ai',
        text: `Reasoning Engine Notice: ${err.message}`,
        errorState: { code: errorCode, message: errorMsg, action: errorAction }
      });
      setIsThinking(false);
    }
  };

  const safeOllamaStatus =
    ollamaStatus && typeof ollamaStatus === 'object'
      ? ollamaStatus
      : { online: false, model: 'qwen2.5:3b-instruct' };
  const isStartingLocalEngine = !safeOllamaStatus.online && safeOllamaStatus.phase === 'starting';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: '#060907',
        color: '#f8fafc',
        fontFamily: 'var(--font-ui)',
        position: 'relative'
      }}
    >
      {/* Dynamic Keyframe & Style Injection */}
      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes cursorBlink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
        .spin { animation: spin 1s linear infinite; }
        .pulse-subtle { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        .streaming-cursor {
          display: inline-block;
          margin-left: 2px;
          color: #34d399;
          font-weight: 900;
          animation: cursorBlink 0.7s infinite;
        }
      `}</style>

      {/* Top Controls: Save File & Read Screen Header */}
      <div
        style={{
          padding: '6px 12px',
          background: '#0a0e0c',
          borderBottom: '1px solid #1a2921',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0
        }}
      >
        {/* Ollama Status Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, fontWeight: 600 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              color: safeOllamaStatus.online ? '#34d399' : isStartingLocalEngine ? '#fbbf24' : '#f43f5e',
              background: safeOllamaStatus.online
                ? 'rgba(52, 211, 153, 0.1)'
                : isStartingLocalEngine ? 'rgba(251, 191, 36, 0.1)' : 'rgba(244, 63, 94, 0.1)',
              padding: '3px 8px',
              borderRadius: 12,
              border: `1px solid ${
                safeOllamaStatus.online
                  ? 'rgba(52, 211, 153, 0.3)'
                  : isStartingLocalEngine ? 'rgba(251, 191, 36, 0.3)' : 'rgba(244, 63, 94, 0.3)'
              }`
            }}
          >
            <span className={isStartingLocalEngine ? 'pulse-subtle' : ''} style={{ fontSize: 12, lineHeight: 1 }}>●</span>
            {safeOllamaStatus.online
              ? 'AI Brain Online'
              : isStartingLocalEngine ? 'Starting local AI service…' : 'Ollama Offline'}
          </div>
          {!safeOllamaStatus.online && !isStartingLocalEngine && (
            <details style={{ color: '#94a3b8', cursor: 'pointer' }}>
              <summary>Setup</summary>
              <div
                style={{
                  position: 'absolute',
                  background: '#0a0e0c',
                  border: '1px solid #1a2921',
                  padding: 8,
                  borderRadius: 6,
                  zIndex: 50,
                  marginTop: 4,
                  fontSize: 11,
                  boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                }}
              >
                Run <code>ollama serve</code> to enable local neural planning.
              </div>
            </details>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
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
              title="Save changes back to dataset"
            >
              <Save size={12} />
              {isSaving ? 'Saving…' : '💾 Save'}
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
              cursor: isReadingScreen ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              boxShadow: '0 2px 8px rgba(16,185,129,0.35)',
              transition: 'all 0.15s ease'
            }}
            title="Scan current active window or dashboard context"
          >
            <Camera size={12} className={isReadingScreen ? 'spin' : ''} />
            {isReadingScreen ? 'Reading Screen…' : '📸 Read Screen'}
          </button>
        </div>
      </div>

      {/* Chat Messages Feed */}
      <div
        ref={chatFeedRef}
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '14px',
          display: 'flex',
          flexDirection: 'column',
          gap: 14
        }}
      >
        {messages.length === 0 && (
          <div
            style={{
              textAlign: 'center',
              padding: '30px 16px',
              color: '#64748b',
              fontSize: 12
            }}
          >
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: '50%',
                background: 'rgba(52, 211, 153, 0.1)',
                border: '1px solid rgba(52, 211, 153, 0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 12px auto',
                color: '#34d399'
              }}
            >
              <Bot size={22} />
            </div>
            <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 13, marginBottom: 4 }}>
              DataPilot Reasoning Copilot
            </div>
            <p style={{ maxWidth: 280, margin: '0 auto 14px auto', lineHeight: 1.45 }}>
              Ask questions about your data, generate ML models, run audits, or click a quick action below.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => {
          const isUser = msg.sender === 'user';
          const msgKey = msg.id || `msg-${idx}`;
          const isThoughtOpen = !!openThoughtDrawers[msgKey];
          const hasThoughtInfo =
            Boolean(msg.thoughtProcess) ||
            Boolean(msg.artifacts?.expected_outputs) ||
            Boolean(msg.artifacts?.dependencies) ||
            Boolean(msg.executionTime);

          return (
            <div
              key={msgKey}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignSelf: isUser ? 'flex-end' : 'flex-start',
                maxWidth: isUser ? '85%' : '94%',
                background: isUser
                  ? 'linear-gradient(135deg, #132219, #192b20)'
                  : 'linear-gradient(135deg, #0d1310, #111a14)',
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
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  marginBottom: 6,
                  fontSize: 11,
                  fontWeight: 700,
                  color: isUser ? '#6ee7b7' : '#34d399'
                }}
              >
                <div
                  style={{
                    width: 20,
                    height: 20,
                    borderRadius: '50%',
                    background: isUser ? 'rgba(52, 211, 153, 0.2)' : 'rgba(16, 185, 129, 0.25)',
                    border: '1px solid rgba(52, 211, 153, 0.4)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#34d399'
                  }}
                >
                  {isUser ? <User size={11} /> : <Bot size={11} />}
                </div>
                <span style={{ letterSpacing: '0.02em' }}>
                  {isUser ? 'You' : 'DataPilot DS Brain'}
                </span>
              </div>

              {/* Validation Guard Badge */}
              {msg.validationBadge && (
                <div
                  style={{
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
                  }}
                >
                  <CheckCircle2 size={13} color="#34d399" />
                  <span>Validation Guard: {msg.validationBadge}</span>
                </div>
              )}

              {/* Error State Display */}
              {msg.errorState && (
                <div
                  style={{
                    marginTop: 10,
                    marginBottom: 8,
                    background: '#1a0f14',
                    border: '1px solid #f43f5e',
                    borderRadius: 8,
                    padding: 12
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      color: '#f43f5e',
                      fontWeight: 700,
                      marginBottom: 6
                    }}
                  >
                    <ServerCrash size={16} />
                    <span>Error Code: {msg.errorState.code || 'UNKNOWN_ERROR'}</span>
                    {msg.errorState.recoverable && (
                      <span
                        style={{
                          fontSize: 9,
                          background: '#f43f5e',
                          color: '#fff',
                          padding: '2px 6px',
                          borderRadius: 4,
                          marginLeft: 'auto'
                        }}
                      >
                        RECOVERABLE
                      </span>
                    )}
                  </div>
                  <div style={{ color: '#f1f5f9', fontSize: 12, marginBottom: 6 }}>
                    {msg.errorState.message}
                  </div>
                  {msg.errorState.action && (
                    <div
                      style={{
                        background: 'rgba(244, 63, 94, 0.1)',
                        padding: 8,
                        borderRadius: 4,
                        color: '#fb7185',
                        fontSize: 11,
                        fontWeight: 600
                      }}
                    >
                      <AlertTriangle size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                      Required Action: {msg.errorState.action}
                    </div>
                  )}
                </div>
              )}

              {/* Intent Plan Display */}
              {msg.intentData && (
                <div
                  style={{
                    marginTop: 8,
                    marginBottom: 8,
                    background: '#0a0f12',
                    border: '1px solid #0369a1',
                    borderRadius: 8,
                    padding: 10
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      color: '#38bdf8',
                      fontSize: 12,
                      fontWeight: 700,
                      marginBottom: 6
                    }}
                  >
                    <ListTree size={14} />
                    <span>Analysis Plan ({msg.intentData.analysis_type || msg.intentData.intent_type || 'General DS'})</span>
                  </div>
                  {msg.intentData.metrics && msg.intentData.metrics.length > 0 && (
                    <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                      <strong style={{ color: '#bae6fd' }}>Metrics:</strong>{' '}
                      {msg.intentData.metrics.join(', ')}
                    </div>
                  )}
                  {msg.intentData.operations && msg.intentData.operations.length > 0 && (
                    <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                      <strong style={{ color: '#bae6fd' }}>Ops:</strong>{' '}
                      {msg.intentData.operations.join(' → ')}
                    </div>
                  )}
                  {msg.intentData.requires_clarification && (
                    <div
                      style={{
                        marginTop: 6,
                        color: '#fbbf24',
                        fontSize: 11,
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 4
                      }}
                    >
                      <AlertTriangle size={12} />
                      Clarification needed before proceeding
                    </div>
                  )}
                </div>
              )}

              {/* Thought Process & Reasoning Plan Collapsible Drawer */}
              {!isUser && hasThoughtInfo && (
                <div
                  style={{
                    margin: '6px 0 8px 0',
                    background: '#070c09',
                    border: '1px solid #1c3224',
                    borderRadius: 8,
                    overflow: 'hidden'
                  }}
                >
                  <button
                    onClick={() => toggleThoughtDrawer(msgKey)}
                    style={{
                      width: '100%',
                      background: 'transparent',
                      border: 'none',
                      padding: '6px 10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      color: '#6ee7b7',
                      fontSize: 11,
                      fontWeight: 600,
                      cursor: 'pointer',
                      textAlign: 'left'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <BrainCircuit size={13} color="#34d399" />
                      <span>Reasoning & Execution Details</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      {msg.executionTime && (
                        <span
                          style={{
                            background: 'rgba(52, 211, 153, 0.15)',
                            color: '#a7f3d0',
                            padding: '1px 6px',
                            borderRadius: 10,
                            fontSize: 10
                          }}
                        >
                          ⚡ {msg.executionTime}s
                        </span>
                      )}
                      {isThoughtOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    </div>
                  </button>

                  {isThoughtOpen && (
                    <div
                      style={{
                        padding: '8px 12px 10px 12px',
                        borderTop: '1px solid #16281d',
                        background: '#040705',
                        fontSize: 11,
                        color: '#94a3b8',
                        lineHeight: 1.5
                      }}
                    >
                      {msg.thoughtProcess && (
                        <div style={{ marginBottom: 6 }}>
                          <strong style={{ color: '#34d399' }}>Thought Trace:</strong>
                          <div style={{ marginTop: 3, color: '#cbd5e1', whiteSpace: 'pre-wrap' }}>
                            {msg.thoughtProcess}
                          </div>
                        </div>
                      )}
                      {msg.artifacts?.expected_outputs && (
                        <div style={{ marginTop: 4 }}>
                          <strong style={{ color: '#6ee7b7' }}>Expected Outputs:</strong>{' '}
                          {Array.isArray(msg.artifacts.expected_outputs)
                            ? msg.artifacts.expected_outputs.join(', ')
                            : JSON.stringify(msg.artifacts.expected_outputs)}
                        </div>
                      )}
                      {msg.artifacts?.dependencies && (
                        <div style={{ marginTop: 4 }}>
                          <strong style={{ color: '#6ee7b7' }}>Dependencies:</strong>{' '}
                          {Array.isArray(msg.artifacts.dependencies)
                            ? msg.artifacts.dependencies.join(', ')
                            : JSON.stringify(msg.artifacts.dependencies)}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Text Body with Streaming Cursor */}
              <div
                style={{
                  color: '#f1f5f9',
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'var(--font-ui)',
                  fontSize: 12.5,
                  lineHeight: 1.55
                }}
              >
                {renderFormattedText(msg.text)}
                {msg.isStreaming && <span className="streaming-cursor">▌</span>}
              </div>

              {/* Clarification Prompt */}
              {!isUser && (msg.decision === 'clarify' || msg.status === 'NEEDS_CLARIFICATION' || (msg.candidateColumns && msg.candidateColumns.length > 0) || msg.slotFilling) && (
                <ClarificationPrompt
                  prompt={msg.clarification || msg.text}
                  candidateColumns={msg.candidateColumns || []}
                  slotFilling={msg.slotFilling || null}
                  onSelectCandidate={(col) => setInputPrompt(prev => prev ? `${prev.trim()} ${col}` : col)}
                  onSubmitClarification={(text) => handleSendQuery(null, text)}
                />
              )}

              {/* Scanned Text Content Drawer */}
              {msg.extractedText && (
                <details
                  open
                  style={{
                    marginTop: 8,
                    background: '#050806',
                    border: '1px solid #1a2921',
                    borderRadius: 6,
                    padding: '6px 10px',
                    fontSize: 11,
                    color: '#94a3b8'
                  }}
                >
                  <summary style={{ cursor: 'pointer', fontWeight: 600, color: '#34d399' }}>
                    📑 Scanned Active Window Content ({msg.extractedText.length} chars)
                  </summary>
                  <pre
                    style={{
                      marginTop: 6,
                      marginBottom: 0,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      color: '#cbd5e1',
                      fontSize: 11,
                      fontFamily: 'var(--font-mono)'
                    }}
                  >
                    {msg.extractedText}
                  </pre>
                </details>
              )}

              {/* Multi-Intent Sub-Results Sequential Card List */}
              {msg.multiResults && msg.multiResults.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 10 }}>
                  {msg.multiResults.map((sub, sIdx) => {
                    const subKey = `sub-${msgKey}-${sIdx}`;
                    const isSubCopied = copiedKey === subKey;

                    return (
                      <div
                        key={sIdx}
                        style={{
                          background: '#070b09',
                          borderRadius: 8,
                          border: '1px solid #1a2921',
                          padding: 10
                        }}
                      >
                        <div
                          style={{
                            fontSize: 11,
                            fontWeight: 700,
                            color: '#34d399',
                            marginBottom: 4
                          }}
                        >
                          Part {sIdx + 1}: "{sub.sub_query}" ({sub.label || 'Step'})
                        </div>
                        <div
                          style={{
                            color: '#e2e8f0',
                            fontSize: 12,
                            lineHeight: 1.5,
                            marginBottom: 8
                          }}
                        >
                          {renderFormattedText(sub.response)}
                        </div>

                        {sub.code && (
                          <div style={{ marginTop: 6, marginBottom: 8 }}>
                            <div
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                marginBottom: 4
                              }}
                            >
                              <span style={{ fontSize: 10, color: '#64748b', fontWeight: 600 }}>
                                Python Script
                              </span>
                              <div style={{ display: 'flex', gap: 6 }}>
                                <button
                                  onClick={() => handleCopyCode(sub.code, subKey)}
                                  style={{
                                    background: '#121c17',
                                    color: isSubCopied ? '#34d399' : '#a7f3d0',
                                    border: '1px solid #1a2921',
                                    borderRadius: 4,
                                    padding: '2px 8px',
                                    fontSize: 10,
                                    fontWeight: 600,
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 3
                                  }}
                                  title="Copy code to clipboard"
                                >
                                  {isSubCopied ? <Check size={10} /> : <Copy size={10} />}
                                  {isSubCopied ? 'Copied!' : 'Copy'}
                                </button>
                                <button
                                  onClick={() => handleRunCode(sub.code)}
                                  style={{
                                    background: 'linear-gradient(135deg, #10b981, #059669)',
                                    color: '#ffffff',
                                    border: 'none',
                                    borderRadius: 4,
                                    padding: '2px 8px',
                                    fontSize: 10,
                                    fontWeight: 700,
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 3
                                  }}
                                  title="Execute in bottom Terminal HUD"
                                >
                                  <Play size={9} fill="#fff" />
                                  Run
                                </button>
                              </div>
                            </div>
                            <pre
                              style={{
                                background: '#040605',
                                border: '1px solid #1a2921',
                                borderRadius: 6,
                                padding: 8,
                                fontSize: 11,
                                color: '#a7f3d0',
                                overflowX: 'auto'
                              }}
                            >
                              <code>{sub.code}</code>
                            </pre>
                          </div>
                        )}

                        {sub.plot_image && (
                          <div
                            style={{
                              marginTop: 8,
                              background: '#040605',
                              borderRadius: 8,
                              overflow: 'hidden',
                              border: '1px solid #1a2921',
                              padding: 6
                            }}
                          >
                            <div
                              style={{
                                fontSize: 10,
                                fontWeight: 700,
                                color: '#34d399',
                                marginBottom: 4,
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                              }}
                            >
                              <span>📊 Plot: {sub.label}</span>
                              <div style={{ display: 'flex', gap: 8 }}>
                                <button
                                  onClick={() => handleZoomImage(sub.plot_image)}
                                  style={{
                                    background: 'transparent',
                                    border: 'none',
                                    color: '#34d399',
                                    fontSize: 10,
                                    fontWeight: 700,
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 2
                                  }}
                                >
                                  <Maximize2 size={10} /> Inspect
                                </button>
                                <a
                                  href={sub.plot_image}
                                  download={`plot_part_${sIdx + 1}.png`}
                                  style={{
                                    color: '#34d399',
                                    fontSize: 10,
                                    fontWeight: 700,
                                    textDecoration: 'none'
                                  }}
                                >
                                  Download PNG
                                </a>
                              </div>
                            </div>
                            <img
                              src={sub.plot_image}
                              alt={sub.label}
                              onClick={() => handleZoomImage(sub.plot_image)}
                              style={{ width: '100%', borderRadius: 6, cursor: 'zoom-in' }}
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Clean Action Badges: Direct navigation to dedicated Code Panel / Results Panel */}
              {!isUser && (msg.actionCode || msg.code || msg.plotImage) && (
                <div style={{ display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
                  {(msg.actionCode || msg.code) && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <button
                        onClick={() => {
                          if (typeof onSelectTab === 'function') onSelectTab('code');
                          if (typeof onOpenCodePanel === 'function') onOpenCodePanel();
                        }}
                        style={{
                          background: 'rgba(16, 185, 129, 0.15)',
                          border: '1px solid rgba(52, 211, 153, 0.35)',
                          color: '#6ee7b7',
                          padding: '4px 10px',
                          borderRadius: 6,
                          fontSize: 11,
                          fontWeight: 600,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 5,
                          transition: 'all 0.15s ease'
                        }}
                        title="View and edit code in dedicated Center Code Editor"
                      >
                        <Terminal size={12} color="#34d399" />
                        <span>Python Script in Editor ↗</span>
                      </button>
                      <button
                        onClick={() => toggleCodeDrawer(msgKey)}
                        style={{
                          background: '#0a0e0c',
                          border: '1px solid #1a2921',
                          color: '#94a3b8',
                          padding: '4px 6px',
                          borderRadius: 6,
                          fontSize: 10,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 2
                        }}
                        title={openCodeDrawers[msgKey] ? "Hide inline code preview" : "Show inline code preview"}
                      >
                        {openCodeDrawers[msgKey] ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                        <span>{openCodeDrawers[msgKey] ? 'Hide' : 'Peek'}</span>
                      </button>
                    </div>
                  )}

                  {msg.plotImage && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <button
                        onClick={() => {
                          if (typeof onSelectTab === 'function') onSelectTab('results');
                          if (typeof onZoomImage === 'function') onZoomImage(msg.plotImage);
                        }}
                        style={{
                          background: 'rgba(56, 189, 248, 0.15)',
                          border: '1px solid rgba(56, 189, 248, 0.35)',
                          color: '#7dd3fc',
                          padding: '4px 10px',
                          borderRadius: 6,
                          fontSize: 11,
                          fontWeight: 600,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 5,
                          transition: 'all 0.15s ease'
                        }}
                        title="View full chart in dedicated Results Panel"
                      >
                        <Activity size={12} color="#38bdf8" />
                        <span>Chart in Results Panel ↗</span>
                      </button>
                      <button
                        onClick={() => togglePlotDrawer(msgKey)}
                        style={{
                          background: '#0a0e0c',
                          border: '1px solid #1a2921',
                          color: '#94a3b8',
                          padding: '4px 6px',
                          borderRadius: 6,
                          fontSize: 10,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 2
                        }}
                        title={openPlotDrawers[msgKey] ? "Hide inline plot preview" : "Show inline plot preview"}
                      >
                        {openPlotDrawers[msgKey] ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                        <span>{openPlotDrawers[msgKey] ? 'Hide' : 'Peek'}</span>
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Collapsible Inline Code Preview (Only when explicitly peeked) */}
              {openCodeDrawers[msgKey] && (msg.actionCode || msg.code) && (
                <div
                  style={{
                    marginTop: 8,
                    background: '#050806',
                    borderRadius: 8,
                    overflow: 'hidden',
                    border: '1px solid #1a2921'
                  }}
                >
                  <div
                    style={{
                      background: '#0a0e0c',
                      padding: '6px 10px',
                      borderBottom: '1px solid #1a2921',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      fontSize: 11,
                      color: '#a7f3d0',
                      fontWeight: 700
                    }}
                  >
                    <span>🐍 Python Code Snippet</span>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button
                        onClick={() => handleCopyCode(msg.actionCode || msg.code, `main-${msgKey}`)}
                        style={{
                          background: '#121c17',
                          color: copiedKey === `main-${msgKey}` ? '#34d399' : '#a7f3d0',
                          border: '1px solid #1a2921',
                          borderRadius: 4,
                          padding: '2px 8px',
                          fontSize: 10,
                          fontWeight: 600,
                          cursor: 'pointer'
                        }}
                      >
                        {copiedKey === `main-${msgKey}` ? 'Copied!' : 'Copy'}
                      </button>
                      <button
                        onClick={() => handleRunCode(msg.actionCode || msg.code)}
                        style={{
                          background: 'linear-gradient(135deg, #10b981, #059669)',
                          color: '#ffffff',
                          border: 'none',
                          borderRadius: 4,
                          padding: '2px 8px',
                          fontSize: 10,
                          fontWeight: 700,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 3
                        }}
                      >
                        <Play size={9} fill="#fff" />
                        Run
                      </button>
                    </div>
                  </div>
                  <pre
                    style={{
                      margin: 0,
                      padding: 10,
                      fontSize: 11,
                      fontFamily: 'var(--font-mono)',
                      color: '#34d399',
                      overflowX: 'auto',
                      whiteSpace: 'pre-wrap',
                      maxHeight: 200
                    }}
                  >
                    {msg.actionCode || msg.code}
                  </pre>
                </div>
              )}

              {/* Collapsible Inline Plot Preview (Only when explicitly peeked) */}
              {openPlotDrawers[msgKey] && msg.plotImage && (
                <div
                  style={{
                    marginTop: 8,
                    background: '#050806',
                    borderRadius: 8,
                    overflow: 'hidden',
                    border: '1px solid #1a2921',
                    padding: 6
                  }}
                >
                  <img
                    src={msg.plotImage}
                    alt="Generated Plot"
                    onClick={() => handleZoomImage(msg.plotImage)}
                    style={{ width: '100%', borderRadius: 6, cursor: 'zoom-in' }}
                  />
                </div>
              )}
            </div>
          );
        })}

        {isThinking && (
          <div
            style={{
              alignSelf: 'flex-start',
              background: '#0a0e0c',
              border: '1px solid #1a2921',
              padding: '8px 14px',
              borderRadius: 16,
              fontSize: 11,
              color: '#94a3b8',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              boxShadow: '0 2px 10px rgba(0,0,0,0.3)'
            }}
          >
            <Cpu size={14} className="spin" color="#34d399" />
            <span style={{ color: '#e2e8f0' }}>AI Brain reasoning & executing query…</span>
          </div>
        )}
      </div>

      {/* Attached Active Window Context Badge Bar */}
      {activeScreenContext && (
        <div
          style={{
            padding: '6px 12px',
            background: '#0a1610',
            borderTop: '1px solid #163828',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: 11,
            color: '#34d399',
            flexShrink: 0
          }}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            <span>
              {activeScreenContext.contentType === 'dataset'
                ? '📊'
                : activeScreenContext.contentType === 'document'
                ? '📄'
                : '📸'}{' '}
              Context Attached:
            </span>
            <strong style={{ color: '#6ee7b7' }}>
              {activeScreenContext.summary || activeScreenContext.activeTitle}
            </strong>
          </span>
          <button
            onClick={handleDetachContext}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#f43f5e',
              cursor: 'pointer',
              fontSize: 11,
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: 2
            }}
            title="Detach active context from future queries"
          >
            <X size={12} /> Detach
          </button>
        </div>
      )}

      {/* Suggested Follow-up Action Chips */}
      {recommendations.length > 0 && (
        <div
          style={{
            padding: '6px 12px',
            background: '#060907',
            display: 'flex',
            gap: 6,
            overflowX: 'auto',
            flexShrink: 0,
            borderTop: '1px solid #1a2921'
          }}
        >
          {recommendations.map((rec, idx) => (
            <button
              key={idx}
              onClick={() => handleSendQuery(null, rec)}
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
                transition: 'all 0.15s ease',
                display: 'flex',
                alignItems: 'center',
                gap: 4
              }}
              onMouseOver={e => {
                e.currentTarget.style.background = '#16221c';
                e.currentTarget.style.borderColor = '#34d399';
              }}
              onMouseOut={e => {
                e.currentTarget.style.background = '#0d1310';
                e.currentTarget.style.borderColor = '#1a2921';
              }}
              title={`Run: ${rec}`}
            >
              <Zap size={10} /> {rec}
            </button>
          ))}
        </div>
      )}

      {/* Suggested Quick Question Pills */}
      {showQuickPills && (
        <div
          style={{
            padding: '6px 12px',
            background: '#060907',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            overflowX: 'auto',
            flexShrink: 0,
            borderTop: '1px solid #1a2921'
          }}
        >
          <span style={{ fontSize: 10, color: '#64748b', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap' }}>
            Quick:
          </span>
          {quickPills.map((pill, idx) => (
            <button
              key={idx}
              onClick={() => {
                logUserAction('QUICK_PILL_CLICKED', { pill: pill.label, prompt: pill.prompt });
                handleSendQuery(null, pill.prompt);
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
                transition: 'all 0.15s ease',
                display: 'flex',
                alignItems: 'center',
                gap: 4
              }}
              onMouseEnter={e => {
                e.currentTarget.style.background = '#10b981';
                e.currentTarget.style.color = '#ffffff';
                e.currentTarget.style.borderColor = '#10b981';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = '#0d1310';
                e.currentTarget.style.color = '#34d399';
                e.currentTarget.style.borderColor = '#1a2921';
              }}
              title={pill.prompt}
            >
              {pill.label}
            </button>
          ))}
        </div>
      )}

      {/* Workflow Phase Indicator */}
      <div
        style={{
          padding: '4px 14px',
          background: '#0a0e0c',
          borderTop: '1px solid #1a2921',
          fontSize: 11,
          color: '#94a3b8',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          flexShrink: 0
        }}
      >
        {workflowPhase === 'idle' && (
          <>
            <Activity size={12} /> Phase: Idle
          </>
        )}
        {workflowPhase === 'analyzing' && (
          <>
            <Cpu size={12} className="spin" color="#38bdf8" />{' '}
            <span style={{ color: '#38bdf8' }}>Phase: Analyzing Request...</span>
          </>
        )}
        {workflowPhase === 'planning' && (
          <>
            <ListTree size={12} color="#facc15" />{' '}
            <span style={{ color: '#facc15' }}>Phase: Planning Execution Graph...</span>
          </>
        )}
        {workflowPhase === 'executing' && (
          <>
            <Play size={12} color="#10b981" />{' '}
            <span style={{ color: '#10b981' }}>Phase: Executing Graph Nodes...</span>
          </>
        )}
        {workflowPhase === 'validated' && (
          <>
            <CheckCircle2 size={12} color="#34d399" />{' '}
            <span style={{ color: '#34d399' }}>Phase: Results Validated</span>
          </>
        )}
        {workflowPhase === 'complete' && (
          <>
            <Sparkles size={12} color="#a78bfa" />{' '}
            <span style={{ color: '#a78bfa' }}>Phase: Workflow Complete</span>
          </>
        )}
        {workflowPhase === 'error' && (
          <>
            <ServerCrash size={12} color="#f43f5e" />{' '}
            <span style={{ color: '#f43f5e' }}>Phase: Execution Error</span>
          </>
        )}
      </div>

      {/* Chat Prompt Input Form */}
      <form
        onSubmit={handleSendQuery}
        style={{
          padding: '10px 14px',
          background: '#0a0e0c',
          borderTop: '1px solid #1a2921',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          flexShrink: 0
        }}
      >
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
          onFocus={e => (e.currentTarget.style.borderColor = '#10b981')}
          onBlur={e => (e.currentTarget.style.borderColor = '#23352b')}
        />
        <button
          type="submit"
          disabled={isThinking || !inputPrompt.trim()}
          style={{
            width: 36,
            height: 36,
            borderRadius: 18,
            background:
              isThinking || !inputPrompt.trim()
                ? '#1e2922'
                : 'linear-gradient(135deg, #10b981, #059669)',
            color: isThinking || !inputPrompt.trim() ? '#64748b' : '#ffffff',
            border: 'none',
            cursor: isThinking || !inputPrompt.trim() ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            boxShadow:
              isThinking || !inputPrompt.trim()
                ? 'none'
                : '0 3px 10px rgba(16, 185, 129, 0.45)',
            transition: 'all 0.15s ease'
          }}
          onMouseDown={e => {
            if (!isThinking && inputPrompt.trim()) e.currentTarget.style.transform = 'scale(0.95)';
          }}
          onMouseUp={e => {
            if (!isThinking && inputPrompt.trim()) e.currentTarget.style.transform = 'scale(1)';
          }}
          title="Send query"
        >
          {isThinking ? <Cpu size={14} className="spin" /> : <Send size={14} />}
        </button>
      </form>

      {/* Fallback Internal Plot Zoom Modal (when parent onZoomImage is unavailable) */}
      {previewImage && !onZoomImage && (
        <PlotZoomModal imageSrc={previewImage} onClose={() => setPreviewImage(null)} />
      )}
    </div>
  );
}
