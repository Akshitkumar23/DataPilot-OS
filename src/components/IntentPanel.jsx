import React, { useState } from 'react';
import { Brain, AlertTriangle, CheckCircle, ChevronDown, ChevronRight, Activity } from 'lucide-react';

export default function IntentPanel({ intent, clarification, onClarificationAnswer, isAnalyzing, ollamaStatus }) {
  const [showAssumptions, setShowAssumptions] = useState(false);
  const [clarificationAnswers, setClarificationAnswers] = useState({});

  if (ollamaStatus && ollamaStatus.is_running === false) {
    return (
      <div style={{ padding: '20px', backgroundColor: '#0f172a', color: '#f8fafc', height: '100%' }}>
        <div style={{ 
          border: '1px solid #dc2626', 
          backgroundColor: '#450a0a', 
          padding: '20px', 
          borderRadius: '8px',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#fca5a5' }}>
            <AlertTriangle size={24} />
            <h2 style={{ margin: 0, fontSize: '18px' }}>Ollama Not Running</h2>
          </div>
          <p style={{ margin: 0, fontSize: '14px', color: '#fecaca' }}>
            DataPilot requires a local Ollama instance.
          </p>
          <ol style={{ margin: 0, paddingLeft: '20px', color: '#fecaca', fontSize: '14px' }}>
            <li>Download Ollama: ollama.com</li>
            <li>Run: <code>ollama serve</code></li>
            <li>Pull model: <code>ollama pull qwen2.5:3b-instruct</code></li>
          </ol>
          <button style={{
            background: '#dc2626',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
            marginTop: '8px',
            alignSelf: 'flex-start'
          }}>
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (isAnalyzing) {
    return (
      <div style={{ padding: '40px 20px', textAlign: 'center', backgroundColor: '#0f172a', color: '#f8fafc', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <Brain size={48} color="#10b981" style={{ animation: 'spin 3s linear infinite', marginBottom: '16px' }} />
        <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '500' }}>Understanding your request...</h3>
        <style>{`
          @keyframes spin { 100% { transform: rotate(360deg); } }
        `}</style>
      </div>
    );
  }

  if (!intent && !clarification) {
    return (
      <div style={{ padding: '20px', backgroundColor: '#0f172a', color: '#94a3b8', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p>Awaiting input...</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px', backgroundColor: '#0f172a', color: '#f8fafc', height: '100%', overflowY: 'auto' }}>
      {clarification && (
        <div style={{ 
          border: '1px solid #d97706', 
          backgroundColor: '#422006', 
          padding: '16px', 
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h3 style={{ margin: '0 0 12px 0', color: '#fcd34d', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertTriangle size={18} />
            Before proceeding, I need to clarify:
          </h3>
          
          {clarification.questions?.map((q, idx) => (
            <div key={idx} style={{ marginBottom: '16px' }}>
              <p style={{ margin: '0 0 8px 0', fontSize: '14px' }}>{q.text}</p>
              
              {q.type === 'single_choice' && q.options?.map(opt => (
                <label key={opt} style={{ display: 'block', marginBottom: '4px', fontSize: '13px', cursor: 'pointer' }}>
                  <input 
                    type="radio" 
                    name={`q-${idx}`} 
                    value={opt}
                    onChange={(e) => setClarificationAnswers({...clarificationAnswers, [q.id]: e.target.value})}
                    style={{ marginRight: '8px' }}
                  />
                  {opt}
                </label>
              ))}

              {q.type === 'multi_choice' && q.options?.map(opt => (
                <label key={opt} style={{ display: 'block', marginBottom: '4px', fontSize: '13px', cursor: 'pointer' }}>
                  <input 
                    type="checkbox"
                    onChange={(e) => {
                      const current = clarificationAnswers[q.id] || [];
                      if (e.target.checked) {
                        setClarificationAnswers({...clarificationAnswers, [q.id]: [...current, opt]});
                      } else {
                        setClarificationAnswers({...clarificationAnswers, [q.id]: current.filter(x => x !== opt)});
                      }
                    }}
                    style={{ marginRight: '8px' }}
                  />
                  {opt}
                </label>
              ))}

              {q.type === 'free_text' && (
                <input 
                  type="text" 
                  style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d97706', background: '#fff', color: '#000' }}
                  onChange={(e) => setClarificationAnswers({...clarificationAnswers, [q.id]: e.target.value})}
                />
              )}
            </div>
          ))}

          <button 
            onClick={() => onClarificationAnswer(clarificationAnswers)}
            style={{
              background: '#d97706',
              color: 'white',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            Continue Analysis
          </button>
        </div>
      )}

      {intent && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ background: '#1e293b', padding: '4px 8px', borderRadius: '4px', fontSize: '12px', color: '#38bdf8', fontWeight: 'bold' }}>
                {intent.analysis_type || 'ANALYSIS'}
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', color: '#10b981' }}>
                <CheckCircle size={14} />
                {(intent.confidence || 0) * 100}% Confidence
              </div>
            </div>
            <h2 style={{ margin: 0, fontSize: '20px', fontWeight: 'bold', lineHeight: '1.4' }}>
              {intent.objective}
            </h2>
          </div>

          <div style={{ background: '#1e293b', padding: '16px', borderRadius: '8px' }}>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#94a3b8', textTransform: 'uppercase' }}>Operations</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {intent.operations?.map((op, i) => (
                <span key={i} style={{ background: '#334155', padding: '4px 10px', borderRadius: '12px', fontSize: '12px' }}>
                  {op}
                </span>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '16px' }}>
            <div style={{ flex: 1, background: '#1e293b', padding: '16px', borderRadius: '8px' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#94a3b8', textTransform: 'uppercase' }}>Metrics</h4>
              <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px' }}>
                {intent.metrics?.map((m, i) => <li key={i}>{m}</li>)}
              </ul>
            </div>
            <div style={{ flex: 1, background: '#1e293b', padding: '16px', borderRadius: '8px' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#94a3b8', textTransform: 'uppercase' }}>Dimensions</h4>
              <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px' }}>
                {intent.dimensions?.map((d, i) => <li key={i}>{d}</li>)}
              </ul>
            </div>
          </div>

          {intent.assumptions && intent.assumptions.length > 0 && (
            <div style={{ background: '#332919', border: '1px solid #78350f', borderRadius: '8px', overflow: 'hidden' }}>
              <div 
                style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', background: '#422006' }}
                onClick={() => setShowAssumptions(!showAssumptions)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#fbbf24', fontSize: '13px', fontWeight: 'bold' }}>
                  <AlertTriangle size={16} />
                  Assumptions Made ({intent.assumptions.length})
                </div>
                {showAssumptions ? <ChevronDown size={16} color="#fbbf24" /> : <ChevronRight size={16} color="#fbbf24" />}
              </div>
              {showAssumptions && (
                <div style={{ padding: '12px 16px' }}>
                  <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', color: '#fde68a' }}>
                    {intent.assumptions.map((assump, i) => <li key={i} style={{ marginBottom: '4px' }}>{assump}</li>)}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
