import React, { useState, useEffect } from 'react';
import { Play, CheckCircle, AlertCircle, Copy, Save, Code, Zap } from 'lucide-react';

export default function CodePanel({ 
  codeVersion, 
  onValidate, 
  onRun, 
  onSaveUserEdit, 
  onCopyCode, 
  isValidating, 
  isRunning, 
  validationResult,
  executionEnabled = false
}) {
  const [editedCode, setEditedCode] = useState('');

  useEffect(() => {
    if (codeVersion?.code) {
      setEditedCode(codeVersion.code);
    }
  }, [codeVersion]);

  if (!codeVersion) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8', backgroundColor: '#0f172a', height: '100%' }}>
        <Code size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
        <p>Generate code by submitting a query in the AI CoPilot</p>
      </div>
    );
  }

  const isEdited = editedCode !== codeVersion.code;
  const showUserModified = codeVersion.is_user_edited || isEdited;
  // Typed-query execution supplies an evidence report (`evidence_sufficient`),
  // while the manual code endpoint supplies `is_valid` / `isValid`.  Both are
  // authoritative validation outcomes; treating the former as undefined made
  // a successfully executed analysis look like a failed code validation.
  const isExecutionValidation = typeof validationResult?.evidence_sufficient === 'boolean';
  const validationPassed = validationResult?.is_valid
    ?? validationResult?.isValid
    ?? validationResult?.evidence_sufficient;
  const validationIssues = validationResult?.issues ?? validationResult?.missing_evidence ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', backgroundColor: '#0f172a', color: '#f8fafc' }}>
      
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #334155', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h2 style={{ margin: 0, fontSize: '14px', fontWeight: 'bold', letterSpacing: '1px' }}>GENERATED CODE</h2>
          <span style={{ background: '#334155', padding: '2px 8px', borderRadius: '12px', fontSize: '11px', fontWeight: 'bold' }}>
            {codeVersion.version_id?.substring(0, 7) || 'v1'}
          </span>
          {showUserModified && (
            <span style={{ background: '#78350f', color: '#fbbf24', padding: '2px 8px', borderRadius: '12px', fontSize: '11px', fontWeight: 'bold' }}>
              User Modified
            </span>
          )}
        </div>
        <div style={{ fontSize: '12px', color: '#94a3b8', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {codeVersion.purpose}
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', position: 'relative' }}>
        <div style={{ 
          width: '40px', 
          backgroundColor: '#1e293b', 
          borderRight: '1px solid #334155',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          paddingTop: '16px',
          color: '#64748b',
          fontSize: '12px',
          fontFamily: 'monospace',
          userSelect: 'none'
        }}>
          {editedCode.split('\n').map((_, i) => (
            <div key={i} style={{ height: '21px' }}>{i + 1}</div>
          ))}
        </div>
        <textarea
          value={editedCode}
          onChange={(e) => setEditedCode(e.target.value)}
          spellCheck="false"
          style={{
            flex: 1,
            backgroundColor: '#0f172a',
            color: '#e2e8f0',
            fontFamily: '"JetBrains Mono", "Fira Code", monospace',
            fontSize: '13px',
            lineHeight: '21px',
            padding: '16px',
            border: 'none',
            outline: 'none',
            resize: 'none',
            whiteSpace: 'pre'
          }}
        />
      </div>

      {validationResult && (
        <div style={{ 
          padding: '12px 16px', 
          backgroundColor: validationPassed ? '#064e3b' : '#450a0a',
          borderTop: `1px solid ${validationPassed ? '#10b981' : '#dc2626'}`,
          color: validationPassed ? '#34d399' : '#fca5a5',
          fontSize: '13px'
        }}>
          {validationPassed ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle size={16} /> ✓ {isExecutionValidation ? 'Execution Validated — Result contract satisfied' : 'Code Validated — Ready to Run'}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold' }}>
                <AlertCircle size={16} /> Validation Issues Found
              </div>
              <ul style={{ margin: 0, paddingLeft: '24px' }}>
                {validationIssues.map((issue, i) => (
                  <li key={i}>{typeof issue === 'string' ? issue : `${issue.severity || 'Issue'}: ${issue.message || String(issue)}`}</li>
                ))}
              </ul>
              {validationResult.hasAutofix && (
                <button style={{ 
                  alignSelf: 'flex-start',
                  marginTop: '8px',
                  background: '#dc2626',
                  color: 'white',
                  border: 'none',
                  padding: '4px 12px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '12px'
                }}>
                  <Zap size={14} /> Apply Autofix
                </button>
              )}
            </div>
          )}
        </div>
      )}

      <div style={{ padding: '12px 16px', borderTop: '1px solid #334155', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#1e293b' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            onClick={() => onValidate(editedCode)}
            disabled={isValidating}
            style={{ 
              background: '#3b82f6', 
              color: 'white', 
              border: 'none', 
              padding: '6px 16px', 
              borderRadius: '4px', 
              cursor: isValidating ? 'wait' : 'pointer',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '13px',
              opacity: isValidating ? 0.7 : 1
            }}
          >
            <CheckCircle size={14} /> {isValidating ? 'Validating...' : 'Validate Code'}
          </button>
          
          {executionEnabled && <button 
            onClick={() => onRun(editedCode)}
            disabled={isRunning || (validationResult && !validationPassed)}
            style={{ 
              background: 'linear-gradient(to right, #10b981, #059669)', 
              color: 'white', 
              border: 'none', 
              padding: '6px 16px', 
              borderRadius: '4px', 
              cursor: (isRunning || (validationResult && !validationPassed)) ? 'not-allowed' : 'pointer',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '13px',
              opacity: (isRunning || (validationResult && !validationPassed)) ? 0.5 : 1
            }}
          >
            <Play size={14} fill="currentColor" /> {isRunning ? 'Running...' : 'Run Code'}
          </button>}
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          {isEdited && (
            <button 
              onClick={() => onSaveUserEdit(editedCode)}
              style={{ 
                background: 'transparent', 
                color: '#94a3b8', 
                border: '1px solid #475569', 
                padding: '6px 12px', 
                borderRadius: '4px', 
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '12px'
              }}
            >
              <Save size={14} /> Save as Version
            </button>
          )}
          <button 
            onClick={() => {
              if (onCopyCode) onCopyCode(editedCode);
              else navigator.clipboard.writeText(editedCode);
            }}
            style={{ 
              background: 'transparent', 
              color: '#94a3b8', 
              border: '1px solid #475569', 
              padding: '6px 12px', 
              borderRadius: '4px', 
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '12px'
            }}
          >
            <Copy size={14} /> Copy
          </button>
        </div>
      </div>
      
      {codeVersion.dependencies && codeVersion.dependencies.length > 0 && (
        <div style={{ padding: '8px 16px', backgroundColor: '#0f172a', borderTop: '1px solid #334155', display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontSize: '11px', color: '#64748b' }}>Dependencies:</span>
          {codeVersion.dependencies.map(dep => (
            <span key={dep} style={{ background: '#1e293b', color: '#94a3b8', padding: '2px 8px', borderRadius: '12px', fontSize: '11px' }}>
              {dep}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
