import { useState, useMemo } from 'react';
import { HelpCircle, Send, ArrowRight, Check, Sparkles } from 'lucide-react';
import { logUserAction } from '../utils/telemetry';

export default function ClarificationPrompt({
  prompt,
  candidateColumns = [],
  slotFilling = null,
  onSelectCandidate,
  onSubmitClarification,
}) {
  const [customText, setCustomText] = useState('');
  const [selectedSlots, setSelectedSlots] = useState({});

  // Parse card details if provided
  const cardType = slotFilling?.card_type || 'generic';
  const cardTitle = slotFilling?.title || null;
  const slots = useMemo(() => slotFilling?.slots || [], [slotFilling]);
  const suggestedQueries = useMemo(() => slotFilling?.suggested_queries || [], [slotFilling]);

  // Handle slot candidate toggle/select
  const handleSelectSlotCandidate = (slotName, candidate) => {
    logUserAction('CLARIFICATION_SLOT_SELECTED', { slot: slotName, candidate });
    setSelectedSlots(prev => ({
      ...prev,
      [slotName]: candidate
    }));
  };

  // Determine if all required slots are fulfilled
  const isFormComplete = useMemo(() => {
    if (!slots.length) return false;
    return slots.every(slot => {
      if (!slot.required) return true;
      return Boolean(selectedSlots[slot.slot_name]);
    });
  }, [slots, selectedSlots]);

  // Construct structured query from selected slots
  const handleExecuteSlots = (e) => {
    e?.preventDefault();
    if (!isFormComplete || typeof onSubmitClarification !== 'function') return;

    let constructedQuery = '';
    if (cardType === 'target_selector' && selectedSlots.target_column) {
      constructedQuery = `Train machine learning model predicting target column ${selectedSlots.target_column}`;
    } else if (cardType === 'forecast_selector') {
      const metric = selectedSlots.metric_column;
      const timeCol = selectedSlots.time_column;
      if (metric && timeCol) {
        constructedQuery = `Forecast metric ${metric} over time column ${timeCol}`;
      } else if (metric) {
        constructedQuery = `Forecast metric ${metric}`;
      }
    } else if (cardType === 'correlation_selector') {
      const x = selectedSlots.x_column;
      const y = selectedSlots.y_column;
      if (x && y) {
        constructedQuery = `Calculate correlation between ${x} and ${y}`;
      }
    } else if (cardType === 'dimension_selector') {
      const dim = selectedSlots.dimension_column;
      const met = selectedSlots.metric_column;
      if (dim && met) {
        constructedQuery = `Group data by ${dim} and aggregate metric ${met}`;
      } else if (dim) {
        constructedQuery = `Analyze distribution grouped by ${dim}`;
      }
    } else {
      const parts = Object.entries(selectedSlots).map(([k, v]) => `${k}: ${v}`);
      constructedQuery = `Execute analysis with parameters: ${parts.join(', ')}`;
    }

    if (constructedQuery) {
      logUserAction('CLARIFICATION_SLOTS_EXECUTED', { query: constructedQuery, slots: selectedSlots });
      onSubmitClarification(constructedQuery);
    }
  };

  const handleSelectFallbackCandidate = (column) => {
    logUserAction('CLARIFICATION_CANDIDATE_CHIP_CLICKED', { column });
    if (typeof onSubmitClarification === 'function') {
      onSubmitClarification(`Analyze column ${column}`);
    } else if (typeof onSelectCandidate === 'function') {
      onSelectCandidate(column);
    }
  };

  const handleSubmitCustom = (e) => {
    e?.preventDefault();
    if (customText.trim() && typeof onSubmitClarification === 'function') {
      logUserAction('CLARIFICATION_CUSTOM_SUBMITTED', { text: customText.trim() });
      onSubmitClarification(customText.trim());
      setCustomText('');
    }
  };

  const handleRunSuggestedQuery = (query) => {
    logUserAction('CLARIFICATION_SUGGESTED_QUERY_CLICKED', { query });
    if (typeof onSubmitClarification === 'function') {
      onSubmitClarification(query);
    }
  };

  return (
    <div
      style={{
        marginTop: 10,
        marginBottom: 10,
        background: '#131b14',
        border: '1px solid #eab308',
        borderRadius: 8,
        padding: 14,
        boxShadow: '0 4px 14px rgba(234, 179, 8, 0.08)',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          color: '#eab308',
          fontWeight: 700,
          fontSize: 13,
          marginBottom: 6,
        }}
      >
        <HelpCircle size={16} />
        <span>{cardTitle || 'Clarification Required'}</span>
      </div>

      {/* Main explanation / prompt */}
      <div style={{ color: '#f1f5f9', fontSize: 12, marginBottom: 12, lineHeight: 1.5 }}>
        {prompt || 'More information is required before this analysis can proceed.'}
      </div>

      {/* Suggested Quick Actions */}
      {suggestedQueries.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: '#fef08a', marginBottom: 6, fontWeight: 600 }}>
            <Sparkles size={12} />
            <span>Recommended One-Click Actions:</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {suggestedQueries.map((q) => (
              <button
                key={q}
                onClick={() => handleRunSuggestedQuery(q)}
                style={{
                  textAlign: 'left',
                  background: 'rgba(234, 179, 8, 0.12)',
                  border: '1px solid rgba(234, 179, 8, 0.35)',
                  borderRadius: 6,
                  padding: '6px 10px',
                  color: '#fef9c3',
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  transition: 'all 0.15s ease',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'rgba(234, 179, 8, 0.25)';
                  e.currentTarget.style.borderColor = '#eab308';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'rgba(234, 179, 8, 0.12)';
                  e.currentTarget.style.borderColor = 'rgba(234, 179, 8, 0.35)';
                }}
              >
                <span>{q}</span>
                <ArrowRight size={12} />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Slot Filling Interactive Selectors */}
      {slots.length > 0 && (
        <div style={{ marginBottom: 12, background: 'rgba(0,0,0,0.25)', padding: 10, borderRadius: 6, border: '1px solid rgba(234, 179, 8, 0.2)' }}>
          <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8, fontWeight: 600 }}>
            Configure Analytical Roles:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {slots.map((slot) => {
              const currentVal = selectedSlots[slot.slot_name];
              return (
                <div key={slot.slot_name}>
                  <div style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 4, fontWeight: 600 }}>
                    {slot.label}: {slot.required && <span style={{ color: '#f87171' }}>*</span>}
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {(slot.candidates || []).map((col) => {
                      const isSelected = currentVal === col;
                      return (
                        <button
                          key={col}
                          onClick={() => handleSelectSlotCandidate(slot.slot_name, col)}
                          style={{
                            background: isSelected ? 'rgba(52, 211, 153, 0.25)' : 'rgba(234, 179, 8, 0.12)',
                            border: isSelected ? '1px solid #10b981' : '1px solid rgba(234, 179, 8, 0.35)',
                            borderRadius: 6,
                            padding: '4px 9px',
                            color: isSelected ? '#6ee7b7' : '#fef08a',
                            fontSize: 11,
                            fontWeight: 600,
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 4,
                            transition: 'all 0.15s ease',
                          }}
                        >
                          {isSelected && <Check size={11} />}
                          <span>`{col}`</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Action button to proceed with selected slots */}
          {isFormComplete && (
            <div style={{ marginTop: 10 }}>
              <button
                onClick={handleExecuteSlots}
                style={{
                  width: '100%',
                  background: '#10b981',
                  color: '#064e3b',
                  border: 'none',
                  borderRadius: 6,
                  padding: '7px 12px',
                  fontSize: 11,
                  fontWeight: 700,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
                  transition: 'background 0.15s ease',
                }}
                onMouseOver={(e) => (e.currentTarget.style.background = '#34d399')}
                onMouseOut={(e) => (e.currentTarget.style.background = '#10b981')}
              >
                <span>Proceed with Selected Roles</span>
                <ArrowRight size={13} />
              </button>
            </div>
          )}
        </div>
      )}

      {/* Fallback Candidate Column Chips (if slots not populated) */}
      {!slots.length && candidateColumns.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6, fontWeight: 600 }}>
            Select one of the candidate columns:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {candidateColumns.map((col) => (
              <button
                key={col}
                onClick={() => handleSelectFallbackCandidate(col)}
                style={{
                  background: 'rgba(234, 179, 8, 0.15)',
                  border: '1px solid #eab308',
                  borderRadius: 6,
                  padding: '4px 10px',
                  color: '#fef08a',
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  transition: 'background 0.2s',
                }}
                onMouseOver={(e) => (e.currentTarget.style.background = 'rgba(234, 179, 8, 0.3)')}
                onMouseOut={(e) => (e.currentTarget.style.background = 'rgba(234, 179, 8, 0.15)')}
              >
                <span>`{col}`</span>
                <ArrowRight size={11} />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Natural language free-text input */}
      <form onSubmit={handleSubmitCustom} style={{ display: 'flex', gap: 6, marginTop: 8 }}>
        <input
          type="text"
          value={customText}
          onChange={(e) => setCustomText(e.target.value)}
          placeholder="Or write your clarification response..."
          style={{
            flex: 1,
            background: '#070c09',
            border: '1px solid #334155',
            borderRadius: 6,
            padding: '6px 10px',
            color: '#f8fafc',
            fontSize: 11,
            outline: 'none',
          }}
        />
        <button
          type="submit"
          disabled={!customText.trim()}
          style={{
            background: customText.trim() ? '#eab308' : '#334155',
            color: customText.trim() ? '#0f172a' : '#64748b',
            border: 'none',
            borderRadius: 6,
            padding: '6px 12px',
            fontSize: 11,
            fontWeight: 700,
            cursor: customText.trim() ? 'pointer' : 'default',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <span>Submit</span>
          <Send size={11} />
        </button>
      </form>
    </div>
  );
}
