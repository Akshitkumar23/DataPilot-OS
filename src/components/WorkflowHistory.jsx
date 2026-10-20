import React, { useState } from 'react';
import { History, Search, Play, Trash2, Clock, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';

export default function WorkflowHistory({ workflows = [], onLoadWorkflow, onDeleteWorkflow }) {
  const [search, setSearch] = useState('');

  const filtered = (workflows || []).filter(w =>
    (w.user_query || '').toLowerCase().includes(search.toLowerCase()) ||
    (w.dataset_name || '').toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', backgroundColor: '#0b0f19', color: '#f8fafc' }}>
      <div style={{ padding: '12px 14px', borderBottom: '1px solid #1e293b', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#0f172a' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <History size={15} color="#38bdf8" />
          <h2 style={{ fontSize: '12px', fontWeight: 800, margin: 0, letterSpacing: '0.8px', color: '#f1f5f9' }}>
            WORKFLOW HISTORY
          </h2>
        </div>
        <span style={{ fontSize: '10px', background: '#1e293b', color: '#94a3b8', padding: '1px 6px', borderRadius: '10px', fontWeight: 700 }}>
          {workflows.length}
        </span>
      </div>

      <div style={{ padding: '10px 12px', borderBottom: '1px solid #1e293b', background: '#0b0f19' }}>
        <div style={{ position: 'relative' }}>
          <Search size={13} style={{ position: 'absolute', left: '10px', top: '9px', color: '#64748b' }} />
          <input
            type="text"
            placeholder="Search workflows..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              width: '100%',
              paddingLeft: '28px',
              paddingRight: '10px',
              paddingTop: '6px',
              paddingBottom: '6px',
              background: '#141d2f',
              border: '1px solid #1e293b',
              borderRadius: '4px',
              color: '#f8fafc',
              fontSize: '11px',
              outline: 'none',
              boxSizing: 'border-box'
            }}
          />
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {filtered.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#64748b', marginTop: '36px', fontSize: '12px', padding: '0 12px' }}>
            <Clock size={28} style={{ color: '#334155', margin: '0 auto 8px' }} />
            <p style={{ margin: 0, color: '#94a3b8', fontWeight: 600 }}>No previous workflows</p>
            <p style={{ margin: '4px 0 0', fontSize: '11px' }}>Queries and analysis runs will be logged here.</p>
          </div>
        ) : (
          filtered.map(wf => {
            const isCompleted = wf.status === 'completed' || wf.status === 'SUCCEEDED';
            const isFailed = wf.status === 'failed' || wf.status === 'FAILED';

            return (
              <div
                key={wf.workflow_id}
                style={{
                  border: '1px solid #1e293b',
                  borderRadius: '6px',
                  padding: '10px 12px',
                  background: '#0f172a',
                  transition: 'border-color 0.15s ease',
                  position: 'relative'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px', gap: '8px' }}>
                  <div style={{
                    fontWeight: 600,
                    fontSize: '12px',
                    color: '#f1f5f9',
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    lineHeight: '1.3'
                  }}>
                    "{wf.user_query}"
                  </div>
                  <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }}>
                    {onLoadWorkflow && (
                      <button
                        onClick={() => onLoadWorkflow(wf.workflow_id)}
                        style={{ background: 'transparent', border: 'none', color: '#38bdf8', cursor: 'pointer', padding: '2px' }}
                        title="Load Workflow"
                      >
                        <Play size={13} />
                      </button>
                    )}
                    {onDeleteWorkflow && (
                      <button
                        onClick={() => onDeleteWorkflow(wf.workflow_id)}
                        style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '2px' }}
                        title="Delete"
                      >
                        <Trash2 size={13} />
                      </button>
                    )}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '8px', fontSize: '10px', color: '#94a3b8', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <span style={{
                      width: '6px', height: '6px', borderRadius: '50%',
                      background: isCompleted ? '#10b981' : isFailed ? '#ef4444' : '#f59e0b'
                    }}></span>
                    {wf.status || 'completed'}
                  </span>
                  {wf.dataset_name && (
                    <>
                      <span>•</span>
                      <span style={{ color: '#cbd5e1' }}>{wf.dataset_name}</span>
                    </>
                  )}
                  {wf.graph_versions && (
                    <>
                      <span>•</span>
                      <span>{wf.graph_versions} {wf.graph_versions === 1 ? 'ver' : 'vers'}</span>
                    </>
                  )}
                </div>

                {wf.insight_summary && (
                  <div style={{ fontSize: '11px', color: '#cbd5e1', background: '#141d2f', padding: '6px 8px', borderRadius: '4px', border: '1px solid #1e293b' }}>
                    <span style={{ fontWeight: 700, color: '#38bdf8' }}>Insight: </span>
                    {wf.insight_summary.substring(0, 80)}{wf.insight_summary.length > 80 ? '...' : ''}
                  </div>
                )}

                {wf.created_at && (
                  <div style={{ fontSize: '9px', color: '#64748b', marginTop: '6px', textAlign: 'right' }}>
                    {new Date(wf.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
