import React, { useEffect } from 'react';
import { API, getAuthHeaders } from '../apiConfig';
import { CheckCircle, Circle, AlertCircle, XCircle, RefreshCw, Loader } from 'lucide-react';

export default function AnalysisGraph({ workflowId, graphState, onRefreshState }) {
  
  useEffect(() => {
    let interval;
    if (workflowId && graphState?.phase === 'EXECUTING') {
      interval = setInterval(() => {
        if (onRefreshState) onRefreshState(workflowId);
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [workflowId, graphState, onRefreshState]);

  if (!graphState || !graphState.nodes) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8', backgroundColor: '#0f172a', height: '100%' }}>
        <p>Analysis graph will appear here once you submit a query</p>
      </div>
    );
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'PENDING': return <Circle size={16} color="#64748b" />;
      case 'RUNNING': return <Loader size={16} color="#3b82f6" className="animate-spin" />;
      case 'SUCCEEDED': return <CheckCircle size={16} color="#10b981" />;
      case 'NEEDS_REPAIR': return <AlertCircle size={16} color="#f59e0b" />;
      case 'FAILED': return <XCircle size={16} color="#ef4444" />;
      case 'REPLANNED': return <RefreshCw size={16} color="#8b5cf6" />;
      default: return <Circle size={16} color="#64748b" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'SUCCEEDED': return '#064e3b';
      case 'RUNNING': return '#1e3a8a';
      case 'FAILED': return '#7f1d1d';
      case 'NEEDS_REPAIR': return '#78350f';
      case 'REPLANNED': return '#4c1d95';
      default: return '#1e293b';
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', backgroundColor: '#0f172a', color: '#f8fafc' }}>
      <div style={{ padding: '16px', borderBottom: '1px solid #334155', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h2 style={{ margin: 0, fontSize: '14px', fontWeight: 'bold', letterSpacing: '1px' }}>ANALYSIS GRAPH</h2>
          <span style={{ background: '#334155', padding: '2px 8px', borderRadius: '12px', fontSize: '11px', fontWeight: 'bold' }}>
            v{graphState.version || 1}
          </span>
        </div>
        {graphState.replan_reason && (
          <div style={{ fontSize: '12px', color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <AlertCircle size={14} /> Replan: {graphState.replan_reason}
          </div>
        )}
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: '40px', position: 'relative' }}>
        <style>{`
          .animate-spin { animation: spin 2s linear infinite; }
          @keyframes spin { 100% { transform: rotate(360deg); } }
          .node-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
          @keyframes pulse { 0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); } 50% { opacity: .8; box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); } }
        `}</style>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '40px', alignItems: 'center' }}>
          {graphState.nodes.map((node, i) => (
            <div key={node.id} style={{ position: 'relative' }}>
              {/* Edge line to next node if exists */}
              {i < graphState.nodes.length - 1 && (
                <div style={{ 
                  position: 'absolute', 
                  width: '2px', 
                  height: '40px', 
                  backgroundColor: '#334155', 
                  left: '50%', 
                  bottom: '-40px',
                  transform: 'translateX(-50%)',
                  zIndex: 0
                }} />
              )}
              
              <div 
                className={node.status === 'RUNNING' ? 'node-pulse' : ''}
                style={{
                  background: getStatusColor(node.status),
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  padding: '16px 24px',
                  minWidth: '250px',
                  position: 'relative',
                  zIndex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {getStatusIcon(node.status)}
                    <span style={{ fontWeight: 'bold', fontSize: '14px' }}>{node.name}</span>
                  </div>
                </div>
                
                {(node.started_at || node.completed_at) && (
                  <div style={{ fontSize: '11px', color: '#94a3b8', display: 'flex', justifyContent: 'space-between' }}>
                    <span>{node.status}</span>
                    <span>
                      {node.started_at && !node.completed_at && 'Running...'}
                      {node.completed_at && 'Completed'}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ padding: '12px 16px', borderTop: '1px solid #334155', display: 'flex', gap: '8px', overflowX: 'auto' }}>
        <span style={{ fontSize: '12px', color: '#94a3b8', marginRight: '8px', alignSelf: 'center' }}>History:</span>
        {Array.from({ length: graphState.graph_version || graphState.version || 1 }).map((_, i) => (
          <div key={i} style={{ 
            padding: '4px 12px', 
            borderRadius: '12px', 
            background: i + 1 === (graphState.graph_version || graphState.version || 1) ? '#3b82f6' : '#334155',
            fontSize: '11px',
            fontWeight: 'bold'
          }}>
            v{i + 1}
          </div>
        ))}
      </div>
    </div>
  );
}
