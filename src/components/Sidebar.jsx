import React from 'react';
import { Layers, GitBranch, History, Database, Sparkles } from 'lucide-react';
import DatasetManager from './DatasetManager';
import AnalysisGraph from './AnalysisGraph';
import WorkflowHistory from './WorkflowHistory';

export default function Sidebar({
  activeTab = 'datasets',
  onTabChange,
  datasets = [],
  activeDatasetId,
  onConnect,
  onDisconnect,
  onSetActive,
  onRefresh,
  engineStatus,
  workflowState,
  onRefreshGraph,
  workflows = [],
  onLoadWorkflow,
  onDeleteWorkflow
}) {
  const tabs = [
    {
      id: 'datasets',
      label: 'Datasets',
      icon: Layers,
      badge: datasets.length > 0 ? datasets.length : null
    },
    {
      id: 'graph',
      label: 'Graph',
      icon: GitBranch,
      badge: workflowState?.graphState?.nodes?.length ? `${workflowState.graphState.nodes.length}` : null
    },
    {
      id: 'history',
      label: 'History',
      icon: History,
      badge: workflows?.length ? `${workflows.length}` : null
    }
  ];

  return (
    <div
      style={{
        width: 280,
        borderRight: '1px solid #1e293b',
        background: '#0b0f19',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        height: '100%',
        overflow: 'hidden'
      }}
    >
      {/* Tab Navigation Header */}
      <div
        style={{
          display: 'flex',
          borderBottom: '1px solid #1e293b',
          background: '#0f172a',
          flexShrink: 0
        }}
      >
        {tabs.map((t) => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => onTabChange && onTabChange(t.id)}
              style={{
                flex: 1,
                padding: '9px 4px',
                background: isActive ? '#0b0f19' : 'transparent',
                border: 'none',
                borderBottom: isActive ? '2px solid #38bdf8' : '2px solid transparent',
                color: isActive ? '#38bdf8' : '#64748b',
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 5,
                transition: 'all 0.15s ease'
              }}
            >
              <Icon size={13} />
              <span>{t.label}</span>
              {t.badge && (
                <span
                  style={{
                    fontSize: 9,
                    background: isActive ? '#0284c7' : '#334155',
                    color: '#fff',
                    padding: '0 4px',
                    borderRadius: '8px',
                    lineHeight: '13px'
                  }}
                >
                  {t.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Tab Panels */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {activeTab === 'datasets' && (
          <DatasetManager
            datasets={datasets}
            activeDatasetId={activeDatasetId}
            onConnect={onConnect}
            onDisconnect={onDisconnect}
            onSetActive={onSetActive}
            onRefresh={onRefresh}
            engineStatus={engineStatus}
          />
        )}

        {activeTab === 'graph' && (
          <AnalysisGraph
            workflowId={workflowState?.graphState?.workflow_id}
            graphState={workflowState?.graphState}
            onRefreshState={onRefreshGraph}
          />
        )}

        {activeTab === 'history' && (
          <WorkflowHistory
            workflows={workflows}
            onLoadWorkflow={onLoadWorkflow}
            onDeleteWorkflow={onDeleteWorkflow}
          />
        )}
      </div>
    </div>
  );
}
