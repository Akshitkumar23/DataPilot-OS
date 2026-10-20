import {
  Home, PlugZap, Eraser, Shuffle, BrainCircuit, BarChart3, LineChart,
  FolderOpen, Save, Undo2, Redo2, Code2, Database, Table, ChevronUp, ChevronDown,
  Bot
} from 'lucide-react';

const TABS = [
  { id: 'home',      label: 'Home',            icon: Home         },
  { id: 'agent',     label: 'Workspace Agent', icon: Bot          },
  { id: 'preview',   label: 'Data Grid',       icon: Table        },
  { id: 'connect',   label: 'Connect',         icon: PlugZap      },
  { id: 'clean',     label: 'Clean',           icon: Eraser       },
  { id: 'transform', label: 'Transform',       icon: Shuffle      },
  { id: 'stats',     label: 'EDA & Stats',     icon: LineChart    },
  { id: 'ml',        label: 'ML Studio',       icon: BrainCircuit },
  { id: 'reports',   label: 'Reports',         icon: BarChart3    },
  { id: 'code',      label: 'Code & SQL',      icon: Code2        },
];

export default function Ribbon({
  activeModule,
  onModuleChange,
  datasetLoaded,
  onOpenFile,
  onOpenDbModal,
  onSaveFile,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  isSaving,
  onOpenCodeInspector,
  isCollapsed,
  onToggleCollapse
}) {
  return (
    <div className={`ribbon ${isCollapsed ? 'ribbon--collapsed' : ''}`} style={{
      height: isCollapsed ? 32 : 'var(--ribbon-h)',
      transition: 'height 0.2s ease',
      WebkitAppRegion: 'no-drag'
    }}>
      {/* Module tabs */}
      <div className="ribbon__tabs">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const disabled = tab.id !== 'home' && tab.id !== 'connect' && tab.id !== 'agent' && !datasetLoaded;
          const isActive = activeModule === tab.id || (tab.id === 'reports' && activeModule === 'report') || (tab.id === 'report' && activeModule === 'reports');
          return (
            <button
              key={tab.id}
              className={`ribbon__tab ${isActive ? 'active' : ''}`}
              onClick={() => !disabled && onModuleChange?.(tab.id)}
              disabled={disabled}
              style={{
                opacity: disabled ? 0.4 : 1,
                cursor: disabled ? 'not-allowed' : 'pointer',
                padding: isCollapsed ? '0 10px' : undefined
              }}
              title={disabled ? 'Load a dataset first to access this module' : tab.label}
            >
              <Icon size={isCollapsed ? 15 : 14} />
              {!isCollapsed && <span>{tab.label}</span>}
            </button>
          );
        })}
      </div>

      <div className="ribbon__divider" />

      {/* Quick Actions */}
      <div className="ribbon__actions">
        {!isCollapsed && (
          <>
            <button className="btn btn--ghost btn--sm" onClick={onOpenFile} title="Open CSV / Excel / Parquet File (Ctrl+O)" style={{ color: 'var(--accent)', fontWeight: 700 }}>
              <FolderOpen size={13} color="var(--accent)" />
              <span className="ribbon-action-label">Open File</span>
            </button>
            <button className="btn btn--ghost btn--sm" onClick={onOpenDbModal} title="Connect to Database (SQL/Cloud)">
              <Database size={13} />
              <span className="ribbon-action-label">Database</span>
            </button>
          </>
        )}

        {datasetLoaded && !isCollapsed && (
          <>
            <div className="ribbon__divider" style={{ height: 20 }} />
            <button
              className="btn btn--ghost btn--sm"
              onClick={onSaveFile}
              disabled={isSaving}
              title="Save / Export Dataset"
              style={{ color: '#10b981', fontWeight: 600 }}
            >
              <Save size={13} />
              <span className="ribbon-action-label">{isSaving ? 'Saving...' : 'Save'}</span>
            </button>
            {/* Undo/Redo buttons are hidden until Phase 8 historical state rollback is connected */}
            <div className="ribbon__divider" style={{ height: 20 }} />
            <button
              className="btn btn--sm btn--ghost"
              onClick={onOpenCodeInspector ? onOpenCodeInspector : () => onModuleChange?.('code')}
              title="Inspect generated Python & DuckDB SQL code"
              style={{ color: 'var(--accent)', borderColor: 'var(--border-accent)' }}
            >
              <Code2 size={13} />
              Code & SQL
            </button>
          </>
        )}

        {/* Feature 2: Collapse / Expand Ribbon Toggle Button */}
        <button
          className="btn btn--ghost btn--sm"
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Expand Ribbon' : 'Collapse Ribbon (Maximize Workspace)'}
          style={{ padding: '3px 8px', marginLeft: 4 }}
        >
          {isCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </button>
      </div>
    </div>
  );
}
