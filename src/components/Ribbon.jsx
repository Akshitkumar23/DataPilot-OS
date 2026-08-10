import {
  Home, PlugZap, Eraser, Shuffle, BrainCircuit, BarChart3, LineChart,
  FolderOpen, Save, Undo2, Redo2, Code2, Database, Table, ChevronUp, ChevronDown
} from 'lucide-react';

const TABS = [
  { id: 'home',      label: 'Home',        icon: Home         },
  { id: 'preview',   label: 'Data Grid',   icon: Table        },
  { id: 'connect',   label: 'Connect',     icon: PlugZap      },
  { id: 'clean',     label: 'Clean',       icon: Eraser       },
  { id: 'transform', label: 'Transform',   icon: Shuffle      },
  { id: 'stats',     label: 'EDA & Stats', icon: LineChart    },
  { id: 'ml',        label: 'ML Studio',   icon: BrainCircuit },
  { id: 'report',    label: 'Report',      icon: BarChart3    },
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
      transition: 'height 0.2s ease'
    }}>
      {/* Module tabs */}
      <div className="ribbon__tabs">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const disabled = tab.id !== 'home' && tab.id !== 'connect' && !datasetLoaded;
          return (
            <button
              key={tab.id}
              className={`ribbon__tab ${activeModule === tab.id ? 'active' : ''}`}
              onClick={() => !disabled && onModuleChange(tab.id)}
              disabled={disabled}
              style={{
                opacity: disabled ? 0.4 : 1,
                cursor: disabled ? 'not-allowed' : 'pointer',
                padding: isCollapsed ? '0 10px' : undefined
              }}
              title={disabled ? 'Load a dataset first' : tab.label}
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
              onClick={onUndo}
              disabled={!canUndo}
              title="Undo last operation"
              style={{ opacity: canUndo ? 1 : 0.35 }}
            >
              <Undo2 size={13} />
            </button>
            <button
              className="btn btn--ghost btn--sm"
              onClick={onRedo}
              disabled={!canRedo}
              title="Redo operation"
              style={{ opacity: canRedo ? 1 : 0.35 }}
            >
              <Redo2 size={13} />
            </button>
            <div className="ribbon__divider" style={{ height: 20 }} />
            <button
              className="btn btn--sm btn--ghost"
              onClick={onOpenCodeInspector}
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
