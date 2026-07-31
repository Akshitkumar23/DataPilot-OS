import {
  Home, PlugZap, Eraser, Shuffle, BrainCircuit, BarChart3, LineChart,
  FolderOpen, Save, Undo2, Redo2, Code2, Database, Table
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
}) {
  return (
    <div className="ribbon">
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
              style={{ opacity: disabled ? 0.4 : 1, cursor: disabled ? 'not-allowed' : 'pointer' }}
              title={disabled ? 'Load a dataset first' : tab.label}
            >
              <Icon size={14} />
              {tab.label}
            </button>
          );
        })}
      </div>

      <div className="ribbon__divider" />

      {/* Quick Actions */}
      <div className="ribbon__actions">
        <button className="btn btn--ghost btn--sm" onClick={onOpenFile} title="Open file (Ctrl+O)">
          <FolderOpen size={13} />
          Open Local File
        </button>
        <button className="btn btn--ghost btn--sm" onClick={onOpenDbModal} title="Connect to Database">
          <Database size={13} />
          Database Connection
        </button>

        {datasetLoaded && (
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
      </div>
    </div>
  );
}
