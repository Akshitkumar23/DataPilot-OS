import { Activity, Database, Rows3, AlertTriangle } from 'lucide-react';

export default function StatusBar({
  connectionStatus,
  datasetName,
  rowCount,
  colCount,
  dirtyCount,
  activeModule,
}) {
  const connected = connectionStatus === 'connected';

  return (
    <div className="status-bar">
      {/* Backend status */}
      <div className={`status-item ${connected ? 'status-item--success' : ''}`}>
        <div className={`status-dot ${connected ? 'status-dot--green' : 'status-dot--red'}`} />
        {connected ? 'DataPilot OS Ready 🟢' : 'Engine Offline 🔴'}
      </div>

      {datasetName && (
        <>
          <div className="status-divider" />
          <div className="status-item status-item--active">
            <Database size={11} />
            {datasetName}
          </div>

          <div className="status-divider" />
          <div className="status-item">
            <Rows3 size={11} />
            {rowCount?.toLocaleString() ?? 0} rows × {colCount ?? 0} cols
          </div>

          {dirtyCount > 0 && (
            <>
              <div className="status-divider" />
              <div className="status-item" style={{ color: 'var(--warning)' }}>
                <AlertTriangle size={11} />
                {dirtyCount} column{dirtyCount !== 1 ? 's' : ''} with null values
              </div>
            </>
          )}
        </>
      )}

      <div className="status-bar__spacer" />

      <div className="status-item">
        <Activity size={11} />
        {activeModule ? activeModule.toUpperCase() : 'GRID VIEW'}
      </div>
    </div>

  );
}
