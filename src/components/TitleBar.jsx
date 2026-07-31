import { useState, useEffect } from 'react';
import { Minus, Square, X, Database, Pin, PinOff, Sparkles, FolderOpen } from 'lucide-react';

const isElectron = typeof window !== 'undefined' && window.electron;

export default function TitleBar({ datasetName, onOpenFile, onSaveFile, isSaving }) {
  const [maximized, setMaximized] = useState(false);
  const [isPinned, setIsPinned]   = useState(true);

  useEffect(() => {
    if (!isElectron) return;
    window.electron.isMaximized().then(setMaximized);
    if (window.electron.isAlwaysOnTop) {
      window.electron.isAlwaysOnTop().then(setIsPinned);
    }
    const off = window.electron.onWindowStateChange((state) => {
      setMaximized(state.maximized);
    });
    return off;
  }, []);

  const handleMinimize = () => isElectron && window.electron.minimize();
  const handleMaximize = () => isElectron && window.electron.maximize();
  const handleClose = () => isElectron && window.electron.close();
  
  const handleTogglePin = async () => {
    if (isElectron && window.electron.toggleAlwaysOnTop) {
      const newState = await window.electron.toggleAlwaysOnTop();
      setIsPinned(newState);
    }
  };

  const title = datasetName ? `${datasetName}` : '';

  return (
    <div className="title-bar" style={{
      background: '#0f172a',
      color: '#f8fafc',
      borderBottom: '1px solid #1e293b',
      height: 48,
      padding: '0 14px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      userSelect: 'none',
      WebkitAppRegion: 'drag',
    }}>
      {/* Left side: Premium AI Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
        <div style={{
          width: 28,
          height: 28,
          background: 'linear-gradient(135deg, #0ea5e9, #6366f1)',
          borderRadius: 8,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#ffffff',
          boxShadow: '0 3px 10px rgba(14,165,233,0.4)',
          border: '1px solid rgba(255,255,255,0.2)'
        }}>
          <Sparkles size={16} color="#ffffff" />
        </div>
        <span style={{ fontSize: 14, fontWeight: 800, color: '#f8fafc', letterSpacing: '0.02em' }}>
          DataPilot <span style={{ color: '#38bdf8' }}>OS</span>
        </span>
      </div>


      {/* Right side: Fixed Action Buttons & Window Controls (no-drag) */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, WebkitAppRegion: 'no-drag', flexShrink: 0 }}>
        <button
          onClick={onOpenFile}
          title="Open CSV / Excel File"
          style={{
            background: '#1e293b',
            color: '#38bdf8',
            border: '1px solid #334155',
            borderRadius: 5,
            padding: '4px 10px',
            fontSize: 11,
            fontWeight: 700,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            WebkitAppRegion: 'no-drag'
          }}
        >
          <FolderOpen size={12} />
          Open File
        </button>


        {isElectron && (
          <div className="title-bar__controls" style={{ display: 'flex', alignItems: 'center', gap: 2, marginLeft: 4 }}>
            <button
              className="title-bar__btn"
              onClick={handleTogglePin}
              title={isPinned ? 'Unpin (Float off)' : 'Pin on top'}
              style={{ color: isPinned ? '#38bdf8' : '#64748b' }}
            >
              {isPinned ? <Pin size={13} style={{ transform: 'rotate(45deg)' }} /> : <PinOff size={13} />}
            </button>

            <button className="title-bar__btn" onClick={handleMinimize} title="Minimize" style={{ color: '#cbd5e1' }}>
              <Minus size={13} />
            </button>
            <button className="title-bar__btn" onClick={handleMaximize} title={maximized ? 'Restore' : 'Maximize'} style={{ color: '#cbd5e1' }}>
              {maximized ? (
                <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                  <rect x="2" y="0" width="8" height="8" rx="1" stroke="currentColor" strokeWidth="1.2" />
                  <rect x="0" y="2" width="8" height="8" rx="1" stroke="currentColor" strokeWidth="1.2" fill="#0f172a" />
                </svg>
              ) : (
                <Square size={11} />
              )}
            </button>
            <button className="title-bar__btn title-bar__btn--close" onClick={handleClose} title="Close" style={{ color: '#cbd5e1' }}>
              <X size={13} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}



