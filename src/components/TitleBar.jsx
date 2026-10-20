import { useState, useEffect } from 'react';
import { Minus, Square, X, Database, Pin, PinOff, Sparkles, FolderOpen, Sun, Moon } from 'lucide-react';

const isElectron = typeof window !== 'undefined' && window.electron;

export default function TitleBar({ datasetName, rowCount, colCount, onOpenFile, onSaveFile, isSaving, theme, onToggleTheme, onNavigateHome }) {
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

  return (
    <div className="title-bar" style={{
      background: '#0a0e0c',
      color: '#f8fafc',
      borderBottom: '1px solid #1a2921',
      height: 48,
      padding: '0 14px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      userSelect: 'none',
      WebkitAppRegion: 'drag',
    }}>
      {/* Left side: Premium AI Logo */}
      <div
        onClick={onNavigateHome}
        title="Go to Home Workspace"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          flexShrink: 0,
          cursor: onNavigateHome ? 'pointer' : 'default',
          WebkitAppRegion: 'no-drag'
        }}
      >
        <div style={{
          width: 28,
          height: 28,
          background: 'linear-gradient(135deg, #10b981, #059669)',
          borderRadius: 8,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#ffffff',
          boxShadow: '0 3px 10px rgba(16,185,129,0.4)',
          border: '1px solid rgba(255,255,255,0.2)'
        }}>
          <Sparkles size={16} color="#ffffff" />
        </div>
        <span style={{ fontSize: 14, fontWeight: 800, color: '#f8fafc', letterSpacing: '0.02em' }}>
          DataPilot <span style={{ color: '#34d399' }}>OS</span>
        </span>
      </div>

      {/* Feature 3: Header Pill Badge (Live Dataset Context) */}
      <div style={{ display: 'flex', alignItems: 'center', WebkitAppRegion: 'no-drag', flexShrink: 1, minWidth: 0, overflow: 'hidden' }}>
        {datasetName ? (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            background: 'rgba(16, 185, 129, 0.14)',
            border: '1px solid rgba(52, 211, 153, 0.35)',
            borderRadius: 14,
            padding: '3px 10px',
            fontSize: 11,
            fontWeight: 700,
            color: '#34d399',
            boxShadow: '0 2px 8px rgba(16, 185, 129, 0.18)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            maxWidth: 240
          }}>
            <Database size={12} color="#34d399" style={{ flexShrink: 0 }} />
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 150, flexShrink: 1 }}>{datasetName}</span>
            <span style={{ color: '#475569', fontWeight: 400, flexShrink: 0 }}>•</span>
            <span style={{ color: '#cbd5e1', fontWeight: 600, flexShrink: 0 }}>
              {rowCount ? rowCount.toLocaleString() : 0} rows {colCount ? `× ${colCount} cols` : ''}
            </span>
          </div>
        ) : (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: 14,
            padding: '3px 10px',
            fontSize: 11,
            color: '#94a3b8',
            fontWeight: 600,
            whiteSpace: 'nowrap'
          }}>
            <Database size={11} color="#34d399" />
            <span>No Dataset Loaded</span>
          </div>
        )}
      </div>

      {/* Right side: Fixed Action Buttons, Theme Switcher & Window Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, WebkitAppRegion: 'no-drag', flexShrink: 0 }}>
        {/* Feature 1: Theme Switcher Button */}
        <button
          onClick={onToggleTheme}
          title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          style={{
            background: '#1e293b',
            color: theme === 'dark' ? '#fbbf24' : '#34d399',
            border: '1px solid #334155',
            borderRadius: 5,
            padding: '5px 9px',
            fontSize: 11,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4,
            WebkitAppRegion: 'no-drag'
          }}
        >
          {theme === 'dark' ? <Sun size={13} /> : <Moon size={13} />}
        </button>

        {isElectron && (
          <div className="title-bar__controls" style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 2 }}>
            <button
              className="title-bar__btn"
              onClick={handleTogglePin}
              title={isPinned ? 'Unpin (Float off)' : 'Pin on top'}
              style={{ color: isPinned ? '#34d399' : '#64748b' }}
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



