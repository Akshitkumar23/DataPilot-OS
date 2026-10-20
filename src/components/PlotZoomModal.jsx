import { useState, useEffect, useCallback } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Download, X, Maximize2, Move, Copy, Check, Sun, Moon } from 'lucide-react';
import { copyImageToClipboard } from '../utils/imageClipboard';

const normalizeImageSrc = (src) => {
  if (!src || typeof src !== 'string') return null;
  if (src.startsWith('data:image') || src.startsWith('http://') || src.startsWith('https://') || src.startsWith('blob:')) {
    return src;
  }
  return `data:image/png;base64,${src}`;
};

export default function PlotZoomModal({ imageSrc, onClose, title = "Data Plot Visualization" }) {
  const normalizedSrc = normalizeImageSrc(imageSrc);
  const [zoom, setZoom] = useState(1);
  const [copied, setCopied] = useState(false);
  const [lightBg, setLightBg] = useState(false);
  const [rotation, setRotation] = useState(0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Reset zoom & pan when image changes
  useEffect(() => {
    setZoom(1);
    setRotation(0);
    setPan({ x: 0, y: 0 });
  }, [normalizedSrc]);

  // Close on Escape key press
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && onClose) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!normalizedSrc) return null;

  const handleZoomIn = (e) => {
    if (e) e.stopPropagation();
    setZoom(prev => Math.min(prev + 0.25, 4));
  };

  const handleZoomOut = (e) => {
    if (e) e.stopPropagation();
    setZoom(prev => Math.max(prev - 0.25, 0.5));
  };

  const handleReset = (e) => {
    if (e) e.stopPropagation();
    setZoom(1);
    setRotation(0);
    setPan({ x: 0, y: 0 });
  };

  const handleRotate = (e) => {
    if (e) e.stopPropagation();
    setRotation(prev => (prev + 90) % 360);
  };

  const handleWheel = (e) => {
    e.stopPropagation();
    if (e.deltaY < 0) {
      setZoom(prev => Math.min(prev + 0.25, 4));
    } else {
      setZoom(prev => Math.max(prev - 0.25, 0.5));
    }
  };

  const handleMouseDown = (e) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging && zoom > 1) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleCopy = async (e) => {
    if (e) e.stopPropagation();
    try {
      await copyImageToClipboard(normalizedSrc);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy image', err);
    }
  };

  const handleDownload = (e) => {
    if (e) e.stopPropagation();
    const link = document.createElement('a');
    link.href = normalizedSrc;
    link.download = `datapilot_plot_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCloseClick = (e) => {
    if (e) e.stopPropagation();
    if (onClose) onClose();
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 99999,
        background: lightBg ? '#ffffff' : '#090d16',
        backdropFilter: lightBg ? 'none' : 'blur(16px)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        userSelect: 'none',
        overflow: 'hidden'
      }}
      onWheel={handleWheel}
      onMouseUp={handleMouseUp}
    >
      {/* Top Floating Control Bar */}
      <div
        onMouseDown={(e) => e.stopPropagation()}
        onMouseUp={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
        style={{
          position: 'fixed',
          top: 20,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 999999,
          background: '#0d1310',
          border: '1px solid #1a2921',
          borderRadius: 24,
          padding: '6px 16px',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.7)',
          WebkitAppRegion: 'no-drag',
          pointerEvents: 'auto'
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 700, color: '#34d399', display: 'flex', alignItems: 'center', gap: 6, marginRight: 6 }}>
          <Maximize2 size={14} />
          <span>{title}</span>
        </div>

        {/* Zoom Percentage Chip */}
        <span style={{ fontSize: 11, fontWeight: 700, color: '#a7f3d0', background: '#121c17', padding: '3px 8px', borderRadius: 10 }}>
          {Math.round(zoom * 100)}%
        </span>

        {/* Action Buttons */}
        <button
          onClick={handleZoomIn}
          onMouseDown={(e) => e.stopPropagation()}
          title="Zoom In (+)"
          style={btnStyle}
        >
          <ZoomIn size={15} />
        </button>

        <button
          onClick={handleZoomOut}
          onMouseDown={(e) => e.stopPropagation()}
          title="Zoom Out (-)"
          style={btnStyle}
        >
          <ZoomOut size={15} />
        </button>

        <button
          onClick={handleRotate}
          onMouseDown={(e) => e.stopPropagation()}
          title="Rotate 90°"
          style={btnStyle}
        >
          <RotateCcw size={14} />
        </button>

        <button
          onClick={(e) => { if (e) e.stopPropagation(); setLightBg(!lightBg); }}
          onMouseDown={(e) => e.stopPropagation()}
          title="Toggle Canvas Background"
          style={btnStyle}
        >
          {lightBg ? <Moon size={14} /> : <Sun size={14} />}
        </button>

        <button
          onClick={handleReset}
          onMouseDown={(e) => e.stopPropagation()}
          title="Reset Zoom (100%)"
          style={{ ...btnStyle, fontSize: 11, fontWeight: 700, padding: '4px 10px' }}
        >
          Reset
        </button>

        <div style={{ width: 1, height: 16, background: '#1a2921' }} />

        <button
          onClick={handleCopy}
          onMouseDown={(e) => e.stopPropagation()}
          title="Copy Image to Clipboard"
          style={btnStyle}
        >
          {copied ? <Check size={14} color="#4ade80" /> : <Copy size={14} />}
        </button>

        <button
          onClick={handleDownload}
          onMouseDown={(e) => e.stopPropagation()}
          title="Download Plot PNG"
          style={{
            ...btnStyle,
            background: 'linear-gradient(135deg, #10b981, #059669)',
            color: '#ffffff',
            border: 'none',
            fontWeight: 700,
            padding: '4px 12px'
          }}
        >
          <Download size={14} />
          <span>Download</span>
        </button>

        <button
          onClick={handleCloseClick}
          onMouseDown={(e) => e.stopPropagation()}
          title="Close (Esc)"
          style={{
            ...btnStyle,
            background: '#ef4444',
            color: '#ffffff',
            border: 'none',
            padding: '4px 10px'
          }}
        >
          <X size={15} />
        </button>
      </div>

      {/* Main Zoomable Image Canvas Container */}
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: zoom > 1 ? (isDragging ? 'grabbing' : 'grab') : 'zoom-in'
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onClick={(e) => {
          if (e.target === e.currentTarget && zoom === 1) onClose();
        }}
      >
        <img
          src={normalizedSrc}
          alt="Zoomed Plot Visualization"
          style={{
            maxWidth: '85vw',
            maxHeight: '82vh',
            objectFit: 'contain',
            borderRadius: 12,
            boxShadow: '0 12px 48px rgba(0, 0, 0, 0.7)',
            border: '1px solid #1a2921',
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom}) rotate(${rotation}deg)`,
            transition: isDragging ? 'none' : 'transform 0.15s cubic-bezier(0.16, 1, 0.3, 1)',
            pointerEvents: 'auto'
          }}
          draggable={false}
        />
      </div>

      {/* Footer Helper Note */}
      <div style={{
        position: 'fixed',
        bottom: 16,
        left: '50%',
        transform: 'translateX(-50%)',
        fontSize: 11,
        color: '#64748b',
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        zIndex: 999999,
        pointerEvents: 'none'
      }}>
        <Move size={12} color="#34d399" />
        <span>Use mouse wheel to zoom • Drag to pan image • Press Esc or click ❌ to close</span>
      </div>
    </div>
  );
}

const btnStyle = {
  background: '#121c17',
  color: '#34d399',
  border: '1px solid #1a2921',
  borderRadius: 16,
  padding: '5px 9px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 4,
  transition: 'all 0.15s ease',
  userSelect: 'none',
  WebkitAppRegion: 'no-drag'
};
