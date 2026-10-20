import React, { useState, useMemo, useRef } from 'react';
import {
  ZoomIn, ZoomOut, RotateCcw, Eye, EyeOff, Image, BarChart2,
  Maximize2, Download, Copy, Check
} from 'lucide-react';

/**
 * DataPilot OS — Professional Interactive Vector Chart Component.
 * Renders SVG charts from `plotly_spec` specifications with zero external dependencies.
 * Features:
 * - Hover tooltip with exact coordinates/values
 * - Legend series toggling (show/hide series)
 * - Zoom & Pan / Reset view controls
 * - Confidence interval shaded envelopes (e.g. SARIMAX forecast 95% bands)
 * - Dual-mode switch: Interactive SVG <-> High-Res Matplotlib PNG
 */
export default function InteractiveChart({
  spec,
  fallbackImage,
  title,
  height = 300,
  onZoomImage
}) {
  const [viewMode, setViewMode] = useState(spec ? 'interactive' : 'image');
  const [hiddenSeries, setHiddenSeries] = useState(new Set());
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [copied, setCopied] = useState(false);
  const containerRef = useRef(null);

  // Default color palette (professional scientific dark theme)
  const PALETTE = [
    '#38bdf8', // sky blue
    '#10b981', // emerald
    '#f59e0b', // amber
    '#a855f7', // purple
    '#ec4899', // pink
    '#6366f1', // indigo
    '#06b6d4', // cyan
    '#f97316'  // orange
  ];

  const traces = useMemo(() => {
    if (!spec || !Array.isArray(spec.data)) return [];
    return spec.data.map((trace, idx) => {
      const color = trace.marker?.color || trace.line?.color || PALETTE[idx % PALETTE.length];
      return {
        ...trace,
        id: trace.name || `Series ${idx + 1}`,
        color: typeof color === 'string' ? color : PALETTE[idx % PALETTE.length]
      };
    });
  }, [spec]);

  const visibleTraces = useMemo(() => {
    return traces.filter(t => !hiddenSeries.has(t.id));
  }, [traces, hiddenSeries]);

  const maxSeriesLength = useMemo(() => {
    return Math.max(0, ...visibleTraces.map(t => (Array.isArray(t.x) ? t.x.length : 0)));
  }, [visibleTraces]);

  const isDownsampled = maxSeriesLength > 1000;

  // Compute bounding boxes for numeric coordinates
  const bounds = useMemo(() => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    let isCategoricalX = false;
    const catLabels = [];

    visibleTraces.forEach(t => {
      const xVals = Array.isArray(t.x) ? t.x : [];
      const yVals = Array.isArray(t.y) ? t.y : [];

      if (xVals.length > 0 && typeof xVals[0] === 'string' && isNaN(Number(xVals[0]))) {
        isCategoricalX = true;
        xVals.forEach(c => {
          if (!catLabels.includes(c)) catLabels.push(c);
        });
      } else {
        xVals.forEach(v => {
          const num = Number(v);
          if (Number.isFinite(num)) {
            if (num < minX) minX = num;
            if (num > maxX) maxX = num;
          }
        });
      }

      yVals.forEach(v => {
        const num = Number(v);
        if (Number.isFinite(num)) {
          if (num < minY) minY = num;
          if (num > maxY) maxY = num;
        }
      });
    });

    if (minX === Infinity) { minX = 0; maxX = 1; }
    if (minY === Infinity) { minY = 0; maxY = 1; }
    if (minX === maxX) { minX -= 1; maxX += 1; }
    if (minY === maxY) { minY -= 1; maxY += 1; }

    // Padding
    const padY = (maxY - minY) * 0.08 || 1;
    const padX = (maxX - minX) * 0.05 || 1;

    return {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: Math.min(0, minY - padY),
      maxY: maxY + padY,
      isCategoricalX,
      catLabels
    };
  }, [visibleTraces]);

  const toggleSeries = (id) => {
    const next = new Set(hiddenSeries);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setHiddenSeries(next);
  };

  const handleCopy = async () => {
    if (!fallbackImage) return;
    try {
      const res = await fetch(fallbackImage);
      const blob = await res.blob();
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      setCopied(false);
    }
  };

  // Dimensions
  const svgWidth = 600;
  const svgHeight = height;
  const margin = { top: 25, right: 30, bottom: 40, left: 55 };
  const plotWidth = svgWidth - margin.left - margin.right;
  const plotHeight = svgHeight - margin.top - margin.bottom;

  const scaleX = (x) => {
    if (bounds.isCategoricalX) {
      const idx = bounds.catLabels.indexOf(x);
      const step = plotWidth / Math.max(1, bounds.catLabels.length);
      return margin.left + idx * step + step / 2;
    }
    const ratio = (Number(x) - bounds.minX) / (bounds.maxX - bounds.minX);
    return margin.left + ratio * plotWidth;
  };

  const scaleY = (y) => {
    const ratio = (Number(y) - bounds.minY) / (bounds.maxY - bounds.minY);
    return margin.top + plotHeight - ratio * plotHeight;
  };

  // Ticks
  const xTicks = useMemo(() => {
    if (bounds.isCategoricalX) {
      return bounds.catLabels.slice(0, 10).map((label, i) => ({
        val: label,
        x: scaleX(label)
      }));
    }
    const count = 5;
    const step = (bounds.maxX - bounds.minX) / count;
    return Array.from({ length: count + 1 }, (_, i) => {
      const val = bounds.minX + i * step;
      return { val: val.toFixed(1), x: scaleX(val) };
    });
  }, [bounds, plotWidth]);

  const yTicks = useMemo(() => {
    const count = 5;
    const step = (bounds.maxY - bounds.minY) / count;
    return Array.from({ length: count + 1 }, (_, i) => {
      const val = bounds.minY + i * step;
      return { val: val >= 1000 ? val.toFixed(0) : val.toFixed(1), y: scaleY(val) };
    });
  }, [bounds, plotHeight]);

  // If no interactive spec, render image fallback directly
  if (!spec || !Array.isArray(spec.data) || spec.data.length === 0 || viewMode === 'image') {
    return (
      <div style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 6 }}>
            <BarChart2 size={14} color="#38bdf8" />
            {title || spec?.layout?.title?.text || spec?.layout?.title || 'Visual Analysis Plot'}
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {spec && (
              <button
                onClick={() => setViewMode('interactive')}
                style={{ background: '#1e293b', color: '#38bdf8', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                title="Switch to Interactive Vector Chart"
              >
                <BarChart2 size={11} /> Interactive View
              </button>
            )}
            {fallbackImage && (
              <>
                <button
                  onClick={handleCopy}
                  style={{ background: '#1e293b', color: '#e2e8f0', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                  title="Copy PNG to Clipboard"
                >
                  {copied ? <Check size={11} color="#4ade80" /> : <Copy size={11} />} {copied ? 'Copied' : 'Copy'}
                </button>
                <button
                  onClick={() => onZoomImage && onZoomImage(fallbackImage)}
                  style={{ background: '#1e293b', color: '#38bdf8', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                  title="Inspect Fullscreen"
                >
                  <Maximize2 size={11} /> Zoom
                </button>
              </>
            )}
          </div>
        </div>
        {fallbackImage ? (
          <img
            src={fallbackImage}
            alt={title || 'Chart'}
            style={{ width: '100%', maxHeight: height, objectFit: 'contain', borderRadius: 4, cursor: onZoomImage ? 'zoom-in' : 'default' }}
            onClick={() => onZoomImage && onZoomImage(fallbackImage)}
          />
        ) : (
          <div style={{ padding: 30, textAlign: 'center', color: '#64748b', fontSize: 12 }}>
            No visual chart generated.
          </div>
        )}
      </div>
    );
  }

  return (
    <div ref={containerRef} style={{ background: '#0c101d', border: '1px solid #1e293b', borderRadius: 8, padding: 12, display: 'flex', flexDirection: 'column', gap: 10, position: 'relative' }}>
      {/* Header bar with controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 6 }}>
          <BarChart2 size={14} color="#38bdf8" />
          {title || spec?.layout?.title?.text || spec?.layout?.title || 'Interactive Vector Plot'}
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {/* Zoom controls */}
          <button
            onClick={() => setZoomLevel(prev => Math.min(2.0, prev + 0.2))}
            style={{ background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 3, padding: '2px 6px', fontSize: 10, cursor: 'pointer' }}
            title="Zoom In"
          >
            <ZoomIn size={12} />
          </button>
          <button
            onClick={() => setZoomLevel(prev => Math.max(0.6, prev - 0.2))}
            style={{ background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 3, padding: '2px 6px', fontSize: 10, cursor: 'pointer' }}
            title="Zoom Out"
          >
            <ZoomOut size={12} />
          </button>
          <button
            onClick={() => { setZoomLevel(1); setHiddenSeries(new Set()); }}
            style={{ background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 3, padding: '2px 6px', fontSize: 10, cursor: 'pointer' }}
            title="Reset Chart View"
          >
            <RotateCcw size={12} />
          </button>

          {/* Toggle to static PNG mode */}
          {fallbackImage && (
            <button
              onClick={() => setViewMode('image')}
              style={{ background: '#1e293b', color: '#e2e8f0', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
              title="Switch to Matplotlib Static Image View"
            >
              <Image size={11} /> Static PNG
            </button>
          )}
        </div>
      </div>

      {/* Downsampling / Large Series Disclosure */}
      {isDownsampled && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          background: 'rgba(56, 189, 248, 0.08)',
          border: '1px solid rgba(56, 189, 248, 0.2)',
          borderRadius: 4,
          padding: '4px 8px',
          fontSize: 10,
          color: '#7dd3fc'
        }}>
          <span>ℹ️</span>
          <span>
            <strong>Large Series Disclosure:</strong> Displaying {Math.min(1000, maxSeriesLength).toLocaleString()} representative points from {maxSeriesLength.toLocaleString()} total observations for responsive interaction. All {maxSeriesLength.toLocaleString()} rows are preserved in the result table.
          </span>
        </div>
      )}

      {/* Interactive Legend Bar */}
      {traces.length > 1 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, padding: '4px 8px', background: '#090d16', borderRadius: 4, border: '1px solid #1e293b' }}>
          {traces.map(t => {
            const isHidden = hiddenSeries.has(t.id);
            return (
              <button
                key={t.id}
                onClick={() => toggleSeries(t.id)}
                style={{
                  background: isHidden ? 'transparent' : 'rgba(30, 41, 59, 0.6)',
                  border: isHidden ? '1px dashed #334155' : `1px solid ${t.color}`,
                  borderRadius: 3,
                  padding: '2px 8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  fontSize: 10,
                  color: isHidden ? '#64748b' : '#f8fafc',
                  cursor: 'pointer',
                  opacity: isHidden ? 0.5 : 1
                }}
              >
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: isHidden ? '#64748b' : t.color }} />
                <span>{t.id}</span>
                {isHidden ? <EyeOff size={10} color="#64748b" /> : <Eye size={10} color="#94a3b8" />}
              </button>
            );
          })}
        </div>
      )}

      {/* SVG Vector Canvas */}
      <div style={{ overflow: 'hidden', position: 'relative', width: '100%', minHeight: height }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ width: '100%', height: height, display: 'block' }}
          onMouseLeave={() => setHoveredPoint(null)}
        >
          {/* Grid lines */}
          {yTicks.map((tick, idx) => (
            <g key={idx}>
              <line
                x1={margin.left}
                y1={tick.y}
                x2={margin.left + plotWidth}
                y2={tick.y}
                stroke="#1e293b"
                strokeDasharray="2,2"
                strokeWidth={1}
              />
              <text
                x={margin.left - 8}
                y={tick.y + 3}
                fill="#64748b"
                fontSize={9}
                textAnchor="end"
                fontFamily="monospace"
              >
                {tick.val}
              </text>
            </g>
          ))}

          {xTicks.map((tick, idx) => (
            <g key={idx}>
              <line
                x1={tick.x}
                y1={margin.top}
                x2={tick.x}
                y2={margin.top + plotHeight}
                stroke="#1e293b"
                strokeDasharray="2,2"
                strokeWidth={1}
              />
              <text
                x={tick.x}
                y={margin.top + plotHeight + 14}
                fill="#64748b"
                fontSize={9}
                textAnchor="middle"
                fontFamily="monospace"
              >
                {tick.val}
              </text>
            </g>
          ))}

          {/* Axes */}
          <line
            x1={margin.left}
            y1={margin.top + plotHeight}
            x2={margin.left + plotWidth}
            y2={margin.top + plotHeight}
            stroke="#334155"
            strokeWidth={1.5}
          />
          <line
            x1={margin.left}
            y1={margin.top}
            x2={margin.left}
            y2={margin.top + plotHeight}
            stroke="#334155"
            strokeWidth={1.5}
          />

          {/* Axis Labels */}
          {spec.layout?.xaxis?.title && (
            <text
              x={margin.left + plotWidth / 2}
              y={svgHeight - 4}
              fill="#94a3b8"
              fontSize={10}
              fontWeight={600}
              textAnchor="middle"
            >
              {typeof spec.layout.xaxis.title === 'string' ? spec.layout.xaxis.title : spec.layout.xaxis.title.text}
            </text>
          )}

          {spec.layout?.yaxis?.title && (
            <text
              x={14}
              y={margin.top + plotHeight / 2}
              fill="#94a3b8"
              fontSize={10}
              fontWeight={600}
              textAnchor="middle"
              transform={`rotate(-90, 14, ${margin.top + plotHeight / 2})`}
            >
              {typeof spec.layout.yaxis.title === 'string' ? spec.layout.yaxis.title : spec.layout.yaxis.title.text}
            </text>
          )}

          {/* Traces */}
          {visibleTraces.map((trace, tIdx) => {
            const xVals = Array.isArray(trace.x) ? trace.x : [];
            const yVals = Array.isArray(trace.y) ? trace.y : [];
            const traceType = trace.type || 'scatter';

            // Bars
            if (traceType === 'bar' || traceType === 'histogram') {
              const barWidth = Math.max(3, (plotWidth / Math.max(1, xVals.length)) * 0.7);
              return (
                <g key={tIdx}>
                  {xVals.map((x, i) => {
                    const y = yVals[i] !== undefined ? yVals[i] : 0;
                    const px = scaleX(x) - barWidth / 2;
                    const py = scaleY(y);
                    const zeroY = scaleY(0);
                    const h = Math.max(2, Math.abs(zeroY - py));
                    const topY = Math.min(py, zeroY);

                    return (
                      <rect
                        key={i}
                        x={px}
                        y={topY}
                        width={barWidth}
                        height={h}
                        fill={trace.color}
                        opacity={0.85}
                        rx={2}
                        onMouseEnter={() => setHoveredPoint({ x, y, series: trace.id, color: trace.color })}
                      />
                    );
                  })}
                </g>
              );
            }

            // Scatter / Line
            if (traceType === 'scatter') {
              const stride = xVals.length > 1000 ? Math.ceil(xVals.length / 1000) : 1;
              const sampledIndices = [];
              for (let i = 0; i < xVals.length; i += stride) {
                sampledIndices.push(i);
              }
              if (sampledIndices.length > 0 && sampledIndices[sampledIndices.length - 1] !== xVals.length - 1) {
                sampledIndices.push(xVals.length - 1);
              }

              const points = sampledIndices.map(i => ({
                x: xVals[i],
                y: yVals[i],
                px: scaleX(xVals[i]),
                py: scaleY(yVals[i])
              })).filter(p => Number.isFinite(p.px) && Number.isFinite(p.py));

              // Scientific integrity check:
              // Never connect unordered categorical points with continuous line segments.
              // Lines are only valid if X is continuous / numeric or explicitly requested.
              const explicitLines = trace.mode && trace.mode.includes('lines');
              const explicitMarkers = trace.mode && trace.mode.includes('markers');

              const isLine = explicitLines || (!trace.mode && !bounds.isCategoricalX);
              const isMarkers = explicitMarkers || (!trace.mode) || points.length < 100 || bounds.isCategoricalX;

              const pathData = (isLine && points.length > 1)
                ? points.map((p, i) => (i === 0 ? `M ${p.px} ${p.py}` : `L ${p.px} ${p.py}`)).join(' ')
                : '';

              return (
                <g key={tIdx}>
                  {/* Shaded Confidence Band if specified */}
                  {trace.lower_ci && trace.upper_ci && (
                    <polygon
                      points={[
                        ...points.map(p => `${p.px},${scaleY(trace.lower_ci[points.indexOf(p)])}`),
                        ...points.slice().reverse().map(p => `${p.px},${scaleY(trace.upper_ci[points.indexOf(p)])}`)
                      ].join(' ')}
                      fill={trace.color}
                      opacity={0.15}
                    />
                  )}

                  {isLine && pathData && (
                    <path
                      d={pathData}
                      fill="none"
                      stroke={trace.color}
                      strokeWidth={2}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  )}

                  {isMarkers && points.map((p, i) => (
                    <circle
                      key={i}
                      cx={p.px}
                      cy={p.py}
                      r={hoveredPoint?.x === p.x && hoveredPoint?.series === trace.id ? 5 : 3}
                      fill={trace.color}
                      stroke="#0f172a"
                      strokeWidth={1}
                      style={{ cursor: 'pointer', transition: 'r 0.1s ease' }}
                      onMouseEnter={() => setHoveredPoint({ x: p.x, y: p.y, series: trace.id, color: trace.color })}
                    />
                  ))}
                </g>
              );
            }

            return null;
          })}
        </svg>

        {/* Hover Tooltip Overlay */}
        {hoveredPoint && (
          <div style={{
            position: 'absolute',
            top: 8,
            right: 8,
            background: 'rgba(15, 23, 42, 0.95)',
            border: `1px solid ${hoveredPoint.color || '#38bdf8'}`,
            borderRadius: 6,
            padding: '6px 10px',
            fontSize: 11,
            color: '#f8fafc',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.4)',
            pointerEvents: 'none',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            zIndex: 10
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700, color: hoveredPoint.color || '#38bdf8' }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: hoveredPoint.color || '#38bdf8' }} />
              {hoveredPoint.series}
            </div>
            <div style={{ color: '#cbd5e1', fontFamily: 'monospace' }}>
              X: <span style={{ color: '#ffffff' }}>{String(hoveredPoint.x)}</span>
            </div>
            <div style={{ color: '#cbd5e1', fontFamily: 'monospace' }}>
              Y: <span style={{ color: '#ffffff' }}>{typeof hoveredPoint.y === 'number' ? hoveredPoint.y.toFixed(4) : String(hoveredPoint.y)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
