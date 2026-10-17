import { useEffect, useMemo, useRef, useState } from 'react';
import type { Imu, Prediction, Range } from '../../data/types';
import { boundsFor } from '../../data/prediction';
import { duration, fmtTime } from '../../data/format';
import { clampRange, panView, timeAtX, xAtTime, zoomView } from './timelineMath';

type Props = { prediction: Prediction; sessionId: string; imu?: Imu; analyzing?: boolean; selection: Range | null; onSelect: (range: Range | null) => void; playhead: number | null };
const labels = [{ title: 'Motion', sub: 'ACC magnitude' }, { title: 'Macro', sub: 'Model score' }, { title: 'Micro', sub: 'Model score' }, { title: 'Events', sub: 'Eating / candidate' }];
const colors = ['#22775f', '#597edf', '#9476cb'];
export default function Timeline({ prediction, sessionId, imu, analyzing, selection, onSelect, playhead }: Props) {
  const bounds = useMemo(() => boundsFor(prediction, sessionId) || (imu?.t.length ? { start_ms: imu.t[0], end_ms: imu.t[imu.t.length - 1] } : null), [prediction, sessionId, imu]);
  const [view, setView] = useState<Range | null>(bounds);
  const [zoom, setZoom] = useState('Full');
  const canvas = useRef<HTMLCanvasElement>(null), overlay = useRef<HTMLDivElement>(null);
  const gesture = useRef<{ mode: 'new' | 'left' | 'right' | 'pan'; anchor: number; anchorX: number; startView: Range; origin: Range | null; moved: boolean } | null>(null);
  const session = prediction.timeline?.sessions?.find(s => s.session_id === sessionId);
  const events = prediction.events.filter(e => e.session_id === sessionId), candidates = (prediction.candidates || []).filter(c => c.session_id === sessionId);
  const gaps = (prediction.gaps || []).filter(g => g.session_id === sessionId);
  useEffect(() => { setView(bounds); setZoom('Full'); }, [bounds?.start_ms, bounds?.end_ms, sessionId]);
  useEffect(() => {
    const el = canvas.current; if (!el || !view) return;
    const render = () => {
      const width = el.clientWidth, height = el.clientHeight, dpr = Math.min(devicePixelRatio || 1, 2);
      el.width = Math.max(1, Math.round(width * dpr)); el.height = Math.max(1, Math.round(height * dpr));
      const ctx = el.getContext('2d'); if (!ctx) return;
      ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
      const track = height / 4;
      ctx.strokeStyle = '#e9e9e7'; ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) { ctx.beginPath(); ctx.moveTo(0, i * track + .5); ctx.lineTo(width, i * track + .5); ctx.stroke(); }
      for (let i = 0; i <= 8; i++) { const x = i * width / 8 + .5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
      for (const gap of gaps) { const x0 = xAtTime(gap.start_ms, width, view), x1 = xAtTime(gap.end_ms, width, view); ctx.fillStyle = '#f0f0ee'; ctx.fillRect(x0, 0, x1 - x0, height); }
      if (imu?.t.length) {
        let max = 1; const step = Math.max(1, Math.floor(imu.t.length / Math.max(width * 5, 1)));
        for (let i = 0; i < imu.t.length; i += step) if (imu.t[i] >= view.start_ms && imu.t[i] <= view.end_ms) max = Math.max(max, Math.hypot(imu.ax[i], imu.ay[i], imu.az[i]));
        ctx.strokeStyle = colors[0]; ctx.lineWidth = 1.25; ctx.beginPath(); let active = false;
        const pixelMin = new Float64Array(Math.ceil(width)).fill(Infinity), pixelMax = new Float64Array(Math.ceil(width)).fill(-Infinity);
        for (let i = 0; i < imu.t.length; i++) {
          if (imu.t[i] < view.start_ms || imu.t[i] > view.end_ms) continue;
          const col = Math.floor(xAtTime(imu.t[i], width, view)); if (col < 0 || col >= pixelMin.length) continue;
          const v = Math.hypot(imu.ax[i], imu.ay[i], imu.az[i]); pixelMin[col] = Math.min(pixelMin[col], v); pixelMax[col] = Math.max(pixelMax[col], v);
        }
        for (let x = 0; x < pixelMin.length; x++) if (pixelMin[x] !== Infinity) { const y0 = track - 8 - pixelMin[x] / max * (track - 16), y1 = track - 8 - pixelMax[x] / max * (track - 16); if (!active) { ctx.moveTo(x, y0); active = true; } else ctx.lineTo(x, y0); ctx.lineTo(x, y1); }
        ctx.stroke();
      }
      for (const [index, key] of (['macro', 'micro'] as const).entries()) {
        const data = session?.[key]; if (!data?.timestamp_ms.length) continue;
        ctx.strokeStyle = colors[index + 1]; ctx.lineWidth = 1.4; ctx.beginPath(); let begun = false, previous = -Infinity;
        data.timestamp_ms.forEach((t, i) => {
          if (t < view.start_ms || t > view.end_ms) return;
          const x = xAtTime(t, width, view), y = (index + 2) * track - 8 - data.probability[i] * (track - 16);
          if (!begun || t - previous > (key === 'macro' ? 60_000 : 30_000)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          begun = true; previous = t;
        }); ctx.stroke();
      }
      for (const c of candidates) { const a = xAtTime(c.start_ms, width, view), b = xAtTime(c.end_ms, width, view); ctx.fillStyle = c.admitted ? '#c5d4d1' : '#dce0e4'; ctx.fillRect(a, 3 * track + track * .22, Math.max(2, b - a), Math.min(14,track*.18)); }
      for (const e of events) { const a = xAtTime(e.start_ms, width, view), b = xAtTime(e.end_ms, width, view); ctx.fillStyle = '#439f7d'; ctx.fillRect(a, 3 * track + track * .56, Math.max(2, b - a), Math.min(19,track*.24)); }
    };
    const observer = new ResizeObserver(render); observer.observe(el); render(); return () => observer.disconnect();
  }, [view, session, imu, prediction, sessionId]);
  if (!bounds || !view) return <div className="no-telemetry">{analyzing ? 'Analyzing…' : 'No timestamped timeline is available in this prediction.'}</div>;
  const span = view.end_ms - view.start_ms;
  const xPercent = (t: number) => `${Math.max(0, Math.min(100, (t - view.start_ms) / Math.max(span, 1) * 100))}%`;
  const point = (event: React.PointerEvent<HTMLDivElement>) => { const r = overlay.current!.getBoundingClientRect(); return { x: event.clientX - r.left, y: event.clientY - r.top, width: r.width, height: r.height }; };
  function down(e: React.PointerEvent<HTMLDivElement>) {
    const p = point(e), t = timeAtX(p.x, p.width, view!);
    if (p.y > p.height * .75) {
      const hit = [...events, ...candidates].find(row => t >= row.start_ms && t <= row.end_ms);
      if (hit) { onSelect(clampRange(hit.start_ms, hit.end_ms, bounds!)); return; }
    }
    const edge = selection && Math.min(Math.abs(p.x - xAtTime(selection.start_ms, p.width, view!)), Math.abs(p.x - xAtTime(selection.end_ms, p.width, view!)));
    const mode = e.shiftKey ? 'pan' : edge !== null && edge !== undefined && edge < 8 ? Math.abs(p.x - xAtTime(selection!.start_ms, p.width, view!)) < Math.abs(p.x - xAtTime(selection!.end_ms, p.width, view!)) ? 'left' : 'right' : 'new';
    gesture.current = { mode, anchor: t, anchorX: p.x, startView: view!, origin: selection, moved: false }; overlay.current?.setPointerCapture(e.pointerId);
    if (mode === 'new') onSelect({ start_ms: t, end_ms: t });
  }
  function move(e: React.PointerEvent<HTMLDivElement>) {
    if (!gesture.current) return;
    const p = point(e), t = timeAtX(p.x, p.width, view!), g = gesture.current;
    if (Math.abs(t - g.anchor) > span / Math.max(p.width, 1) * 3) g.moved = true;
    if (g.mode === 'pan') { setView(panView(bounds!, g.startView, (g.anchorX-p.x)/p.width*(g.startView.end_ms-g.startView.start_ms))); return; }
    if (g.mode === 'new') onSelect(clampRange(g.anchor, t, bounds!));
    else if (g.origin) onSelect(clampRange(g.mode === 'left' ? t : g.origin.start_ms, g.mode === 'right' ? t : g.origin.end_ms, bounds!));
  }
  function up() { if (gesture.current?.mode === 'new' && !gesture.current.moved) onSelect(null); gesture.current = null; }
  return <section className="panel timeline-panel" aria-label="Synchronized motion and model timeline">
    <div className="timeline-toolbar" title="Drag to select · Shift + drag or wheel to pan"><h2>Evidence timeline</h2><div className="segmented">{[['5 min', 300_000], ['30 min', 1_800_000], ['1 hour', 3_600_000], ['Full', Infinity]].map(([label, ms]) => <button key={label} className={zoom === label ? 'active' : ''} onClick={() => { setZoom(String(label)); setView(zoomView(bounds, selection ? (selection.start_ms + selection.end_ms) / 2 : (view.start_ms + view.end_ms) / 2, Number(ms))); }}>{label}</button>)}</div></div>
    <div className="timeline-body"><div className="track-labels">{labels.map((l, i) => <div key={l.title}><strong>{l.title}</strong><small>{i === 0 && !imu ? 'No IMU telemetry' : i > 0 && i < 3 && !session?.[i === 1 ? 'macro' : 'micro']?.timestamp_ms.length ? 'No series in prediction' : l.sub}</small>{i === 3 && <div className="track-legend"><span className="accepted-dot"/>Eating <span className="candidate-dot"/>Candidate</div>}</div>)}</div><div className="plot-wrap"><canvas ref={canvas} className="timeline-canvas"/><div ref={overlay} className="timeline-overlay" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onWheel={e => { e.preventDefault(); setView(panView(bounds, view, e.deltaY * span / 1500)); }}>
      {selection && selection.end_ms > selection.start_ms && <><div className="selection" style={{ left: xPercent(selection.start_ms), width: `${(selection.end_ms - selection.start_ms) / span * 100}%` }}/><div className="selection-edge left" style={{ left: xPercent(selection.start_ms) }}/><div className="selection-edge right" style={{ left: xPercent(selection.end_ms) }}/><div className="selection-label" style={{ left: xPercent((selection.start_ms + selection.end_ms) / 2) }}>{duration(selection.end_ms - selection.start_ms)}<br/>{fmtTime(selection.start_ms, true)} – {fmtTime(selection.end_ms, true)}</div></>}
      {playhead !== null && playhead >= view.start_ms && playhead <= view.end_ms && <div className="playhead" style={{ left: xPercent(playhead) }}/>}</div>
      {analyzing && <div className="timeline-analyzing" role="status" aria-live="polite"><span className="analyzing-spinner" aria-hidden="true"/><strong>Analyzing…</strong><small>canonical inference is running locally · about 30 s per session</small></div>}
      <div className="time-ticks">{Array.from({ length: 7 }, (_, i) => <span key={i}>{fmtTime(view.start_ms + i / 6 * span)}</span>)}</div></div></div>
  </section>;
}

