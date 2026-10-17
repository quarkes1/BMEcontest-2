import type { Range } from '../../data/types';
export function clampRange(start: number, end: number, bounds: Range): Range {
  const a = Math.max(bounds.start_ms, Math.min(bounds.end_ms, Math.min(start, end)));
  const b = Math.max(bounds.start_ms, Math.min(bounds.end_ms, Math.max(start, end)));
  return { start_ms: a, end_ms: Math.max(a, b) };
}
export function timeAtX(x: number, width: number, view: Range): number {
  return view.start_ms + Math.max(0, Math.min(1, x / Math.max(width, 1))) * (view.end_ms - view.start_ms);
}
export function xAtTime(time: number, width: number, view: Range): number {
  return (time - view.start_ms) / Math.max(1, view.end_ms - view.start_ms) * width;
}
export function zoomView(bounds: Range, center: number, spanMs: number): Range {
  const total = bounds.end_ms - bounds.start_ms;
  if (spanMs >= total) return bounds;
  const start = Math.max(bounds.start_ms, Math.min(bounds.end_ms - spanMs, center - spanMs / 2));
  return { start_ms: start, end_ms: start + spanMs };
}
export function panView(bounds: Range, view: Range, shiftMs: number): Range {
  return zoomView(bounds, (view.start_ms + view.end_ms) / 2 + shiftMs, view.end_ms - view.start_ms);
}
