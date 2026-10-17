import type { MotionData, Prediction, Range, SessionTimeline } from './types';

/** Parse and validate the published prediction contract (schema v1.0). */
export function parsePrediction(value: unknown): Prediction {
  if (!value || typeof value !== 'object') throw new Error('Prediction must be a JSON object.');
  const p = value as Prediction;
  if (p.schema_version !== '1.0' || !p.model?.run_key || !Array.isArray(p.events) || !p.diagnostics || !p.input)
    throw new Error('Unsupported prediction schema. Expected version 1.0.');
  for (const e of p.events) if (!Number.isFinite(e.start_ms) || !Number.isFinite(e.end_ms) || e.end_ms <= e.start_ms) throw new Error('Invalid event timestamps.');
  if (p.timeline && !p.timeline.sessions && p.timeline.series?.length) {
    const grouped = new Map<string, SessionTimeline>();
    for (const point of p.timeline.series) {
      if (!Number.isFinite(point.timestamp_ms) || !Number.isFinite(point.macro_probability) || !Number.isFinite(point.micro_probability) || point.macro_probability < 0 || point.macro_probability > 1 || point.micro_probability < 0 || point.micro_probability > 1) throw new Error('Invalid timeline series.');
      if (point.gap || !point.valid) continue;
      const current = grouped.get(point.session_id) || { session_id: point.session_id, start_ms: point.timestamp_ms, end_ms: point.timestamp_ms, macro: { timestamp_ms: [], probability: [] }, micro: { timestamp_ms: [], probability: [] } };
      current.start_ms = Math.min(current.start_ms, point.timestamp_ms); current.end_ms = Math.max(current.end_ms, point.timestamp_ms);
      current.macro.timestamp_ms.push(point.timestamp_ms); current.macro.probability.push(point.macro_probability);
      current.micro.timestamp_ms.push(point.timestamp_ms); current.micro.probability.push(point.micro_probability);
      grouped.set(point.session_id, current);
    }
    p.timeline.sessions = [...grouped.values()];
  }
  for (const session of p.timeline?.sessions || []) {
    if (!Number.isFinite(session.start_ms) || !Number.isFinite(session.end_ms) || session.end_ms <= session.start_ms) throw new Error('Invalid timeline timestamps.');
    for (const key of ['macro', 'micro'] as const) {
      const s = session[key];
      if (!s || s.timestamp_ms.length !== s.probability.length || s.timestamp_ms.some((t, i) => !Number.isFinite(t) || (i > 0 && t < s.timestamp_ms[i - 1])) || s.probability.some(x => !Number.isFinite(x) || x < 0 || x > 1)) throw new Error('Invalid timeline stream.');
    }
  }
  return p;
}

export function boundsFor(p: Prediction, sessionId: string): Range | null {
  const session = p.timeline?.sessions?.find(s => s.session_id === sessionId);
  if (session) return session;
  const rows = [...p.events, ...(p.candidates || []), ...(p.gaps || [])].filter(r => r.session_id === sessionId);
  if (!rows.length) return null;
  return { start_ms: Math.min(...rows.map(r => r.start_ms)), end_ms: Math.max(...rows.map(r => r.end_ms)) };
}

/** Ordered session identifiers: contract order where available, else the union of records. */
export function sessionIds(p: Prediction, motions: Record<string, MotionData> = {}): string[] {
  const ids = new Set<string>();
  for (const id of p.timeline?.session_ids || []) ids.add(id);
  for (const row of [...p.events, ...(p.candidates || []), ...(p.gaps || [])]) ids.add(row.session_id);
  for (const id of Object.keys(motions)) ids.add(id);
  return [...ids];
}
