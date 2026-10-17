import type { Imu, MotionManifest, Prediction } from './types';

const start = new Date('2026-09-17T18:10:00+08:00').getTime();
const sid = 'DEMO SESSION 04';
const minute = 60_000;
const episodes = [[7.2, 10.5, 0.87], [21.1, 25.3, 0.79], [36.5, 40.0, 0.91]] as const;
function gaussian(t: number, center: number, width: number) { const x = (t - center) / width; return Math.exp(-x * x); }
function evidence(t: number) { return episodes.reduce((v, [a, b]) => Math.max(v, gaussian(t, (a + b) / 2, (b - a) / 1.6)), 0); }
const macroTimes: number[] = [], macroProbability: number[] = [], microTimes: number[] = [], microProbability: number[] = [];
for (let s = 0; s <= 42 * 60; s += 15) { const t = s / 60; macroTimes.push(start + s * 1000); macroProbability.push(Math.min(.94, .05 + .82 * evidence(t))); }
for (let s = 0; s <= 42 * 60; s += 7.5) { const t = s / 60; microTimes.push(start + s * 1000); microProbability.push(Math.min(.99, .05 + .72 * evidence(t) * (.57 + .43 * Math.sin(s * 1.7) ** 2))); }
export const demoPrediction: Prediction = {
  schema_version: '1.0', model: { name: 'event-stack', run_key: 'DEMO — not a release' }, input: { source: 'DEMO DATA', duration_seconds: 42 * 60 },
  events: episodes.map(([a, b, score], id) => ({ id, session_id: sid, start_ms: start + a * minute, end_ms: start + b * minute, duration_s: (b - a) * 60, confidence: score })),
  candidates: [
    ...episodes.map(([a, b, score]) => ({ session_id: sid, start_ms: start + a * minute, end_ms: start + b * minute, score, admitted: true })),
    { session_id: sid, start_ms: start + 16.1 * minute, end_ms: start + 17.7 * minute, score: .21, admitted: false },
    { session_id: sid, start_ms: start + 32.2 * minute, end_ms: start + 33.4 * minute, score: .31, admitted: false },
  ],
  gaps: [], diagnostics: { coverage: 1, warnings: [], resolved_device: 'cpu' },
  timeline: { session_ids: [sid], macro_windows: macroTimes.length, micro_windows: microTimes.length,
    sessions: [{ session_id: sid, start_ms: start, end_ms: start + 42 * minute, macro: { timestamp_ms: macroTimes, probability: macroProbability }, micro: { timestamp_ms: microTimes, probability: microProbability } }] },
};
const n = 42 * 60 * 20;
const t = new Float64Array(n), ax = new Float32Array(n), ay = new Float32Array(n), az = new Float32Array(n), gx = new Float32Array(n), gy = new Float32Array(n), gz = new Float32Array(n);
for (let i = 0; i < n; i++) {
  const s = i / 20, e = evidence(s / 60), gesture = Math.sin(2 * Math.PI * .28 * s);
  t[i] = start + s * 1000;
  ax[i] = .12 * e * gesture + .018 * Math.sin(s * 2.2); ay[i] = .08 * e * Math.cos(s * 1.75) + .015 * Math.sin(s * .7); az[i] = 1 + .24 * e * Math.abs(gesture) + .02 * Math.sin(s * 2.7);
  gx[i] = .13 * e * Math.sin(s * 1.8); gy[i] = .62 * e * Math.cos(s * 1.75); gz[i] = .16 * e * Math.sin(s * 1.1);
}
export const demoMotion: { manifest: MotionManifest; imu: Imu } = {
  manifest: { telemetry_version: '1.0', session_id: sid, record_format: 'f64_ms_6xf32_le', sample_count: n, start_ms: start, end_ms: start + 42 * minute,
    binary_file: 'synthetic-in-memory', units: { acceleration: 'g', gyroscope: 'rad/s' }, calibration: { acceleration_counts_per_g: 1, gyroscope_counts_per_rad_s: 1, viewer_from_sensor: [1, 0, 0, 0, 0, 1, 0, -1, 0] } },
  imu: { t, ax, ay, az, gx, gy, gz },
};
