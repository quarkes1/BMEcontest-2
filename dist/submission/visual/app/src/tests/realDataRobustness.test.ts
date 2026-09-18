import { describe, expect, it, vi } from 'vitest';
import { parseMotion } from '../data/telemetry';
import { demoPrediction } from '../data/demo';
import { analyzeRawSelection, analyzeTokens, mergeOutcomes, uploadSessionFile } from '../runtime/inferenceClient';
import { canOrient } from '../motion/coordinateFrame';
import { reconstructOrientation } from '../motion/orientation';
import type { MotionManifest } from '../data/types';

const manifest: MotionManifest = {
  telemetry_version: '1.0', session_id: 's', record_format: 'f64_ms_6xf32_le', sample_count: 3,
  start_ms: 10, end_ms: 20, binary_file: 'motion.bin',
  units: { acceleration: 'raw_adc', gyroscope: 'raw_adc' },
  calibration: { acceleration_counts_per_g: 100, viewer_from_sensor: [1, 0, 0, 0, 0, 1, 0, -1, 0] },
};

describe('real data robustness', () => {
  it('accepts finite timestamp regressions and gravity-only replay', () => {
    const bytes = new ArrayBuffer(96), view = new DataView(bytes);
    [10, 20, 15].forEach((t, i) => { view.setFloat64(i * 32, t, true); view.setFloat32(i * 32 + 8 + 8, 100, true); });
    const motion = parseMotion(manifest, bytes);
    expect([...motion.imu.t]).toEqual([10, 20, 15]);
    expect(canOrient(manifest)).toBe(true);
    expect(canOrient({ ...manifest, calibration: undefined })).toBe(false);
    expect(canOrient({ ...manifest, calibration: { ...manifest.calibration!, viewer_from_sensor: [1, 0, 0, 0, 1, 0, 0, 0, -1] } })).toBe(false);
    expect(reconstructOrientation(motion.imu, manifest)?.every(p => Number.isFinite(p.w))).toBe(true);
  });

  it('anchors a rewound run immediately instead of dropping its samples', () => {
    const times: number[] = [];
    for (let i = 0; i <= 10; i++) times.push(i * 100);        // first run 0..1000 ms
    for (let i = 1; i <= 10; i++) times.push(i * 100);        // rewound run 100..1000 ms
    const bytes = new ArrayBuffer(times.length * 32), view = new DataView(bytes);
    times.forEach((time, i) => { view.setFloat64(i * 32, time, true); view.setFloat32(i * 32 + 16, 100, true); });
    const motion = parseMotion({ ...manifest, sample_count: times.length }, bytes);
    const points = reconstructOrientation(motion.imu, manifest);
    expect(points).not.toBeNull();
    // the rewound run must start a new segment at its first sample (t=100), not be
    // silently skipped until its clock overtakes the previous maximum
    expect(points!.some(point => point.segment === 1 && point.t <= 500)).toBe(true);
    expect(points!.some(point => point.t === 100 && point.segment === 1)).toBe(true);
  });

  it('keeps HTTP detail separate from network failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('NetworkError when attempting to fetch resource')));
    await expect(uploadSessionFile(new File(['x'], 'collect_data1_2_3.txt'), null)).rejects.toThrow('NetworkError：本地推理服务连接中断');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, json: async () => ({ error: 'Server detail' }) }));
    await expect(analyzeTokens(['token'])).rejects.toThrow('Server detail');
    vi.unstubAllGlobals();
  });

  it('merges independent prediction documents with valid event IDs', () => {
    const first = { prediction: structuredClone(demoPrediction), motions: {}, warnings: [] };
    const second = { prediction: structuredClone(demoPrediction), motions: {}, warnings: ['skipped'] };
    second.prediction.events[0].session_id = 'other';
    const merged = mergeOutcomes([first, second]);
    expect(merged.prediction.events.map(e => e.id)).toEqual([...merged.prediction.events.keys()]);
    expect(merged.warnings).toContain('skipped');
  });

  it('continues after one per-session HTTP failure', async () => {
    let uploads = 0, analyses = 0;
    const relatives: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === './api/upload') {
        relatives.push(String((init?.headers as Record<string, string>)['X-Relative-Path']));
        return { ok: true, json: async () => ({ token: `token-${++uploads}` }) };
      }
      if (url === './api/analyze') {
        analyses++;
        if (analyses === 1) return { ok: false, json: async () => ({ error: 'bad timeline' }) };
        return { ok: true, json: async () => ({ prediction: structuredClone(demoPrediction), sessions: [], warnings: [] }) };
      }
      throw new Error(`Unexpected URL ${url}`);
    }));
    const files = [new File(['x'], 'collect_data1_2_3.txt'), new File(['x'], 'collect_data1_2_3.txt')];
    const phases: string[] = [];
    const result = await analyzeRawSelection(files, false, (phase, done, total) => phases.push(`${phase}:${done}/${total}`));
    expect(analyses).toBe(2);
    expect(result.prediction.events.length).toBeGreaterThan(0);
    expect(result.warnings.join(' ')).toContain('bad timeline');
    expect(phases).toContain('running:1/2');
    expect(new Set(relatives).size).toBe(2);
    expect(relatives.every(path => path.endsWith('.txt'))).toBe(true);
    vi.unstubAllGlobals();
  });
});
