import type { Imu, MotionData, MotionManifest } from './types';

/** Motion telemetry record. Segment boundaries may contain timestamp regressions. */
export function parseMotion(manifestValue: unknown, buffer: ArrayBuffer): MotionData {
  const m = manifestValue as MotionManifest;
  if (!m || m.telemetry_version !== '1.0' || m.record_format !== 'f64_ms_6xf32_le' || !Number.isSafeInteger(m.sample_count) || m.sample_count < 0 || buffer.byteLength !== m.sample_count * 32) throw new Error('Invalid motion telemetry manifest or binary size.');
  const t = new Float64Array(m.sample_count), channels = Array.from({ length: 6 }, () => new Float32Array(m.sample_count));
  const view = new DataView(buffer);
  for (let i = 0; i < m.sample_count; i++) {
    t[i] = view.getFloat64(i * 32, true);
    if (!Number.isFinite(t[i])) throw new Error('IMU timestamps must be finite.');
    for (let c = 0; c < 6; c++) {
      channels[c][i] = view.getFloat32(i * 32 + 8 + c * 4, true);
      if (!Number.isFinite(channels[c][i])) throw new Error('IMU values must be finite.');
    }
  }
  return { manifest: m, imu: { t, ax: channels[0], ay: channels[1], az: channels[2], gx: channels[3], gy: channels[4], gz: channels[5] } as Imu };
}
