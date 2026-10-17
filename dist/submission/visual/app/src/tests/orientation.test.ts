import { describe, expect, it } from 'vitest';
import type { Imu, MotionManifest } from '../data/types';
import { validMapping, canOrient } from '../motion/coordinateFrame';
import { normalize, slerp } from '../motion/quaternion';
import { reconstructOrientation } from '../motion/orientation';
import { interpolateOrientation } from '../motion/interpolation';

function manifest(overrides: Partial<MotionManifest> = {}): MotionManifest {
  return {
    telemetry_version: '1.0', session_id: 's', record_format: 'f64_ms_6xf32_le', sample_count: 4,
    start_ms: 0, end_ms: 900, binary_file: 'motion.bin',
    units: { acceleration: 'g', gyroscope: 'rad/s' },
    calibration: { acceleration_counts_per_g: 1, gyroscope_counts_per_rad_s: 1, viewer_from_sensor: [1, 0, 0, 0, 1, 0, 0, 0, 1] },
    ...overrides,
  };
}

describe('IMU orientation', () => {
  it('rejects reflected frames and uncalibrated telemetry', () => {
    expect(validMapping([1, 0, 0, 0, 1, 0, 0, 0, -1])).toBe(false);
    expect(validMapping([1, 0, 0, 0, 0, 1, 0, -1, 0])).toBe(true);
    expect(canOrient(manifest({ calibration: undefined }))).toBe(false);
    expect(canOrient(manifest())).toBe(true);
  });

  it('normalizes quaternions and slerps along the shortest arc', () => {
    const scaled = normalize({ x: 0, y: 0, z: 0, w: 4 });
    expect([scaled.x, scaled.y, scaled.z, scaled.w]).toEqual([0, 0, 0, 1]);
    const mid = slerp({ x: 0, y: 0, z: 0, w: 1 }, { x: 0, y: 0, z: Math.SQRT1_2, w: Math.SQRT1_2 }, .5);
    expect(mid.z).toBeCloseTo(Math.sin(Math.PI / 8), 6);
    expect(mid.w).toBeCloseTo(Math.cos(Math.PI / 8), 6);
  });

  it('integrates a known rotation, interpolates, and refuses to bridge gaps', () => {
    const imu: Imu = {
      t: new Float64Array([0, 100, 200, 900]),
      ax: new Float32Array(4), ay: new Float32Array([1, 1, 1, 1]), az: new Float32Array(4),
      gx: new Float32Array(4), gy: new Float32Array(4), gz: new Float32Array([0, Math.PI / 2, Math.PI / 2, 0]),
    };
    const m = manifest({ calibration: { acceleration_counts_per_g: 1, gyroscope_counts_per_rad_s: 1, viewer_from_sensor: [1, 0, 0, 0, 1, 0, 0, 0, 1] } });
    const points = reconstructOrientation(imu, m)!;
    expect(points[2].z).toBeGreaterThan(.13);
    expect(interpolateOrientation(points, 150)).not.toBeNull();
    expect(interpolateOrientation(points, 500)).toBeNull();
    expect(reconstructOrientation(imu, { ...m, calibration: undefined })).toBeNull();
  });

  it('converts degrees per second exactly once at the telemetry boundary', () => {
    const seconds = [0, 100, 200, 300];
    const degrees = new Float32Array([0, 45, 90, 90]);
    const radians = new Float32Array([0, Math.PI / 4, Math.PI / 2, Math.PI / 2]);
    const base = {
      ax: new Float32Array(4), ay: new Float32Array([0, 0, 0, 0]), az: new Float32Array(4),
      gx: new Float32Array(4), gy: new Float32Array(4),
    };
    const degImu: Imu = { t: new Float64Array(seconds), ...base, gz: degrees };
    const radImu: Imu = { t: new Float64Array(seconds), ...base, gz: radians };
    const deg = reconstructOrientation(degImu, manifest({ units: { acceleration: 'g', gyroscope: 'deg/s' } }))!;
    const rad = reconstructOrientation(radImu, manifest())!;
    expect(deg).toHaveLength(rad.length);
    deg.forEach((point, i) => expect(point.z).toBeCloseTo(rad[i].z, 6));
  });
});
