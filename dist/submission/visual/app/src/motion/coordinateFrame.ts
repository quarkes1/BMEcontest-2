import type { MotionManifest } from '../data/types';
import { fromAxisAngle, identity, type Quaternion } from './quaternion';

/** A valid sensor→viewer frame is a proper rotation: orthonormal rows, determinant +1. */
export function validMapping(matrix: number[]): boolean {
  if (!Array.isArray(matrix) || matrix.length !== 9 || matrix.some(v => !Number.isFinite(v))) return false;
  const rows = [matrix.slice(0, 3), matrix.slice(3, 6), matrix.slice(6, 9)];
  const determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
    - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
    + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  return rows.every(r => Math.abs(Math.hypot(...r) - 1) < 1e-5)
    && Math.abs(rows[0].reduce((s, x, i) => s + x * rows[1][i], 0)) < 1e-5
    && Math.abs(rows[0].reduce((s, x, i) => s + x * rows[2][i], 0)) < 1e-5
    && Math.abs(rows[1].reduce((s, x, i) => s + x * rows[2][i], 0)) < 1e-5
    && Math.abs(determinant - 1) < 1e-5;
}

/**
 * Gravity-only replay requires calibrated acceleration and a proper signed mapping.
 * Unknown gyro scale is allowed, but gyro integration must then be disabled.
 */
export function canOrient(m: MotionManifest): boolean {
  const c = m.calibration;
  return !!c && validMapping(c.viewer_from_sensor)
    && (m.units.acceleration === 'g' || c.acceleration_counts_per_g > 0);
}

export function mapVector(v: [number, number, number], m: number[]): [number, number, number] {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

/** Quaternion that tilts the viewer "up" axis (Y) onto the measured gravity direction. */
export function gravityTilt(up: [number, number, number]): Quaternion {
  const n = Math.hypot(...up);
  if (n < 1e-8) return identity;
  const u = up.map(x => x / n) as [number, number, number];
  const axis: [number, number, number] = [-u[2], 0, u[0]];
  const dot = Math.max(-1, Math.min(1, u[1]));
  return fromAxisAngle(...axis, Math.acos(dot));
}
