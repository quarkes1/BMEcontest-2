import type { Imu, MotionManifest } from '../data/types';
import { gravityTilt, canOrient, mapVector } from './coordinateFrame';
import { fromAxisAngle, identity, multiply, normalize, rotate, type QuaternionPoint } from './quaternion';

/**
 * Visualization-only orientation: a deterministic complementary gravity/gyro filter over
 * calibrated telemetry. It estimates orientation of the rigid forearm/wrist/hand/watch
 * body — never absolute position, wrist-joint articulation, or elbow motion. Runs longer
 * than the gap threshold start a new segment and are never bridged by interpolation.
 */
export function reconstructOrientation(imu: Imu, m: MotionManifest): QuaternionPoint[] | null {
  if (!canOrient(m)) return null;
  const c = m.calibration!, basis = c.viewer_from_sensor, points: QuaternionPoint[] = [];
  let q = identity, last = -Infinity, lastStored = -Infinity, segment = 0, regressed = false;
  const gyroAvailable = m.units.gyroscope !== 'raw_adc' || (c.gyroscope_counts_per_rad_s ?? 0) > 0;
  for (let i = 0; i < imu.t.length; i++) {
    const time = imu.t[i], dt = (time - last) / 1000;
    if (!Number.isFinite(time)) continue;
    if (time < last) { regressed = true; continue; }
    if (time === last) continue;
    const accFactor = m.units.acceleration === 'g' ? 1 : 1 / c.acceleration_counts_per_g;
    const gyroFactor = !gyroAvailable ? 0 : m.units.gyroscope === 'rad/s' ? 1 : m.units.gyroscope === 'deg/s' ? Math.PI / 180 : 1 / c.gyroscope_counts_per_rad_s!;
    const acc = mapVector([imu.ax[i] * accFactor, imu.ay[i] * accFactor, imu.az[i] * accFactor], basis);
    const gyro = mapVector([imu.gx[i] * gyroFactor, imu.gy[i] * gyroFactor, imu.gz[i] * gyroFactor], basis);
    if (!acc.every(Number.isFinite) || !gyro.every(Number.isFinite)) { last = time; continue; }
    const norm = Math.hypot(...acc);
    if (!Number.isFinite(dt) || dt > .5 || regressed) {
      if (last !== -Infinity) segment++;
      q = norm > .4 && norm < 1.6 ? gravityTilt(acc) : identity;
    } else if (!gyroAvailable) {
      if (norm > .4 && norm < 1.6) q = gravityTilt(acc);
    } else if (dt > 0) {
      const omega = Math.hypot(...gyro);
      q = multiply(q, fromAxisAngle(...gyro, omega * dt));
      if (norm > .8 && norm < 1.2) {
        const up = rotate(q, [acc[0] / norm, acc[1] / norm, acc[2] / norm]);
        const error: [number, number, number] = [-up[2], 0, up[0]];
        const e = Math.hypot(...error);
        if (e > 1e-8) q = multiply(fromAxisAngle(...error, Math.min(.025, e * .025)), q);
      }
    }
    last = time;
    regressed = false;
    if (time - lastStored >= 40 || points.length === 0) {
      points.push({ t: time, ...normalize(q), segment });
      lastStored = time;
    }
  }
  return points;
}
