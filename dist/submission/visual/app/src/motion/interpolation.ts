import { slerp, type Quaternion, type QuaternionPoint } from './quaternion';

/** 按回放时间轴确定性查询球面线性插值；超出采样范围、跨片段边界或采样缺口时返回空值，不虚构缺失运动。 */
export function interpolateOrientation(points: QuaternionPoint[], time: number): Quaternion | null {
  if (!points.length || time < points[0].t || time > points[points.length - 1].t) return null;
  let lo = 0, hi = points.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (points[mid].t < time) lo = mid + 1; else hi = mid;
  }
  if (lo === 0) return points[0];
  const a = points[lo - 1], b = points[lo];
  if (a.segment !== b.segment || b.t - a.t > 500) return null;
  return slerp(a, b, (time - a.t) / (b.t - a.t));
}
