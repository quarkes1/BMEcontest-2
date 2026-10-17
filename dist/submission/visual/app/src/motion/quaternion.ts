export type Quaternion = { x: number; y: number; z: number; w: number };
/** A quaternion sample on the replay timeline; `segment` separates gap-broken runs. */
export type QuaternionPoint = Quaternion & { t: number; segment: number };

export const identity: Quaternion = { x: 0, y: 0, z: 0, w: 1 };

export function normalize(q: Quaternion): Quaternion {
  const n = Math.hypot(q.x, q.y, q.z, q.w);
  return n > 1e-12 && Number.isFinite(n) ? { x: q.x / n, y: q.y / n, z: q.z / n, w: q.w / n } : identity;
}

function rawMultiply(a: Quaternion, b: Quaternion): Quaternion {
  return {
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

export function multiply(a: Quaternion, b: Quaternion): Quaternion {
  return normalize(rawMultiply(a, b));
}

export function fromAxisAngle(x: number, y: number, z: number, angle: number): Quaternion {
  const n = Math.hypot(x, y, z);
  if (n < 1e-12) return identity;
  const s = Math.sin(angle / 2) / n;
  return normalize({ x: x * s, y: y * s, z: z * s, w: Math.cos(angle / 2) });
}

export function rotate(q: Quaternion, v: [number, number, number]): [number, number, number] {
  const [x, y, z] = v;
  const qv = { x, y, z, w: 0 };
  const inv = { x: -q.x, y: -q.y, z: -q.z, w: q.w };
  const a = rawMultiply(rawMultiply(q, qv), inv);
  return [a.x, a.y, a.z];
}

export function slerp(a: Quaternion, b: Quaternion, alpha: number): Quaternion {
  let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  let bb = b;
  if (dot < 0) {
    dot = -dot;
    bb = { x: -b.x, y: -b.y, z: -b.z, w: -b.w };
  }
  if (dot > 0.9995) {
    return normalize({ x: a.x + (bb.x - a.x) * alpha, y: a.y + (bb.y - a.y) * alpha, z: a.z + (bb.z - a.z) * alpha, w: a.w + (bb.w - a.w) * alpha });
  }
  const theta = Math.acos(Math.max(-1, Math.min(1, dot)));
  const s = Math.sin(theta);
  return normalize({
    x: (a.x * Math.sin((1 - alpha) * theta) + bb.x * Math.sin(alpha * theta)) / s,
    y: (a.y * Math.sin((1 - alpha) * theta) + bb.y * Math.sin(alpha * theta)) / s,
    z: (a.z * Math.sin((1 - alpha) * theta) + bb.z * Math.sin(alpha * theta)) / s,
    w: (a.w * Math.sin((1 - alpha) * theta) + bb.w * Math.sin(alpha * theta)) / s,
  });
}
