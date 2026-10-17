export type Capabilities = {
  inference: boolean;
  service?: string;
  runKey?: string;
  /** Why inference is unavailable: opened as a local file, or no service reachable. */
  reason?: 'file-protocol' | 'unreachable';
};

/**
 * Probe the optional local inference bridge (`GET ./api/health`). The static/demo
 * application must keep working when the bridge is absent: failures resolve to
 * `{ inference: false }` instead of throwing.
 */
export async function probeCapabilities(timeoutMs = 1500): Promise<Capabilities> {
  // Under file:// no fetch is permitted (CORS), so the bridge is known to be absent;
  // short-circuit instead of provoking a console error.
  if (typeof location !== 'undefined' && location.protocol === 'file:') return { inference: false, reason: 'file-protocol' };
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch('./api/health', { signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) return { inference: false, reason: 'unreachable' };
    const health = await response.json();
    if (health?.status !== 'ok') return { inference: false, reason: 'unreachable' };
    return { inference: true, service: health.service, runKey: health.run_key };
  } catch {
    return { inference: false, reason: 'unreachable' };
  }
}
