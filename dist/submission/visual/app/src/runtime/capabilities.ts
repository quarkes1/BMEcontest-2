export type Capabilities = {
  inference: boolean;
  service?: string;
  runKey?: string;
  /** 推理不可用的原因：本地文件方式打开，或无法连接服务。 */
  reason?: 'file-protocol' | 'unreachable';
};

/** 探测可选本地推理服务 GET ./api/health；服务缺失时静态演示仍可使用，返回 inference: false 而不抛出异常。 */
export async function probeCapabilities(timeoutMs = 1500): Promise<Capabilities> {
  // 通过 file:// 打开时受跨域限制，不允许请求本地服务。
  // 直接返回不可用，避免触发控制台错误。
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
