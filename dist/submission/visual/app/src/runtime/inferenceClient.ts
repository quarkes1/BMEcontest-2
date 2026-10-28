import { parsePrediction } from '../data/prediction';
import { parseMotion } from '../data/telemetry';
import type { MotionData, Prediction } from '../data/types';

export type AnalyzeOutcome = {
  prediction: Prediction;
  motions: Record<string, MotionData>;
  warnings: string[];
};

type SessionArtifact = {
  session_id: string;
  source_name?: string;
  sample_count?: number;
  motion?: { manifest: unknown; bin_url: string };
};

/** 本地推理服务客户端，对应 start.bat 或 serve.py 入口；仅传输原始 TXT 并接收公开预测和运动数据，浏览器不实现模型。 */
async function ensureOk(response: Response, fallback: string): Promise<any> {
  if (response.ok) return response.json();
  let detail = fallback;
  try {
    const body = await response.json();
    if (body?.error) detail = String(body.error);
  } catch {
    /** 保留默认回退值。 */
  }
  throw new Error(detail);
}

async function bridgeFetch(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (error) {
    if (error instanceof TypeError) throw new Error('NetworkError：本地推理服务连接中断。请确认 start.bat 窗口仍在运行，然后重试。');
    throw error;
  }
}

export async function uploadSessionFile(file: File, relativePath: string | null): Promise<string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/octet-stream', 'X-Session-Name': file.name };
  if (relativePath) headers['X-Relative-Path'] = relativePath.slice(0, 400);
  const response = await bridgeFetch('./api/upload', { method: 'POST', headers, body: file });
  const body = await ensureOk(response, `Upload failed for ${file.name}.`);
  if (!body?.token) throw new Error(`Upload failed for ${file.name}.`);
  return String(body.token);
}

export async function analyzeTokens(tokens: string[], sessionId?: string): Promise<AnalyzeOutcome> {
  const response = await bridgeFetch('./api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tokens, session_id: sessionId ?? null }),
  });
  const body = await ensureOk(response, 'Local inference failed.');
  const prediction = parsePrediction(body.prediction);
  const warnings: string[] = Array.isArray(body.warnings) ? body.warnings.map(String) : [];
  const motions: Record<string, MotionData> = {};
  for (const session of (body.sessions || []) as SessionArtifact[]) {
    if (!session?.motion?.bin_url) continue;
    const bin = await bridgeFetch(`.${session.motion.bin_url}`);
    if (!bin.ok) { warnings.push(`Telemetry unavailable for ${session.session_id}.`); continue; }
    motions[session.session_id] = parseMotion(session.motion.manifest, await bin.arrayBuffer());
  }
  return { prediction, motions, warnings };
}

export type AnalyzeProgress = (phase: 'preparing' | 'running' | 'timeline', done: number, total: number) => void;

export function mergeOutcomes(outcomes: AnalyzeOutcome[]): AnalyzeOutcome {
  if (!outcomes.length) throw new Error('No session analysis succeeded.');
  const first = outcomes[0].prediction;
  const warnings = [...new Set(outcomes.flatMap(o => o.warnings))];
  const events = outcomes.flatMap(o => o.prediction.events).map((event, id) => ({ ...event, id }));
  const input = { ...first.input, source: `${outcomes.length} sessions (local upload)`,
    duration_seconds: outcomes.reduce((sum, o) => sum + o.prediction.input.duration_seconds, 0) };
  const duration = input.duration_seconds;
  const coverage = duration ? outcomes.reduce((sum, o) => sum + o.prediction.diagnostics.coverage * o.prediction.input.duration_seconds, 0) / duration : 0;
  const prediction: Prediction = {
    ...first, input, events,
    diagnostics: { ...first.diagnostics, coverage, warnings: [...new Set([...outcomes.flatMap(o => o.prediction.diagnostics.warnings), ...warnings])] },
  };
  if (outcomes.some(o => o.prediction.candidates)) prediction.candidates = outcomes.flatMap(o => o.prediction.candidates || []);
  if (outcomes.some(o => o.prediction.gaps)) prediction.gaps = outcomes.flatMap(o => o.prediction.gaps || []);
  if (outcomes.some(o => o.prediction.timeline)) prediction.timeline = {
    session_ids: [...new Set(outcomes.flatMap(o => o.prediction.timeline?.session_ids || []))],
    macro_windows: outcomes.reduce((sum, o) => sum + (o.prediction.timeline?.macro_windows || 0), 0),
    micro_windows: outcomes.reduce((sum, o) => sum + (o.prediction.timeline?.micro_windows || 0), 0),
    series: outcomes.flatMap(o => o.prediction.timeline?.series || []),
  };
  return { prediction: parsePrediction(prediction), motions: Object.assign({}, ...outcomes.map(o => o.motions)), warnings };
}

/** 上传文件或文件夹选择器中的原始 TXT，运行统一推理。 */
export async function analyzeRawSelection(files: File[], folder: boolean, onProgress?: AnalyzeProgress): Promise<AnalyzeOutcome> {
  const total = files.length;
  const tokens: string[] = [];
  for (let i = 0; i < files.length; i++) {
    onProgress?.('preparing', i, total);
    const file = files[i];
    // 每次独立分析会发现一个仅包含单文件的目录。
    // 多文件选择时为各文件建立独立目录，避免合并结果的会话标识冲突。
    const relative = total > 1 ? `${file.name.replace(/\.txt$/i, '')}-${i + 1}/${file.name}`
      : folder ? (file as File & { webkitRelativePath?: string }).webkitRelativePath || null : null;
    tokens.push(await uploadSessionFile(file, relative));
  }
  const outcomes: AnalyzeOutcome[] = [];
  const failures: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    onProgress?.('running', i, total);
    try {
      outcomes.push(await analyzeTokens([tokens[i]]));
    } catch (error) {
      failures.push(`${files[i].name}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  if (!outcomes.length) throw new Error(failures[0] || 'All sessions failed.');
  const outcome = mergeOutcomes(outcomes);
  outcome.warnings.push(...failures);
  onProgress?.('timeline', total, total);
  return outcome;
}
