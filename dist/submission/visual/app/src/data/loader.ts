import { parsePrediction } from './prediction';
import { parseMotion } from './telemetry';
import type { MotionManifest, Prediction } from './types';

/** 使用最小结构化输入，使文件分类逻辑可在浏览器外测试。 */
export type FileLike = { name: string; text(): Promise<string>; arrayBuffer(): Promise<ArrayBuffer> };

export type Selection = {
  predictionFiles: FileLike[];
  motionManifests: FileLike[];
  binaries: FileLike[];
  sessionTexts: FileLike[];
  ignored: string[];
};

/** 统一原始会话文件命名约定：collect_data*.txt。 */
export function isSessionTxt(name: string): boolean {
  return /^collect_data.*\.txt$/i.test(name);
}

export function classifySelection(files: FileLike[]): Selection {
  const selection: Selection = { predictionFiles: [], motionManifests: [], binaries: [], sessionTexts: [], ignored: [] };
  for (const file of files) {
    const lower = file.name.toLowerCase();
    if (lower.endsWith('.json')) {
      // 预测清单和运动清单都使用 .json 扩展名。
      // 调用方读取内容后区分类型，因此两组均保留全部 JSON 文件。
      selection.predictionFiles.push(file);
      selection.motionManifests.push(file);
    } else if (lower.endsWith('.bin')) {
      selection.binaries.push(file);
    } else if (isSessionTxt(file.name)) {
      selection.sessionTexts.push(file);
    } else {
      selection.ignored.push(file.name);
    }
  }
  return selection;
}

export type ExistingArtifacts = { prediction: Prediction | null; motion: { manifest: MotionManifest; imu: import('./types').Imu } | null };

/** 载入已有 prediction.json 及 motion.json、motion.bin；忽略无关文件，运动二进制必须与清单配套，且对应预测中已有会话。 */
export async function loadExistingArtifacts(files: FileLike[], current: Prediction | null): Promise<ExistingArtifacts> {
  const selection = classifySelection(files);
  let prediction: Prediction | null = null;
  let manifest: MotionManifest | null = null;
  let binary: FileLike | null = null;
  for (const file of selection.binaries) binary = file;
  for (const file of selection.motionManifests) {
    const json = JSON.parse(await file.text());
    if (json?.schema_version) prediction = parsePrediction(json);
    else if (json?.telemetry_version) manifest = json as MotionManifest;
  }
  if (!prediction && !manifest) throw new Error('Choose a prediction JSON or motion telemetry files.');
  let motion: ExistingArtifacts['motion'] = null;
  if (manifest) {
    if (!binary) throw new Error('motion.json requires its matching motion.bin in the same selection.');
    motion = parseMotion(manifest, await binary.arrayBuffer());
    const reference = prediction ?? current;
    const known = reference
      ? (reference.timeline?.session_ids || []).includes(manifest.session_id) || reference.events.some(e => e.session_id === manifest.session_id) || (reference.candidates || []).some(c => c.session_id === manifest.session_id) || (reference.gaps || []).some(g => g.session_id === manifest.session_id)
      : true;
    if (!known) throw new Error('Motion session ID does not match prediction.');
  }
  return { prediction, motion };
}
