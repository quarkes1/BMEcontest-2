import { useRef } from 'react';
import { isSessionTxt, loadExistingArtifacts } from '../data/loader';
import type { Dataset, Prediction } from '../data/types';
import { analyzeRawSelection } from '../runtime/inferenceClient';
import type { Capabilities } from '../runtime/capabilities';

type Props = {
  capabilities: Capabilities | null;
  prediction: Prediction;
  onDataset: (dataset: Dataset, sessionId?: string) => void;
  onError: (message: string) => void;
  onStatus: (message: string) => void;
  onResetDemo: () => void;
};

/**
 * Raw TXT files and folders are the primary competition workflow; they are analyzed by
 * the local canonical-inference bridge. Loading already-produced artifacts stays
 * available for debugging, replay, and development.
 */
export default function DataLoader({ capabilities, prediction, onDataset, onError, onStatus, onResetDemo }: Props) {
  const txtInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const existingInput = useRef<HTMLInputElement>(null);
  const bridgeReady = capabilities?.inference === true;
  const unavailableHint = capabilities?.reason === 'file-protocol'
    ? 'Opened as a local file: raw TXT analysis needs the local inference service — double-click dist\\start.bat and use the page it opens. Demo and existing prediction files stay available here.'
    : 'Local inference service unavailable — start it with dist\\start.bat. Demo and existing prediction files are still available.';

  async function handleRaw(files: FileList | null, folder: boolean) {
    if (!files?.length) return;
    onError('');
    const all = Array.from(files);
    const sessions = all.filter(file => isSessionTxt(file.name));
    const ignored = all.length - sessions.length;
    if (!sessions.length) {
      onError(folder ? 'No collect_data*.txt session files found in this folder.' : 'Select collect_data*.txt session files.');
      return;
    }
    const started = Date.now();
    let progress = '';
    const timer = window.setInterval(() => {
      if (progress) onStatus(`${progress} · 已用 ${Math.floor((Date.now() - started) / 1000)} 秒`);
    }, 1000);
    try {
      const outcome = await analyzeRawSelection(sessions, folder, (phase, done, total) => {
        if (phase === 'preparing') progress = `正在上传 ${done + 1}/${total} 个会话`;
        else if (phase === 'running') progress = `正在分析 ${done + 1}/${total} 个会话`;
        else progress = `${total}/${total} 完成 · 正在加载时间轴`;
        onStatus(progress);
      });
      const warnings = [...outcome.warnings];
      if (ignored > 0) warnings.push(`${ignored} unrelated file(s) were ignored.`);
      const folderName = folder ? (sessions[0] as File & { webkitRelativePath?: string }).webkitRelativePath?.split('/')[0] : undefined;
      const dataset: Dataset = {
        prediction: outcome.prediction,
        motions: outcome.motions,
        label: folder ? (folderName || 'Selected folder') : `${sessions.length} session file(s)`,
        demo: false,
      };
      onStatus(warnings.length ? `Ready · ${sessions.length}/${sessions.length} 完成 · ${warnings.join(' ')}` : `Ready · ${sessions.length}/${sessions.length} 完成`);
      onDataset(dataset);
    } catch (error) {
      onStatus('');
      onError(error instanceof Error ? error.message : String(error));
    } finally {
      window.clearInterval(timer);
      if (txtInput.current) txtInput.current.value = '';
      if (folderInput.current) folderInput.current.value = '';
    }
  }

  async function handleExisting(files: FileList | null) {
    if (!files?.length) return;
    onError('');
    try {
      const { prediction: nextPrediction, motion } = await loadExistingArtifacts(Array.from(files), prediction);
      const base = nextPrediction ?? prediction;
      const motions: Dataset['motions'] = motion ? { [motion.manifest.session_id]: motion } : {};
      const dataset: Dataset = {
        prediction: base,
        motions,
        label: base.input.source.split(/[\\/]/).pop() || base.input.source,
        demo: false,
      };
      onDataset(dataset, motion?.manifest.session_id);
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error));
    } finally {
      if (existingInput.current) existingInput.current.value = '';
    }
  }

  return (
    <div className="data-loader">
      <input ref={txtInput} type="file" multiple accept=".txt,text/plain" hidden onChange={e => handleRaw(e.target.files, false)} />
      <input ref={folderInput} type="file" multiple hidden {...({ webkitdirectory: '' } as Record<string, string>)} onChange={e => handleRaw(e.target.files, true)} />
      <input ref={existingInput} type="file" multiple accept=".json,.bin" hidden onChange={e => handleExisting(e.target.files)} />
      <div className="loader-buttons">
        <button onClick={() => txtInput.current?.click()} disabled={!bridgeReady} title={bridgeReady ? 'Run canonical inference on collect_data*.txt files' : 'Local inference service unavailable'}>Select TXT files</button>
        <button onClick={() => folderInput.current?.click()} disabled={!bridgeReady} title={bridgeReady ? 'Analyze a folder of session files' : 'Local inference service unavailable'}>Select folder</button>
        <button className="secondary" onClick={() => existingInput.current?.click()}>Open existing prediction</button>
        <button className="secondary" onClick={onResetDemo}>Reset demo</button>
      </div>
      {!bridgeReady && <span className="loader-hint">{unavailableHint}</span>}
    </div>
  );
}
