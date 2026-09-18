import { useEffect, useMemo, useRef, useState } from 'react';
import type { Dataset, Event, Range } from './data/types';
import { boundsFor, sessionIds } from './data/prediction';
import { demoMotion, demoPrediction } from './data/demo';
import { reconstructOrientation } from './motion/orientation';
import { probeCapabilities, type Capabilities } from './runtime/capabilities';
import TopNav, { type Page } from './components/TopNav';
import SessionBrowser from './components/SessionBrowser';
import DataLoader from './components/DataLoader';
import MonitorPage from './pages/MonitorPage';
import EventsPage from './pages/EventsPage';
import ModelPage from './pages/ModelPage';

const DEMO_SESSION = demoPrediction.timeline!.session_ids[0];
const demoDataset: Dataset = { prediction: demoPrediction, motions: { [demoMotion.manifest.session_id]: demoMotion }, label: 'Demo data', demo: true };
const demoSelection: Range = { start_ms: demoPrediction.events[0].start_ms + 60_000, end_ms: demoPrediction.events[0].start_ms + 72_400 };

/**
 * Application shell: page routing, the loaded dataset (one prediction plus per-session
 * motion telemetry), the selected session/interval, and the single playback clock that
 * drives Timeline, Raw IMU, Orientation, and Motion Replay.
 */
export default function App() {
  const [page, setPage] = useState<Page>('Monitor');
  const [dataset, setDataset] = useState<Dataset>(demoDataset);
  const [sessionId, setSessionId] = useState(DEMO_SESSION);
  const [selection, setSelection] = useState<Range | null>(demoSelection);
  const [playhead, setPlayhead] = useState<number | null>(demoSelection.start_ms);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);

  useEffect(() => { probeCapabilities().then(setCapabilities); }, []);

  const sessions = useMemo(() => sessionIds(dataset.prediction, dataset.motions), [dataset]);
  const motion = dataset.motions[sessionId] ?? null;
  const orientation = useMemo(() => motion ? reconstructOrientation(motion.imu, motion.manifest) : null, [motion]);
  const bounds = useMemo(() => boundsFor(dataset.prediction, sessionId), [dataset.prediction, sessionId]);

  const playheadRef = useRef<number | null>(playhead);
  playheadRef.current = playhead;
  useEffect(() => {
    if (!playing || !selection || !orientation) return;
    let frame = 0, last = 0;
    const tick = (now: number) => {
      if (last) {
        const next = (playheadRef.current ?? selection.start_ms) + (now - last) * speed;
        if (next >= selection.end_ms) { setPlayhead(selection.end_ms); setPlaying(false); return; }
        setPlayhead(next);
      }
      last = now;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, selection, speed, orientation]);

  function select(range: Range | null) {
    setSelection(range);
    setPlaying(false);
    setPlayhead(range?.start_ms ?? null);
  }

  function applyDataset(next: Dataset, nextSessionId?: string) {
    setDataset(next);
    const ids = sessionIds(next.prediction, next.motions);
    setSessionId(nextSessionId && ids.includes(nextSessionId) ? nextSessionId : ids[0] || '');
    select(null);
    setPage('Monitor');
  }

  function openEvent(event: Event) {
    setSessionId(event.session_id);
    select(event);
    setPage('Monitor');
  }

  function switchSession(nextSessionId: string) {
    setSessionId(nextSessionId);
    select(null);
  }

  function resetDemo() {
    setDataset(demoDataset);
    setSessionId(DEMO_SESSION);
    select(demoSelection);
    setPage('Monitor');
    setError('');
    setStatus('');
  }

  function togglePlay() {
    if (!selection || !orientation) return;
    if (playhead === selection.end_ms) setPlayhead(selection.start_ms);
    setPlaying(!playing);
  }

  return (
    <div className="app">
      <TopNav page={page} onPage={setPage} demo={dataset.demo} />
      {error && <div role="alert" className="error-banner">{error}<button onClick={() => setError('')}>Dismiss</button></div>}
      {status && <div className="status-line" title={status}>{status}</div>}
      <main className={page === 'Monitor' ? 'monitor-workspace' : undefined}>
        <div className="data-bar">
          <SessionBrowser label={dataset.label} demo={dataset.demo} sessions={sessions} sessionId={sessionId} onSession={switchSession} bounds={bounds} />
          <DataLoader capabilities={capabilities} prediction={dataset.prediction} onDataset={applyDataset} onError={setError} onStatus={setStatus} onResetDemo={resetDemo} />
        </div>
        {page === 'Monitor' && (
          <MonitorPage
            demo={dataset.demo}
            sessionId={sessionId}
            prediction={dataset.prediction}
            motion={motion}
            orientation={orientation}
            selection={selection}
            playhead={playhead}
            playing={playing}
            speed={speed}
            onSelect={select}
            onTogglePlay={togglePlay}
            onSeek={time => { setPlaying(false); setPlayhead(time); }}
            onSpeed={setSpeed}
          />
        )}
        {page === 'Events' && <EventsPage prediction={dataset.prediction} demo={dataset.demo} onOpen={openEvent} />}
        {page === 'Model' && <ModelPage prediction={dataset.prediction} />}
      </main>
    </div>
  );
}
