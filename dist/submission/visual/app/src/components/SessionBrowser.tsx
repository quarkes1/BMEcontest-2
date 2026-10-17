import type { Range } from '../data/types';
import { fmtTime } from '../data/format';

type Props = {
  label: string;
  demo: boolean;
  sessions: string[];
  sessionId: string;
  onSession: (sessionId: string) => void;
  bounds: Range | null;
};

/** Compact dataset/session selector that lives in the Monitor header workspace. */
export default function SessionBrowser({ label, demo, sessions, sessionId, onSession, bounds }: Props) {
  return (
    <div className="session-browser">
      <span className="dataset-label" title={label}>{demo ? 'Session workspace · demo data' : label}</span>
      <label className="session-picker">
        <span>Session</span>
        <select value={sessionId} onChange={e => onSession(e.target.value)} disabled={sessions.length < 2} aria-label="Session">
          {sessions.map(id => <option key={id} value={id}>{id}</option>)}
        </select>
      </label>
      {bounds && <span className="session-range mono">{fmtTime(bounds.start_ms)} ─────────── {fmtTime(bounds.end_ms)}</span>}
    </div>
  );
}
