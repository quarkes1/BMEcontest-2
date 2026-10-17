import type { Event, Prediction } from '../data/types';
import { duration, fmtTime } from '../data/format';

type Props = { prediction: Prediction; demo: boolean; onOpen: (event: Event) => void };

/** All detected events across the loaded dataset; a record opens its session on Monitor. */
export default function EventsPage({ prediction, demo, onOpen }: Props) {
  return (
    <section className="panel events-page">
      <div className="page-title"><div><h1>Events</h1><p>Canonical model output · select a record to inspect it on Monitor.</p></div><span>{demo ? 'DEMO DATA' : 'REAL PREDICTION'}</span></div>
      {prediction.events.length ? prediction.events.map(e => (
        <button key={`${e.session_id}-${e.id}`} className="event-record" onClick={() => onOpen(e)}>
          <span className="record-date">{new Date(e.start_ms).toLocaleDateString('en-CA')}</span>
          <strong className="mono">{fmtTime(e.start_ms, true)} – {fmtTime(e.end_ms, true)}</strong>
          <span>{duration(e.end_ms - e.start_ms)}</span>
          <span>Event score {e.confidence.toFixed(3)}</span>
          <span className="record-session" title={e.session_id}>{e.session_id}</span>
          <b>Inspect →</b>
        </button>
      )) : <div className="panel-empty">No eating events in this prediction.</div>}
    </section>
  );
}
