import type { Event } from '../../data/types';
import { duration, fmtTime } from '../../data/format';

type Props = { selectedEvent: Event | undefined };

export default function EventContext({ selectedEvent }: Props) {
  return (
    <section className="panel context">
      <h2>Event context</h2>
      {selectedEvent ? (
        <dl>
          <dt>Start time</dt><dd className="mono">{fmtTime(selectedEvent.start_ms, true)}</dd>
          <dt>End time</dt><dd className="mono">{fmtTime(selectedEvent.end_ms, true)}</dd>
          <dt>Duration</dt><dd>{duration(selectedEvent.end_ms - selectedEvent.start_ms)}</dd>
          <dt>Event score</dt><dd>{selectedEvent.confidence.toFixed(3)}</dd>
          <dt>Session</dt><dd>{selectedEvent.session_id}</dd>
        </dl>
      ) : <div className="panel-empty">Select an eating event to inspect its context.</div>}
    </section>
  );
}
