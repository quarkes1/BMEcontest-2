import type { Event, MotionData, Prediction } from '../data/types';
import { duration, fmtTime } from '../data/format';

type Props = {
  demo: boolean;
  sessionId: string;
  prediction: Prediction;
  motion: MotionData | null;
  selectedEvent: Event | undefined;
};

export default function StatusHeader({ demo, sessionId, prediction, motion, selectedEvent }: Props) {
  const events = prediction.events.filter(e => e.session_id === sessionId);
  const eventTotal = events.reduce((sum, e) => sum + e.duration_s, 0);
  const hz = motion?.imu.t.length
    ? `${Math.round(motion.imu.t.length / Math.max(1, (motion.imu.t[motion.imu.t.length - 1] - motion.imu.t[0]) / 1000))} Hz`
    : '—';
  return (
    <section className="panel status-header">
      <div><strong>{demo ? 'Demo monitoring' : 'Session review'}</strong><span>{sessionId || 'No session'}</span></div>
      <div className="status-main">
        <span className="status-indicator" />
        <div>
          <strong>{selectedEvent ? 'Eating event' : 'No eating event selected'}</strong>
          <span>{selectedEvent ? `${fmtTime(selectedEvent.start_ms, true)} – ${fmtTime(selectedEvent.end_ms, true)}` : 'Choose an interval on the timeline'}</span>
        </div>
        {selectedEvent && <b className="mono">{duration(selectedEvent.end_ms - selectedEvent.start_ms)}</b>}
      </div>
      <div className="status-metric"><small>IMU</small><strong>{hz}</strong></div>
      <div className="status-metric"><small>Coverage</small><strong>{(prediction.diagnostics.coverage * 100).toFixed(1)}%</strong></div>
      <div className="status-metric"><small>Events in session</small><strong>{events.length}</strong></div>
      <div className="status-metric"><small>Total duration</small><strong>{duration(eventTotal * 1000)}</strong></div>
    </section>
  );
}
