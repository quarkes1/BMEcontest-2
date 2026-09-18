import { useMemo } from 'react';
import type { MotionData, Prediction, Range } from '../data/types';
import type { QuaternionPoint } from '../motion/quaternion';
import StatusHeader from '../components/StatusHeader';
import Timeline from '../components/timeline/Timeline';
import MotionReplay from '../components/motion/MotionReplay';
import SelectedInterval from '../components/inspector/SelectedInterval';
import RawImuChart from '../components/inspector/RawImuChart';
import EventContext from '../components/inspector/EventContext';

type Props = {
  demo: boolean;
  sessionId: string;
  prediction: Prediction;
  motion: MotionData | null;
  orientation: QuaternionPoint[] | null;
  selection: Range | null;
  playhead: number | null;
  playing: boolean;
  speed: number;
  onSelect: (range: Range | null) => void;
  onTogglePlay: () => void;
  onSeek: (time: number) => void;
  onSpeed: (speed: number) => void;
};

const mean = (values: number[]) => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;

export default function MonitorPage({ demo, sessionId, prediction, motion, orientation, selection, playhead, playing, speed, onSelect, onTogglePlay, onSeek, onSpeed }: Props) {
  const events = prediction.events.filter(e => e.session_id === sessionId);
  const candidates = (prediction.candidates || []).filter(c => c.session_id === sessionId);
  const selectedEvent = selection ? events.find(e => e.start_ms < selection.end_ms && e.end_ms > selection.start_ms) : undefined;
  const selectedCandidate = selection ? candidates.find(c => c.start_ms < selection.end_ms && c.end_ms > selection.start_ms) : undefined;
  const selectedImu = useMemo(() => {
    if (!selection || !motion) return null;
    const a = motion.imu, acc: number[] = [], gyro: number[] = [];
    for (let i = 0; i < a.t.length; i++) {
      if (a.t[i] < selection.start_ms) continue;
      if (a.t[i] > selection.end_ms) break;
      acc.push(Math.hypot(a.ax[i], a.ay[i], a.az[i]));
      gyro.push(Math.hypot(a.gx[i], a.gy[i], a.gz[i]));
    }
    return { acc: mean(acc), gyro: mean(gyro) };
  }, [selection, motion]);
  return (
    <>
      <StatusHeader demo={demo} sessionId={sessionId} prediction={prediction} motion={motion} selectedEvent={selectedEvent} />
      <div className="monitor-grid">
        <Timeline prediction={prediction} sessionId={sessionId} imu={motion?.imu} selection={selection} onSelect={onSelect} playhead={playhead} />
        <MotionReplay orientation={orientation} manifest={motion?.manifest} selection={selection} playhead={playhead} playing={playing} speed={speed} onPlay={onTogglePlay} onSeek={onSeek} onSpeed={onSpeed} />
      </div>
      <div className="detail-grid">
        <SelectedInterval selection={selection} motion={motion} selectedImu={selectedImu} selectedCandidate={selectedCandidate} selectedEvent={selectedEvent} />
        <RawImuChart imu={motion?.imu} selection={selection} playhead={playhead} orientation={orientation} />
        <EventContext selectedEvent={selectedEvent} />
      </div>
    </>
  );
}
