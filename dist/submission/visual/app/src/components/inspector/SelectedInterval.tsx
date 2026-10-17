import type { Candidate, Event, MotionData, Range } from '../../data/types';
import { duration, fmtTime } from '../../data/format';

type Props = {
  selection: Range | null;
  motion: MotionData | null;
  selectedImu: { acc: number | null; gyro: number | null } | null;
  selectedCandidate: Candidate | undefined;
  selectedEvent: Event | undefined;
};

export default function SelectedInterval({ selection, motion, selectedImu, selectedCandidate, selectedEvent }: Props) {
  const accUnit = motion?.manifest.units.acceleration === 'g' ? 'g' : 'counts';
  const gyroUnit = motion?.manifest.units.gyroscope === 'raw_adc' ? 'counts' : motion?.manifest.units.gyroscope;
  return (
    <section className="panel inspector">
      <h2>Selected interval</h2>
      {selection && selection.end_ms > selection.start_ms ? (
        <>
          <div className="interval-time mono">◷ &nbsp; {fmtTime(selection.start_ms, true)} – {fmtTime(selection.end_ms, true)} <span>({duration(selection.end_ms - selection.start_ms)})</span></div>
          <dl>
            {selectedImu?.acc !== null && selectedImu?.acc !== undefined && <><dt>ACC magnitude (mean)</dt><dd>{selectedImu.acc.toFixed(2)} {accUnit}</dd></>}
            {selectedImu?.gyro !== null && selectedImu?.gyro !== undefined && <><dt>Angular velocity (mean)</dt><dd>{selectedImu.gyro.toFixed(2)} {gyroUnit}</dd></>}
            <dt>Candidate</dt><dd>{selectedCandidate ? (selectedCandidate.admitted ? 'Admitted' : 'Not admitted') : 'None'}</dd>
            {selectedCandidate && <><dt>Candidate score</dt><dd>{selectedCandidate.score.toFixed(3)}</dd></>}
            <dt>Model decision</dt><dd><span className={selectedEvent ? 'decision eating' : 'decision'}>{selectedEvent ? 'Eating event' : 'No event overlaps'}</span></dd>
          </dl>
        </>
      ) : <div className="panel-empty">Drag on the timeline or click an event.</div>}
    </section>
  );
}
