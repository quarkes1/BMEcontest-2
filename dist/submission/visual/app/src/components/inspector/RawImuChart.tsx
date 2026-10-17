import { useMemo, useState } from 'react';
import type { Imu, Range } from '../../data/types';
import type { QuaternionPoint } from '../../motion/quaternion';

type Props = { imu?: Imu; selection: Range | null; playhead: number | null; orientation: QuaternionPoint[] | null };

/** Raw sensor chart for the selected interval: raw ACC / raw GYRO / derived orientation. */
export default function RawImuChart({ imu, selection, playhead, orientation }: Props) {
  const [tab, setTab] = useState<'ACC' | 'GYRO' | 'Orientation'>('ACC');
  const plot = useMemo(() => {
    if (!imu || !selection) return null;
    const times = tab === 'Orientation' && orientation ? orientation.map(p => p.t) : imu.t;
    const series = tab === 'Orientation' && orientation ? [orientation.map(p => p.x), orientation.map(p => p.y), orientation.map(p => p.z), orientation.map(p => p.w)] : tab === 'ACC' ? [imu.ax, imu.ay, imu.az] : [imu.gx, imu.gy, imu.gz];
    let first = 0; while (first < times.length && times[first] < selection.start_ms) first++;
    let end = first; while (end < times.length && times[end] <= selection.end_ms) end++;
    if (end <= first) return null;
    let max = tab === 'Orientation' ? 1 : .1;
    if (tab !== 'Orientation') for (let i = first; i < end; i++) for (const channel of series) max = Math.max(max, Math.abs(channel[i]));
    const step = Math.max(1, Math.floor((end - first) / 1000));
    const paths = series.map(channel => {
      let path = '';
      for (let i = first; i < end; i += step) {
        const x = 35 + (times[i] - selection.start_ms) / Math.max(1, selection.end_ms - selection.start_ms) * 557;
        const y = 82.5 - channel[i] / max * 55;
        let gap = false;
        for (let j = Math.max(first + 1, i - step + 1); j <= i; j++) if (times[j] - times[j - 1] > 500) { gap = true; break; }
        path += `${!path || gap ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)} `;
      }
      return path;
    });
    return { paths, max };
  }, [imu, selection, tab, orientation]);
  const colors = ['#d56f6a', '#459775', '#5d82d9', '#9a79c9'];
  return (
    <section className="panel raw-panel">
      <div className="panel-title">
        <h2>{tab === 'Orientation' ? 'Orientation' : 'Raw IMU'} <span>(selected interval)</span></h2>
        <div className="axis-legend">
          <span className="axis-x">●</span>{tab === 'Orientation' ? 'qx' : 'X'} <span className="axis-y">●</span>{tab === 'Orientation' ? 'qy' : 'Y'} <span className="axis-z">●</span>{tab === 'Orientation' ? 'qz' : 'Z'}{tab === 'Orientation' && <> <span style={{ color: '#9a79c9' }}>●</span>qw</>}
        </div>
      </div>
      <div className="segmented raw-tabs">
        {(['ACC', 'GYRO', 'Orientation'] as const).map(x => <button key={x} className={tab === x ? 'active' : ''} disabled={x === 'Orientation' && !orientation} onClick={() => setTab(x)}>{x}</button>)}
      </div>
      {selection && imu ? (
        <div className="raw-chart-wrap">
          <svg viewBox="0 0 600 165" preserveAspectRatio="none" role="img" aria-label={`${tab} channels for selected interval`}>
            {[28, 64, 101, 137].map(y => <line key={y} x1="35" y1={y} x2="592" y2={y} stroke="#e7e8e9" strokeWidth="1" />)}
            {plot?.paths.map((path, i) => <path key={i} d={path} fill="none" stroke={colors[i]} strokeWidth="1.25" vectorEffect="non-scaling-stroke" />)}
            <text x="2" y="29" fill="#7a7e88" fontSize="11">{plot?.max.toFixed(1) || '—'}</text>
            <text x="18" y="86" fill="#7a7e88" fontSize="11">0</text>
            <text x="2" y="141" fill="#7a7e88" fontSize="11">{plot ? (-plot.max).toFixed(1) : '—'}</text>
            {playhead !== null && <line x1={35 + (playhead - selection.start_ms) / Math.max(1, selection.end_ms - selection.start_ms) * 557} x2={35 + (playhead - selection.start_ms) / Math.max(1, selection.end_ms - selection.start_ms) * 557} y1="8" y2="151" stroke="#303d4d" strokeWidth="1" vectorEffect="non-scaling-stroke" />}
          </svg>
        </div>
      ) : <div className="panel-empty">Load motion telemetry and select an interval to inspect raw channels.</div>}
      <p className="unit-note">{tab === 'Orientation' ? 'Quaternion · derived from calibrated IMU' : tab === 'GYRO' ? 'GYRO' : 'ACC'} · Data gaps are not interpolated</p>
    </section>
  );
}
