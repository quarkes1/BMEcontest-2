import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createForearmModel } from './ForearmModel';
import type { QuaternionPoint } from '../../motion/quaternion';
import { interpolateOrientation } from '../../motion/interpolation';
import type { MotionManifest, Range } from '../../data/types';
import { duration } from '../../data/format';

type Props = { orientation: QuaternionPoint[] | null; manifest?: MotionManifest; selection: Range | null; playhead: number | null; playing: boolean; speed: number; onPlay: () => void; onSeek: (time: number) => void; onSpeed: (speed: number) => void };
export default function MotionReplay({ orientation, manifest, selection, playhead, playing, speed, onPlay, onSeek, onSpeed }: Props) {
  const mount = useRef<HTMLDivElement>(null), motion = useRef<THREE.Group | null>(null), camera = useRef<THREE.PerspectiveCamera | null>(null), controls = useRef<OrbitControls | null>(null);
  const [view, setView] = useState<'3D' | 'Side' | 'Top'>('3D'), [debug, setDebug] = useState(false);
  useEffect(() => {
    const container = mount.current!; const scene = new THREE.Scene(); scene.background = new THREE.Color('#eef0ef');
    const cam = new THREE.PerspectiveCamera(45,1,.1,100); cam.position.set(2.2,5.8,7.4); cam.lookAt(0,0,0); camera.current = cam;
    scene.add(new THREE.HemisphereLight(0xffffff,0xa0aaa6,1.4));
    const key = new THREE.DirectionalLight(0xffffff,2.5); key.position.set(-3,6,4); scene.add(key);
    const fill = new THREE.DirectionalLight(0xe8e8e5,.55); fill.position.set(4,1,-3); scene.add(fill);
    const { motionGroup,dispose } = createForearmModel(); scene.add(motionGroup); motion.current = motionGroup;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false }); renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
    container.appendChild(renderer.domElement);
    const orbit = new OrbitControls(cam, renderer.domElement); orbit.target.set(0,0,0); orbit.enablePan = false; orbit.minDistance = 3.5; orbit.maxDistance = 13; orbit.enableDamping = true; orbit.dampingFactor = .07; controls.current = orbit;
    const fit = () => { const d=Math.max(4.25,5.5/(2*Math.tan(THREE.MathUtils.degToRad(cam.fov/2))*cam.aspect*.82));cam.position.sub(orbit.target).normalize().multiplyScalar(d).add(orbit.target);cam.lookAt(orbit.target); };
    const resize = () => { const w = container.clientWidth, h = container.clientHeight; if (!w || !h) return; cam.aspect = w/h; fit(); cam.updateProjectionMatrix(); renderer.setSize(w,h,false); }; const observer = new ResizeObserver(resize); observer.observe(container); resize();
    let frame = 0, active = true; const animate = () => { if (!active) return; orbit.update(); renderer.render(scene,cam); frame = requestAnimationFrame(animate); }; animate();
    return () => { active = false; cancelAnimationFrame(frame); observer.disconnect(); orbit.dispose(); renderer.dispose(); dispose(); container.removeChild(renderer.domElement); motion.current = null; camera.current = null; controls.current = null; };
  }, []);
  useEffect(() => {
    const cam = camera.current, orbit = controls.current; if (!cam || !orbit) return;
    const target = view === '3D' ? [2.2,5.8,7.4] : view === 'Side' ? [0,.35,9.6] : [0,9.6,.01];
    const d=Math.max(4.25,5.5/(2*Math.tan(THREE.MathUtils.degToRad(cam.fov/2))*cam.aspect*.82)); cam.position.set(...target as [number,number,number]).normalize().multiplyScalar(d); orbit.target.set(0,0,0); cam.lookAt(0,0,0); orbit.update(); orbit.enabled = view === '3D';
  }, [view]);
  useEffect(() => {
    if (!motion.current) return;
    const q = playhead !== null && orientation ? interpolateOrientation(orientation, playhead) : null;
    motion.current.quaternion.set(q?.x ?? 0,q?.y ?? 0,q?.z ?? 0,q?.w ?? 1);
    const axis = motion.current.userData.axis as THREE.Group; if (axis) axis.visible = !!orientation || debug;
  }, [orientation,playhead,debug]);
  const progress = selection && playhead !== null ? Math.max(0,Math.min(1,(playhead-selection.start_ms)/Math.max(1,selection.end_ms-selection.start_ms))) : 0;
  const approximate = !!orientation && manifest?.units.acceleration === 'raw_adc' && !(manifest.calibration?.gyroscope_counts_per_rad_s && manifest.calibration.gyroscope_counts_per_rad_s > 0);
  return <section className="panel motion-panel"><div className="panel-title"><h2>Motion replay</h2><div className="segmented small">{(['3D','Side','Top'] as const).map(label => <button key={label} className={view===label?'active':''} onClick={()=>setView(label)}>{label}</button>)}</div></div>
    {approximate && <div className="motion-approximate" role="note">Approximate：自动重力标定；陀螺仪未标定；轴向为约定值</div>}
    <div ref={mount} className="motion-viewport" aria-label="3D forearm, wrist, hand and smartwatch model" />
    {!orientation && <div className="motion-unavailable">{selection ? 'Orientation unavailable · raw sensor calibration required' : 'Select an interval to replay motion'}</div>}
    <div className="motion-controls"><button className="play-button" onClick={onPlay} disabled={!selection || !orientation} aria-label={playing?'Pause':'Play'}>{playing?'Ⅱ':'▶'}</button><span className="mono">{selection && playhead !== null ? duration(playhead-selection.start_ms) : '0.0 s'} / {selection ? duration(selection.end_ms-selection.start_ms) : '0.0 s'}</span><input type="range" min="0" max="1000" value={Math.round(progress*1000)} disabled={!selection || !orientation} onChange={e=>selection&&onSeek(selection.start_ms+Number(e.target.value)/1000*(selection.end_ms-selection.start_ms))} aria-label="Replay position"/><select value={speed} onChange={e=>onSpeed(Number(e.target.value))} aria-label="Playback speed">{[.25,.5,1,2].map(s=><option key={s} value={s}>{s}×</option>)}</select></div>
    <div className="motion-meta"><span>IMU orientation · rigid body</span><label><input type="checkbox" checked={debug} onChange={e=>setDebug(e.target.checked)}/> Details</label></div>
    {debug && <div className="debug-readout mono">Viewer axes: X arm → hand · Y up · Z depth<br/>q {orientation&&playhead!==null ? JSON.stringify(interpolateOrientation(orientation,playhead)) : 'unavailable'}<br/>t {playhead??'—'}</div>}
  </section>;
}

