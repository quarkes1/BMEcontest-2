export type Range = { start_ms: number; end_ms: number };
export type Event = Range & { id: number; session_id: string; duration_s: number; confidence: number };
export type Candidate = Range & { session_id: string; score: number; admitted: boolean };
export type Gap = Range & { session_id: string };
export type Stream = { timestamp_ms: number[]; probability: number[] };
export type SessionTimeline = Range & { session_id: string; macro: Stream; micro: Stream };
export type Prediction = {
  schema_version: string;
  model: { name: string; run_key: string };
  input: { source: string; duration_seconds: number };
  events: Event[];
  candidates?: Candidate[];
  gaps?: Gap[];
  diagnostics: { coverage: number; warnings: string[]; resolved_device?: string };
  timeline?: {
    session_ids: string[];
    macro_windows: number;
    micro_windows: number;
    sessions?: SessionTimeline[];
    series?: Array<{ session_id: string; timestamp_ms: number; macro_probability: number; micro_probability: number; valid: boolean; gap: boolean }>;
  };
};
export type MotionManifest = {
  telemetry_version: '1.0';
  session_id: string;
  record_format: 'f64_ms_6xf32_le';
  sample_count: number;
  start_ms: number;
  end_ms: number;
  binary_file: string;
  units: { acceleration: 'raw_adc' | 'g'; gyroscope: 'raw_adc' | 'rad/s' | 'deg/s' };
  calibration?: { acceleration_counts_per_g: number; gyroscope_counts_per_rad_s?: number; viewer_from_sensor: number[] };
  provenance?: Record<string, unknown>;
};
export type Imu = { t: Float64Array; ax: Float32Array; ay: Float32Array; az: Float32Array; gx: Float32Array; gy: Float32Array; gz: Float32Array };
export type MotionData = { manifest: MotionManifest; imu: Imu };

/** Everything the application displays: one prediction plus per-session motion telemetry. */
export type Dataset = {
  prediction: Prediction;
  motions: Record<string, MotionData>;
  label: string;
  demo: boolean;
};
