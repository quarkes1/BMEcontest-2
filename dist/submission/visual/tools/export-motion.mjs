import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { once } from 'node:events';

const [source, output, ...options] = process.argv.slice(2);
if (!source || !output) {
  console.error('Usage: node tools/export-motion.mjs <collect_data*.txt> <output-dir> [--session-id=ID]');
  process.exit(2);
}
const sessionId = options.find(x => x.startsWith('--session-id='))?.slice(13) || basename(source, '.txt');
const out = resolve(output);
await mkdir(out, { recursive: true });
const writer = createWriteStream(join(out, 'motion.bin'));
let columns, count = 0, start = Infinity, end = -Infinity, previous = -Infinity, skipped = 0;
const keys = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z'];
for await (const line of createInterface({ input: createReadStream(source), crlfDelay: Infinity })) {
  if (!columns) {
    columns = new Map(line.replace(/^\uFEFF/, '').split('\t').map((name, i) => [name.trim(), i]));
    if (!columns.has('ACC_TIME') || keys.some(key => !columns.has(key))) throw new Error('Missing required ACC_TIME or IMU columns.');
    continue;
  }
  const fields = line.split('\t');
  const t = Number(fields[columns.get('ACC_TIME')]);
  const values = keys.map(key => Number(fields[columns.get(key)]));
  if (!Number.isFinite(t) || t <= 0 || values.some(v => !Number.isFinite(v))) { skipped++; continue; }
  if (t < previous) throw new Error(`Non-monotonic ACC_TIME at record ${count + skipped + 1}`);
  const record = Buffer.allocUnsafe(32); record.writeDoubleLE(t, 0);
  values.forEach((value, i) => record.writeFloatLE(value, 8 + i * 4));
  if (!writer.write(record)) await once(writer, 'drain');
  previous = t; start = Math.min(start, t); end = t; count++;
}
writer.end(); await once(writer, 'finish');
if (!count) throw new Error('No valid IMU samples found.');
const manifest = {
  telemetry_version: '1.0', session_id: sessionId, record_format: 'f64_ms_6xf32_le',
  sample_count: count, start_ms: start, end_ms: end, binary_file: 'motion.bin',
  units: { acceleration: 'raw_adc', gyroscope: 'raw_adc' },
  provenance: { source_file: basename(source), skipped_rows: skipped, timestamps: 'ACC_TIME', gyro_timing_note: 'GYRO values paired by source row; GYRO_TIME may differ.' }
};
await writeFile(join(out, 'motion.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify({ manifest: join(out, 'motion.json'), samples: count, skipped_rows: skipped, session_id: sessionId }));
