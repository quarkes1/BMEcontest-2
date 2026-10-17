import { describe, expect, it } from 'vitest';
import { classifySelection, isSessionTxt, loadExistingArtifacts, type FileLike } from '../data/loader';
import { demoPrediction } from '../data/demo';

function textFile(name: string, content: string): FileLike {
  return { name, text: async () => content, arrayBuffer: async () => new ArrayBuffer(0) };
}

function binaryFile(name: string, buffer: ArrayBuffer): FileLike {
  return { name, text: async () => '', arrayBuffer: async () => buffer };
}

function motionPair(sessionId: string): { manifest: FileLike; binary: FileLike } {
  const bytes = new ArrayBuffer(32);
  const view = new DataView(bytes);
  view.setFloat64(0, 1000, true);
  [1, 2, 3, 4, 5, 6].forEach((v, i) => view.setFloat32(8 + i * 4, v, true));
  const manifest = {
    telemetry_version: '1.0', session_id: sessionId, record_format: 'f64_ms_6xf32_le',
    sample_count: 1, start_ms: 1000, end_ms: 1000, binary_file: 'motion.bin',
    units: { acceleration: 'raw_adc', gyroscope: 'raw_adc' },
  };
  return { manifest: textFile('motion.json', JSON.stringify(manifest)), binary: binaryFile('motion.bin', bytes) };
}

describe('selection classification', () => {
  it('recognizes the canonical raw session convention', () => {
    expect(isSessionTxt('collect_data1_2_3.txt')).toBe(true);
    expect(isSessionTxt('collect_data.txt')).toBe(true);
    expect(isSessionTxt('notes.txt')).toBe(false);
    expect(isSessionTxt('collect_data1.csv')).toBe(false);
  });

  it('classifies session files, artifacts, and unrelated files', () => {
    const selection = classifySelection([
      textFile('collect_data1_2_3.txt', 'x'),
      textFile('prediction.json', '{}'),
      binaryFile('motion.bin', new ArrayBuffer(0)),
      textFile('README.md', 'unrelated'),
    ]);
    expect(selection.sessionTexts.map(f => f.name)).toEqual(['collect_data1_2_3.txt']);
    expect(selection.binaries.map(f => f.name)).toEqual(['motion.bin']);
    expect(selection.ignored).toEqual(['README.md']);
  });
});

describe('existing prediction and telemetry loading', () => {
  it('loads a prediction document', async () => {
    const { prediction } = await loadExistingArtifacts([textFile('prediction.json', JSON.stringify(demoPrediction))], null);
    expect(prediction?.events).toEqual(demoPrediction.events);
  });

  it('loads a motion pair and validates the session against the prediction', async () => {
    const session = demoPrediction.timeline!.session_ids[0];
    const pair = motionPair(session);
    const { motion } = await loadExistingArtifacts([pair.manifest, pair.binary], demoPrediction);
    expect(motion?.manifest.session_id).toBe(session);
    expect(motion?.imu.ax[0]).toBe(1);
    const stranger = motionPair('UNKNOWN SESSION');
    await expect(loadExistingArtifacts([stranger.manifest, stranger.binary], demoPrediction)).rejects.toThrow('does not match prediction');
  });

  it('requires the binary companion and reports unrelated selections', async () => {
    const pair = motionPair('s');
    await expect(loadExistingArtifacts([pair.manifest], demoPrediction)).rejects.toThrow('matching motion.bin');
    await expect(loadExistingArtifacts([textFile('README.md', 'x')], null)).rejects.toThrow('Choose a prediction JSON');
  });
});
