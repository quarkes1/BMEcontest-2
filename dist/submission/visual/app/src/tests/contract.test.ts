import { describe, expect, it } from 'vitest';
import { parsePrediction } from '../data/prediction';
import { demoPrediction } from '../data/demo';

describe('visualization contract', () => {
  it('accepts timestamped model evidence while preserving canonical events', () => {
    expect(parsePrediction(demoPrediction).events).toEqual(demoPrediction.events);
  });

  it('rejects unsupported schemas and malformed records', () => {
    expect(() => parsePrediction({ ...demoPrediction, schema_version: '2.0' })).toThrow('Unsupported prediction schema');
    expect(() => parsePrediction({ ...demoPrediction, events: [{ ...demoPrediction.events[0], end_ms: demoPrediction.events[0].start_ms }] })).toThrow('Invalid event timestamps');
    expect(() => parsePrediction({ ...demoPrediction, timeline: { ...demoPrediction.timeline, sessions: [{ ...demoPrediction.timeline!.sessions![0], macro: { timestamp_ms: [2, 1], probability: [.4, .5] } }] } })).toThrow('Invalid timeline stream');
  });

  it('handles optional timeline series and optional candidates', () => {
    const seriesPrediction = {
      ...demoPrediction,
      timeline: {
        session_ids: ['s'], macro_windows: 2, micro_windows: 0,
        series: [
          { session_id: 's', timestamp_ms: 1000, macro_probability: .2, micro_probability: .1, valid: true, gap: false },
          { session_id: 's', timestamp_ms: 2000, macro_probability: .9, micro_probability: .8, valid: true, gap: false },
          { session_id: 's', timestamp_ms: 3000, macro_probability: 0, micro_probability: 0, valid: false, gap: true },
        ],
      },
    };
    const parsed = parsePrediction(seriesPrediction);
    expect(parsed.timeline!.sessions![0].macro.probability).toEqual([.2, .9]);
    expect(parsed.timeline!.sessions![0].micro.probability).toEqual([.1, .8]);
    const invalidSeries = {
      ...demoPrediction,
      timeline: {
        session_ids: ['s'], macro_windows: 1, micro_windows: 0,
        series: [{ session_id: 's', timestamp_ms: 1000, macro_probability: 1.4, micro_probability: .1, valid: true, gap: false }],
      },
    };
    expect(() => parsePrediction(invalidSeries)).toThrow('Invalid timeline series');
    const withoutOptional = parsePrediction({ ...demoPrediction, candidates: undefined, timeline: { session_ids: ['s'], macro_windows: 0, micro_windows: 0 } });
    expect(withoutOptional.candidates).toBeUndefined();
  });
});
