import { describe, expect, it } from 'vitest';
import { clampRange, panView, timeAtX, xAtTime, zoomView } from '../components/timeline/timelineMath';

describe('timeline viewport math', () => {
  const bounds = { start_ms: 0, end_ms: 1000 };

  it('clamps selection to the session bounds', () => {
    expect(clampRange(1200, -100, bounds)).toEqual(bounds);
    expect(clampRange(400, 700, bounds)).toEqual({ start_ms: 400, end_ms: 700 });
    expect(clampRange(700, 400, bounds)).toEqual({ start_ms: 400, end_ms: 700 });
  });

  it('zooms around the requested center and keeps the full view as the limit', () => {
    expect(zoomView(bounds, 900, 200)).toEqual({ start_ms: 800, end_ms: 1000 });
    expect(zoomView(bounds, 100, 200)).toEqual({ start_ms: 0, end_ms: 200 });
    expect(zoomView(bounds, 500, 5000)).toEqual(bounds);
  });

  it('pans without leaving the session bounds', () => {
    expect(panView(bounds, { start_ms: 800, end_ms: 1000 }, 500)).toEqual({ start_ms: 800, end_ms: 1000 });
    expect(panView(bounds, { start_ms: 400, end_ms: 600 }, -500)).toEqual({ start_ms: 0, end_ms: 200 });
  });

  it('maps pixels and time consistently in both directions', () => {
    const view = { start_ms: 200, end_ms: 1200 };
    expect(timeAtX(0, 500, view)).toBe(200);
    expect(timeAtX(500, 500, view)).toBe(1200);
    expect(timeAtX(250, 500, view)).toBe(700);
    expect(xAtTime(700, 500, view)).toBe(250);
  });
});
