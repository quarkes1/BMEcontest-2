export function fmtTime(ms: number, detailed = false): string {
  const d = new Date(ms);
  return d.toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', second: detailed ? '2-digit' : undefined, fractionalSecondDigits: detailed ? 1 : undefined });
}

export function duration(ms: number): string {
  const s = Math.max(0, ms / 1000);
  return s < 60 ? `${s.toFixed(1)} s` : s < 3600 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${(s / 3600).toFixed(1)} h`;
}
