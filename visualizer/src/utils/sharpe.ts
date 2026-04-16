import type { ActivityLogRow } from '../models.ts';

export function sortedActivityTimestamps(activityLogs: ActivityLogRow[]): number[] {
  const unique = new Set<number>();
  for (const row of activityLogs) {
    unique.add(row.timestamp);
  }
  return [...unique].sort((a, b) => a - b);
}

/** Per-timestamp PnL increments; missing timestamps count as 0. */
export function periodPnLByTimestamps(
  activityLogs: ActivityLogRow[],
  timestamps: number[],
  product: 'total' | string,
): number[] {
  const byTimestamp = new Map<number, number>();
  for (const row of activityLogs) {
    if (product !== 'total' && row.product !== product) {
      continue;
    }
    byTimestamp.set(row.timestamp, (byTimestamp.get(row.timestamp) ?? 0) + row.profitLoss);
  }
  return timestamps.map(ts => byTimestamp.get(ts) ?? 0);
}

/**
 * Sharpe from period PnL: sqrt(n) * mean / std (sample std, ddof=1), n = number of periods.
 */
export function sharpeFromPeriodPnL(periodPnL: number[]): number | null {
  const n = periodPnL.length;
  if (n < 2) {
    return null;
  }
  let sum = 0;
  for (const x of periodPnL) {
    sum += x;
  }
  const mean = sum / n;
  let sq = 0;
  for (const x of periodPnL) {
    const d = x - mean;
    sq += d * d;
  }
  const variance = sq / (n - 1);
  const std = Math.sqrt(variance);
  if (std === 0) {
    return mean === 0 ? 0 : null;
  }
  return (Math.sqrt(n) * mean) / std;
}

export function formatSharpeRatio(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return '—';
  }
  return value.toLocaleString(undefined, { minimumFractionDigits: 3, maximumFractionDigits: 3 });
}
