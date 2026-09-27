export const pts = (n: number | null | undefined, dp = 2) =>
  n == null ? "\u2014" : n.toFixed(dp);

export const pct = (n: number | null | undefined, dp = 0) =>
  n == null ? "\u2014" : `${(n * 100).toFixed(dp)}%`;

export const money = (n: number | null | undefined) =>
  n == null ? "\u2014" : `\u00a3${n.toFixed(1)}m`;

/**
 * Two projections separated by less than the Monte Carlo standard error are
 * not distinguishable. The simulator reports that error and the UI respects
 * it rather than presenting a 0.03 gap as though it were a decision.
 */
export const indistinguishable = (a: number, b: number, se: number | null) =>
  se != null && se > 0 && Math.abs(a - b) < 2 * se;

export const positionColour: Record<string, string> = {
  GK: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
  DEF: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
  MID: "bg-sky-500/15 text-sky-700 dark:text-sky-300",
  FWD: "bg-rose-500/15 text-rose-700 dark:text-rose-300",
};

/**
 * recharts v3 widened Tooltip's formatter `value` parameter from a
 * guaranteed `number` to `ValueType | undefined` (it can be a string or an
 * array when a chart has multiple series of mixed types). Every formatter in
 * this project expects a plain number, so this coerces once at the call
 * site rather than repeating the same undefined/array guard six times.
 */
export const toNum = (v: unknown): number => {
  if (typeof v === "number") return v;
  if (Array.isArray(v)) return toNum(v[0]);
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
};
