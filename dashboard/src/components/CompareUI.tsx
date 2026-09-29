"use client";

import clsx from "clsx";
import Link from "next/link";
import type { ComparePlayer, CompareFixture } from "@/lib/types";
import { money, pct, pts } from "@/lib/format";
import { PositionBadge } from "@/components/ui";

export type CompareLens = "output" | "floor" | "ceiling" | "minutes" | "value" | "differential" | "fixtures";

const lensMeta: Record<CompareLens, { label: string; title: string; description: string }> = {
  output: { label: "Core output", title: "Expected points", description: "Mean and median projection with outcome probabilities." },
  floor: { label: "Floor", title: "Downside control", description: "P10, downside and the low-outcome profile." },
  ceiling: { label: "Ceiling", title: "Explosive upside", description: "P90, upside and haul probability." },
  minutes: { label: "Minutes", title: "Availability", description: "Appearance, starting and 60+ security." },
  value: { label: "Value", title: "Points per £m", description: "Price efficiency and value under the projection distribution." },
  differential: { label: "Differential", title: "Ownership leverage", description: "Upside relative to selected-by / effective ownership." },
  fixtures: { label: "Fixtures", title: "Upcoming run", description: "Model-rated difficulty of the next fixtures, judged for each player's position (clean sheets for defenders, goals for forwards)." },
};

/** Short, plain labels for the model's consistency flags. */
const FLAG_LABEL: Record<string, string> = {
  high_xp_low_pplay: "Fragile: low chance of playing",
  wide_distribution: "Unstable: very wide range",
  diverging_week_over_week: "Mixed signals vs last week",
};

function difficultyTone(d: number) {
  if (d <= 2.4) return "border-[var(--good)]/40 bg-[var(--good)]/12 text-[var(--good)]";
  if (d <= 3.4) return "border-line bg-bg/60 text-muted";
  return "border-[var(--bad)]/40 bg-[var(--bad)]/12 text-[var(--bad)]";
}

export function FixtureStrip({ fixtures }: { fixtures: CompareFixture[] }) {
  if (!fixtures.length) return <div className="text-[10px] text-muted">No fixtures in the window.</div>;
  return <div className="flex flex-wrap gap-1">
    {fixtures.map((f, i) => <span key={`${f.gw}-${f.opponent}-${i}`} title={`GW${f.gw} ${f.home ? "home" : "away"} v ${f.opponent}, difficulty ${f.difficulty.toFixed(1)} of 5`} className={clsx("rounded-md border px-1.5 py-1 font-mono text-[9px] font-semibold", difficultyTone(f.difficulty))}>{f.opponent} {f.home ? "(H)" : "(A)"}</span>)}
  </div>;
}

export function FlagBadges({ p, compact = false }: { p: ComparePlayer; compact?: boolean }) {
  if (!p.flags?.length) return null;
  return <div className="mt-2 space-y-1">
    {p.flags.map(f => <div key={f.code} title={f.message} className="rounded-md border border-[var(--warn)]/30 bg-[var(--warn)]/10 px-2 py-1 text-[9px] leading-4 text-[var(--warn)]"><span className="font-semibold">{FLAG_LABEL[f.code] ?? f.code}</span>{compact ? null : <span className="mt-0.5 block text-muted">{f.message}</span>}</div>)}
  </div>;
}

export function CompareLensTabs({ value, onChange }: { value: CompareLens; onChange: (v: CompareLens) => void }) {
  return <div className="flex flex-wrap gap-1.5">
    {(Object.keys(lensMeta) as CompareLens[]).map((key) => <button key={key} onClick={() => onChange(key)} className={clsx("rounded-lg border px-2.5 py-1.5 text-[10px] font-semibold", value === key ? "border-[var(--accent)]/30 bg-[var(--accent)]/12 text-[var(--accent)]" : "border-line bg-bg text-muted hover:text-fg")}>{lensMeta[key].label}</button>)}
  </div>;
}

export function CompareHeader({ players, anchorId, onAnchorChange, onRemove }: { players: ComparePlayer[]; anchorId: number; onAnchorChange: (id: number) => void; onRemove: (id: number) => void }) {
  return <div className="grid gap-2" style={{ gridTemplateColumns: `180px repeat(${Math.max(players.length, 1)}, minmax(180px, 1fr))` }}>
    <div className="rounded-2xl border border-line bg-bg/50 p-3">
      <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-muted">Reference</div>
      <div className="mt-1 text-xs font-medium">All deltas are measured from this player.</div>
    </div>
    {players.map((p) => <div key={p.player_id} className={clsx("rounded-2xl border p-3", anchorId === p.player_id ? "border-[var(--accent)]/30 bg-[var(--accent)]/6" : "border-line bg-panel")}>
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0"><Link href={`/players/${p.player_id}`} className="truncate text-sm font-semibold hover:text-[var(--accent)]">{p.web_name}</Link><div className="mt-1 flex items-center gap-1.5"><PositionBadge position={p.position}/><span className="text-[10px] text-muted">{p.short_name ?? "—"}</span></div></div>
        <button onClick={() => onRemove(p.player_id)} aria-label={`Remove ${p.web_name}`} className="rounded-md px-1.5 py-1 text-muted hover:bg-bg hover:text-fg">×</button>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-[10px]">
        <div className="rounded-lg bg-bg/60 p-2"><div className="text-muted">Price</div><div className="mt-0.5 font-mono font-semibold">{money(p.now_cost)}</div></div>
        <div className="rounded-lg bg-bg/60 p-2"><div className="text-muted">xP</div><div className="mt-0.5 font-mono font-semibold">{pts(p.xp_mean)}</div></div>
      </div>
      <FlagBadges p={p} compact />
      <button onClick={() => onAnchorChange(p.player_id)} className={clsx("mt-2 w-full rounded-lg border px-2 py-1.5 text-[9px] font-semibold uppercase tracking-wide", anchorId === p.player_id ? "border-[var(--accent)]/25 text-[var(--accent)]" : "border-line text-muted hover:text-fg")}>{anchorId === p.player_id ? "Reference player" : "Use as reference"}</button>
    </div>)}
  </div>;
}

type Better = "higher" | "lower" | "neutral";

/** Colour a delta by whether the change is an improvement for THIS metric.
 *  A higher P(blank) or bench risk is worse, so it must not turn green. */
function toneForDelta(v: number | null | undefined, better: Better = "higher") {
  if (v == null || !Number.isFinite(v) || Math.abs(v) < 0.05 || better === "neutral") return "text-muted";
  const improved = better === "higher" ? v > 0 : v < 0;
  return improved ? "text-[var(--good)]" : "text-[var(--bad)]";
}

export function ComparisonGrid({ players, anchorId, lens }: { players: ComparePlayer[]; anchorId: number; lens: CompareLens }) {
  const anchor = players.find((p) => p.player_id === anchorId) ?? players[0];
  if (!anchor) return null;

  const rows: { label: string; get: (p: ComparePlayer) => number | null | undefined; fmt: (v: number | null | undefined) => string; better?: Better }[] = lens === "output" ? [
    { label: "Expected points", get: p => p.xp_mean, fmt: v => pts(v) },
    { label: "P50", get: p => p.xp_p50, fmt: v => pts(v,1) },
    { label: "P(haul 10+)", get: p => p.p_haul_10plus, fmt: v => pct(v) },
    { label: "P(blank ≤2)", get: p => p.p_blank_le2, fmt: v => pct(v), better: "lower" },
    { label: "Simulation SE", get: p => p.xp_se, fmt: v => pts(v,2), better: "lower" },
  ] : lens === "floor" ? [
    { label: "P10", get: p => p.xp_p10, fmt: v => pts(v,1) },
    { label: "P25", get: p => p.xp_p25, fmt: v => pts(v,1) },
    { label: "Downside", get: p => p.downside_points, fmt: v => pts(v,1), better: "lower" },
    { label: "Floor-adjusted xP", get: p => p.floor_adjusted_xp, fmt: v => pts(v,2) },
    { label: "Range P10→P90", get: p => p.range_points, fmt: v => pts(v,1), better: "neutral" },
    { label: "Quartile range P25→P75", get: p => p.quartile_range_points, fmt: v => pts(v,1), better: "neutral" },
  ] : lens === "ceiling" ? [
    { label: "P75", get: p => p.xp_p75, fmt: v => pts(v,1) },
    { label: "P90", get: p => p.xp_p90, fmt: v => pts(v,1) },
    { label: "Upside", get: p => p.upside_points, fmt: v => `+${pts(v,1)}` },
    { label: "Ceiling-adjusted xP", get: p => p.ceiling_adjusted_xp, fmt: v => pts(v,2) },
    { label: "Haul probability", get: p => p.p_haul_10plus, fmt: v => pct(v) },
  ] : lens === "minutes" ? [
    { label: "P(play)", get: p => p.p_play, fmt: v => pct(v) },
    { label: "P(start)", get: p => p.p_start, fmt: v => pct(v) },
    { label: "P60+", get: p => p.p_sixty, fmt: v => pct(v) },
    { label: "Start security", get: p => p.start_security, fmt: v => pct(v) },
    { label: "60+ security", get: p => p.sixty_security, fmt: v => pct(v) },
    { label: "Bench risk", get: p => p.bench_risk, fmt: v => pct(v), better: "lower" },
    { label: "Start gap", get: p => p.start_gap, fmt: v => pct(v), better: "lower" },
    { label: "Minutes stability", get: p => p.minutes_stability, fmt: v => pct(v) },
  ] : lens === "value" ? [
    { label: "Price", get: p => p.now_cost, fmt: v => money(v), better: "neutral" },
    { label: "xP / £m", get: p => p.value_per_million ?? p.value, fmt: v => pts(v,2) },
    { label: "Floor xP / £m", get: p => p.floor_value_per_million ?? (p.now_cost && p.floor_adjusted_xp != null ? p.floor_adjusted_xp / p.now_cost : null), fmt: v => pts(v,2) },
    { label: "Ceiling xP / £m", get: p => p.ceiling_value_per_million ?? (p.now_cost && p.ceiling_adjusted_xp != null ? p.ceiling_adjusted_xp / p.now_cost : null), fmt: v => pts(v,2) },
  ] : lens === "fixtures" ? [
    { label: "Avg difficulty (next fixtures)", get: p => p.avg_difficulty, fmt: v => pts(v,2), better: "lower" },
    { label: "Next fixture difficulty", get: p => p.next_fixtures?.[0]?.difficulty, fmt: v => pts(v,2), better: "lower" },
    { label: "Easiest in window", get: p => p.next_fixtures?.length ? Math.min(...p.next_fixtures.map(f => f.difficulty)) : null, fmt: v => pts(v,2), better: "lower" },
    { label: "Hardest in window", get: p => p.next_fixtures?.length ? Math.max(...p.next_fixtures.map(f => f.difficulty)) : null, fmt: v => pts(v,2), better: "lower" },
    { label: "Fixtures in window", get: p => p.next_fixtures?.length ?? 0, fmt: v => (v == null ? "n/a" : String(v)), better: "neutral" },
  ] : [
    { label: "Ownership", get: p => p.ownership, fmt: v => pct(v), better: "neutral" },
    { label: "Ownership-adjusted upside", get: p => p.ownership_adjusted_upside, fmt: v => pts(v,2) },
    { label: "Ownership-adjusted downside", get: p => p.ownership_adjusted_downside, fmt: v => pts(v,2), better: "lower" },
    { label: "Ceiling", get: p => p.xp_p90, fmt: v => pts(v,1) },
  ];

  return <div className="overflow-x-auto rounded-2xl border border-line bg-panel">
    <table className="w-full min-w-[880px] text-xs">
      <thead><tr className="border-b border-line bg-bg/45 text-[9px] uppercase tracking-[0.14em] text-muted"><th className="w-[180px] px-3 py-3 text-left">Metric</th>{players.map(p => <th key={p.player_id} className="px-3 py-3 text-right">{p.web_name}</th>)}</tr></thead>
      <tbody>{rows.map((r) => {
        const anchorValue = r.get(anchor);
        return <tr key={r.label} className="border-b border-line/70 last:border-0"><td className="px-3 py-3 font-medium">{r.label}</td>{players.map((p) => {
          const value = r.get(p);
          const num = typeof value === "number" ? value : null;
          const aNum = typeof anchorValue === "number" ? anchorValue : null;
          const delta = num != null && aNum != null ? num - aNum : null;
          return <td key={p.player_id} className="px-3 py-3 text-right"><div className="font-mono font-semibold">{r.fmt(value)}</div>{p.player_id === anchor.player_id ? <div className="mt-0.5 text-[9px] text-[var(--accent)]">reference</div> : delta != null ? <div className={clsx("mt-0.5 font-mono text-[9px]", toneForDelta(delta, r.better))}>{delta > 0 ? "+" : ""}{r.fmt(delta)} vs ref</div> : null}</td>;
        })}</tr>;
      })}</tbody>
    </table>
  </div>;
}

export function ProfileBars({ players, lens }: { players: ComparePlayer[]; lens: CompareLens }) {
  const metric = lens === "floor" ? (p: ComparePlayer) => p.floor_adjusted_xp ?? p.xp_p10 ?? 0 : lens === "ceiling" ? (p: ComparePlayer) => p.ceiling_adjusted_xp ?? p.xp_p90 ?? 0 : lens === "minutes" ? (p: ComparePlayer) => p.minutes_stability ?? 0 : lens === "value" ? (p: ComparePlayer) => p.value_per_million ?? p.value ?? 0 : lens === "differential" ? (p: ComparePlayer) => p.ownership_adjusted_upside ?? 0 : lens === "fixtures" ? (p: ComparePlayer) => p.avg_difficulty ?? 0 : (p: ComparePlayer) => p.xp_mean;
  const vals = players.map(metric);
  const max = Math.max(0.0001, ...vals.map(v => Math.abs(v)));
  return <div className="space-y-2">
    {players.map((p, i) => <div key={p.player_id} className="grid grid-cols-[110px_1fr_70px] items-center gap-2 text-[10px]"><div className="truncate font-medium">{p.web_name}</div><div className="h-2 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full bg-[var(--accent)]" style={{ width: `${Math.max(2, Math.min(100, Math.abs(vals[i] ?? 0) / max * 100))}%` }} /></div><div className="text-right font-mono">{lens === "minutes" ? pct(vals[i] ?? null) : pts(vals[i] ?? null, 2)}</div></div>)}
  </div>;
}

export function CompareSummaryCard({ p, lens, anchorId }: { p: ComparePlayer; lens: CompareLens; anchorId: number }) {
  const primary = lens === "floor" ? pts(p.floor_adjusted_xp, 2) : lens === "ceiling" ? pts(p.ceiling_adjusted_xp, 2) : lens === "minutes" ? pct(p.minutes_stability) : lens === "value" ? pts(p.value_per_million ?? p.value, 2) : lens === "differential" ? pts(p.ownership_adjusted_upside, 2) : lens === "fixtures" ? pts(p.avg_difficulty, 2) : pts(p.xp_mean);
  return <div className={clsx("rounded-2xl border p-4", p.player_id === anchorId ? "border-[var(--accent)]/30 bg-[var(--accent)]/6" : "border-line bg-panel")}>
    <div className="flex items-start justify-between gap-3"><div><div className="text-sm font-semibold">{p.web_name}</div><div className="mt-1 flex items-center gap-1.5"><PositionBadge position={p.position}/><span className="text-[10px] text-muted">{p.short_name ?? "—"}</span></div></div><div className="text-right"><div className="text-[9px] uppercase tracking-wider text-muted">{lensMeta[lens].label}</div><div className="mt-0.5 font-mono text-xl font-semibold">{primary}</div></div></div>
    <div className="mt-4 grid grid-cols-2 gap-2 text-[10px]"><MetricMini label="xP" value={pts(p.xp_mean,1)}/><MetricMini label="P90" value={pts(p.xp_p90,1)}/><MetricMini label="P(play)" value={pct(p.p_play)}/><MetricMini label="Ownership" value={pct(p.ownership)}/></div>
    <div className="mt-3"><div className="mb-1 text-[9px] uppercase tracking-wider text-muted">Next fixtures</div><FixtureStrip fixtures={p.next_fixtures ?? []}/></div>
    <FlagBadges p={p}/>
  </div>;
}

function MetricMini({ label, value }: { label: string; value: string }) { return <div className="rounded-lg bg-bg/60 p-2"><div className="text-muted">{label}</div><div className="mt-0.5 font-mono font-semibold">{value}</div></div>; }

export { lensMeta };
