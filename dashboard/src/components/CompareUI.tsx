"use client";

import clsx from "clsx";
import Link from "next/link";
import type { DashboardPlayer } from "@/lib/types";
import { money, pct, pts } from "@/lib/format";
import { PositionBadge } from "@/components/ui";

export type CompareLens = "output" | "floor" | "ceiling" | "minutes" | "value" | "differential";

const lensMeta: Record<CompareLens, { label: string; title: string; description: string }> = {
  output: { label: "Core output", title: "Expected points", description: "Mean and median projection with outcome probabilities." },
  floor: { label: "Floor", title: "Downside control", description: "P10, downside and the low-outcome profile." },
  ceiling: { label: "Ceiling", title: "Explosive upside", description: "P90, upside and haul probability." },
  minutes: { label: "Minutes", title: "Availability", description: "Appearance, starting and 60+ security." },
  value: { label: "Value", title: "Points per £m", description: "Price efficiency and value under the projection distribution." },
  differential: { label: "Differential", title: "Ownership leverage", description: "Upside relative to selected-by / effective ownership." },
};

export function CompareLensTabs({ value, onChange }: { value: CompareLens; onChange: (v: CompareLens) => void }) {
  return <div className="flex flex-wrap gap-1.5">
    {(Object.keys(lensMeta) as CompareLens[]).map((key) => <button key={key} onClick={() => onChange(key)} className={clsx("rounded-lg border px-2.5 py-1.5 text-[10px] font-semibold", value === key ? "border-[var(--accent)]/30 bg-[var(--accent)]/12 text-[var(--accent)]" : "border-line bg-bg text-muted hover:text-fg")}>{lensMeta[key].label}</button>)}
  </div>;
}

export function CompareHeader({ players, anchorId, onAnchorChange, onRemove }: { players: DashboardPlayer[]; anchorId: number; onAnchorChange: (id: number) => void; onRemove: (id: number) => void }) {
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
      <button onClick={() => onAnchorChange(p.player_id)} className={clsx("mt-2 w-full rounded-lg border px-2 py-1.5 text-[9px] font-semibold uppercase tracking-wide", anchorId === p.player_id ? "border-[var(--accent)]/25 text-[var(--accent)]" : "border-line text-muted hover:text-fg")}>{anchorId === p.player_id ? "Reference player" : "Use as reference"}</button>
    </div>)}
  </div>;
}

function toneForDelta(v: number | null | undefined) {
  if (v == null || !Number.isFinite(v) || Math.abs(v) < 0.05) return "text-muted";
  return v > 0 ? "text-[var(--good)]" : "text-[var(--bad)]";
}

export function ComparisonGrid({ players, anchorId, lens }: { players: DashboardPlayer[]; anchorId: number; lens: CompareLens }) {
  const anchor = players.find((p) => p.player_id === anchorId) ?? players[0];
  if (!anchor) return null;

  const rows: { label: string; get: (p: DashboardPlayer) => number | null | undefined; fmt: (v: number | null | undefined) => string }[] = lens === "output" ? [
    { label: "Expected points", get: p => p.xp_mean, fmt: v => pts(v) },
    { label: "P50", get: p => p.xp_p50, fmt: v => pts(v,1) },
    { label: "P(haul 10+)", get: p => p.p_haul_10plus, fmt: v => pct(v) },
    { label: "P(blank ≤2)", get: p => p.p_blank_le2, fmt: v => pct(v) },
    { label: "Simulation SE", get: p => p.xp_se, fmt: v => pts(v,2) },
  ] : lens === "floor" ? [
    { label: "P10", get: p => p.xp_p10, fmt: v => pts(v,1) },
    { label: "P25", get: p => p.xp_p25, fmt: v => pts(v,1) },
    { label: "Downside", get: p => p.downside_points, fmt: v => pts(v,1) },
    { label: "Floor-adjusted xP", get: p => p.floor_adjusted_xp, fmt: v => pts(v,2) },
    { label: "Range P10→P90", get: p => p.range_points, fmt: v => pts(v,1) },
    { label: "Quartile range P25→P75", get: p => p.quartile_range_points, fmt: v => pts(v,1) },
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
    { label: "Bench risk", get: p => p.bench_risk, fmt: v => pct(v) },
    { label: "Start gap", get: p => p.start_gap, fmt: v => pct(v) },
    { label: "Minutes stability", get: p => p.minutes_stability, fmt: v => pct(v) },
  ] : lens === "value" ? [
    { label: "Price", get: p => p.now_cost, fmt: v => money(v) },
    { label: "xP / £m", get: p => p.value_per_million ?? p.value, fmt: v => pts(v,2) },
    { label: "Floor xP / £m", get: p => p.floor_value_per_million ?? (p.now_cost && p.floor_adjusted_xp != null ? p.floor_adjusted_xp / p.now_cost : null), fmt: v => pts(v,2) },
    { label: "Floor xP / £m", get: p => p.floor_value_per_million ?? (p.now_cost && p.floor_adjusted_xp != null ? p.floor_adjusted_xp / p.now_cost : null), fmt: v => pts(v,2) },
    { label: "Ceiling xP / £m", get: p => p.ceiling_value_per_million ?? (p.now_cost && p.ceiling_adjusted_xp != null ? p.ceiling_adjusted_xp / p.now_cost : null), fmt: v => pts(v,2) },
  ] : [
    { label: "Ownership", get: p => p.ownership, fmt: v => pct(v) },
    { label: "Ownership-adjusted upside", get: p => p.ownership_adjusted_upside, fmt: v => pts(v,2) },
    { label: "Ownership-adjusted downside", get: p => p.ownership_adjusted_downside, fmt: v => pts(v,2) },
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
          return <td key={p.player_id} className="px-3 py-3 text-right"><div className="font-mono font-semibold">{r.fmt(value)}</div>{p.player_id === anchor.player_id ? <div className="mt-0.5 text-[9px] text-[var(--accent)]">reference</div> : delta != null ? <div className={clsx("mt-0.5 font-mono text-[9px]", toneForDelta(delta))}>{delta > 0 ? "+" : ""}{r.fmt(delta)} vs ref</div> : null}</td>;
        })}</tr>;
      })}</tbody>
    </table>
  </div>;
}

export function ProfileBars({ players, lens }: { players: DashboardPlayer[]; lens: CompareLens }) {
  const metric = lens === "floor" ? (p: DashboardPlayer) => p.floor_adjusted_xp ?? p.xp_p10 ?? 0 : lens === "ceiling" ? (p: DashboardPlayer) => p.ceiling_adjusted_xp ?? p.xp_p90 ?? 0 : lens === "minutes" ? (p: DashboardPlayer) => p.minutes_stability ?? 0 : lens === "value" ? (p: DashboardPlayer) => p.value_per_million ?? p.value ?? 0 : lens === "differential" ? (p: DashboardPlayer) => p.ownership_adjusted_upside ?? 0 : (p: DashboardPlayer) => p.xp_mean;
  const vals = players.map(metric);
  const max = Math.max(0.0001, ...vals.map(v => Math.abs(v)));
  return <div className="space-y-2">
    {players.map((p, i) => <div key={p.player_id} className="grid grid-cols-[110px_1fr_70px] items-center gap-2 text-[10px]"><div className="truncate font-medium">{p.web_name}</div><div className="h-2 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full bg-[var(--accent)]" style={{ width: `${Math.max(2, Math.min(100, Math.abs(vals[i]) / max * 100))}%` }} /></div><div className="text-right font-mono">{lens === "minutes" ? pct(vals[i]) : pts(vals[i], 2)}</div></div>)}
  </div>;
}

export function CompareSummaryCard({ p, lens, anchorId }: { p: DashboardPlayer; lens: CompareLens; anchorId: number }) {
  const primary = lens === "floor" ? pts(p.floor_adjusted_xp, 2) : lens === "ceiling" ? pts(p.ceiling_adjusted_xp, 2) : lens === "minutes" ? pct(p.minutes_stability) : lens === "value" ? pts(p.value_per_million ?? p.value, 2) : lens === "differential" ? pts(p.ownership_adjusted_upside, 2) : pts(p.xp_mean);
  return <div className={clsx("rounded-2xl border p-4", p.player_id === anchorId ? "border-[var(--accent)]/30 bg-[var(--accent)]/6" : "border-line bg-panel")}>
    <div className="flex items-start justify-between gap-3"><div><div className="text-sm font-semibold">{p.web_name}</div><div className="mt-1 flex items-center gap-1.5"><PositionBadge position={p.position}/><span className="text-[10px] text-muted">{p.short_name ?? "—"}</span></div></div><div className="text-right"><div className="text-[9px] uppercase tracking-wider text-muted">{lensMeta[lens].label}</div><div className="mt-0.5 font-mono text-xl font-semibold">{primary}</div></div></div>
    <div className="mt-4 grid grid-cols-2 gap-2 text-[10px]"><MetricMini label="xP" value={pts(p.xp_mean,1)}/><MetricMini label="P90" value={pts(p.xp_p90,1)}/><MetricMini label="P(play)" value={pct(p.p_play)}/><MetricMini label="Ownership" value={pct(p.ownership)}/></div>
  </div>;
}

function MetricMini({ label, value }: { label: string; value: string }) { return <div className="rounded-lg bg-bg/60 p-2"><div className="text-muted">{label}</div><div className="mt-0.5 font-mono font-semibold">{value}</div></div>; }

export { lensMeta };
