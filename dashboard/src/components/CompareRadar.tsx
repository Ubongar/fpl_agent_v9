"use client";

import { useMemo } from "react";
import {
  Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  ResponsiveContainer, Tooltip,
} from "recharts";
import type { ComparePlayer } from "@/lib/types";
import { playerColor } from "@/lib/comparePalette";

type Axis = { key: string; label: string; get: (p: ComparePlayer) => number | null | undefined };

/**
 * Seven dimensions, each already "higher is better" so a bigger polygon is
 * genuinely a stronger profile. Fixture ease is 5 minus the average upcoming
 * difficulty (difficulty is 1 easy to 5 hard), so an easy run scores high.
 */
const AXES: Axis[] = [
  { key: "output", label: "Output", get: p => p.xp_mean },
  { key: "floor", label: "Floor", get: p => p.floor_adjusted_xp },
  { key: "ceiling", label: "Ceiling", get: p => p.ceiling_adjusted_xp },
  { key: "minutes", label: "Minutes", get: p => p.minutes_stability },
  { key: "value", label: "Value", get: p => p.value_per_million ?? p.value },
  { key: "leverage", label: "Leverage", get: p => p.ownership_adjusted_upside },
  { key: "fixtures", label: "Fixtures", get: p => (p.avg_difficulty == null ? null : 5 - p.avg_difficulty) },
];

/**
 * Each axis is scaled to the best value in the CURRENT selection (best = 100).
 * This is a relative view for comparing these players against each other, not
 * an absolute rating: add a stronger player and everyone else's polygon
 * shrinks. Missing data plots as 0 and is called out beneath the chart rather
 * than silently drawn as a real zero.
 */
export function CompareRadar({ players }: { players: ComparePlayer[] }) {
  const { rows, missing } = useMemo(() => {
    const missing: string[] = [];
    const rows = AXES.map(axis => {
      const raw = players.map(p => axis.get(p));
      const best = Math.max(0, ...raw.map(v => (typeof v === "number" && Number.isFinite(v) ? v : 0)));
      const row: Record<string, number | string> = { axis: axis.label };
      players.forEach((p, i) => {
        const v = raw[i];
        if (typeof v !== "number" || !Number.isFinite(v)) {
          missing.push(`${p.web_name}: ${axis.label}`);
          row[`p${p.player_id}`] = 0;
        } else {
          row[`p${p.player_id}`] = best > 0 ? Math.round((Math.max(0, v) / best) * 100) : 0;
        }
      });
      return row;
    });
    return { rows, missing };
  }, [players]);

  if (players.length < 2) return null;

  return <div>
    <div className="h-[340px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={rows} outerRadius="72%">
          <PolarGrid stroke="var(--line)" />
          <PolarAngleAxis dataKey="axis" tick={{ fill: "var(--muted)", fontSize: 11 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: "var(--muted)", fontSize: 9 }} tickCount={5} axisLine={false} />
          {players.map((p, i) => <Radar key={p.player_id} name={p.web_name} dataKey={`p${p.player_id}`} stroke={playerColor(i)} fill={playerColor(i)} fillOpacity={players.length > 4 ? 0.08 : 0.16} strokeWidth={2} />)}
          <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 12, fontSize: 11 }} formatter={(v) => `${Math.round(Number(v))} / 100`} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
    <p className="mt-2 text-[10px] leading-4 text-muted">Each axis is scaled so the best player in this selection scores 100. It compares these players with each other, not against the whole league, and every axis is oriented so bigger is better (Fixtures = easier run).</p>
    {missing.length ? <p className="mt-1 text-[10px] leading-4 text-[var(--warn)]">No data, plotted as 0: {missing.join("; ")}.</p> : null}
  </div>;
}
