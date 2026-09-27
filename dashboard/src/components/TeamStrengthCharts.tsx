"use client";

import { toNum } from "@/lib/format";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TeamStrength } from "@/lib/types";

const AXIS = { stroke: "var(--muted)", fontSize: 11, tickLine: false } as const;
const TIP = {
  contentStyle: {
    background: "var(--panel)",
    border: "1px solid var(--line)",
    borderRadius: 6,
    fontSize: 12,
  },
} as const;

/**
 * Attack against defence, fitted from Dixon-Coles. Bottom-right is the
 * quadrant to target for attacking picks against; top-left is where clean
 * sheets get backed.
 */
export function AttackDefenceScatter({ teams }: { teams: TeamStrength[] }) {
  const data = teams.map((t) => ({
    attack: t.attack,
    defence: t.defence,
    name: t.short_name ?? String(t.team_id),
  }));
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 20, left: -8 }}>
          <CartesianGrid stroke="var(--line)" />
          <XAxis
            type="number"
            dataKey="attack"
            name="attack"
            {...AXIS}
            label={{ value: "Attack rating (higher = stronger)", position: "insideBottom", offset: -8, fontSize: 11, fill: "var(--muted)" }}
          />
          <YAxis
            type="number"
            dataKey="defence"
            name="defence"
            reversed
            {...AXIS}
            label={{ value: "Defence rating (lower = stronger)", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--muted)" }}
          />
          <Tooltip
            {...TIP}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]!.payload as (typeof data)[number];
              return (
                <div className="rounded-md border border-line bg-panel px-2.5 py-1.5 text-xs shadow-sm">
                  <div className="font-medium">{d.name}</div>
                  <div className="text-muted">
                    attack {d.attack.toFixed(2)} &middot; defence {d.defence.toFixed(2)}
                  </div>
                </div>
              );
            }}
          />
          <Scatter data={data} fill="var(--accent)" fillOpacity={0.75} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Fitted home advantage by team -- how much of the home/away split is real. */
export function HomeAdvantageBar({ teams }: { teams: TeamStrength[] }) {
  const data = [...teams]
    .sort((a, b) => b.home_adv - a.home_adv)
    .map((t) => ({ name: t.short_name ?? String(t.team_id), home_adv: t.home_adv }));
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
          <CartesianGrid stroke="var(--line)" horizontal={false} />
          <XAxis type="number" {...AXIS} />
          <YAxis type="category" dataKey="name" {...AXIS} width={44} />
          <Tooltip {...TIP} formatter={(v) => [toNum(v).toFixed(2), "home advantage"]} />
          <Bar dataKey="home_adv" fill="var(--accent)" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
