"use client";

import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { PlayerPrediction } from "@/lib/types";

const AXIS = { stroke: "var(--muted)", fontSize: 11, tickLine: false } as const;

const POS_COLOUR: Record<string, string> = {
  GK: "#d97706",
  DEF: "#059669",
  MID: "#0284c7",
  FWD: "#e11d48",
};

/**
 * Price against expected points, one dot per player. The players worth
 * looking at twice are the ones sitting above the pack for their price band
 * -- exactly what a table sorted by xP alone hides, because it never shows
 * what price bought a nearby rank.
 */
export function ValueScatter({ players }: { players: PlayerPrediction[] }) {
  const data = players
    .filter((p) => p.now_cost != null)
    .map((p) => ({
      price: p.now_cost as number,
      xp: p.xp_mean,
      name: p.web_name,
      club: p.short_name ?? "",
      position: p.position,
    }));

  if (data.length === 0) {
    return <p className="text-sm text-muted">No priced players yet.</p>;
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
          <CartesianGrid stroke="var(--line)" />
          <XAxis
            type="number"
            dataKey="price"
            name="price"
            domain={["dataMin - 0.3", "dataMax + 0.3"]}
            tickFormatter={(v: number) => `£${v.toFixed(0)}m`}
            {...AXIS}
          />
          <YAxis type="number" dataKey="xp" name="xP" {...AXIS} />
          <ZAxis range={[26, 26]} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3", stroke: "var(--line)" }}
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              fontSize: 12,
            }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]!.payload as (typeof data)[number];
              return (
                <div className="rounded-md border border-line bg-panel px-2.5 py-1.5 text-xs shadow-sm">
                  <div className="font-medium">
                    {d.name} <span className="text-muted">{d.club}</span>
                  </div>
                  <div className="text-muted">
                    £{d.price.toFixed(1)}m &middot; {d.xp.toFixed(2)} xP
                  </div>
                </div>
              );
            }}
          />
          <Scatter data={data} fillOpacity={0.75}>
            {data.map((d, i) => (
              <Cell key={i} fill={POS_COLOUR[d.position] ?? "var(--accent)"} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
        {Object.entries(POS_COLOUR).map(([pos, colour]) => (
          <span key={pos} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: colour }}
            />
            {pos}
          </span>
        ))}
      </div>
    </div>
  );
}
