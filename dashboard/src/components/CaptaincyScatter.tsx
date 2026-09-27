"use client";

import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { CaptainCandidate } from "@/lib/types";

const AXIS = { stroke: "var(--muted)", fontSize: 11, tickLine: false } as const;

/**
 * Floor against ceiling, one bubble per candidate, sized by haul
 * probability. xP alone can't show this trade-off: two players with the
 * same mean can have very different distributions, and captaincy is a bet
 * on the shape, not the mean.
 */
export function CaptaincyScatter({ candidates }: { candidates: CaptainCandidate[] }) {
  const data = candidates
    .filter((c) => c.floor_p10 != null && c.ceiling_p90 != null)
    .map((c) => ({
      floor: c.floor_p10 as number,
      ceiling: c.ceiling_p90 as number,
      haul: c.p_haul ?? 0,
      name: c.name,
    }));

  if (data.length === 0) {
    return <p className="text-sm text-muted">No floor/ceiling data yet.</p>;
  }

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 4, left: -12 }}>
          <CartesianGrid stroke="var(--line)" />
          <XAxis
            type="number"
            dataKey="floor"
            name="floor"
            {...AXIS}
            label={{ value: "Floor (p10)", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--muted)" }}
          />
          <YAxis
            type="number"
            dataKey="ceiling"
            name="ceiling"
            {...AXIS}
            label={{ value: "Ceiling (p90)", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--muted)" }}
          />
          <ZAxis type="number" dataKey="haul" range={[50, 400]} name="P(haul)" />
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
                  <div className="font-medium">{d.name}</div>
                  <div className="text-muted">
                    floor {d.floor.toFixed(1)} &middot; ceiling {d.ceiling.toFixed(1)} &middot;{" "}
                    {(d.haul * 100).toFixed(0)}% haul
                  </div>
                </div>
              );
            }}
          />
          <Scatter data={data} fill="var(--accent)" fillOpacity={0.55} />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="mt-1 text-[11px] text-muted">Bubble size is haul probability (10+ points).</p>
    </div>
  );
}
