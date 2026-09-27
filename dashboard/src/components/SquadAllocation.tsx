"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SquadPlayer } from "@/lib/types";

const AXIS = { stroke: "var(--muted)", fontSize: 11, tickLine: false } as const;
const ORDER = ["GK", "DEF", "MID", "FWD"];

/**
 * Where the £100m and the eleven expected points actually went, by
 * position. A squad that reads fine as a formation can still be badly
 * lopsided in spend -- this is the check for that.
 */
export function SquadAllocation({
  xi,
  bench,
}: {
  xi: SquadPlayer[];
  bench: SquadPlayer[];
}) {
  const all = [...xi, ...bench];
  const data = ORDER.map((pos) => {
    const players = all.filter((p) => p.position === pos);
    return {
      position: pos,
      spend: Math.round(players.reduce((a, p) => a + p.price, 0) * 10) / 10,
      xp: Math.round(xi.filter((p) => p.position === pos).reduce((a, p) => a + p.xp, 0) * 100) / 100,
    };
  });

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -16 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="position" {...AXIS} />
          <YAxis {...AXIS} />
          <Tooltip
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="spend" name="Spend (£m)" fill="var(--muted)" radius={[3, 3, 0, 0]} />
          <Bar dataKey="xp" name="XI expected points" fill="var(--accent)" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
