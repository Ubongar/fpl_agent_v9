"use client";

import { toNum } from "@/lib/format";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TransferMove } from "@/lib/types";

const AXIS = { stroke: "var(--muted)", fontSize: 11, tickLine: false } as const;

/** Net xP added by each week's moves (incoming minus outgoing), hit cost included. */
export function TransferGainChart({ moves }: { moves: TransferMove[] }) {
  const data = moves.map((m) => {
    const inXp = m.in.reduce((a, p) => a + p.xp, 0);
    const outXp = m.out.reduce((a, p) => a + p.xp, 0);
    return {
      gw: `GW${m.gw}`,
      net: Math.round((inXp - outXp - m.hits * 4) * 100) / 100,
    };
  });

  if (data.length === 0) {
    return <p className="text-sm text-muted">No moves in this horizon -- every week is a roll.</p>;
  }

  return (
    <div className="h-52 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -16 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="gw" {...AXIS} />
          <YAxis {...AXIS} />
          <Tooltip
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(v) => { const n = toNum(v); return [`${n >= 0 ? "+" : ""}${n.toFixed(2)}`, "net xP (after hits)"]; }}
          />
          <Bar dataKey="net" radius={[3, 3, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.net >= 0 ? "var(--good)" : "var(--bad)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
