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
import type { AttributionChange } from "@/lib/types";

const AXIS = { stroke: "var(--muted)", fontSize: 11, tickLine: false } as const;

/** Biggest risers and fallers, one bar per player, coloured by direction. */
export function MoversChart({ changes }: { changes: AttributionChange[] }) {
  const data = [...changes]
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, 12)
    .map((c) => ({ name: c.name, delta: Math.round(c.delta * 100) / 100 }))
    .sort((a, b) => a.delta - b.delta);

  if (data.length === 0) {
    return <p className="text-sm text-muted">No movers to show.</p>;
  }

  return (
    <div style={{ height: Math.max(220, data.length * 28) }} className="w-full">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
          <CartesianGrid stroke="var(--line)" horizontal={false} />
          <XAxis type="number" {...AXIS} />
          <YAxis type="category" dataKey="name" {...AXIS} width={92} />
          <Tooltip
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(v) => { const n = toNum(v); return [`${n >= 0 ? "+" : ""}${n.toFixed(2)}`, "xP change"]; }}
          />
          <Bar dataKey="delta" radius={[0, 3, 3, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.delta >= 0 ? "var(--good)" : "var(--bad)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
