"use client";

import { toNum } from "@/lib/format";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/**
 * The stored quantile grid rendered as a density.
 *
 * This is the reason the simulation stores a full CDF with every prediction
 * rather than a mean: a captaincy decision is about the shape of the
 * distribution, not its centre. Two players with the same xP and different
 * spreads are different assets, and this is where that becomes visible.
 */
export function DistributionChart({
  grid,
  mean,
}: {
  grid: number[];
  mean: number;
}) {
  if (!grid || grid.length < 3) {
    return (
      <p className="text-sm text-muted">
        No stored quantile grid for this prediction.
      </p>
    );
  }

  // Convert the quantile grid into a histogram over integer point values.
  const lo = Math.floor(Math.min(...grid));
  const hi = Math.ceil(Math.max(...grid));
  const buckets: { points: number; density: number }[] = [];
  const levels = grid.map((_, i) => i / (grid.length - 1));

  const cdfAt = (x: number) => {
    if (x <= grid[0]!) return 0;
    if (x >= grid[grid.length - 1]!) return 1;
    for (let i = 1; i < grid.length; i++) {
      const a = grid[i - 1]!;
      const b = grid[i]!;
      if (x <= b) {
        const t = b === a ? 0 : (x - a) / (b - a);
        return levels[i - 1]! + t * (levels[i]! - levels[i - 1]!);
      }
    }
    return 1;
  };

  for (let x = lo; x <= hi; x++) {
    buckets.push({ points: x, density: cdfAt(x + 0.5) - cdfAt(x - 0.5) });
  }

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer>
        <AreaChart data={buckets} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis
            dataKey="points"
            stroke="var(--muted)"
            fontSize={11}
            tickLine={false}
          />
          <YAxis
            stroke="var(--muted)"
            fontSize={11}
            tickLine={false}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <Tooltip
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(v) => [`${(toNum(v) * 100).toFixed(1)}%`, "probability"]}
            labelFormatter={(l) => `${l} points`}
          />
          <ReferenceLine
            x={Math.round(mean)}
            stroke="var(--accent)"
            strokeDasharray="4 3"
            label={{ value: "mean", fontSize: 10, fill: "var(--accent)" }}
          />
          <Area
            type="monotone"
            dataKey="density"
            stroke="var(--accent)"
            fill="var(--accent)"
            fillOpacity={0.18}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
