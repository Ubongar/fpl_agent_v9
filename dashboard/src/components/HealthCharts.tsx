"use client";

import { toNum } from "@/lib/format";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { CalibrationBucket, EvaluationRun } from "@/lib/types";

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
 * PIT histogram.
 *
 * If the predicted distributions are honest these bars are flat. A U shape
 * means the model is overconfident and real outcomes land in the tails more
 * often than it expects; a hump in the middle means it is underconfident.
 * No accuracy metric can tell you either of those things, which is why the
 * simulation stores a full CDF with every prediction.
 */
export function PitHistogram({ hist }: { hist: number[] }) {
  const target = 1 / hist.length;
  const data = hist.map((v, i) => ({
    bin: `${(i / hist.length).toFixed(1)}`,
    share: v,
    off: Math.abs(v - target) / target,
  }));
  return (
    <div className="h-52 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -20 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="bin" {...AXIS} />
          <YAxis {...AXIS} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
          <Tooltip
            {...TIP}
            formatter={(v) => [`${(toNum(v) * 100).toFixed(1)}%`, "share"]}
          />
          <ReferenceLine
            y={target}
            stroke="var(--good)"
            strokeDasharray="4 3"
            label={{ value: "uniform", fontSize: 10, fill: "var(--good)" }}
          />
          <Bar dataKey="share">
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d.off > 0.5 ? "var(--bad)" : "var(--accent)"}
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Predicted rate against observed rate. The diagonal is perfect. */
export function CalibrationPlot({ curve }: { curve: CalibrationBucket[] }) {
  const data = curve.map((b) => ({
    predicted: b.predicted_rate,
    actual: b.actual_rate,
    n: b.n,
  }));
  return (
    <div className="h-52 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: -20 }}>
          <CartesianGrid stroke="var(--line)" />
          <XAxis
            type="number"
            dataKey="predicted"
            domain={[0, 1]}
            {...AXIS}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <YAxis
            type="number"
            dataKey="actual"
            domain={[0, 1]}
            {...AXIS}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <ZAxis type="number" dataKey="n" range={[40, 340]} />
          <Tooltip
            {...TIP}
            formatter={(v, k) =>
              k === "n" ? [toNum(v), "players"] : [`${(toNum(v) * 100).toFixed(1)}%`, String(k)]
            }
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="var(--good)"
            strokeDasharray="4 3"
          />
          <Scatter data={data} fill="var(--accent)" fillOpacity={0.65} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Error over time (MAE, and CRPS if stored). Spearman is what captaincy and
 * squad decisions actually consume, but a model can hold rank while its
 * point estimates drift, and MAE is what catches that.
 */
export function AccuracyHistory({ runs }: { runs: EvaluationRun[] }) {
  const data = [...runs]
    .reverse()
    .map((r) => ({
      label: `GW${r.gw_from}-${r.gw_to}`,
      mae: r.mae ?? null,
      crps: r.crps ?? null,
    }));
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -14 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="label" {...AXIS} />
          <YAxis {...AXIS} />
          <Tooltip {...TIP} />
          <Line
            type="monotone"
            dataKey="mae"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={false}
            name="MAE"
          />
          <Line
            type="monotone"
            dataKey="crps"
            stroke="var(--warn)"
            strokeDasharray="4 3"
            dot={false}
            name="CRPS"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Rank correlation over time, which is the metric decisions consume. */
export function SpearmanHistory({ runs }: { runs: EvaluationRun[] }) {
  const data = [...runs]
    .reverse()
    .map((r) => ({
      label: `GW${r.gw_from}-${r.gw_to}`,
      model: r.spearman_overall ?? 0,
      rolling: r.baselines?.["rolling_6gw"]?.spearman ?? null,
      price: r.baselines?.["price"]?.spearman ?? null,
    }));
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -22 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="label" {...AXIS} />
          <YAxis {...AXIS} />
          <Tooltip {...TIP} />
          <Line
            type="monotone"
            dataKey="model"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={false}
            name="model"
          />
          <Line
            type="monotone"
            dataKey="rolling"
            stroke="var(--warn)"
            strokeDasharray="4 3"
            dot={false}
            name="6-GW rolling mean"
          />
          <Line
            type="monotone"
            dataKey="price"
            stroke="var(--muted)"
            strokeDasharray="2 3"
            dot={false}
            name="price"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
