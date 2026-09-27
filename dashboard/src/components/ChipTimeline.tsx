"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChipSchedule } from "@/lib/types";

const CHIP_LABEL: Record<string, string> = {
  wildcard: "Wildcard",
  free_hit: "Free Hit",
  bench_boost: "Bench Boost",
  triple_captain: "Triple Captain",
};

const FIRST_HALF_LAST_GW = 19;
const TOTAL_GWS = 38;

function Track({
  slot,
  gw,
}: {
  slot: "first" | "second";
  gw: number | null;
}) {
  const start = slot === "first" ? 1 : FIRST_HALF_LAST_GW + 1;
  const end = slot === "first" ? FIRST_HALF_LAST_GW : TOTAL_GWS;
  const span = end - start;
  const pct = gw != null ? ((gw - start) / span) * 100 : null;
  return (
    <div className="relative h-6 flex-1 rounded bg-line">
      {pct != null ? (
        <div
          className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded bg-[var(--accent)] px-1.5 py-0.5 text-[10px] font-medium text-white"
          style={{ left: `${Math.min(96, Math.max(4, pct))}%` }}
        >
          GW{gw}
        </div>
      ) : (
        <div className="flex h-full items-center justify-center text-[10px] text-muted">
          not scheduled
        </div>
      )}
    </div>
  );
}

/** GW1-19 and GW20-38 tracks per chip, with the assigned week marked. */
export function ChipTimeline({ schedule }: { schedule: ChipSchedule }) {
  return (
    <div className="space-y-4">
      <div className="flex gap-2 pl-24 text-[10px] text-muted">
        <span className="flex-1">First half (GW1-{FIRST_HALF_LAST_GW})</span>
        <span className="flex-1">Second half (GW{FIRST_HALF_LAST_GW + 1}-{TOTAL_GWS})</span>
      </div>
      {Object.entries(schedule.assignments).map(([chip, a]) => (
        <div key={chip} className="flex items-center gap-2">
          <span className="w-24 shrink-0 text-xs text-muted">{CHIP_LABEL[chip] ?? chip}</span>
          <Track slot="first" gw={a.first} />
          <Track slot="second" gw={a.second} />
        </div>
      ))}
    </div>
  );
}

/** What each scheduled chip is worth, given the values entered in the form. */
export function ChipValueBar({
  schedule,
  values,
}: {
  schedule: ChipSchedule;
  values: Record<string, Record<number, number>>;
}) {
  const data = Object.entries(schedule.assignments).map(([chip, a]) => {
    const v =
      (a.first != null ? values[chip]?.[a.first] ?? 0 : 0) +
      (a.second != null ? values[chip]?.[a.second] ?? 0 : 0);
    return { name: CHIP_LABEL[chip] ?? chip, value: Math.round(v * 100) / 100 };
  });
  return (
    <div className="h-48 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -16 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="name" stroke="var(--muted)" fontSize={11} tickLine={false} />
          <YAxis stroke="var(--muted)" fontSize={11} tickLine={false} />
          <Tooltip
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              fontSize: 12,
            }}
          />
          <Bar dataKey="value" fill="var(--accent)" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
