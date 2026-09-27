"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

export function DistributionViz({ p10, p50, p90, mean }: { p10: number; p50: number; p90: number; mean: number }) {
  const points = [
    { q: "P10", value: p10 },
    { q: "P25", value: p10 + (p50 - p10) * 0.5 },
    { q: "P50", value: p50 },
    { q: "P75", value: p50 + (p90 - p50) * 0.5 },
    { q: "P90", value: p90 },
  ];
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer>
        <AreaChart data={points} margin={{ top: 8, right: 12, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="q" stroke="var(--muted)" fontSize={11} tickLine={false} />
          <YAxis stroke="var(--muted)" fontSize={11} tickLine={false} />
          <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 10, fontSize: 12 }} formatter={(v) => [Number(v ?? 0).toFixed(2), "points"]} />
          <Area type="monotone" dataKey="value" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.16} />
          <Line type="monotone" dataKey={() => mean} stroke="var(--good)" strokeDasharray="4 4" dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TeamStrengthBars({ data }: { data: { name?: string; short_name?: string; attack: number; defence: number }[] }) {
  const rows = data.slice().sort((a, b) => b.attack - a.attack).slice(0, 10).map((r) => ({ name: r.short_name ?? r.name ?? "?", attack: r.attack, defence: r.defence }));
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="name" stroke="var(--muted)" fontSize={10} tickLine={false} />
          <YAxis stroke="var(--muted)" fontSize={10} tickLine={false} />
          <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 10, fontSize: 12 }} />
          <Bar dataKey="attack" fill="var(--accent)" radius={[4, 4, 0, 0]} />
          <Bar dataKey="defence" fill="var(--good)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function FixtureHeatmap({ fixtures }: { fixtures: { gw: number; team: string; opponent: string; home: boolean; difficulty: number }[] }) {
  const data = fixtures.map((f) => ({
    gw: `GW${f.gw}`,
    fixture: `${f.team} ${f.home ? "H" : "A"} ${f.opponent}`,
    difficulty: f.difficulty,
  }));
  return (
    <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4">
      {data.map((f, i) => (
        <div key={`${f.gw}-${f.fixture}-${i}`} className="rounded-xl border border-line bg-bg/60 p-3">
          <div className="flex items-center justify-between gap-3 text-[11px] uppercase tracking-wider text-muted">
            <span>{f.gw}</span>
            <span>{f.difficulty.toFixed(1)}/5</span>
          </div>
          <div className="mt-2 text-sm font-medium">{f.fixture}</div>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-line">
            <div className="h-full rounded-full bg-[var(--accent)]" style={{ width: `${Math.min(100, f.difficulty / 5 * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export function MarketBars({ rows }: { rows: { name: string; value: number }[] }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 28 }}>
          <CartesianGrid stroke="var(--line)" horizontal={false} />
          <XAxis type="number" stroke="var(--muted)" fontSize={10} tickLine={false} />
          <YAxis dataKey="name" type="category" width={54} stroke="var(--muted)" fontSize={10} tickLine={false} />
          <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 10, fontSize: 12 }} />
          <Bar dataKey="value" fill="var(--accent)" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
