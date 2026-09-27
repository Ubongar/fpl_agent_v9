"use client";

import { useMemo, useState } from "react";
import type { PlayerPrediction } from "@/lib/types";
import { money, pct, pts } from "@/lib/format";
import { PlayerLink, PositionBadge } from "./ui";

type SortKey = "xp_mean" | "now_cost" | "value" | "p_play" | "p_haul_10plus";

const COLUMNS: { key: SortKey; label: string; hint: string }[] = [
  { key: "xp_mean", label: "xP", hint: "Mean simulated points" },
  { key: "now_cost", label: "Price", hint: "Current FPL price" },
  { key: "value", label: "xP / £m", hint: "Points per million" },
  { key: "p_play", label: "Play", hint: "P(appears)" },
  { key: "p_haul_10plus", label: "Haul", hint: "P(10 or more points)" },
];

export function PlayerTable({ players }: { players: PlayerPrediction[] }) {
  const [position, setPosition] = useState<string>("ALL");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<SortKey>("xp_mean");

  const rows = useMemo(() => {
    let r = players;
    if (position !== "ALL") r = r.filter((p) => p.position === position);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      r = r.filter(
        (p) =>
          p.web_name.toLowerCase().includes(q) ||
          (p.short_name ?? "").toLowerCase().includes(q),
      );
    }
    return [...r].sort((a, b) => (b[sort] ?? 0) - (a[sort] ?? 0));
  }, [players, position, search, sort]);

  // The largest standard error on screen sets the bar for what counts as a
  // real difference. Two projections closer together than this are noise,
  // and the table says so rather than implying a ranking it cannot support.
  const maxSe = useMemo(
    () => Math.max(0, ...rows.map((r) => r.xp_se ?? 0)),
    [rows],
  );

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search player or club"
          className="rounded-md border border-line bg-bg px-3 py-1.5 text-sm outline-none focus:border-[var(--accent)]"
        />
        <div className="flex gap-1">
          {["ALL", "GK", "DEF", "MID", "FWD"].map((p) => (
            <button
              key={p}
              onClick={() => setPosition(p)}
              className={
                "rounded-md border px-2.5 py-1.5 text-xs " +
                (position === p
                  ? "border-[var(--accent)] text-[var(--accent)]"
                  : "border-line text-muted hover:text-fg")
              }
            >
              {p}
            </button>
          ))}
        </div>
        <span className="ml-auto text-xs text-muted">
          {rows.length} players
          {maxSe > 0
            ? ` · gaps under ${(2 * maxSe).toFixed(2)} pts are within simulation noise`
            : ""}
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="bg-panel text-left text-xs text-muted">
              <th className="px-3 py-2 font-medium">Player</th>
              <th className="px-3 py-2 font-medium">Pos</th>
              <th className="px-3 py-2 font-medium">Club</th>
              {COLUMNS.map((c) => (
                <th key={c.key} className="px-3 py-2 text-right font-medium">
                  <button
                    title={c.hint}
                    onClick={() => setSort(c.key)}
                    className={
                      sort === c.key ? "text-[var(--accent)]" : "hover:text-fg"
                    }
                  >
                    {c.label}
                  </button>
                </th>
              ))}
              <th className="px-3 py-2 text-right font-medium" title="Simulated p10 to p90">
                Range
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.player_id} className="border-t border-line">
                <td className="px-3 py-2">
                  <PlayerLink id={p.player_id} name={p.web_name} />
                  {p.n_fixtures > 1 ? (
                    <span
                      className="ml-1.5 rounded bg-[var(--accent)]/15 px-1 text-[10px] text-[var(--accent)]"
                      title="Double gameweek: the distribution is the convolution of both fixtures, not the maximum"
                    >
                      DGW
                    </span>
                  ) : null}
                </td>
                <td className="px-3 py-2">
                  <PositionBadge position={p.position} />
                </td>
                <td className="px-3 py-2 text-muted">{p.short_name ?? "-"}</td>
                <td className="px-3 py-2 text-right font-mono">{pts(p.xp_mean)}</td>
                <td className="px-3 py-2 text-right font-mono">{money(p.now_cost)}</td>
                <td className="px-3 py-2 text-right font-mono">{pts(p.value, 2)}</td>
                <td className="px-3 py-2 text-right font-mono">{pct(p.p_play)}</td>
                <td className="px-3 py-2 text-right font-mono">
                  {pct(p.p_haul_10plus)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-xs text-muted">
                  {pts(p.xp_p10, 1)} to {pts(p.xp_p90, 1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
