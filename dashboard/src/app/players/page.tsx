"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { getJSON, BrowserApiError } from "@/lib/browserApi";
import { money, pct, pts } from "@/lib/format";
import { EmptyState, PositionBadge } from "@/components/ui";
import type { DashboardPlayer, PredictionsPage } from "@/lib/types";

const POS = ["ALL", "GK", "DEF", "MID", "FWD"];
const COMPARE_STORAGE = "fpl-agent-compare-v3";

export default function PlayerLab() {
  const [rows, setRows] = useState<DashboardPlayer[]>([]);
  const [query, setQuery] = useState("");
  const [position, setPosition] = useState("ALL");
  const [sort, setSort] = useState("xp_mean");
  const [onlyPlayable, setOnlyPlayable] = useState(false);
  const [selected, setSelected] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);

  // localStorage is unavailable during server render, so the saved comparison
  // set must be read after mount; this one-time setState-in-effect is intentional.
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(COMPARE_STORAGE) ?? "[]");
      if (Array.isArray(saved)) setSelected(saved.filter((x: unknown): x is number => typeof x === "number").slice(0, 8));
    } catch {}
    getJSON<PredictionsPage & { players: DashboardPlayer[] }>("/predictions?limit=700&sort=xp_mean&desc=true")
      .then((r) => setRows(r.players))
      .catch((e) => setError(e instanceof BrowserApiError ? e.message : "Could not load players."));
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rows
      .filter((p) => position === "ALL" || p.position === position)
      .filter((p) => !q || p.web_name.toLowerCase().includes(q) || (p.short_name ?? "").toLowerCase().includes(q))
      .filter((p) => !onlyPlayable || p.p_play >= 0.8)
      .slice()
      .sort((a, b) => Number(b[sort as keyof DashboardPlayer] ?? 0) - Number(a[sort as keyof DashboardPlayer] ?? 0));
  }, [rows, query, position, sort, onlyPlayable]);

  function toggle(id: number) {
    setSelected((prev) => {
      const next = prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 8 ? [...prev, id] : prev;
      localStorage.setItem(COMPARE_STORAGE, JSON.stringify(next));
      return next;
    });
  }

  if (error) return <EmptyState title="Player Lab offline" detail={error} command="fplagent predict" />;

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div><div className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent)]">Intelligence surface</div><h1 className="mt-2 text-3xl font-semibold tracking-tight">Player Lab</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-muted">Search the full player pool, inspect the distribution instead of only the mean, and build an explicit comparison set before you decide. Select up to eight players.</p></div>
        {selected.length >= 2 ? <Link href={`/compare?players=${selected.join(",")}`} className="rounded-xl bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white">Compare {selected.length} →</Link> : null}
      </div>

      <div className="rounded-2xl border border-line bg-panel p-4">
        <div className="grid gap-2 md:grid-cols-[1.5fr_0.7fr_0.8fr_auto]">
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search player or club" className="rounded-xl border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-[var(--accent)]" />
          <select value={position} onChange={(e) => setPosition(e.target.value)} className="rounded-xl border border-line bg-bg px-3 py-2 text-sm outline-none">{POS.map(p=><option key={p}>{p}</option>)}</select>
          <select value={sort} onChange={(e) => setSort(e.target.value)} className="rounded-xl border border-line bg-bg px-3 py-2 text-sm outline-none">
            {[["xp_mean","Expected points"],["value","Value / £m"],["xp_p90","P90 ceiling"],["ownership_adjusted_upside","Differential upside"],["certainty","Model certainty"],["p_play","P(play)"],["minutes_stability","Minutes stability"]].map(([v,l])=><option key={v} value={v}>{l}</option>)}
          </select>
          <label className="flex items-center justify-center gap-2 rounded-xl border border-line bg-bg px-3 py-2 text-xs text-muted"><input type="checkbox" checked={onlyPlayable} onChange={e=>setOnlyPlayable(e.target.checked)} /> P(play) ≥ 80%</label>
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-line bg-panel">
        <div className="max-h-[720px] overflow-auto scrollbar-thin">
          <table className="w-full min-w-[1120px] text-sm">
            <thead className="sticky top-0 z-10 bg-panel">
              <tr className="border-b border-line text-left text-[10px] uppercase tracking-wider text-muted">
                <th className="px-3 py-3">Compare</th><th className="px-3 py-3">Player</th><th className="px-3 py-3 text-right">Price</th><th className="px-3 py-3 text-right">xP</th><th className="px-3 py-3 text-right">P10</th><th className="px-3 py-3 text-right">P90</th><th className="px-3 py-3 text-right">P(play)</th><th className="px-3 py-3 text-right">P(start)</th><th className="px-3 py-3 text-right">P60+</th><th className="px-3 py-3 text-right">Value</th><th className="px-3 py-3 text-right">Upside</th><th className="px-3 py-3 text-right">Certainty</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => (
                <tr key={p.player_id} className="border-b border-line/70 last:border-0 hover:bg-bg/45">
                  <td className="px-3 py-3"><input aria-label={`Select ${p.web_name}`} type="checkbox" checked={selected.includes(p.player_id)} onChange={() => toggle(p.player_id)} /></td>
                  <td className="px-3 py-3"><Link href={`/players/${p.player_id}`} className="group flex items-center gap-2"><PositionBadge position={p.position}/><span className="font-semibold group-hover:text-[var(--accent)]">{p.web_name}</span><span className="text-[11px] text-muted">{p.short_name}</span></Link></td>
                  <td className="px-3 py-3 text-right font-mono">{money(p.now_cost)}</td>
                  <td className="px-3 py-3 text-right font-mono font-semibold">{pts(p.xp_mean)}</td>
                  <td className="px-3 py-3 text-right font-mono text-muted">{pts(p.xp_p10,1)}</td>
                  <td className="px-3 py-3 text-right font-mono">{pts(p.xp_p90,1)}</td>
                  <td className="px-3 py-3 text-right font-mono">{pct(p.p_play)}</td>
                  <td className="px-3 py-3 text-right font-mono">{pct(p.p_start)}</td>
                  <td className="px-3 py-3 text-right font-mono">{pct(p.p_sixty)}</td>
                  <td className="px-3 py-3 text-right font-mono">{p.value?.toFixed(2) ?? "—"}</td>
                  <td className="px-3 py-3 text-right font-mono">+{p.upside_points?.toFixed(1) ?? "—"}</td>
                  <td className="px-3 py-3 text-right font-mono">{pct(p.certainty)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="sticky bottom-0 flex flex-col gap-2 border-t border-line bg-panel/95 px-4 py-3 text-[11px] backdrop-blur md:flex-row md:items-center md:justify-between"><span className="text-muted">{filtered.length} players shown · {selected.length} selected · selections persist while you move between Player Lab and Compare.</span><div className="flex items-center gap-2">{selected.length ? <button onClick={() => { setSelected([]); localStorage.removeItem(COMPARE_STORAGE); }} className="rounded-lg border border-line px-2.5 py-1.5 text-[10px] text-muted hover:text-fg">Clear</button> : null}{selected.length >= 2 ? <Link href={`/compare?players=${selected.join(",")}`} className="rounded-lg bg-[var(--accent)] px-3 py-1.5 text-[10px] font-semibold text-white">Open comparison desk →</Link> : <span className="text-[10px] text-muted">Select 2–8 players to compare</span>}</div></div>
      </div>
    </div>
  );
}
