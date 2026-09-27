"use client";

import { useState } from "react";
import { managerTeam } from "@/lib/api";
import { EmptyState, Panel, PositionBadge, Stat } from "@/components/ui";
import { pct, pts, money } from "@/lib/format";

type SquadRow = {
  player_id: number; name: string; position: string | null; club: string | null;
  price: number | null; xp: number | null; p_play: number | null;
  is_captain: boolean; is_vice_captain: boolean; multiplier: number;
  sell_price?: number | null;
};
type ManagerData = {
  entry_id: number; manager_name: string; team_name: string; season: number; gw: number;
  overall_rank: number | null; overall_points: number | null; bank: number; team_value: number;
  free_transfers: number; free_transfers_is_estimated: boolean; active_chip: string | null;
  squad: SquadRow[]; unmatched_player_ids: number[];
};

function num(v: unknown) { const n = Number(v); return Number.isFinite(n) ? n : 0; }

export default function ManagerCockpit() {
  const [entry, setEntry] = useState("");
  const [data, setData] = useState<ManagerData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    const id = Number(entry);
    if (!Number.isInteger(id) || id <= 0) return setError("Enter a valid FPL team ID.");
    setLoading(true); setError(null);
    try { setData(await managerTeam(id) as ManagerData); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not load manager team."); }
    finally { setLoading(false); }
  }

  const totalXp = data?.squad.reduce((s, p) => s + num(p.xp) * Math.max(1, p.multiplier || 1), 0) ?? 0;
  const active = data?.squad.filter(p => p.multiplier > 0) ?? [];
  const captain = data?.squad.find(p => p.is_captain);
  const availability = data?.squad.reduce((s, p) => s + (p.p_play == null ? 0 : num(p.p_play)), 0) ?? 0;
  const availabilityPct = data?.squad.length ? availability / data.squad.length : 0;

  return <div className="space-y-5">
    <div>
      <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent)]">Manager cockpit</div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">Your squad, interpreted by the model</h1>
      <p className="mt-2 max-w-3xl text-sm text-muted">Load a live FPL squad and see expected output, availability, captain exposure and squad-level risk before entering the transfer workflow.</p>
    </div>

    <Panel title="Load manager">
      <div className="flex flex-col gap-2 md:flex-row">
        <input value={entry} onChange={e => setEntry(e.target.value)} onKeyDown={e => { if (e.key === "Enter") load(); }} placeholder="FPL Entry ID e.g. 1234567" inputMode="numeric" className="flex-1 rounded-xl border border-line bg-bg px-3 py-2 text-sm" />
        <button onClick={load} disabled={loading} className="rounded-xl bg-[var(--accent)] px-5 py-2 text-sm font-semibold text-white disabled:opacity-50">{loading ? "Loading…" : "Load squad"}</button>
      </div>
      {error ? <div className="mt-3 rounded-xl border border-[var(--bad)]/30 bg-[var(--bad)]/5 p-3 text-xs text-[var(--bad)]">{error}</div> : null}
    </Panel>

    {data ? <>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
        <Stat label="Gameweek" value={`GW${data.gw}`} />
        <Stat label="Overall rank" value={data.overall_rank?.toLocaleString() ?? "—"} />
        <Stat label="Bank" value={money(data.bank)} />
        <Stat label="Team value" value={money(data.team_value)} />
        <Stat label="Free transfers" value={`${data.free_transfers}${data.free_transfers_is_estimated ? "*" : ""}`} />
        <Stat label="Squad xP" value={pts(totalXp)} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title={data.team_name} subtitle={`${data.manager_name} · ${data.season}/${String(data.season + 1).slice(-2)}`}>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between"><span className="text-muted">Overall points</span><span className="font-mono">{data.overall_points?.toLocaleString() ?? "—"}</span></div>
            <div className="flex justify-between"><span className="text-muted">Captain</span><span className="font-medium">{captain?.name ?? "—"}</span></div>
            <div className="flex justify-between"><span className="text-muted">Active chip</span><span className="font-medium">{data.active_chip ?? "None"}</span></div>
            <div className="flex justify-between"><span className="text-muted">Players loaded</span><span className="font-mono">{data.squad.length}/15</span></div>
          </div>
        </Panel>
        <Panel title="Availability surface" subtitle="Average P(play) across the loaded squad">
          <div className="text-4xl font-semibold">{pct(availabilityPct)}</div>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full bg-[var(--accent)]" style={{width:`${Math.max(0, Math.min(100, availabilityPct * 100))}%`}} /></div>
          <p className="mt-3 text-xs text-muted">This is an availability estimate, not a guarantee of minutes or starts.</p>
        </Panel>
        <Panel title="On-pitch projection" subtitle="Projected contribution from players currently carrying multipliers">
          <div className="text-4xl font-semibold">{pts(active.reduce((s,p)=>s+num(p.xp)*(p.multiplier || 1),0))}</div>
          <p className="mt-3 text-xs text-muted">Use the Transfer Planner next for hit cost, sell-price effects, replacement chains and multi-GW consequences.</p>
        </Panel>
      </div>

      <Panel title="Squad X-ray" subtitle="Prediction context attached to each current pick">
        <div className="overflow-x-auto"><table className="w-full min-w-[900px] text-sm"><thead><tr className="border-b border-line text-[10px] uppercase tracking-wider text-muted"><th className="px-3 py-3 text-left">Player</th><th className="px-3 py-3 text-left">Club</th><th className="px-3 py-3 text-right">Price</th><th className="px-3 py-3 text-right">xP</th><th className="px-3 py-3 text-right">P(play)</th><th className="px-3 py-3 text-right">Multiplier</th><th className="px-3 py-3 text-right">Sell</th></tr></thead><tbody>{data.squad.map(p=><tr key={p.player_id} className="border-b border-line/70"><td className="px-3 py-3"><div className="flex items-center gap-2"><PositionBadge position={p.position ?? ""}/><span className="font-medium">{p.name}</span>{p.is_captain ? <span className="rounded-md bg-[var(--accent)]/15 px-1.5 py-0.5 text-[10px] text-[var(--accent)]">C</span> : null}{p.is_vice_captain ? <span className="rounded-md border border-line px-1.5 py-0.5 text-[10px] text-muted">VC</span> : null}</div></td><td className="px-3 py-3 text-muted">{p.club ?? "—"}</td><td className="px-3 py-3 text-right font-mono">{p.price == null ? "—" : money(p.price)}</td><td className="px-3 py-3 text-right font-mono">{p.xp == null ? "—" : pts(p.xp)}</td><td className="px-3 py-3 text-right font-mono">{p.p_play == null ? "—" : pct(p.p_play)}</td><td className="px-3 py-3 text-right font-mono">{p.multiplier}</td><td className="px-3 py-3 text-right font-mono">{p.sell_price == null ? "—" : money(p.sell_price)}</td></tr>)}</tbody></table></div>
        {data.unmatched_player_ids.length ? <p className="mt-3 text-xs text-[var(--warn)]">{data.unmatched_player_ids.length} squad player(s) have no stored prediction for GW{data.gw}. Run the prediction pipeline before using transfer or captain decisions.</p> : <p className="mt-3 text-xs text-muted">All 15 squad picks have a stored model prediction.</p>}
      </Panel>
    </> : <EmptyState title="No manager loaded" detail="Enter your FPL Entry ID to turn the dashboard into a personal decision cockpit." />}
  </div>;
}
