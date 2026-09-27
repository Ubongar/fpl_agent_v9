"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { BrowserApiError, getJSON, postJSON } from "@/lib/browserApi";
import type { CompareResponse, DashboardPlayer, PredictionsPage } from "@/lib/types";
import { EmptyState } from "@/components/ui";
import { CompareHeader, CompareLensTabs, CompareSummaryCard, ComparisonGrid, ProfileBars, type CompareLens, lensMeta } from "@/components/CompareUI";
import { money, pct, pts } from "@/lib/format";

type Pairwise = {
  player_a: { player_id: number; name: string; xp: number };
  player_b: { player_id: number; name: string; xp: number };
  p_a_wins: number;
  p_b_wins: number;
  p_tie: number;
  note?: string;
};

const STORAGE_KEY = "fpl-agent-compare-v3";

export default function PlayerComparePage() {
  const [pool, setPool] = useState<DashboardPlayer[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [query, setQuery] = useState("");
  const [position, setPosition] = useState("ALL");
  const [maxPrice, setMaxPrice] = useState(15);
  const [lens, setLens] = useState<CompareLens>("output");
  const [data, setData] = useState<CompareResponse | null>(null);
  const [anchorId, setAnchorId] = useState(0);
  const [pairwise, setPairwise] = useState<Pairwise | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPicker, setShowPicker] = useState(true);

  useEffect(() => {
    let initial: number[] = [];
    const params = new URLSearchParams(window.location.search);
    const urlPlayers = params.get("players");
    if (urlPlayers) initial = urlPlayers.split(",").map(Number).filter(Number.isInteger).slice(0, 8);
    else {
      try { initial = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]").filter((x: unknown): x is number => typeof x === "number").slice(0, 8); } catch { initial = []; }
    }
    setSelected(initial);
    if (initial.length) setAnchorId(initial[0]);
    getJSON<PredictionsPage & { players: DashboardPlayer[] }>("/predictions?limit=700&sort=xp_mean&desc=true")
      .then(r => setPool(r.players))
      .catch(e => setError(e instanceof BrowserApiError ? e.message : "Could not load the player pool."));
  }, []);

  useEffect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify(selected)); }, [selected]);

  const candidates = useMemo(() => {
    const q = query.trim().toLowerCase();
    return pool.filter(p =>
      (position === "ALL" || p.position === position) &&
      (p.now_cost == null || p.now_cost <= maxPrice) &&
      (!q || p.web_name.toLowerCase().includes(q) || (p.short_name ?? "").toLowerCase().includes(q))
    ).slice(0, 30);
  }, [pool, query, position, maxPrice]);

  const selectedPlayers = useMemo(() => pool.filter(p => selected.includes(p.player_id)), [pool, selected]);

  async function runComparison(ids = selected) {
    if (ids.length < 2) return;
    setLoading(true); setError(null); setPairwise(null);
    try {
      const result = await postJSON<CompareResponse>("/players/compare", { player_ids: ids });
      setData(result);
      if (!ids.includes(anchorId)) setAnchorId(ids[0]);
      setShowPicker(false);
    } catch (e) { setError(e instanceof BrowserApiError ? e.message : "Comparison failed."); }
    finally { setLoading(false); }
  }

  function toggle(id: number) {
    setSelected(prev => {
      const next = prev.includes(id) ? prev.filter(x => x !== id) : prev.length < 8 ? [...prev, id] : prev;
      if (next.length && !next.includes(anchorId)) setAnchorId(next[0]);
      return next;
    });
  }

  async function runPairwise() {
    const others = selectedPlayers.filter(p => p.player_id !== anchorId);
    if (!others.length || !anchorId) return;
    try {
      const result = await postJSON<Pairwise>("/compare/captaincy", { player_a_id: anchorId, player_b_id: others[0].player_id, same_team: anchorId === others[0].player_id ? false : anchorId != null && anchorId !== 0 && (data?.players.find(p => p.player_id === anchorId)?.team_id === others[0].team_id) });
      setPairwise(result);
    } catch (e) { setError(e instanceof BrowserApiError ? e.message : "Could not calculate the head-to-head distribution."); }
  }

  if (error && !pool.length) return <EmptyState title="Player Compare offline" detail={error} command="uvicorn fplagent.api.main:app --port 8000" />;

  return <div className="space-y-5">
    <section className="panel-glow soft-grid overflow-hidden rounded-3xl border border-line p-5 md:p-7">
      <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
        <div className="max-w-4xl"><div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--accent)]">Head-to-head research desk</div><h1 className="mt-2 text-3xl font-semibold tracking-tight md:text-5xl">Player Compare</h1><p className="mt-3 text-sm leading-6 text-muted md:text-base">Build a real comparison set and inspect the trade-off between output, floor, ceiling, minutes security, value and ownership leverage. The screen never collapses those dimensions into one magic score.</p></div>
        <div className="flex shrink-0 flex-wrap gap-2"><span className="rounded-xl border border-line bg-panel px-3 py-2 text-[10px] text-muted">{selected.length}/8 selected</span><button onClick={() => setShowPicker(v => !v)} className="rounded-xl border border-line bg-panel px-3 py-2 text-[10px] font-semibold hover:border-[var(--accent)]">{showPicker ? "Hide player picker" : "Change players"}</button></div>
      </div>
    </section>

    {showPicker ? <section className="rounded-2xl border border-line bg-panel p-4 md:p-5">
      <div className="grid gap-2 md:grid-cols-[1.6fr_.7fr_.6fr_auto]">
        <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search a player or club" className="rounded-xl border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-[var(--accent)]" />
        <select value={position} onChange={e => setPosition(e.target.value)} className="rounded-xl border border-line bg-bg px-3 py-2 text-sm outline-none">{["ALL","GK","DEF","MID","FWD"].map(p => <option key={p}>{p}</option>)}</select>
        <label className="rounded-xl border border-line bg-bg px-3 py-2 text-xs text-muted">Max £{maxPrice.toFixed(1)}m<input aria-label="Maximum price" type="range" min="4" max="16" step="0.1" value={maxPrice} onChange={e => setMaxPrice(Number(e.target.value))} className="mt-1 block w-full" /></label>
        <button disabled={selected.length < 2 || loading} onClick={() => runComparison()} className="rounded-xl bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40">{loading ? "Comparing…" : "Compare now"}</button>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {candidates.map(p => <button key={p.player_id} onClick={() => toggle(p.player_id)} className={`rounded-xl border px-3 py-2 text-left ${selected.includes(p.player_id) ? "border-[var(--accent)]/35 bg-[var(--accent)]/10" : "border-line bg-bg/60 hover:border-[var(--accent)]/25"}`}><div className="flex items-center gap-2"><span className="text-xs font-semibold">{p.web_name}</span><span className="text-[9px] text-muted">{p.position}</span></div><div className="mt-1 text-[9px] text-muted">{money(p.now_cost)} · xP {pts(p.xp_mean,1)} · Play {pct(p.p_play)}</div></button>)}
      </div>
      {selected.length ? <div className="mt-4 flex flex-wrap gap-1.5">{selectedPlayers.map(p => <span key={p.player_id} className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--accent)]/20 bg-[var(--accent)]/8 px-2 py-1 text-[10px] text-[var(--accent)]">{p.web_name}<button onClick={() => toggle(p.player_id)} className="text-muted hover:text-fg">×</button></span>)}</div> : null}
      {error ? <div className="mt-3 rounded-xl border border-[var(--bad)]/20 bg-[var(--bad)]/8 px-3 py-2 text-xs text-[var(--bad)]">{error}</div> : null}
    </section> : null}

    {data ? <>
      <div className="flex flex-col gap-3 rounded-2xl border border-line bg-panel p-4 md:flex-row md:items-center md:justify-between">
        <div><div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-muted">Comparison lens</div><div className="mt-1 text-sm font-semibold">{lensMeta[lens].title}</div><div className="mt-1 text-xs text-muted">{lensMeta[lens].description}</div></div>
        <CompareLensTabs value={lens} onChange={setLens}/>
      </div>

      <CompareHeader players={data.players} anchorId={anchorId} onAnchorChange={setAnchorId} onRemove={(id) => { const next = selected.filter(x => x !== id); setSelected(next); setData(next.length >= 2 ? data : null); if (next.length) setAnchorId(next[0]); }} />

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">{data.players.map(p => <CompareSummaryCard key={p.player_id} p={p} lens={lens} anchorId={anchorId}/>)}</div>

      <div className="grid gap-4 xl:grid-cols-[1.35fr_.65fr]">
        <section><ComparisonGrid players={data.players} anchorId={anchorId} lens={lens}/></section>
        <section className="rounded-2xl border border-line bg-panel p-4"><div className="mb-4"><h2 className="text-sm font-semibold">Lens profile</h2><p className="mt-1 text-xs text-muted">Visualises the selected dimension only. It is not a universal ranking.</p></div><ProfileBars players={data.players} lens={lens}/></section>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="rounded-2xl border border-line bg-panel p-4"><div className="flex items-start justify-between gap-3"><div><h2 className="text-sm font-semibold">Pairwise outcome probability</h2><p className="mt-1 text-xs leading-5 text-muted">A direct simulation comparison between the reference player and another selected player.</p></div><button onClick={runPairwise} disabled={data.players.length < 2} className="rounded-lg border border-line bg-bg px-3 py-2 text-[10px] font-semibold hover:border-[var(--accent)]">Calculate duel</button></div>{pairwise ? <div className="mt-4 grid grid-cols-3 gap-2"><div className="rounded-xl bg-bg/60 p-3 text-center"><div className="text-[9px] text-muted">{pairwise.player_a.name}</div><div className="mt-1 font-mono text-xl font-semibold">{pct(pairwise.p_a_wins)}</div><div className="text-[9px] text-muted">outscores B</div></div><div className="rounded-xl bg-bg/60 p-3 text-center"><div className="text-[9px] text-muted">Tie</div><div className="mt-1 font-mono text-xl font-semibold">{pct(pairwise.p_tie)}</div><div className="text-[9px] text-muted">same simulated points</div></div><div className="rounded-xl bg-bg/60 p-3 text-center"><div className="text-[9px] text-muted">{pairwise.player_b.name}</div><div className="mt-1 font-mono text-xl font-semibold">{pct(pairwise.p_b_wins)}</div><div className="text-[9px] text-muted">outscores A</div></div></div> : <div className="mt-4 rounded-xl border border-dashed border-line p-6 text-center text-xs text-muted">Choose a reference player, then calculate the duel against the first other selected player.</div>}{pairwise?.note ? <div className="mt-3 text-[10px] text-muted">{pairwise.note}</div> : null}</section>
        <section className="rounded-2xl border border-line bg-panel p-4"><h2 className="text-sm font-semibold">What this comparison should answer</h2><div className="mt-3 grid gap-2 text-xs"><div className="rounded-xl border border-line bg-bg/60 p-3"><div className="font-semibold">Output</div><div className="mt-1 text-muted">Who has the stronger mean/median projection?</div></div><div className="rounded-xl border border-line bg-bg/60 p-3"><div className="font-semibold">Risk</div><div className="mt-1 text-muted">How much floor or minutes uncertainty are you accepting?</div></div><div className="rounded-xl border border-line bg-bg/60 p-3"><div className="font-semibold">Economics</div><div className="mt-1 text-muted">How much projection are you buying for each £m?</div></div><div className="rounded-xl border border-line bg-bg/60 p-3"><div className="font-semibold">Ownership</div><div className="mt-1 text-muted">How much upside remains after accounting for crowd exposure?</div></div></div><div className="mt-4"><Link href="/players" className="text-xs font-semibold text-[var(--accent)] hover:underline">Back to Player Lab →</Link></div></section>
      </div>
    </> : <div className="rounded-2xl border border-dashed border-line bg-panel p-10 text-center"><div className="text-sm font-semibold">Select at least two players</div><p className="mx-auto mt-2 max-w-xl text-xs leading-6 text-muted">The comparison desk becomes fully interactive once you select two or more players. Select up to eight for a wider research set.</p></div>}
  </div>;
}
