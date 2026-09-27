import { fixtureOutlook } from "@/lib/api";
import { EmptyState, Panel, Stat } from "@/components/ui";
import { FixtureHeatmap } from "@/components/Viz";

export const dynamic = "force-dynamic";

export default async function FixturesPage() {
  let data: Awaited<ReturnType<typeof fixtureOutlook>> | null = null;
  let error: string | null = null;
  try { data = await fixtureOutlook(undefined, 6); } catch(e){ error = e instanceof Error ? e.message : "Could not load fixture outlook."; }
  if (!data) return <EmptyState title="Fixture Lab unavailable" detail={error ?? "No future fixtures stored."} command="fplagent ingest" />;
  const easiest = data.teams.slice(0,5); const hardest = data.teams.slice(-5).reverse();
  return <div className="space-y-5">
    <div><div className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent)]">Schedule intelligence</div><h1 className="mt-2 text-3xl font-semibold tracking-tight">Fixture Lab</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-muted">A forward-looking schedule surface built from the current team-strength model. Difficulty is a proxy for context, not an official FDR replacement.</p></div>
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4"><Stat label="Horizon" value={`${data.horizon} GWs`} /><Stat label="Fixtures mapped" value={String(data.fixtures.length / 2)} /><Stat label="Teams" value={String(data.teams.length)} /><Stat label="Start GW" value={`GW${data.start_gw}`} /></div>
    <Panel title="Fixture matrix" subtitle="Lower difficulty means the current fitted matchup surface is friendlier. Home/away is shown inside every card."><FixtureHeatmap fixtures={data.fixtures.slice(0,24)} /></Panel>
    <div className="grid gap-4 lg:grid-cols-2"><Panel title="Schedule runways"><div className="space-y-2">{easiest.map(t=><div key={t.team_id} className="flex items-center justify-between rounded-xl border border-line bg-bg/60 p-3"><div><div className="text-sm font-semibold">{t.team}</div><div className="text-[11px] text-muted">Best {t.best_fixture.toFixed(1)} · worst {t.worst_fixture.toFixed(1)}</div></div><div className="font-mono text-sm">{t.avg_difficulty.toFixed(2)}</div></div>)}</div></Panel><Panel title="Schedule friction"><div className="space-y-2">{hardest.map(t=><div key={t.team_id} className="flex items-center justify-between rounded-xl border border-line bg-bg/60 p-3"><div><div className="text-sm font-semibold">{t.team}</div><div className="text-[11px] text-muted">Worst {t.worst_fixture.toFixed(1)} · best {t.best_fixture.toFixed(1)}</div></div><div className="font-mono text-sm">{t.avg_difficulty.toFixed(2)}</div></div>)}</div></Panel></div>
  </div>;
}
