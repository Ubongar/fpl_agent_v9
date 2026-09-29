import Link from "next/link";
import { dashboardOverview, evaluationLatest } from "@/lib/api";
import { EmptyState } from "@/components/ui";
import { Kpi, MetricPill, PlayerStrip } from "@/components/DashboardBits";
import { TeamStrengthBars } from "@/components/Viz";
import { pts } from "@/lib/format";
import type { DashboardOverview, EvaluationRun } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function CommandCenter() {
  let data: DashboardOverview | null = null;
  let health: EvaluationRun | null = null;
  let error: string | null = null;
  try {
    data = await dashboardOverview();
    try {
      const ev = await evaluationLatest(2, data.gw - 1);
      health = ev as EvaluationRun;
    } catch {}
  } catch (e) {
    error = e instanceof Error ? e.message : "Could not load the command center.";
  }

  if (!data) return <EmptyState title="Command center offline" detail={error ?? "No current projections are available."} command="uvicorn fplagent.api.main:app --port 8000\nfplagent run" />;

  const top = data.top_players[0];
  const qualityFail = data.quality.fail ?? 0;
  const qualityWarn = data.quality.warn ?? 0;

  return (
    <div className="space-y-6">
      <section className="panel-glow soft-grid overflow-hidden rounded-3xl border border-line p-5 md:p-7">
        <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
          <div className="max-w-3xl">
            <div className="flex flex-wrap items-center gap-2">
              <MetricPill label="SEASON" value={String(data.season)} />
              <MetricPill label="GW" value={String(data.gw)} />
              <MetricPill label="MODEL" value={data.model_version} />
            </div>
            <h1 className="mt-5 text-3xl font-semibold tracking-tight md:text-5xl">Decision intelligence for every FPL move.</h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-muted md:text-base">
              One surface for projection, uncertainty, fixture context, ownership, market pressure, squad construction, captaincy and chip timing. The point is not one magic number—the point is seeing the trade-offs clearly.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:w-[520px]">
            <Kpi label="Top xP" value={top ? pts(top.xp_mean) : "—"} detail={top?.web_name ?? "No leader"} />
            <Kpi label="Median xP" value={pts(data.summary.median_xp)} detail="Across priced players" />
            <Kpi label="High upside" value={String(data.summary.high_upside_players)} detail="Large p90–p50" tone="good" />
            <Kpi label="Low P(play)" value={String(data.summary.low_play_players)} detail="<75% availability" tone={data.summary.low_play_players > 20 ? "warn" : "neutral"} />
          </div>
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-[1.35fr_0.65fr]">
        <section className="rounded-2xl border border-line bg-panel p-5">
          <div className="mb-4 flex items-end justify-between gap-4">
            <div><h2 className="text-sm font-semibold">This week’s player surface</h2><p className="mt-1 text-xs text-muted">Expected points, ceiling, availability and decision metrics.</p></div>
            <Link href="/players" className="text-xs text-[var(--accent)] hover:underline">Open Player Lab →</Link>
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {data.top_players.slice(0, 6).map((p) => <PlayerStrip key={p.player_id} p={p} />)}
          </div>
          {data.top_players.length >= 2 ? <Link href={`/compare?players=${data.top_players.slice(0,2).map((p) => p.player_id).join(",")}`} className="mt-3 inline-flex rounded-xl border border-[var(--accent)]/25 bg-[var(--accent)]/8 px-3 py-2 text-[10px] font-semibold text-[var(--accent)] hover:bg-[var(--accent)]/12">Open top-pair comparison →</Link> : null}
        </section>

        <section className="rounded-2xl border border-line bg-panel p-5">
          <div className="flex items-start justify-between gap-3"><div><h2 className="text-sm font-semibold">Model watch</h2><p className="mt-1 text-xs text-muted">Validation should be visible beside the projections.</p></div><Link href="/model" className="text-xs text-[var(--accent)] hover:underline">Model Lab →</Link></div>
          <div className="mt-5 grid grid-cols-2 gap-3">
            <Kpi label="MAE" value={health?.mae != null ? health.mae.toFixed(2) : "—"} detail="Latest scored run" />
            <Kpi label="RMSE" value={health?.rmse != null ? health.rmse.toFixed(2) : "—"} detail="Lower is tighter" />
            <Kpi label="Warnings" value={String(qualityWarn)} detail="Recent quality events" tone={qualityWarn ? "warn" : "good"} />
            <Kpi label="Failures" value={String(qualityFail)} detail="Recent quality events" tone={qualityFail ? "bad" : "good"} />
          </div>
        </section>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <section className="rounded-2xl border border-line bg-panel p-5 lg:col-span-2">
          <div className="mb-4 flex items-end justify-between gap-4"><div><h2 className="text-sm font-semibold">Attack vs defensive strength</h2><p className="mt-1 text-xs text-muted">Current fitted team ratings used by the fixture model.</p></div><Link href="/fixtures" className="text-xs text-[var(--accent)] hover:underline">Fixture Lab →</Link></div>
          <TeamStrengthBars data={data.team_strength} />
        </section>
        <section className="rounded-2xl border border-line bg-panel p-5">
          <div className="mb-4"><h2 className="text-sm font-semibold">Action queue</h2><p className="mt-1 text-xs text-muted">Where the model surface says to investigate, not what you must do.</p></div>
          <div className="space-y-2">
            <Link href="/compare" className="block rounded-xl border border-line bg-bg/60 p-3 hover:border-[var(--accent)]"><div className="text-xs font-semibold">Compare two players</div><div className="mt-1 text-[11px] text-muted">See floor, ceiling, minutes stability, value and ownership side-by-side.</div></Link>
            <Link href="/fixtures" className="block rounded-xl border border-line bg-bg/60 p-3 hover:border-[var(--accent)]"><div className="text-xs font-semibold">Scan fixture swings</div><div className="mt-1 text-[11px] text-muted">Find schedule changes before they show up in raw xP.</div></Link>
            <Link href="/market" className="block rounded-xl border border-line bg-bg/60 p-3 hover:border-[var(--accent)]"><div className="text-xs font-semibold">Inspect market pressure</div><div className="mt-1 text-[11px] text-muted">Transfers, ownership and availability in one view.</div></Link>
            <Link href="/manager" className="block rounded-xl border border-line bg-bg/60 p-3 hover:border-[var(--accent)]"><div className="text-xs font-semibold">Load your squad</div><div className="mt-1 text-[11px] text-muted">Turn projections into a manager-specific squad cockpit.</div></Link>
            <Link href="/transfers" className="block rounded-xl border border-line bg-bg/60 p-3 hover:border-[var(--accent)]"><div className="text-xs font-semibold">Run transfer search</div><div className="mt-1 text-[11px] text-muted">Let the multi-GW optimizer see the whole horizon.</div></Link>
          </div>
        </section>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <section className="rounded-2xl border border-line bg-panel p-5"><h2 className="text-sm font-semibold">Best value</h2><p className="mt-1 mb-3 text-xs text-muted">xP per £m, filtered through availability.</p><div className="space-y-2">{data.best_value.slice(0,5).map(p=><PlayerStrip key={p.player_id} p={p} metric="value" />)}</div></section>
        <section className="rounded-2xl border border-line bg-panel p-5"><h2 className="text-sm font-semibold">Differential upside</h2><p className="mt-1 mb-3 text-xs text-muted">Ceiling above median adjusted for ownership.</p><div className="space-y-2">{data.differentials.slice(0,5).map(p=><PlayerStrip key={p.player_id} p={p} metric="upside" />)}</div></section>
        <section className="rounded-2xl border border-line bg-panel p-5"><h2 className="text-sm font-semibold">Model caveat</h2><p className="mt-3 text-sm leading-6 text-muted">The dashboard deliberately keeps uncertainty visible. A 0.2-point xP gap is not meaningful when simulation error and outcome variance are materially larger.</p><div className="mt-4 rounded-xl border border-line bg-bg p-3 text-xs text-muted">Model version: <span className="font-mono text-fg">{data.model_version}</span></div></section>
      </div>
    </div>
  );
}
