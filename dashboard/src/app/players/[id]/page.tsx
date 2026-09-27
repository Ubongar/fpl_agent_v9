import Link from "next/link";
import { playerDetail } from "@/lib/api";
import { ComponentWaterfall } from "@/components/ComponentWaterfall";
import { DistributionViz } from "@/components/Viz";
import { EmptyState, Panel, Stat } from "@/components/ui";
import { pct, pts } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function PlayerXray({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const playerId = Number(id);
  if (!Number.isInteger(playerId) || playerId <= 0) return <EmptyState title="Invalid player" detail="The player identifier is not valid." />;
  let data: any = null;
  let error: string | null = null;
  try { data = await playerDetail(playerId); } catch (e) { error = e instanceof Error ? e.message : "Player projection unavailable."; }
  if (!data) return <EmptyState title="Player projection unavailable" detail={error ?? "No stored prediction for this player."} command="fplagent predict" />;
  const d = data.distribution;
  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div><Link href="/players" className="text-xs text-muted hover:text-fg">← Player Lab</Link><h1 className="mt-2 text-3xl font-semibold tracking-tight">Player X-ray</h1><p className="mt-2 text-sm text-muted">GW{data.gw} · model <span className="font-mono text-fg">{data.model_version}</span> · {data.n_fixtures} fixture{data.n_fixtures === 1 ? "" : "s"}</p></div>
        <div className="flex flex-wrap items-center gap-2">
          <Link href={`/compare?players=${data.player_id}`} className="rounded-xl border border-[var(--accent)]/30 bg-[var(--accent)]/10 px-3 py-2 text-xs font-semibold text-[var(--accent)]">Add to comparison desk</Link>
          {data.lineup_news ? <div className="rounded-xl border border-[var(--warn)]/30 bg-[var(--warn)]/10 px-3 py-2 text-xs text-[var(--warn)]">Live lineup overlay applied</div> : null}
        </div>
      </div>
      <div className="grid gap-3 grid-cols-2 md:grid-cols-4 xl:grid-cols-8">
        <Stat label="xP" value={pts(data.expected_points)} />
        <Stat label="P10" value={pts(d.p10,1)} />
        <Stat label="P50" value={pts(d.p50,1)} />
        <Stat label="P90" value={pts(d.p90,1)} />
        <Stat label="P(play)" value={pct(data.probabilities.play)} />
        <Stat label="P60+" value={pct(data.probabilities.sixty_plus)} />
        <Stat label="P(haul)" value={pct(data.probabilities.haul_10plus)} />
        <Stat label="P(blank)" value={pct(data.probabilities.blank_2_or_less)} />
      </div>
      <Panel title="Decision layer" subtitle="Separate dimensions for floor, ceiling, minutes security, value and ownership leverage.">
        <div className="grid gap-3 grid-cols-2 md:grid-cols-4 xl:grid-cols-8">
          <Stat label="Value / £m" value={data.decision?.value_per_million?.toFixed(2) ?? "—"} />
          <Stat label="Floor-adjusted" value={data.decision?.floor_adjusted_xp?.toFixed(2) ?? "—"} />
          <Stat label="Ceiling-adjusted" value={data.decision?.ceiling_adjusted_xp?.toFixed(2) ?? "—"} />
          <Stat label="Minutes stability" value={pct(data.decision?.minutes_stability)} />
          <Stat label="Start security" value={pct(data.decision?.start_security)} />
          <Stat label="Uncertainty" value={pts(data.decision?.uncertainty_points,2)} />
          <Stat label="Diff. upside" value={data.decision?.ownership_adjusted_upside != null ? `+${data.decision.ownership_adjusted_upside.toFixed(2)}` : "—"} />
          <Stat label="Ownership" value={pct(data.decision?.ownership)} />
        </div>
      </Panel>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Outcome distribution" subtitle="The distribution is the product. The mean is one summary statistic inside it."><DistributionViz p10={d.p10} p50={d.p50} p90={d.p90} mean={data.expected_points} /></Panel>
        <Panel title="Scoring decomposition" subtitle="Expected contribution by FPL scoring mechanism. Negative bars are downside sources."><ComponentWaterfall components={data.components ?? {}} /></Panel>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Consistency flags"><div className="space-y-2">{(data.flags ?? []).length ? data.flags.map((f: any) => <div key={f.code} className="rounded-xl border border-[var(--warn)]/30 bg-[var(--warn)]/10 p-3 text-xs"><div className="font-semibold text-[var(--warn)]">{f.code}</div><div className="mt-1 text-muted">{f.message}</div></div>) : <div className="rounded-xl border border-[var(--good)]/25 bg-[var(--good)]/10 p-3 text-xs text-[var(--good)]">No deterministic consistency flags for this projection.</div>}</div></Panel>
        <Panel title="Model inputs"><div className="max-h-80 overflow-auto rounded-xl border border-line bg-bg p-3"><pre className="whitespace-pre-wrap text-[10px] leading-5 text-muted">{JSON.stringify(data.inputs, null, 2)}</pre></div></Panel>
        <Panel title="Projection freshness"><div className="space-y-3 text-sm"><div><div className="text-xs text-muted">Created</div><div className="mt-1 font-mono">{data.freshness?.age_hours != null ? `${data.freshness.age_hours}h ago` : "Unknown"}</div></div><div className="text-xs leading-5 text-muted">{data.freshness?.note ?? "Freshness metadata unavailable."}</div><div className="rounded-xl border border-line p-3 text-xs">Stored rows remain immutable for evaluation; live news overlays do not rewrite the historical prediction.</div></div></Panel>
      </div>
    </div>
  );
}
