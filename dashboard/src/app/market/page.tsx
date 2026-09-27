import Link from "next/link";
import { marketWatch } from "@/lib/api";
import { EmptyState, Panel, Stat } from "@/components/ui";
import { MarketBars } from "@/components/Viz";
import { money, pct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function MarketPage() {
  let data: Awaited<ReturnType<typeof marketWatch>> | null = null;
  let error: string | null = null;
  try {
    data = await marketWatch(50);
  } catch (e) {
    error = e instanceof Error ? e.message : "Could not load market activity.";
  }

  if (!data) {
    return (
      <EmptyState
        title="Market Watch unavailable"
        detail={error ?? "No current market snapshot."}
        command="fplagent ingest"
      />
    );
  }

  // The bar chart is labelled by player name now, not by player id. The
  // fallback to "player N" only fires if a snapshot references a player
  // who is not in the current season's `players` table -- which happens
  // when a player has been sold mid-season and the snapshot has not yet
  // been pruned.
  const rows = data.players.slice(0, 8).map((p) => ({
    name: p.web_name ?? p.short_name ?? `player ${p.player_id}`,
    value: Number(p.net_transfers),
  }));

  const leader = data.players[0];
  const topOwnership = [...data.players].sort(
    (a, b) => (b.selected_by_percent ?? 0) - (a.selected_by_percent ?? 0),
  )[0];

  return (
    <div className="space-y-5">
      <div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent)]">
          Market intelligence
        </div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">Market Watch</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">
          Ownership, transfer velocity, availability and price context. This
          layer is deliberately a signal dashboard—not a promise about future
          price changes.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Players monitored" value={String(data.players.length)} />
        <Stat
          label="Net inflow leader"
          value={
            leader
              ? `${leader.web_name ?? leader.short_name ?? `#${leader.player_id}`} (${
                  leader.net_transfers > 0 ? "+" : ""
                }${leader.net_transfers})`
              : "—"
          }
          tone="good"
        />
        <Stat
          label="Top ownership"
          value={
            topOwnership
              ? `${topOwnership.web_name ?? topOwnership.short_name ?? `#${topOwnership.player_id}`} (${
                  topOwnership.selected_by_percent != null
                    ? topOwnership.selected_by_percent.toFixed(1)
                    : "—"
                }%)`
              : "—"
          }
        />
        <Stat label="Signal" value="Flow" hint="Not price certainty" />
      </div>

      <div className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <Panel
          title="Transfer pressure"
          subtitle="Net transfers in over the last snapshot period. Bar labels are player names."
        >
          <MarketBars rows={rows} />
        </Panel>

        <Panel title="Live watchlist" subtitle={data.note}>
          <div className="overflow-auto">
            <table className="w-full min-w-[620px] text-sm">
              <thead>
                <tr className="border-b border-line text-[10px] uppercase tracking-wider text-muted">
                  <th className="px-2 py-3 text-left">Player</th>
                  <th className="px-2 py-3 text-left">Club</th>
                  <th className="px-2 py-3 text-right">Ownership</th>
                  <th className="px-2 py-3 text-right">In</th>
                  <th className="px-2 py-3 text-right">Out</th>
                  <th className="px-2 py-3 text-right">Net</th>
                  <th className="px-2 py-3 text-right">Form</th>
                  <th className="px-2 py-3 text-right">Availability</th>
                </tr>
              </thead>
              <tbody>
                {data.players.map((p) => {
                  const display =
                    p.web_name ?? p.short_name ?? `player ${p.player_id}`;
                  return (
                    <tr key={p.player_id} className="border-b border-line/70">
                      <td className="px-2 py-3">
                        <Link
                          href={`/players/${p.player_id}`}
                          className="font-medium underline-offset-2 hover:text-[var(--accent)] hover:underline"
                        >
                          {display}
                        </Link>
                        {p.position ? (
                          <span className="ml-2 rounded-md bg-bg px-1.5 py-0.5 text-[10px] text-muted">
                            {p.position}
                          </span>
                        ) : null}
                      </td>
                      <td className="px-2 py-3 text-muted">
                        {p.short_name ?? "—"}
                      </td>
                      <td className="px-2 py-3 text-right font-mono">
                        {p.selected_by_percent?.toFixed(1) ?? "—"}%
                      </td>
                      <td className="px-2 py-3 text-right font-mono">
                        {p.transfers_in_event ?? "—"}
                      </td>
                      <td className="px-2 py-3 text-right font-mono">
                        {p.transfers_out_event ?? "—"}
                      </td>
                      <td
                        className={`px-2 py-3 text-right font-mono ${
                          p.net_transfers >= 0
                            ? "text-[var(--good)]"
                            : "text-[var(--bad)]"
                        }`}
                      >
                        {p.net_transfers >= 0 ? "+" : ""}
                        {p.net_transfers}
                      </td>
                      <td className="px-2 py-3 text-right font-mono">
                        {p.form ?? "—"}
                      </td>
                      <td className="px-2 py-3 text-right font-mono">
                        {p.chance_of_playing_next_round != null
                          ? `${p.chance_of_playing_next_round}%`
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}