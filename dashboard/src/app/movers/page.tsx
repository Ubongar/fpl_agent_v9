import { changes, currentGameweek } from "@/lib/api";
import { MoversChart } from "@/components/MoversChart";
import { Bar as MiniBar, EmptyState, Panel, PlayerLink } from "@/components/ui";
import { pts } from "@/lib/format";
import type { AttributionChange } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function MoversPage({
  searchParams,
}: {
  searchParams: Promise<{ from?: string; to?: string }>;
}) {
  const sp = await searchParams;

  let diffs: AttributionChange[] | null = null;
  let gwFrom = 0;
  let gwTo = 0;
  let error: string | null = null;
  let noWindowYet = false;

  try {
    const gw = await currentGameweek();
    gwTo = Number(sp.to ?? gw.target_gw);
    gwFrom = Number(sp.from ?? gwTo - 1);

    if (gwFrom < 1) {
      noWindowYet = true;
    } else {
      const result = await changes(gwTo, gwFrom);
      diffs = result.changes;
    }
  } catch (e) {
    error = e instanceof Error ? e.message : "Unknown error.";
  }

  if (error) {
    return (
      <EmptyState title="Could not load movers" detail={error} command="fplagent run" />
    );
  }

  if (noWindowYet) {
    return (
      <EmptyState
        title="Nothing to compare yet"
        detail="Week-over-week attribution needs two gameweeks of stored predictions. There's only one so far."
      />
    );
  }

  if (!diffs || diffs.length === 0) {
    return (
      <EmptyState
        title="No predictions to compare"
        detail={`No stored predictions for both GW${gwFrom} and GW${gwTo}. Run the pipeline again once both exist.`}
        command="fplagent run"
      />
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Movers: GW{gwFrom} &rarr; GW{gwTo}
        </h1>
        <p className="mt-1 text-sm text-muted">
          Not &quot;this player is worth 6.4 points&quot; but what changed and why.
          The xP decomposition is additive by construction, so each driver
          below is a subtraction, not a guess.
        </p>
      </div>

      <Panel title="Biggest movers">
        <MoversChart changes={diffs} />
      </Panel>

      <Panel title="Drivers of change">
        <div className="space-y-4">
          {diffs
            .slice()
            .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
            .slice(0, 15)
            .map((c) => {
              const maxDriver = Math.max(0.01, ...c.drivers.map((d) => Math.abs(d.delta)));
              return (
                <div key={c.player_id} className="rounded-md border border-line p-3">
                  <div className="flex items-center justify-between text-sm">
                    <PlayerLink id={c.player_id} name={c.name} />
                    <span
                      className={
                        "font-mono " + (c.delta >= 0 ? "text-[var(--good)]" : "text-[var(--bad)]")
                      }
                    >
                      {pts(c.xp_prev, 1)} &rarr; {pts(c.xp_now, 1)} ({c.delta >= 0 ? "+" : ""}
                      {c.delta.toFixed(2)})
                    </span>
                  </div>
                  {c.drivers.length > 0 ? (
                    <ul className="mt-2 space-y-1.5">
                      {c.drivers.map((d) => (
                        <li
                          key={d.component}
                          className="grid grid-cols-[minmax(0,1fr)_46px] items-center gap-2"
                        >
                          <div>
                            <div className="text-[11px] text-muted">
                              {d.component.replace(/_/g, " ")}
                            </div>
                            <MiniBar
                              value={d.delta}
                              max={maxDriver}
                              tone={d.delta < 0 ? "bad" : "good"}
                            />
                          </div>
                          <span className="text-right font-mono text-[11px] text-muted">
                            {(d.share_of_change * 100).toFixed(0)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                  {c.context.length > 0 ? (
                    <ul className="mt-2 list-disc pl-4 text-xs text-muted">
                      {c.context.map((ctx, i) => (
                        <li key={i}>{ctx}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              );
            })}
        </div>
      </Panel>
    </div>
  );
}
