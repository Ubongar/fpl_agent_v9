import { optimiseSquad } from "@/lib/api";
import { SquadAllocation } from "@/components/SquadAllocation";
import { EmptyState, Panel, PositionBadge, Stat } from "@/components/ui";
import { money, pts } from "@/lib/format";
import type { SquadPlayer, SquadSolution } from "@/lib/types";

export const dynamic = "force-dynamic";

const ORDER: Record<string, number> = { GK: 0, DEF: 1, MID: 2, FWD: 3 };

function Line({ players, captain }: { players: SquadPlayer[]; captain: string | null }) {
  return (
    <div className="flex flex-wrap justify-center gap-2">
      {players.map((p) => (
        <div
          key={p.player_id}
          className="w-32 rounded-md border border-line bg-bg px-2 py-1.5 text-center"
        >
          <div className="truncate text-xs font-medium">
            {p.name}
            {captain === p.name ? (
              <span className="ml-1 rounded bg-[var(--accent)] px-1 text-[10px] text-white">
                C
              </span>
            ) : null}
          </div>
          <div className="mt-0.5 flex justify-between text-[11px] text-muted">
            <span>{money(p.price)}</span>
            <span className="font-mono">{pts(p.xp, 1)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default async function SquadPage({
  searchParams,
}: {
  searchParams: Promise<{ budget?: string; risk?: string }>;
}) {
  const sp = await searchParams;
  const budget = Number(sp.budget ?? 100);
  const risk = Number(sp.risk ?? 0);

  let sol: SquadSolution | null = null;
  let error: string | null = null;
  try {
    sol = await optimiseSquad({ budget, risk_lambda: risk });
  } catch (e) {
    error = e instanceof Error ? e.message : "Unknown error.";
  }

  if (!sol) {
    return (
      <EmptyState
        title="Could not build a squad"
        detail={error ?? "Unknown error."}
        command="fplagent predict"
      />
    );
  }

  const xi = [...sol.starting_xi].sort(
    (a, b) => (ORDER[a.position] ?? 9) - (ORDER[b.position] ?? 9),
  );
  const byPos = (p: string) => xi.filter((x) => x.position === p);

  return (
    <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Optimal squad</h1>
          <p className="mt-1 text-sm text-muted">
            Integer program over fifteen squad slots, eleven starting slots and
            one captain slot, solved jointly. Optimising the fifteen-man total
            would treat a bench place as worth as much as a starting place.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Formation" value={sol.summary.formation} />
          <Stat label="Cost" value={money(sol.summary.total_cost)} />
          <Stat
            label="XI expected points"
            value={pts(sol.summary.xi_expected_points)}
            hint="Captain counted twice"
          />
          <Stat
            label="Captain"
            value={sol.summary.captain ?? "-"}
            hint={`Vice: ${sol.summary.vice_captain ?? "-"}`}
          />
        </div>

        <Panel
          title="Spend vs. expected points, by position"
          subtitle="Squad spend (starters and bench) against the starting XI's expected points. A position taking a big share of the budget for a small share of XI points is worth a second look."
        >
          <SquadAllocation xi={sol.starting_xi} bench={sol.bench ?? []} />
        </Panel>

        <Panel title="Starting eleven">
          <div className="space-y-3 py-2">
            {["GK", "DEF", "MID", "FWD"].map((p) => (
              <Line key={p} players={byPos(p)} captain={sol.summary.captain} />
            ))}
          </div>
        </Panel>

        {sol.bench && sol.bench.length > 0 ? (
          <Panel
            title="Bench"
            subtitle="Ordered by likelihood of actually playing, since that is what autosubs depend on, not by expected points."
          >
            <div className="flex flex-wrap gap-2">
              {sol.bench.map((p) => (
                <div
                  key={p.player_id}
                  className="flex w-40 items-center justify-between rounded-md border border-line bg-bg px-2 py-1.5 text-xs"
                >
                  <span className="flex items-center gap-1.5 truncate">
                    <PositionBadge position={p.position} />
                    {p.name}
                  </span>
                  <span className="font-mono">{pts(p.xp, 1)}</span>
                </div>
              ))}
            </div>
          </Panel>
        ) : null}
    </div>
  );
}
