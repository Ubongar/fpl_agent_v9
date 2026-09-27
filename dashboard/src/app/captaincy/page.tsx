import { rankCaptains } from "@/lib/api";
import { CaptaincyScatter } from "@/components/CaptaincyScatter";
import { EmptyState, Panel } from "@/components/ui";
import { pct, pts } from "@/lib/format";
import type { CaptainCandidate } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function CaptaincyPage({
  searchParams,
}: {
  searchParams: Promise<{ risk?: string }>;
}) {
  const sp = await searchParams;
  const risk = Number(sp.risk ?? 0);

  let candidates: CaptainCandidate[] | null = null;
  let error: string | null = null;
  try {
    const result = await rankCaptains({ risk_lambda: risk, top_n: 12 });
    candidates = result.candidates;
  } catch (e) {
    error = e instanceof Error ? e.message : "Unknown error.";
  }

  if (!candidates) {
    return (
      <EmptyState
        title="No captaincy candidates"
        detail={error ?? "Unknown error."}
        command="fplagent predict"
      />
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Captaincy</h1>
        <p className="mt-1 text-sm text-muted">
          For pure expected points the captain choice is trivial, because
          doubling a player doubles his mean. Ceiling only matters when the
          objective is rank, so ceiling, floor and differential value are
          shown as context rather than folded into one opaque score.
        </p>
      </div>

      <Panel
        title="Floor vs. ceiling"
        subtitle="A high floor and a high ceiling are different bets. The table below ranks by expected captain points; this chart is where you check which trade-off you're actually taking."
      >
        <CaptaincyScatter candidates={candidates} />
      </Panel>

      <Panel>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="text-left text-xs text-muted">
                <th className="px-2 py-2 font-medium">Player</th>
                <th className="px-2 py-2 text-right font-medium">xP</th>
                <th className="px-2 py-2 text-right font-medium">Doubled</th>
                <th className="px-2 py-2 text-right font-medium">Floor</th>
                <th className="px-2 py-2 text-right font-medium">Ceiling</th>
                <th className="px-2 py-2 text-right font-medium">P(haul)</th>
                <th className="px-2 py-2 text-right font-medium">EO</th>
                <th className="px-2 py-2 text-right font-medium">Differential</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => (
                <tr key={c.player_id} className="border-t border-line align-top">
                  <td className="px-2 py-2">
                    <div className="font-medium">{c.name}</div>
                    {c.risk_note ? (
                      <div className="mt-0.5 max-w-sm text-[11px] text-[var(--warn)]">
                        {c.risk_note}
                      </div>
                    ) : null}
                  </td>
                  <td className="px-2 py-2 text-right font-mono">{pts(c.xp)}</td>
                  <td className="px-2 py-2 text-right font-mono">
                    {pts(c.expected_captain_points)}
                  </td>
                  <td className="px-2 py-2 text-right font-mono">
                    {pts(c.floor_p10, 1)}
                  </td>
                  <td className="px-2 py-2 text-right font-mono">
                    {pts(c.ceiling_p90, 1)}
                  </td>
                  <td className="px-2 py-2 text-right font-mono">{pct(c.p_haul)}</td>
                  <td className="px-2 py-2 text-right font-mono">
                    {pct(c.effective_ownership)}
                  </td>
                  <td className="px-2 py-2 text-right font-mono">
                    {pts(c.differential_points)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
