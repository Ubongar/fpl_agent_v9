import { teamStrength } from "@/lib/api";
import { AttackDefenceScatter, HomeAdvantageBar } from "@/components/TeamStrengthCharts";
import { EmptyState, Panel, Stat } from "@/components/ui";
import { pts } from "@/lib/format";
import type { TeamStrength } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function TeamsPage() {
  let modelVersion = "";
  let teams: TeamStrength[] = [];
  let error: string | null = null;

  try {
    const result = await teamStrength();
    modelVersion = result.model_version;
    teams = result.teams;
  } catch (e) {
    error = e instanceof Error ? e.message : "Unknown error.";
  }

  if (error) {
    return (
      <EmptyState
        title="Could not load team strength"
        detail={error}
        command="fplagent predict"
      />
    );
  }

  if (!teams.length) {
    return (
      <EmptyState
        title="No team-strength fit yet"
        detail="The Dixon-Coles fit runs as part of the prediction pipeline."
        command="fplagent predict"
      />
    );
  }

  const byAttack = [...teams].sort((a, b) => b.attack - a.attack);
  const sorted = [...teams].sort((a, b) =>
    (a.name ?? a.short_name ?? "").localeCompare(b.name ?? b.short_name ?? ""),
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Team strength</h1>
        <p className="mt-1 text-sm text-muted">
          Attack and defence ratings from the Dixon-Coles fit, model {modelVersion}.
          Every player projection is built on top of these two numbers plus the
          opponent&apos;s.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Teams fitted" value={String(teams.length)} />
        <Stat
          label="Strongest attack"
          value={byAttack[0]?.short_name ?? "-"}
          hint={pts(byAttack[0]?.attack, 2)}
        />
        <Stat
          label="League avg. goals"
          value={pts(teams[0]?.league_avg_goals, 2)}
          hint="Per team per match"
        />
        <Stat
          label="Correlation (rho)"
          value={pts(teams[0]?.rho, 3)}
          hint="Low-score dependence between the sides"
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel
          title="Attack vs. defence"
          subtitle="Bottom-right: strong going forward, weak at the back -- good fixtures for attackers. Top-left is the reverse."
        >
          <AttackDefenceScatter teams={teams} />
        </Panel>
        <Panel
          title="Home advantage"
          subtitle="Fitted per team, not assumed to be identical across the league."
        >
          <HomeAdvantageBar teams={teams} />
        </Panel>
      </div>

      <Panel title="Full ratings table">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="text-left text-xs text-muted">
                <th className="px-2 py-2 font-medium">Team</th>
                <th className="px-2 py-2 text-right font-medium">Attack</th>
                <th className="px-2 py-2 text-right font-medium">Defence</th>
                <th className="px-2 py-2 text-right font-medium">Home adv.</th>
                <th className="px-2 py-2 text-right font-medium">Matches fit</th>
                <th className="px-2 py-2 text-right font-medium">Effective n</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((t) => (
                <tr key={t.team_id} className="border-t border-line">
                  <td className="px-2 py-2 font-medium">{t.name ?? t.short_name ?? t.team_id}</td>
                  <td className="px-2 py-2 text-right font-mono">{pts(t.attack, 2)}</td>
                  <td className="px-2 py-2 text-right font-mono">{pts(t.defence, 2)}</td>
                  <td className="px-2 py-2 text-right font-mono">{pts(t.home_adv, 2)}</td>
                  <td className="px-2 py-2 text-right font-mono">{t.n_matches_fit}</td>
                  <td className="px-2 py-2 text-right font-mono">
                    {t.effective_n != null ? t.effective_n.toFixed(1) : "-"}
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
