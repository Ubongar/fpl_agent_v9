import { fixtureOutlook } from "@/lib/api";
import { EmptyState, Panel, Stat } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function FixturesPage() {
  let data: Awaited<ReturnType<typeof fixtureOutlook>> | null = null;
  let error: string | null = null;
  
  try { data = await fixtureOutlook(undefined, 6); } 
  catch(e){ error = e instanceof Error ? e.message : "Could not load fixture outlook."; }
  
  if (!data) return <EmptyState title="Fixture Lab unavailable" detail={error ?? "No future fixtures stored."} command="fplagent ingest" />;
  
  return <div className="space-y-5">
    <div>
      <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent)]">Schedule intelligence</div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">True FDR Lab</h1>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">
        Compare FPL's official Difficulty Rating (FDR) against our Dixon-Coles Elo model. 
        Look for <strong>Traps</strong> (FPL says easy, model says hard) and <strong>Hidden Gems</strong> (FPL says hard, model says easy).
      </p>
    </div>
    
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <Stat label="Horizon" value={`${data.horizon} GWs`} />
      <Stat label="Fixtures mapped" value={String(data.fixtures.length / 2)} />
      <Stat label="Start GW" value={`GW${data.start_gw}`} />
    </div>

    <Panel title="Matchup Breakdown" subtitle="FPL Difficulty is 1-5. Model Attack/Defence Difficulty is how hard it is to score/keep a clean sheet based on our Elo engine.">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-line text-[10px] uppercase tracking-wider text-muted">
              <th className="px-3 py-3 text-left">GW</th>
              <th className="px-3 py-3 text-left">Team</th>
              <th className="px-3 py-3 text-left">Opponent</th>
              <th className="px-3 py-3 text-right">FPL FDR</th>
              <th className="px-3 py-3 text-right">Model Attack Diff</th>
              <th className="px-3 py-3 text-right">Model Def Diff</th>
              <th className="px-3 py-3 text-center">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {data.fixtures.map((fx: any, i: number) => {
              const fplDiff = fx.fpl_difficulty ?? 3;
              const modelAtt = fx.model_attack_difficulty ?? 3;
              const modelDef = fx.model_defence_difficulty ?? 3;
              
              let verdict = "Neutral";
              let color = "text-muted";
              if (fplDiff <= 2.5 && modelAtt >= 3.5) {
                verdict = "Trap";
                color = "text-red-500 font-bold";
              } else if (fplDiff >= 3.5 && modelAtt <= 2.5) {
                verdict = "Hidden Gem";
                color = "text-green-500 font-bold";
              }

              return (
                <tr key={i} className="border-b border-line/50 hover:bg-white/5 transition">
                  <td className="px-3 py-2">GW{fx.gw}</td>
                  <td className="px-3 py-2 font-semibold">
                    {fx.team} <span className="text-[10px] text-muted ml-1">{fx.home ? "(H)" : "(A)"}</span>
                  </td>
                  <td className="px-3 py-2">{fx.opponent}</td>
                  <td className="px-3 py-2 text-right font-mono">{fplDiff.toFixed(1)}</td>
                  <td className="px-3 py-2 text-right font-mono text-orange-400">{modelAtt.toFixed(1)}</td>
                  <td className="px-3 py-2 text-right font-mono text-blue-400">{modelDef.toFixed(1)}</td>
                  <td className={`px-3 py-2 text-center text-xs ${color}`}>{verdict}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  </div>;
}