import { evaluationHistory, evaluationLatest } from "@/lib/api";
import {
  AccuracyHistory,
  CalibrationPlot,
  PitHistogram,
  SpearmanHistory,
} from "@/components/HealthCharts";
import { EmptyState, Panel, Stat } from "@/components/ui";
import { pts } from "@/lib/format";
import type { BacktestResult, EvaluationRun } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function HealthPage() {
  let history: { runs: EvaluationRun[] } = { runs: [] };
  let latest: BacktestResult | null = null;
  let error: string | null = null;

  try {
    history = await evaluationHistory().catch(() => ({ runs: [] }));

    // Score the same window as the most recent persisted run rather than a
    // hardcoded one. Defaulting to a single gameweek produced an empty page
    // whenever the operator had evaluated a different range, which reads as
    // "the model has never been measured" when in fact it has.
    const newest = history.runs[0];
    latest = await evaluationLatest(
      newest?.gw_from ?? 2,
      newest?.gw_to ?? undefined,
    ).catch(() => null);
  } catch (e) {
    error = e instanceof Error ? e.message : "Unknown error.";
  }

  if (error) {
    return (
      <EmptyState
        title="Could not load evaluation"
        detail={error}
        command="fplagent evaluate --from 2 --to 8"
      />
    );
  }

  if (!latest || latest.n === 0) {
    return (
      <EmptyState
        title="Nothing scored yet"
        detail="The backtest needs stored predictions and the actual results they can be scored against."
        command="fplagent evaluate --from 2 --to 8"
      />
    );
  }

  const sig = latest.baseline_significance;
  const beats = sig?.beats_all_significantly ?? false;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Model health</h1>
        <p className="mt-1 text-sm text-muted">
          A model that is not measured against free baselines is decoration.
          These numbers include the weeks the model loses, because those are
          the informative ones.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Rows scored" value={String(latest.n ?? 0)} />
        <Stat
          label="Spearman"
          value={pts(latest.spearman_overall, 3)}
          hint="Rank correlation, what decisions consume"
        />
        <Stat
          label="CRPS"
          value={pts(latest.crps, 3)}
          hint="Scores the whole distribution, lower is better"
        />
        <Stat
          label="Beats every baseline"
          value={beats ? "Yes" : "Not yet"}
          tone={beats ? "good" : "warn"}
          hint="Bootstrapped, not a point estimate"
        />
      </div>

      <Panel
        title="Against the free baselines"
        subtitle="Block-bootstrapped by gameweek. Rows within a gameweek share fixtures and a single weekend, so resampling individual rows would make the interval far too narrow and declare victory on noise."
      >
        {sig?.per_baseline ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted">
                <th className="py-2 font-medium">Baseline</th>
                <th className="py-2 text-right font-medium">Mean gain</th>
                <th className="py-2 text-right font-medium">95% interval</th>
                <th className="py-2 text-right font-medium">Clears zero</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(sig.per_baseline).map(([name, v]) => (
                <tr key={name} className="border-t border-line">
                  <td className="py-2">{name.replace(/_/g, " ")}</td>
                  <td className="py-2 text-right font-mono">
                    {v.mean_spearman_gain?.toFixed(3) ?? "-"}
                  </td>
                  <td className="py-2 text-right font-mono text-xs">
                    {v.ci95 ? `${v.ci95[0].toFixed(3)} to ${v.ci95[1].toFixed(3)}` : (v.note ?? "-")}
                  </td>
                  <td
                    className={
                      "py-2 text-right " +
                      (v.significant ? "text-[var(--good)]" : "text-[var(--warn)]")
                    }
                  >
                    {v.significant ? "yes" : "no"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-muted">
            Needs at least two scored gameweeks before an interval means
            anything.
          </p>
        )}
      </Panel>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel
          title="Distribution calibration (PIT)"
          subtitle="Flat is honest. A U shape means the simulation is overconfident."
        >
          {latest.pit_histogram?.length ? (
            <>
              <PitHistogram hist={latest.pit_histogram} />
              <p className="mt-2 text-xs text-muted">
                {latest.pit_interpretation ?? ""}
              </p>
            </>
          ) : (
            <p className="text-sm text-muted">
              Needs stored quantile grids, which live predictions carry.
            </p>
          )}
        </Panel>

        <Panel
          title="P(60+ minutes) calibration"
          subtitle="Predicted rate against observed. Bubble size is the number of players in the bucket."
        >
          {latest.sixty_calibration?.length ? (
            <>
              <CalibrationPlot curve={latest.sixty_calibration} />
              <p className="mt-2 text-xs text-muted">
                Expected calibration error{" "}
                <span className="font-mono">
                  {pts(latest.sixty_calibration_ece, 3)}
                </span>
                . Scored against minutes actually played, not against the
                starts flag: a starter withdrawn on 55 minutes started and
                did not reach sixty.
              </p>
            </>
          ) : (
            <p className="text-sm text-muted">No calibration data yet.</p>
          )}
        </Panel>
      </div>

      {history.runs.length > 1 ? (
        <div className="grid gap-5 lg:grid-cols-2">
          <Panel
            title="Rank correlation over time"
            subtitle="The model against a six-gameweek rolling mean and against price. Both are free, and most hand-built FPL models lose to at least one of them."
          >
            <SpearmanHistory runs={history.runs} />
          </Panel>
          <Panel
            title="Error over time"
            subtitle="Rank can hold while point estimates drift. MAE and CRPS catch that; Spearman does not."
          >
            <AccuracyHistory runs={history.runs} />
          </Panel>
        </div>
      ) : null}
    </div>
  );
}
