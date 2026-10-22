"use client";

import { useEffect, useState } from "react";
import { modelDashboard, evaluationLatest } from "@/lib/api";
import { EmptyState, Panel, Stat } from "@/components/ui";
import { PitHistogram, CalibrationPlot } from "@/components/HealthCharts";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function ModelLab() {
  const [data, setData] = useState<any>(null);
  const [latest, setLatest] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const d = await modelDashboard();
        setData(d);
        try {
          const l = await evaluationLatest();
          setLatest(l);
        } catch {}
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not load model diagnostics.");
      }
    }
    load();
  }, []);

  if (error) return <EmptyState title="Model Lab unavailable" detail={error} command="fplagent evaluate --from 2 --to 8" />;
  if (!data) return <div className="p-10 text-center text-muted">Loading diagnostics...</div>;

  const run = data.evaluation_runs?.[0];
  
  // Reverse history for chronological chart
  const chartData = [...(data.evaluation_runs || [])].reverse().map((r: any) => ({
    gw: `GW${r.gw_to}`,
    mae: r.mae,
    spearman: r.spearman_overall
  }));

  return <div className="space-y-5">
    <div>
      <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent)]">Validation & provenance</div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">Model Lab</h1>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">A model surface should show where it is strong, where uncertainty widens, how calibration behaves and whether the current run has enough evidence to trust.</p>
    </div>
    
    <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
      <Stat label="Version" value={data.version} />
      <Stat label="MAE" value={run?.mae != null ? run.mae.toFixed(2) : "—"} />
      <Stat label="RMSE" value={run?.rmse != null ? run.rmse.toFixed(2) : "—"} />
      <Stat label="Spearman" value={run?.spearman_overall != null ? run.spearman_overall.toFixed(3) : "—"} />
      <Stat label="CRPS" value={run?.crps != null ? run.crps.toFixed(3) : "—"} />
    </div>

    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title="Performance History" subtitle="Mean Absolute Error (lower is better) and Spearman Correlation (higher is better) over time.">
        <div className="h-[300px] w-full pt-4">
          {chartData.length > 1 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="gw" stroke="#888" fontSize={12} />
                <YAxis yAxisId="left" stroke="#888" fontSize={12} domain={['auto', 'auto']} />
                <YAxis yAxisId="right" orientation="right" stroke="#888" fontSize={12} domain={[0, 1]} />
                <Tooltip contentStyle={{ backgroundColor: '#111', borderColor: '#333' }} />
                <Line yAxisId="left" type="monotone" dataKey="mae" stroke="#f87171" strokeWidth={2} name="MAE" />
                <Line yAxisId="right" type="monotone" dataKey="spearman" stroke="#60a5fa" strokeWidth={2} name="Spearman" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
             <div className="flex h-full items-center justify-center text-sm text-muted">Not enough gameweeks to plot history.</div>
          )}
        </div>
      </Panel>
      
      <Panel title="PIT calibration" subtitle="A flat PIT histogram is consistent with calibrated predictive distributions.">
        {run?.pit_histogram ? <PitHistogram hist={run.pit_histogram} /> : <div className="p-8 text-center text-sm text-muted">No PIT history stored yet.</div>}
      </Panel>
    </div>

    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title="Probability calibration" subtitle="Predicted event rates versus observed outcomes.">
        {latest?.play_calibration ? <CalibrationPlot curve={latest.play_calibration} /> : <div className="p-8 text-center text-sm text-muted">No calibration curve stored yet.</div>}
      </Panel>
      
      <Panel title="Recent evaluation runs">
        <div className="overflow-auto max-h-[300px]">
          <table className="w-full min-w-[800px] text-sm">
            <thead>
              <tr className="border-b border-line text-[10px] uppercase tracking-wider text-muted">
                <th className="px-2 py-3 text-left">Run</th>
                <th className="px-2 py-3 text-right">GW</th>
                <th className="px-2 py-3 text-right">N</th>
                <th className="px-2 py-3 text-right">MAE</th>
                <th className="px-2 py-3 text-right">Spearman</th>
                <th className="px-2 py-3 text-right">Beat baseline</th>
              </tr>
            </thead>
            <tbody>
              {data.evaluation_runs?.map((r: any) => (
                <tr key={r.run_id} className="border-b border-line/70 hover:bg-white/5">
                  <td className="px-2 py-3 font-mono">#{r.run_id}</td>
                  <td className="px-2 py-3 text-right">{r.gw_from}-{r.gw_to}</td>
                  <td className="px-2 py-3 text-right">{r.n}</td>
                  <td className="px-2 py-3 text-right font-mono">{r.mae?.toFixed(2) ?? "—"}</td>
                  <td className="px-2 py-3 text-right font-mono">{r.spearman_overall?.toFixed(3) ?? "—"}</td>
                  <td className="px-2 py-3 text-right">
                    {r.beat_best_baseline == null ? "—" : r.beat_best_baseline ? <span className="text-green-500">Yes</span> : <span className="text-red-500">No</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  </div>;
}