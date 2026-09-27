"use client";

import { useEffect, useMemo, useState } from "react";
import { getJSON, postJSON, BrowserApiError } from "@/lib/browserApi";
import { ChipTimeline, ChipValueBar } from "@/components/ChipTimeline";
import { EmptyState, Panel } from "@/components/ui";
import type { ChipSchedule } from "@/lib/types";

const CHIPS = ["wildcard", "free_hit", "bench_boost", "triple_captain"] as const;
const CHIP_LABEL: Record<string, string> = {
  wildcard: "Wildcard",
  free_hit: "Free Hit",
  bench_boost: "Bench Boost",
  triple_captain: "Triple Captain",
};
const WINDOW = 8; // how many upcoming gameweeks the grid covers

type Values = Record<string, Record<number, number>>;
type AlreadyUsed = Record<string, string>; // raw comma-separated gw text per chip

interface AutoChipResponse extends ChipSchedule {
  per_gw_values: Record<string, Record<string, number>>;
  horizon: number[];
  entry_id_used: number | null;
}

export default function ChipsPage() {
  const [gwNow, setGwNow] = useState<number | null>(null);
  const [values, setValues] = useState<Values>({});
  const [used, setUsed] = useState<AlreadyUsed>({});
  const [loading, setLoading] = useState(true);
  const [prefilling, setPrefilling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [schedule, setSchedule] = useState<ChipSchedule | null>(null);
  const [solving, setSolving] = useState(false);

  // Auto-fill from the pipeline. The endpoint reads stored predictions for
  // the horizon and, when an entry id is supplied, also pulls the manager's
  // real squad so Free Hit and Wildcard values are computed against it
  // rather than against the optimal squad.
  const [autoEntryId, setAutoEntryId] = useState("");
  const [autoLoading, setAutoLoading] = useState(false);
  const [autoError, setAutoError] = useState<string | null>(null);
  const [autoHorizon, setAutoHorizon] = useState(WINDOW);

  const gws = useMemo(
    () => (gwNow ? Array.from({ length: WINDOW }, (_, i) => gwNow + i).filter((g) => g <= 38) : []),
    [gwNow],
  );

  useEffect(() => {
    getJSON<{ target_gw: number }>("/gameweeks/current")
      .then((r) => {
        setGwNow(r.target_gw);
        setValues(Object.fromEntries(CHIPS.map((c) => [c, {}])));
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load the current gameweek."))
      .finally(() => setLoading(false));
  }, []);

  function setCell(chip: string, gw: number, v: number) {
    setValues((prev) => ({ ...prev, [chip]: { ...prev[chip], [gw]: v } }));
  }

  /**
   * Fill the value grid and the schedule from the pipeline's own
   * computation. Reads every gameweek in the horizon from the stored
   * predictions, calls the same `compute_chip_values` the CLI uses, and
   * returns the assigned weeks plus the values that drove them.
   *
   * The endpoint does NOT predict on demand -- if the horizon gameweeks
   * haven't been predicted yet, it 404s with a message telling you which
   * CLI command to run. That is deliberate: predicting on demand inside a
   * web request would take minutes and could easily hit a proxy timeout.
   */
  async function fetchAuto() {
    if (!gwNow) return;
    setAutoLoading(true);
    setAutoError(null);
    try {
      const id = autoEntryId.trim();
      const query = new URLSearchParams();
      query.set("horizon", String(autoHorizon));
      if (id) {
        const n = Number(id);
        if (!Number.isInteger(n) || n <= 0) {
          setAutoError("Team ID must be a positive integer, or left blank.");
          return;
        }
        query.set("entry_id", String(n));
      }
      const result = await getJSON<AutoChipResponse>(
        `/optimise/chips/auto?${query.toString()}`,
      );

      // Populate the grid so the user sees the numbers that produced the
      // schedule, and can adjust any of them and re-solve manually.
      const next: Values = {};
      for (const chip of CHIPS) {
        const byGw = result.per_gw_values[chip] ?? {};
        next[chip] = {};
        for (const [g, v] of Object.entries(byGw)) {
          next[chip][Number(g)] = Number(v);
        }
      }
      setValues(next);
      setSchedule(result);

      if (!result.entry_id_used && id) {
        setAutoError(
          "Could not load that team -- Free Hit and Wildcard values are " +
          "computed against the optimal squad instead. Check the ID.",
        );
      }
    } catch (e) {
      setAutoError(
        e instanceof BrowserApiError
          ? `${e.message}`
          : "Could not compute a chip schedule from stored predictions.",
      );
    } finally {
      setAutoLoading(false);
    }
  }

  async function prefillCurrentWeek() {
    if (!gwNow) return;
    setPrefilling(true);
    setError(null);
    try {
      const [squad, captains] = await Promise.all([
        postJSON<{ bench?: { xp: number }[] }>("/optimise/squad", { budget: 100, risk_lambda: 0 }),
        postJSON<{ candidates: { xp: number }[] }>("/optimise/captaincy", { risk_lambda: 0, top_n: 1 }),
      ]);
      const benchXp = (squad.bench ?? []).reduce((a, p) => a + p.xp, 0);
      const tcXp = captains.candidates[0]?.xp ?? 0;
      setCell("bench_boost", gwNow, Math.round(benchXp * 100) / 100);
      setCell("triple_captain", gwNow, Math.round(tcXp * 100) / 100);
    } catch (e) {
      setError(
        e instanceof BrowserApiError
          ? e.message
          : "Could not prefill from the current squad/captaincy optimisers.",
      );
    } finally {
      setPrefilling(false);
    }
  }

  async function solve() {
    if (!gwNow) return;
    setSolving(true);
    setError(null);
    setSchedule(null);
    try {
      const already_used: Record<string, number[]> = {};
      for (const c of CHIPS) {
        const raw = (used[c] ?? "").trim();
        if (raw) {
          already_used[c] = raw
            .split(",")
            .map((s) => Number(s.trim()))
            .filter((n) => Number.isFinite(n));
        }
      }
      const body = {
        available_from_gw: gwNow,
        chip_values: values,
        already_used,
      };
      const result = await postJSON<ChipSchedule>("/optimise/chips", body);
      setSchedule(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not solve the chip schedule.");
    } finally {
      setSolving(false);
    }
  }

  if (loading) {
    return <p className="text-sm text-muted">Loading&hellip;</p>;
  }

  if (error && !gwNow) {
    return (
      <EmptyState
        title="Could not load chip planner"
        detail={error}
        command="uvicorn fplagent.api.main:app --port 8000"
      />
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Chip planner</h1>
        <p className="mt-1 text-sm text-muted">
          Two Wildcards, two Free Hits, two Bench Boosts and two Triple
          Captains -- one of each per half-season, and an unplayed first-half
          chip expires for good at the GW19 deadline. This solves the
          scheduling problem, not just a ranking, so it can tell you when
          waiting for a better week actually costs you the chip.
        </p>
      </div>

      <Panel
        title="Auto-schedule from stored predictions"
        subtitle="Reads every gameweek in the horizon from the pipeline's own predictions, computes Bench Boost (bench xP), Triple Captain (best captain's xP), Free Hit and Wildcard values against the optimal squad -- or against YOUR squad if you enter a team ID -- then solves the schedule. Predictions must already exist for the horizon; if any gameweek is missing, the CLI `fplagent chips` command will predict on demand."
      >
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={autoEntryId}
            onChange={(e) => setAutoEntryId(e.target.value)}
            placeholder="Optional: your FPL team ID"
            inputMode="numeric"
            className="w-56 rounded-md border border-line bg-bg px-3 py-1.5 text-sm outline-none focus:border-[var(--accent)]"
          />
          <label className="flex items-center gap-2 text-xs text-muted">
            Horizon
            <input
              type="number"
              min={4}
              max={20}
              value={autoHorizon}
              onChange={(e) => setAutoHorizon(Math.max(4, Math.min(20, Number(e.target.value) || WINDOW)))}
              className="w-16 rounded-md border border-line bg-bg px-2 py-1.5 text-sm outline-none focus:border-[var(--accent)]"
            />
          </label>
          <button
            onClick={fetchAuto}
            disabled={autoLoading}
            className="rounded-md bg-[var(--accent)] px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {autoLoading ? "Computing\u2026" : "Auto-fill & solve"}
          </button>
          <span className="text-xs text-muted">
            Team ID fills Free Hit and Wildcard values against your real squad;
            leave blank to use the optimal squad.
          </span>
        </div>
        {autoError ? (
          <p className="mt-2 text-xs text-[var(--bad)]">{autoError}</p>
        ) : null}
      </Panel>

      <Panel
        title="Estimated value by gameweek"
        subtitle="How many extra points you think each chip is worth if played that week. Auto-fill populates this from stored predictions; you can edit any cell and re-solve manually with the button at the bottom."
      >
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <button
            onClick={prefillCurrentWeek}
            disabled={prefilling}
            className="rounded-md border border-line px-3 py-1.5 text-xs hover:text-fg disabled:opacity-50"
          >
            {prefilling ? "Prefilling\u2026" : `Prefill GW${gwNow} from live projections`}
          </button>
          <span className="text-[11px] text-muted">
            Fills Bench Boost (bench xP) and Triple Captain (top captain&apos;s xP) for the
            current gameweek only.
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="text-left text-xs text-muted">
                <th className="px-2 py-2 font-medium">Chip</th>
                {gws.map((g) => (
                  <th key={g} className="px-1 py-2 text-center font-medium">
                    GW{g}
                  </th>
                ))}
                <th className="px-2 py-2 font-medium">Already used (GWs)</th>
              </tr>
            </thead>
            <tbody>
              {CHIPS.map((chip) => (
                <tr key={chip} className="border-t border-line">
                  <td className="px-2 py-2 font-medium">{CHIP_LABEL[chip]}</td>
                  {gws.map((g) => (
                    <td key={g} className="px-1 py-1 text-center">
                      <input
                        type="number"
                        step="0.1"
                        value={values[chip]?.[g] ?? ""}
                        onChange={(e) => setCell(chip, g, Number(e.target.value) || 0)}
                        className="w-14 rounded border border-line bg-bg px-1 py-1 text-center text-xs outline-none focus:border-[var(--accent)]"
                      />
                    </td>
                  ))}
                  <td className="px-2 py-1">
                    <input
                      type="text"
                      placeholder="e.g. 7"
                      value={used[chip] ?? ""}
                      onChange={(e) => setUsed((prev) => ({ ...prev, [chip]: e.target.value }))}
                      className="w-20 rounded border border-line bg-bg px-1.5 py-1 text-xs outline-none focus:border-[var(--accent)]"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <button
          onClick={solve}
          disabled={solving}
          className="mt-4 rounded-md bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {solving ? "Solving\u2026" : "Solve schedule"}
        </button>
        {error && gwNow ? <p className="mt-2 text-xs text-[var(--bad)]">{error}</p> : null}
      </Panel>

      {schedule ? (
        <>
          <Panel
            title="Schedule"
            subtitle={`Status: ${schedule.status}. Total estimated value: ${schedule.total_value.toFixed(2)} points.`}
          >
            <ChipTimeline schedule={schedule} />
          </Panel>

          <Panel title="Value by chip">
            <ChipValueBar schedule={schedule} values={values} />
          </Panel>

          {schedule.notes.length > 0 ? (
            <Panel title="Notes">
              <ul className="list-disc space-y-1 pl-4 text-sm text-muted">
                {schedule.notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            </Panel>
          ) : null}
        </>
      ) : null}
    </div>
  );
}