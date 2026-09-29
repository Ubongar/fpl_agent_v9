"use client";

import { useEffect, useState } from "react";
import type { ComparePlayer } from "@/lib/types";
import { playerColor } from "@/lib/comparePalette";
import { postJSON, BrowserApiError } from "@/lib/browserApi";
import { pct } from "@/lib/format";

type Duel = { p_a_wins: number; p_b_wins: number; p_tie: number };
type Row = { opponent: ComparePlayer; duel: Duel | null; error?: string };

/**
 * The reference player against every other selected player, run as one grid
 * instead of one pairwise call at a time. Reuses /compare/captaincy (the
 * same simulation the single duel uses) fired in parallel; the comparison
 * set is capped at 8 players, so this is at most 7 requests.
 */
export function DuelMatrix({ anchor, opponents, season }: { anchor: ComparePlayer; opponents: ComparePlayer[]; season?: number }) {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const opponentIds = opponents.map(o => o.player_id).join(",");

  // Resetting to a loading state at the start of a fetch effect, then
  // setting the result once the fetch resolves, is the standard
  // data-fetching effect shape (the player pool loaded above follows the
  // same shape for its own setState calls, just split across a .then()).
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    let cancelled = false;
    setRows(null);
    setError(null);
    if (!opponentIds) return;
    setLoading(true);
    Promise.all(opponents.map(async (opp): Promise<Row> => {
      try {
        const path = season != null ? `/compare/captaincy?season=${season}` : "/compare/captaincy";
        const duel = await postJSON<Duel>(path, {
          player_a_id: anchor.player_id, player_b_id: opp.player_id,
          same_team: anchor.team_id === opp.team_id,
        });
        return { opponent: opp, duel };
      } catch (e) {
        return { opponent: opp, duel: null, error: e instanceof BrowserApiError ? e.message : "Failed" };
      }
    })).then(r => { if (!cancelled) setRows(r); })
      .catch(e => { if (!cancelled) setError(e instanceof BrowserApiError ? e.message : "Could not calculate the duel matrix."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
    // opponentIds is the stable, comparable form of `opponents` used below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anchor.player_id, anchor.team_id, opponentIds, season]);
  /* eslint-enable react-hooks/set-state-in-effect */

  if (!opponents.length) return <div className="rounded-xl border border-dashed border-line p-6 text-center text-xs text-muted">Select at least one more player to build a duel matrix.</div>;
  if (loading && !rows) return <div className="rounded-xl border border-dashed border-line p-6 text-center text-xs text-muted">Calculating {opponents.length} duel{opponents.length > 1 ? "s" : ""} against {anchor.web_name}...</div>;
  if (error) return <div className="rounded-xl border border-dashed border-line p-6 text-center text-xs text-[var(--bad)]">{error}</div>;

  return <div className="space-y-1.5">
    {(rows ?? []).map(({ opponent, duel, error: rowError }) => {
      const color = playerColor(opponents.indexOf(opponent) + 1);
      if (rowError || !duel) {
        return <div key={opponent.player_id} className="flex items-center justify-between rounded-lg border border-line bg-bg/60 px-3 py-2 text-[10px] text-muted">
          <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: color }} />{opponent.web_name}</span>
          <span>{rowError ?? "No data"}</span>
        </div>;
      }
      const aPct = duel.p_a_wins * 100, tPct = duel.p_tie * 100, bPct = duel.p_b_wins * 100;
      return <div key={opponent.player_id} className="rounded-lg border border-line bg-bg/60 px-3 py-2">
        <div className="flex items-center justify-between text-[10px]">
          <span className="font-medium">{anchor.web_name}</span>
          <span className="text-muted"><span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: color }} />{opponent.web_name}</span>
        </div>
        <div className="mt-1.5 flex h-3 overflow-hidden rounded-full bg-line/40" role="img" aria-label={`${anchor.web_name} outscores ${opponent.web_name} ${pct(duel.p_a_wins)} of the time, tie ${pct(duel.p_tie)}, ${opponent.web_name} outscores ${pct(duel.p_b_wins)}`}>
          <div style={{ width: `${aPct}%`, background: "var(--accent)" }} />
          <div style={{ width: `${tPct}%`, background: "var(--muted)", opacity: 0.5 }} />
          <div style={{ width: `${bPct}%`, background: color }} />
        </div>
        <div className="mt-1 flex justify-between font-mono text-[9px] text-muted">
          <span>{pct(duel.p_a_wins)}</span>
          <span>tie {pct(duel.p_tie)}</span>
          <span>{pct(duel.p_b_wins)}</span>
        </div>
      </div>;
    })}
  </div>;
}
