"use client";

import { useEffect, useMemo, useState } from "react";
import { getJSON, postJSON, BrowserApiError } from "@/lib/browserApi";
import { TransferGainChart } from "@/components/TransferGainChart";
import { EmptyState, Panel, PositionBadge, Stat } from "@/components/ui";
import { money, pts } from "@/lib/format";
import type { ManagerTeam, PlayerPrediction, TransferPlan } from "@/lib/types";

const REQUIRED: Record<string, number> = { GK: 2, DEF: 5, MID: 5, FWD: 3 };

export default function TransfersPage() {
  const [pool, setPool] = useState<PlayerPrediction[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [search, setSearch] = useState("");
  const [position, setPosition] = useState("ALL");

  const [bank, setBank] = useState(0);
  const [freeTransfers, setFreeTransfers] = useState(1);
  const [horizon, setHorizon] = useState(4);
  const [risk, setRisk] = useState(0);

  const [plan, setPlan] = useState<TransferPlan | null>(null);
  const [solving, setSolving] = useState(false);
  const [solveError, setSolveError] = useState<string | null>(null);

  const [entryId, setEntryId] = useState("");
  const [myTeam, setMyTeam] = useState<ManagerTeam | null>(null);
  const [loadingTeam, setLoadingTeam] = useState(false);
  const [teamError, setTeamError] = useState<string | null>(null);
  // Real purchase prices, player_id -> price, only for players FPL's
  // transfer history actually confirms a buy price for. Fed straight into
  // /optimise/transfers so a risen asset's sell value -- purchase price
  // plus half the rise, not the full current price -- is right, not just
  // the squad selection.
  const [purchasePrices, setPurchasePrices] = useState<Record<number, number>>({});

  useEffect(() => {
    getJSON<{ players: PlayerPrediction[] }>("/predictions?limit=700")
      .then((r) => setPool(r.players))
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load the player pool."));
  }, []);

  const counts = useMemo(() => {
    const c: Record<string, number> = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
    if (pool) {
      for (const p of pool) {
        if (selected.has(p.player_id)) c[p.position] = (c[p.position] ?? 0) + 1;
      }
    }
    return c;
  }, [pool, selected]);

  const validFormation = Object.entries(REQUIRED).every(([pos, n]) => counts[pos] === n);

  const rows = useMemo(() => {
    if (!pool) return [];
    let r = pool;
    if (position !== "ALL") r = r.filter((p) => p.position === position);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      r = r.filter((p) => p.web_name.toLowerCase().includes(q));
    }
    return r.slice(0, 120);
  }, [pool, position, search]);

  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < 15) next.add(id);
      return next;
    });
  }

  async function loadMyTeam() {
    const id = Number(entryId);
    if (!id || id <= 0) {
      setTeamError("Enter your FPL team ID -- it's the number in the URL "
        + "when you view your team on the official FPL site "
        + "(fantasy.premierleague.com/entry/<this number>/event/...).");
      return;
    }
    setLoadingTeam(true);
    setTeamError(null);
    try {
      const team = await getJSON<ManagerTeam>(`/manager/${id}/team`);
      setMyTeam(team);
      setSelected(new Set(team.squad.map((p) => p.player_id)));
      setBank(team.bank);
      setFreeTransfers(team.free_transfers);
      const prices: Record<number, number> = {};
      for (const p of team.squad) {
        if (p.purchase_price_known && p.purchase_price != null) {
          prices[p.player_id] = p.purchase_price;
        }
      }
      setPurchasePrices(prices);
    } catch (e) {
      setTeamError(
        e instanceof BrowserApiError
          ? e.message
          : "Could not fetch that team. Double-check the ID.",
      );
    } finally {
      setLoadingTeam(false);
    }
  }

  async function solve() {
    setSolving(true);
    setSolveError(null);
    setPlan(null);
    try {
      const result = await postJSON<TransferPlan>("/optimise/transfers", {
        squad_player_ids: Array.from(selected),
        bank,
        free_transfers: freeTransfers,
        horizon,
        risk_lambda: risk,
        // Only carries prices for players that came from a real team fetch
        // and whose purchase price FPL's transfer history actually
        // confirmed. A manually-built squad sends none of this, which is
        // correct: there is no real purchase price to report for it.
        purchase_prices: Object.keys(purchasePrices).length > 0 ? purchasePrices : undefined,
      });
      setPlan(result);
    } catch (e) {
      setSolveError(
        e instanceof BrowserApiError
          ? e.message
          : "Could not solve a transfer plan for this squad.",
      );
    } finally {
      setSolving(false);
    }
  }

  if (error) {
    return (
      <EmptyState
        title="Could not load the player pool"
        detail={error}
        command="uvicorn fplagent.api.main:app --port 8000"
      />
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Transfer planner</h1>
        <p className="mt-1 text-sm text-muted">
          Pick your current fifteen, and the planner searches the full
          horizon jointly rather than one gameweek at a time -- a transfer
          that looks bad this week and great in three weeks is exactly what a
          greedy, week-by-week planner misses.
        </p>
      </div>

      <Panel
        title="Load your real team"
        subtitle="Fetches your current fifteen straight from FPL's own site -- no need to click through them by hand. This reads public data only; nothing is ever written back to your FPL account."
      >
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={entryId}
            onChange={(e) => setEntryId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && loadMyTeam()}
            placeholder="Your FPL team ID, e.g. 1234567"
            inputMode="numeric"
            className="w-56 rounded-md border border-line bg-bg px-3 py-1.5 text-sm outline-none focus:border-[var(--accent)]"
          />
          <button
            onClick={loadMyTeam}
            disabled={loadingTeam}
            className="rounded-md bg-[var(--accent)] px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {loadingTeam ? "Loading\u2026" : "Load team"}
          </button>
          <span className="text-xs text-muted">
            Find your ID in the URL when you view your team on{" "}
            <span className="font-mono">fantasy.premierleague.com</span> --
            it&apos;s the number after <span className="font-mono">/entry/</span>.
          </span>
        </div>

        {teamError ? <p className="mt-2 text-xs text-[var(--bad)]">{teamError}</p> : null}

        {myTeam ? (
          <div className="mt-3 rounded-md border border-line bg-bg p-3 text-sm">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <span className="font-medium">{myTeam.team_name}</span>
                <span className="ml-1.5 text-muted">({myTeam.manager_name})</span>
              </div>
              <span className="text-xs text-muted">GW{myTeam.gw}</span>
            </div>
            <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
              <span>Bank: <span className="font-mono text-fg">{money(myTeam.bank)}</span></span>
              <span>Value: <span className="font-mono text-fg">{money(myTeam.team_value)}</span></span>
              <span>
                Free transfers: <span className="font-mono text-fg">{myTeam.free_transfers}</span>
                {myTeam.free_transfers_is_estimated ? " (estimated)" : ""}
              </span>
              {myTeam.active_chip ? (
                <span className="text-[var(--warn)]">Active chip: {myTeam.active_chip}</span>
              ) : null}
            </div>
            {myTeam.note ? (
              <p className="mt-2 text-xs text-[var(--warn)]">{myTeam.note}</p>
            ) : null}
          </div>
        ) : null}
      </Panel>

      <Panel
        title={`Your squad (${selected.size} / 15)`}
        subtitle={
          selected.size === 15
            ? validFormation
              ? "Valid formation: 2 GK, 5 DEF, 5 MID, 3 FWD."
              : `Fifteen selected, but not a legal squad shape yet -- GK ${counts.GK}, DEF ${counts.DEF}, MID ${counts.MID}, FWD ${counts.FWD} (need 2/5/5/3).`
            : "Select exactly fifteen players: 2 GK, 5 DEF, 5 MID, 3 FWD."
        }
      >
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search player"
            className="rounded-md border border-line bg-bg px-3 py-1.5 text-sm outline-none focus:border-[var(--accent)]"
          />
          <div className="flex gap-1">
            {["ALL", "GK", "DEF", "MID", "FWD"].map((p) => (
              <button
                key={p}
                onClick={() => setPosition(p)}
                className={
                  "rounded-md border px-2.5 py-1.5 text-xs " +
                  (position === p
                    ? "border-[var(--accent)] text-[var(--accent)]"
                    : "border-line text-muted hover:text-fg")
                }
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        {pool ? (
          <div className="max-h-80 overflow-y-auto rounded-lg border border-line">
            <table className="w-full min-w-[560px] text-sm">
              <thead className="sticky top-0 bg-panel">
                <tr className="text-left text-xs text-muted">
                  <th className="px-2 py-2 font-medium"> </th>
                  <th className="px-2 py-2 font-medium">Player</th>
                  <th className="px-2 py-2 font-medium">Pos</th>
                  <th className="px-2 py-2 font-medium">Club</th>
                  <th className="px-2 py-2 text-right font-medium">Price</th>
                  <th className="px-2 py-2 text-right font-medium">xP</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((p) => (
                  <tr
                    key={p.player_id}
                    className="cursor-pointer border-t border-line hover:bg-panel"
                    onClick={() => toggle(p.player_id)}
                  >
                    <td className="px-2 py-1.5">
                      <input
                        type="checkbox"
                        readOnly
                        checked={selected.has(p.player_id)}
                        className="pointer-events-none"
                      />
                    </td>
                    <td className="px-2 py-1.5">{p.web_name}</td>
                    <td className="px-2 py-1.5">
                      <PositionBadge position={p.position} />
                    </td>
                    <td className="px-2 py-1.5 text-muted">{p.short_name ?? "-"}</td>
                    <td className="px-2 py-1.5 text-right font-mono">{money(p.now_cost)}</td>
                    <td className="px-2 py-1.5 text-right font-mono">{pts(p.xp_mean)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-muted">Loading player pool&hellip;</p>
        )}
      </Panel>

      <Panel title="Constraints">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <label className="text-xs text-muted">
            Bank (&pound;m)
            <input
              type="number"
              step="0.1"
              value={bank}
              onChange={(e) => setBank(Number(e.target.value) || 0)}
              className="mt-1 w-full rounded-md border border-line bg-bg px-2 py-1.5 text-sm outline-none focus:border-[var(--accent)]"
            />
          </label>
          <label className="text-xs text-muted">
            Free transfers
            <input
              type="number"
              min={0}
              max={5}
              value={freeTransfers}
              onChange={(e) => setFreeTransfers(Number(e.target.value) || 0)}
              className="mt-1 w-full rounded-md border border-line bg-bg px-2 py-1.5 text-sm outline-none focus:border-[var(--accent)]"
            />
          </label>
          <label className="text-xs text-muted">
            Horizon (gameweeks)
            <input
              type="number"
              min={1}
              max={10}
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value) || 1)}
              className="mt-1 w-full rounded-md border border-line bg-bg px-2 py-1.5 text-sm outline-none focus:border-[var(--accent)]"
            />
          </label>
          <label className="text-xs text-muted">
            Risk aversion (0-1)
            <input
              type="number"
              step="0.1"
              min={0}
              max={1}
              value={risk}
              onChange={(e) => setRisk(Number(e.target.value) || 0)}
              className="mt-1 w-full rounded-md border border-line bg-bg px-2 py-1.5 text-sm outline-none focus:border-[var(--accent)]"
            />
          </label>
        </div>
        <p className="mt-3 text-[11px] text-muted">
          Every gameweek in the horizon is scored off this gameweek&apos;s
          projections -- the model doesn&apos;t yet forecast fixture-specific
          swings further ahead, so a long horizon mainly stress-tests price
          and bank rather than future form.
        </p>
        <button
          onClick={solve}
          disabled={solving || selected.size !== 15}
          className="mt-3 rounded-md bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {solving ? "Solving\u2026" : "Plan transfers"}
        </button>
        {solveError ? <p className="mt-2 text-xs text-[var(--bad)]">{solveError}</p> : null}
      </Panel>

      {plan ? (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Status" value={plan.status} />
            <Stat label="Hits taken" value={String(plan.hits)} tone={plan.hits > 0 ? "warn" : "good"} />
            <Stat label="Net gain" value={pts(plan.net_gain)} hint="After hit cost" />
            <Stat label="Gameweeks with moves" value={String(plan.moves.length)} />
          </div>

          <Panel title="Net xP by gameweek">
            <TransferGainChart moves={plan.moves} />
          </Panel>

          <Panel title="Moves">
            {plan.moves.length === 0 ? (
              <p className="text-sm text-muted">
                No moves recommended -- roll every transfer across this horizon.
              </p>
            ) : (
              <div className="space-y-3">
                {plan.moves.map((m) => (
                  <div key={m.gw} className="rounded-md border border-line p-3 text-sm">
                    <div className="mb-1.5 flex items-center justify-between">
                      <span className="font-medium">GW{m.gw}</span>
                      {m.hits > 0 ? (
                        <span className="text-xs text-[var(--warn)]">
                          {m.hits} hit{m.hits > 1 ? "s" : ""} (&minus;{m.hits * 4} pts)
                        </span>
                      ) : null}
                    </div>
                    <div className="grid gap-2 sm:grid-cols-2">
                      <div>
                        <div className="text-[11px] uppercase tracking-wide text-[var(--good)]">In</div>
                        <ul className="mt-1 space-y-0.5 text-xs">
                          {m.in.map((p) => (
                            <li key={p.player_id} className="flex justify-between">
                              <span>{p.name}</span>
                              <span className="font-mono text-muted">
                                {money(p.price)} &middot; {pts(p.xp, 1)}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <div className="text-[11px] uppercase tracking-wide text-[var(--bad)]">Out</div>
                        <ul className="mt-1 space-y-0.5 text-xs">
                          {m.out.map((p) => (
                            <li key={p.player_id} className="flex justify-between">
                              <span>{p.name}</span>
                              <span className="font-mono text-muted">
                                {money(p.price)} &middot; {pts(p.xp, 1)}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Panel>

          {plan.reasoning.length > 0 ? (
            <Panel title="Reasoning">
              <ul className="list-disc space-y-1 pl-4 text-sm text-muted">
                {plan.reasoning.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </Panel>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
