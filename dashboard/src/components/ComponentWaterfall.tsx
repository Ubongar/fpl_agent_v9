import { Bar } from "./ui";

const LABELS: Record<string, string> = {
  appearance: "Appearance",
  goals: "Goals (total)",
  goals_open_play: "  from open play",
  goals_penalty: "  from penalties",
  assists: "Assists",
  clean_sheet: "Clean sheet",
  goals_conceded: "Goals conceded",
  saves: "Saves",
  penalties_saved: "Penalty saves",
  defensive_contribution: "Defensive contribution",
  bonus: "Bonus",
  cards: "Cards",
  own_goals: "Own goals",
  penalty_misses: "Penalty misses",
};

// `goals` is the sum of the two rows beneath it, so showing all three would
// double count the total. The split is displayed because separating penalty
// from open-play goals is the entire point of the set-piece model.
const NESTED = new Set(["goals_open_play", "goals_penalty"]);

export function ComponentWaterfall({
  components,
}: {
  components: Record<string, number>;
}) {
  const entries = Object.entries(components).filter(
    ([k, v]) => !k.startsWith("_") && k !== "total" && (v !== 0 || k === "appearance"),
  );
  const max = Math.max(0.01, ...entries.map(([, v]) => Math.abs(v)));
  const total = entries
    .filter(([k]) => !NESTED.has(k))
    .reduce((a, [, v]) => a + v, 0);

  return (
    <div>
      <ul className="space-y-2">
        {entries.map(([k, v]) => (
          <li key={k} className="grid grid-cols-[minmax(0,1fr)_60px] items-center gap-3">
            <div>
              <div className="flex justify-between text-xs">
                <span
                  className={
                    NESTED.has(k) ? "pl-3 text-muted" : "text-fg"
                  }
                >
                  {LABELS[k] ?? k}
                </span>
              </div>
              <div className="mt-1">
                <Bar value={v} max={max} tone={v < 0 ? "bad" : "accent"} />
              </div>
            </div>
            <span
              className={
                "text-right font-mono text-xs " +
                (v < 0 ? "text-[var(--bad)]" : "text-fg")
              }
            >
              {v >= 0 ? "+" : ""}
              {v.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
      <div className="mt-3 flex justify-between border-t border-line pt-2 text-sm">
        <span className="font-medium">Total</span>
        <span className="font-mono">{total.toFixed(2)}</span>
      </div>
    </div>
  );
}
