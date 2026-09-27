import { dataQuality } from "@/lib/api";
import { EmptyState, Panel, Stat } from "@/components/ui";
import type { QualityEvent } from "@/lib/types";

export const dynamic = "force-dynamic";

const SEVERITY_TONE: Record<string, "good" | "warn" | "bad" | "neutral"> = {
  info: "neutral",
  warn: "warn",
  fail: "bad",
};

export default async function QualityPage() {
  let events: QualityEvent[] = [];
  let error: string | null = null;

  try {
    const result = await dataQuality(100);
    events = result.events;
  } catch (e) {
    error = e instanceof Error ? e.message : "Unknown error.";
  }

  if (error) {
    return (
      <EmptyState
        title="Could not load data-quality events"
        detail={error}
        command="fplagent run"
      />
    );
  }

  if (events.length === 0) {
    return (
      <EmptyState
        title="No data-quality events logged"
        detail="Either nothing has been flagged, or the pipeline hasn't run yet."
        command="fplagent run"
      />
    );
  }

  const counts = { info: 0, warn: 0, fail: 0 } as Record<string, number>;
  for (const e of events) counts[e.severity] = (counts[e.severity] ?? 0) + 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Data quality</h1>
        <p className="mt-1 text-sm text-muted">
          What to check when a projection looks wrong, before rewriting any
          model. Every ingest and predict run can log an event here, most
          recent first.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Stat label="Info" value={String(counts.info ?? 0)} />
        <Stat label="Warnings" value={String(counts.warn ?? 0)} tone="warn" />
        <Stat label="Failures" value={String(counts.fail ?? 0)} tone="bad" />
      </div>

      <Panel title="Recent events">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="text-left text-xs text-muted">
                <th className="px-2 py-2 font-medium">When</th>
                <th className="px-2 py-2 font-medium">Stage</th>
                <th className="px-2 py-2 font-medium">Check</th>
                <th className="px-2 py-2 font-medium">Severity</th>
                <th className="px-2 py-2 font-medium">Detail</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => {
                const tone = SEVERITY_TONE[e.severity] ?? "neutral";
                const colour =
                  tone === "bad"
                    ? "text-[var(--bad)]"
                    : tone === "warn"
                      ? "text-[var(--warn)]"
                      : tone === "good"
                        ? "text-[var(--good)]"
                        : "text-muted";
                return (
                  <tr key={e.event_id} className="border-t border-line align-top">
                    <td className="whitespace-nowrap px-2 py-2 font-mono text-xs text-muted">
                      {new Date(e.created_at).toLocaleString()}
                    </td>
                    <td className="px-2 py-2">{e.stage ?? "-"}</td>
                    <td className="px-2 py-2">{e.check_name ?? "-"}</td>
                    <td className={`px-2 py-2 font-medium uppercase ${colour}`}>
                      {e.severity}
                    </td>
                    <td className="max-w-sm px-2 py-2 font-mono text-xs text-muted">
                      {e.detail ? JSON.stringify(e.detail) : "-"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
