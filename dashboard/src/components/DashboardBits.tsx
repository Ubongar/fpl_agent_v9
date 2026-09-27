import Link from "next/link";
import clsx from "clsx";
import type { DashboardPlayer } from "@/lib/types";
import { money, pct, pts } from "@/lib/format";
import { PositionBadge } from "@/components/ui";

export function Kpi({ label, value, detail, tone = "neutral" }: { label: string; value: string; detail?: string; tone?: "neutral" | "good" | "warn" | "bad" }) {
  return (
    <div className={clsx("rounded-2xl border border-line bg-panel/80 p-4 shadow-sm", tone === "good" && "ring-1 ring-[var(--good)]/25", tone === "warn" && "ring-1 ring-[var(--warn)]/25", tone === "bad" && "ring-1 ring-[var(--bad)]/25")}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-[10px] font-medium uppercase tracking-[0.16em] text-muted">{label}</span>
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
      </div>
      <div className="mt-3 text-2xl font-semibold tracking-tight">{value}</div>
      {detail ? <div className="mt-1 text-xs text-muted">{detail}</div> : null}
    </div>
  );
}

export function PlayerStrip({ p, metric = "xp" }: { p: DashboardPlayer; metric?: "xp" | "value" | "upside" }) {
  const m = metric === "value" ? `${p.value?.toFixed(2) ?? "—"}/£m` : metric === "upside" ? `+${p.upside_points?.toFixed(1) ?? "—"}` : pts(p.xp_mean);
  return (
    <Link href={`/players/${p.player_id}`} className="group flex items-center justify-between gap-3 rounded-xl border border-line bg-bg/60 px-3 py-2.5 hover:border-[var(--accent)] hover:bg-bg">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-semibold">{p.web_name}</span>
          <PositionBadge position={p.position} />
        </div>
        <div className="mt-0.5 text-[11px] text-muted">{p.short_name ?? "—"} · {money(p.now_cost)}</div>
      </div>
      <div className="text-right">
        <div className="font-mono text-sm font-semibold">{m}</div>
        <div className="text-[10px] text-muted">P(play) {pct(p.p_play)}</div>
      </div>
    </Link>
  );
}

export function MetricPill({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "neutral" | "good" | "warn" | "bad" }) {
  const cls = tone === "good" ? "text-[var(--good)] bg-[var(--good)]/10" : tone === "warn" ? "text-[var(--warn)] bg-[var(--warn)]/10" : tone === "bad" ? "text-[var(--bad)] bg-[var(--bad)]/10" : "text-muted bg-bg";
  return <span className={clsx("rounded-lg px-2 py-1 text-[10px] font-medium", cls)}>{label} {value}</span>;
}
