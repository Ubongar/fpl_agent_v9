import clsx from "clsx";
import Link from "next/link";
import { positionColour } from "@/lib/format";

export function Panel({ title, subtitle, children, className }: { title?: string; subtitle?: string; children: React.ReactNode; className?: string }) {
  return <section className={clsx("rounded-2xl border border-line bg-panel p-4 md:p-5", className)}>
    {title ? <header className="mb-4"><h2 className="text-sm font-semibold tracking-tight">{title}</h2>{subtitle ? <p className="mt-1 text-xs leading-5 text-muted">{subtitle}</p> : null}</header> : null}
    {children}
  </section>;
}

export function Stat({ label, value, hint, tone = "neutral" }: { label: string; value: string; hint?: string; tone?: "neutral" | "good" | "warn" | "bad" }) {
  const colour = tone === "good" ? "text-[var(--good)]" : tone === "warn" ? "text-[var(--warn)]" : tone === "bad" ? "text-[var(--bad)]" : "text-fg";
  return <div className="rounded-xl border border-line bg-bg/60 p-3"><div className="text-[10px] uppercase tracking-wider text-muted">{label}</div><div className={clsx("mt-1 text-lg font-semibold font-mono", colour)}>{value}</div>{hint ? <div className="mt-1 text-[10px] text-muted">{hint}</div> : null}</div>;
}

export function PositionBadge({ position }: { position: string }) { return <span className={clsx("rounded-md px-1.5 py-0.5 text-[10px] font-semibold", positionColour[position] ?? "bg-line text-muted")}>{position}</span>; }

export function PlayerLink({ id, name }: { id: number; name: string }) { return <Link href={`/players/${id}`} className="font-medium text-fg underline-offset-2 hover:text-[var(--accent)] hover:underline">{name}</Link>; }

export function EmptyState({ title, detail, command }: { title: string; detail: string; command?: string }) { return <div className="rounded-2xl border border-dashed border-line bg-panel p-10 text-center"><h2 className="text-base font-semibold">{title}</h2><p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-muted">{detail}</p>{command ? <pre className="mx-auto mt-4 w-fit rounded-xl border border-line bg-bg px-4 py-3 text-left font-mono text-xs text-muted">{command}</pre> : null}</div>; }

export function Bar({ value, max, tone = "accent" }: { value: number; max: number; tone?: "accent" | "good" | "bad" }) { const pctWidth = max > 0 ? Math.min(100, (Math.abs(value) / max) * 100) : 0; const colour = tone === "good" ? "bg-[var(--good)]" : tone === "bad" ? "bg-[var(--bad)]" : "bg-[var(--accent)]"; return <div className="h-1.5 w-full overflow-hidden rounded-full bg-line"><div className={clsx("h-full rounded-full", colour)} style={{ width: `${pctWidth}%` }} /></div>; }
