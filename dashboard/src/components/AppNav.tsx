"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { useState } from 'react';

const GROUPS = [
  {
    label: "Core",
    items: [
      ["/", "Command Center", "⌂"],
      ["/players", "Player Lab", "PL"],
      ["/compare", "Player Compare", "VS"],
      ["/manager", "My Team", "MT"],
      ["/fixtures", "Fixture Lab", "FX"],
      ["/market", "Market Watch", "MW"],
    ],
  },
  {
    label: "Planning",
    items: [
      ["/squad", "Squad Builder", "SB"],
      ["/transfers", "Transfers", "TR"],
      ["/captaincy", "Captaincy", "C"],
      ["/chips", "Chips", "CH"],
    ],
  },
  {
    label: "Evidence",
    items: [
      ["/teams", "Teams", "TM"],
      ["/movers", "Movers", "MV"],
      ["/model", "Model Lab", "ML"],
      ["/health", "Health", "HL"],
      ["/quality", "Data Quality", "DQ"],
    ],
  },
] as const;

export function AppNav() {
  const pathname = usePathname();
  const active = (href: string) => href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <aside className="hidden w-60 shrink-0 border-r border-line bg-panel/45 lg:block">
      <div className="sticky top-[73px] flex max-h-[calc(100vh-73px)] flex-col px-3 py-4">
        <div className="px-2 pb-3">
          <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted">Workspace</div>
          <div className="mt-1 text-xs text-muted">Research → compare → act</div>
        </div>
        <div className="scrollbar-thin flex-1 space-y-5 overflow-y-auto pr-1">
          {GROUPS.map((group) => (
            <div key={group.label}>
              <div className="px-2 pb-1.5 text-[9px] font-semibold uppercase tracking-[0.18em] text-muted">{group.label}</div>
              <div className="space-y-0.5">
                {group.items.map(([href, label, mark]) => (
                  <Link key={href} href={href} className={clsx(
                    "group flex items-center gap-2.5 rounded-xl px-2.5 py-2 text-xs transition",
                    active(href) ? "bg-[var(--accent)]/12 text-fg ring-1 ring-[var(--accent)]/20" : "text-muted hover:bg-bg hover:text-fg",
                    href === "/compare" && !active(href) && "text-[var(--accent)]",
                  )}>
                    <span className={clsx(
                      "grid h-6 w-6 place-items-center rounded-lg border text-[9px] font-bold",
                      active(href) ? "border-[var(--accent)]/30 bg-[var(--accent)]/12 text-[var(--accent)]" : "border-line bg-bg/70 text-muted group-hover:text-fg",
                    )}>{mark}</span>
                    <span className="font-medium">{label}</span>
                    {href === "/compare" ? <span className="ml-auto rounded-md border border-[var(--accent)]/20 px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wide text-[var(--accent)]">NEW</span> : null}
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 rounded-xl border border-line bg-bg/60 p-3">
          <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-muted">Method</div>
          <p className="mt-1 text-[10px] leading-5 text-muted">Use the distribution, not just the mean. Small gaps can be simulation noise.</p>
        </div>
      </div>
    </aside>
  );
}

export function MobileNav() {
  const pathname = usePathname();
  const items = [["/", "Home"], ["/players", "Players"], ["/compare", "Compare"], ["/manager", "My Team"], ["/transfers", "Transfers"]] as const;
  return (
    <nav className="border-b border-line bg-bg/95 px-3 py-2 lg:hidden">
      <div className="flex gap-1 overflow-x-auto scrollbar-thin pb-0.5">
        {items.map(([href, label]) => {
          const isActive = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return <Link key={href} href={href} className={clsx("whitespace-nowrap rounded-lg px-3 py-1.5 text-xs font-medium", isActive ? "bg-[var(--accent)]/12 text-[var(--accent)]" : "text-muted hover:text-fg")}>{label}</Link>;
        })}
      </div>
    </nav>
  );
}
