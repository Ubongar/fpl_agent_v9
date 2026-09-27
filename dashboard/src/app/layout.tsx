import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { AppNav, MobileNav } from "@/components/AppNav";

export const metadata: Metadata = {
  title: "FPL Agent — Decision Intelligence",
  description: "Probabilistic FPL prediction, simulation, optimisation and decision intelligence.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-bg">
          <header className="sticky top-0 z-40 border-b border-line/80 bg-bg/95 backdrop-blur-xl">
            <div className="flex h-[73px] items-center justify-between gap-4 px-4 md:px-6">
              <div className="flex min-w-0 items-center gap-3">
                <Link href="/" className="shrink-0 rounded-xl border border-line bg-panel px-3 py-2 text-xs font-bold tracking-tight">
                  FPL <span className="text-[var(--accent)]">AGENT</span>
                </Link>
                <div className="hidden h-6 w-px bg-line sm:block" />
                <div className="hidden min-w-0 sm:block"><div className="text-[9px] uppercase tracking-[0.18em] text-muted">Decision intelligence</div><div className="truncate text-xs font-medium">Projection · simulation · optimisation</div></div>
              </div>
              <div className="flex items-center gap-2">
                <Link href="/compare" className="rounded-xl border border-[var(--accent)]/30 bg-[var(--accent)]/10 px-3 py-2 text-[10px] font-bold uppercase tracking-wide text-[var(--accent)] hover:bg-[var(--accent)]/15">Player Compare</Link>
                <div className="hidden items-center gap-2 rounded-xl border border-line bg-panel px-3 py-2 md:flex"><span className="h-1.5 w-1.5 rounded-full bg-[var(--good)]"/><span className="text-[9px] font-semibold uppercase tracking-wider text-muted">Model surface online</span></div>
              </div>
            </div>
          </header>
          <MobileNav />
          <div className="mx-auto flex max-w-[1700px]">
            <AppNav />
            <main className="min-w-0 flex-1 px-4 py-6 md:px-6 md:py-7">{children}</main>
          </div>
          <footer className="mx-auto max-w-[1700px] px-4 pb-10 pt-1 text-[10px] text-muted md:px-6">Projections are model outputs. Inspect uncertainty, availability, fixture context and market signals before acting.</footer>
        </div>
      </body>
    </html>
  );
}
