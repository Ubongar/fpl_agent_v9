"use client";

import type { ComparePlayer } from "@/lib/types";
import { playerColor } from "@/lib/comparePalette";
import { pts } from "@/lib/format";

/**
 * Every compared player's simulated points distribution on ONE shared axis.
 *
 * Per player: a thin line for P10-P90, a thick bar for P25-P75, a tick at the
 * median and a diamond at the mean. Because the axis is shared, overlap is
 * visible directly: a wide bar sitting left of a narrow bar is a real
 * floor/ceiling trade-off, which a table of numbers hides.
 */
export function DistributionOverlay({ players }: { players: ComparePlayer[] }) {
  const usable = players.filter(p => p.xp_p10 != null && p.xp_p90 != null);
  if (!usable.length) {
    return <div className="rounded-xl border border-dashed border-line p-6 text-center text-xs text-muted">No distribution data for this selection.</div>;
  }

  const lo = Math.min(0, ...usable.map(p => p.xp_p10 ?? 0));
  const hi = Math.max(1, ...usable.map(p => p.xp_p90 ?? 0));
  const span = hi - lo || 1;
  const x = (v: number) => ((v - lo) / span) * 100;

  // A whole-number tick every 2 points (every 5 when the axis is very wide).
  const step = span > 20 ? 5 : 2;
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);

  return <div className="space-y-3">
    <div className="space-y-2.5">
      {usable.map((p) => {
        const i = players.findIndex(q => q.player_id === p.player_id);
        const color = playerColor(i);
        const p10 = p.xp_p10 ?? 0, p90 = p.xp_p90 ?? 0;
        const p25 = p.xp_p25 ?? p10, p75 = p.xp_p75 ?? p90;
        return <div key={p.player_id} className="grid grid-cols-[84px_1fr] items-center gap-2 text-[10px]">
          <div className="truncate font-medium" title={p.web_name}><span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: color }} />{p.web_name}</div>
          <div className="relative h-6" role="img" aria-label={`${p.web_name}: P10 ${pts(p10, 1)}, median ${pts(p.xp_p50, 1)}, mean ${pts(p.xp_mean, 1)}, P90 ${pts(p90, 1)} points`}>
            <div className="absolute top-1/2 h-px -translate-y-1/2" style={{ left: `${x(p10)}%`, width: `${Math.max(0.5, x(p90) - x(p10))}%`, background: color, opacity: 0.55 }} />
            <div className="absolute top-1/2 h-3 -translate-y-1/2 rounded-sm" style={{ left: `${x(p25)}%`, width: `${Math.max(0.8, x(p75) - x(p25))}%`, background: color, opacity: 0.35 }} />
            {p.xp_p50 != null ? <div className="absolute top-1/2 h-4 w-0.5 -translate-y-1/2" style={{ left: `${x(p.xp_p50)}%`, background: color }} /> : null}
            <div className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rotate-45" style={{ left: `${x(p.xp_mean)}%`, background: color, outline: "1.5px solid var(--bg, #fff)" }} />
          </div>
        </div>;
      })}
    </div>
    <div className="grid grid-cols-[84px_1fr] gap-2">
      <div />
      <div className="relative h-4 border-t border-line text-[9px] text-muted">
        {ticks.map(t => <span key={t} className="absolute top-1 -translate-x-1/2 font-mono" style={{ left: `${x(t)}%` }}>{t}</span>)}
      </div>
    </div>
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[9px] text-muted">
      <span>Line = P10 to P90</span><span>Block = P25 to P75</span><span>Bar = median</span><span>Diamond = mean</span>
    </div>
  </div>;
}
