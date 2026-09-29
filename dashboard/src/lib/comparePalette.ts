/**
 * One colour per compared player, shared by every comparison chart so a
 * player keeps the same colour in the distribution overlay, the radar and
 * the legend. Chosen to stay distinguishable on both the light and dark
 * themes and for the common colour-vision deficiencies (Okabe-Ito based).
 * The comparison set is capped at 8 players, matching this palette.
 */
export const COMPARE_PALETTE = [
  "#0072B2", // blue
  "#E69F00", // orange
  "#009E73", // green
  "#D55E00", // vermillion
  "#CC79A7", // pink
  "#56B4E9", // sky
  "#B8A100", // olive
  "#7A6FF0", // violet
] as const;

export function playerColor(index: number): string {
  return COMPARE_PALETTE[index % COMPARE_PALETTE.length] ?? COMPARE_PALETTE[0];
}
