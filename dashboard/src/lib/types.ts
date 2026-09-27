/**
 * Hand-maintained mirror of the API contract.
 *
 * These are a fallback. The real source of truth is the OpenAPI schema
 * FastAPI already emits: run `npm run gen:types` with the API up and
 * `openapi-typescript` writes `api-schema.d.ts` straight from it.
 */

export type Position = "GK" | "DEF" | "MID" | "FWD";

export interface PlayerPrediction {
  player_id: number;
  web_name: string;
  position: Position;
  team_id: number;
  short_name: string | null;
  now_cost: number | null;
  gw: number;
  xp_mean: number;
  xp_se: number | null;
  xp_p10: number | null;
  xp_p25: number | null;
  xp_p50: number | null;
  xp_p75: number | null;
  xp_p90: number | null;
  p_play: number;
  p_start: number | null;
  p_sixty: number;
  p_haul_10plus: number | null;
  n_fixtures: number;
  value: number | null;
  model_version: string;
}

export interface PredictionsPage {
  total: number;
  offset: number;
  limit: number;
  model_version: string | null;
  players: PlayerPrediction[];
}

export interface PlayerDetail {
  player_id: number;
  /** Player identity, resolved server-side. Present on all current responses. */
  web_name?: string | null;
  position?: Position | null;
  team_id?: number | null;
  short_name?: string | null;
  gw: number;
  model_version: string;
  expected_points: number;
  distribution: { p10: number; p25: number; p50: number; p75: number; p90: number };
  probabilities: {
    play: number;
    sixty_plus: number;
    haul_10plus: number;
    blank_2_or_less: number;
  };
  components: Record<string, number>;
  inputs: Record<string, unknown>;
  quantile_grid: number[];
  decision?: DecisionMetrics;
}

export interface TeamStrength {
  team_id: number;
  name?: string;
  short_name?: string;
  attack: number;
  defence: number;
  home_adv: number;
  rho: number;
  league_avg_goals: number;
  n_matches_fit: number;
  effective_n: number | null;
  target: string | null;
}

export interface SetPieceHolder {
  player_id: number;
  web_name: string | null;
  short_name: string | null;
  position: Position | null;
  duty: string;
  order: number;
  share: number | null;
  confidence: number | null;
  source: string | null;
}

export interface CalibrationBucket {
  bucket: string;
  predicted_rate: number;
  actual_rate: number;
  n: number;
}

export interface EvaluationRun {
  run_id: number;
  model_version: string;
  gw_from: number;
  gw_to: number;
  n: number;
  mae: number | null;
  rmse: number | null;
  spearman_overall: number | null;
  spearman_by_position: Record<string, number> | null;
  top5_captain_hit_rate: number | null;
  top10_captain_hit_rate: number | null;
  crps: number | null;
  pit_ks_stat: number | null;
  pit_histogram: number[] | null;
  baselines: Record<string, { mae?: number; spearman?: number; n?: number }> | null;
  beat_best_baseline: boolean | null;
  created_at: string;
}

export interface BacktestResult extends Partial<EvaluationRun> {
  pit_interpretation?: string;
  sixty_calibration?: CalibrationBucket[];
  start_calibration?: CalibrationBucket[];
  play_calibration?: CalibrationBucket[];
  sixty_calibration_ece?: number;
  play_calibration_ece?: number;
  baseline_significance?: {
    n_gameweeks: number;
    beats_all_significantly: boolean;
    per_baseline?: Record<
      string,
      {
        mean_spearman_gain?: number;
        ci95?: [number, number];
        significant?: boolean;
        note?: string;
      }
    >;
  };
  note?: string;
}

export interface SquadPlayer {
  player_id: number;
  name: string;
  position: Position;
  club: string;
  price: number;
  xp: number;
  p_play?: number;
}

export interface SquadSolution {
  summary: {
    formation: string;
    total_cost: number;
    xi_expected_points: number;
    captain: string | null;
    vice_captain: string | null;
    status: string;
  };
  starting_xi: SquadPlayer[];
  bench?: SquadPlayer[];
}

export interface CaptainCandidate {
  player_id: number;
  name: string;
  xp: number;
  captain_score: number;
  expected_captain_points: number;
  ceiling_p90: number | null;
  floor_p10: number | null;
  p_haul: number | null;
  differential_points: number;
  effective_ownership: number;
  risk_note: string;
}

export interface AttributionChange {
  player_id: number;
  name: string;
  xp_now: number;
  xp_prev: number;
  delta: number;
  drivers: { component: string; delta: number; share_of_change: number }[];
  context: string[];
}

export interface ChangesResponse {
  changes: AttributionChange[];
  narrative?: string[];
}

/** One of the four half-season chips. */
export type ChipName = "wildcard" | "free_hit" | "bench_boost" | "triple_captain";

export interface ChipSchedule {
  assignments: Record<string, { first: number | null; second: number | null }>;
  total_value: number;
  status: string;
  notes: string[];
}

export interface TransferPlayer {
  player_id: number;
  name: string;
  price: number;
  xp: number;
}

export interface TransferMove {
  gw: number;
  in: TransferPlayer[];
  out: TransferPlayer[];
  hits: number;
}

export interface ManagerSquadPick {
  player_id: number;
  name: string;
  position: Position | null;
  club: string | null;
  price: number | null;
  xp: number | null;
  p_play: number | null;
  is_captain: boolean;
  is_vice_captain: boolean;
  multiplier: number;
  purchase_price: number | null;
  purchase_price_known: boolean;
  sell_price?: number;
}

export interface ManagerTeam {
  entry_id: number;
  manager_name: string;
  team_name: string;
  season: number;
  gw: number;
  overall_rank: number | null;
  overall_points: number | null;
  bank: number;
  team_value: number;
  free_transfers: number;
  free_transfers_is_estimated: boolean;
  active_chip: string | null;
  chips_used: string[];
  squad: ManagerSquadPick[];
  unmatched_player_ids: number[];
  note: string | null;
}

export interface TransferPlan {
  moves: TransferMove[];
  hits: number;
  net_gain: number;
  status: string;
  reasoning: string[];
}

export interface QualityEvent {
  event_id: number;
  created_at: string;
  stage: string | null;
  check_name: string | null;
  severity: "info" | "warn" | "fail" | string;
  detail: Record<string, unknown> | null;
}

export interface DecisionMetrics {
  certainty?: number | null;
  uncertainty_points?: number | null;
  upside_points?: number | null;
  downside_points?: number | null;
  range_points?: number | null;
  start_security?: number | null;
  sixty_security?: number | null;
  minutes_stability?: number | null;
  value_per_million?: number | null;
  floor_adjusted_xp?: number | null;
  ceiling_adjusted_xp?: number | null;
  ownership_adjusted_upside?: number | null;
  ownership_adjusted_downside?: number | null;
  quartile_range_points?: number | null;
  start_gap?: number | null;
  bench_risk?: number | null;
  floor_value_per_million?: number | null;
  ceiling_value_per_million?: number | null;
  ownership?: number | null;
}

export interface DashboardPlayer extends PlayerPrediction, DecisionMetrics {}

export interface DashboardOverview {
  season: number;
  gw: number;
  target_gw: number | null;
  model_version: string;
  summary: {
    players: number;
    mean_xp: number;
    median_xp: number;
    high_upside_players: number;
    low_play_players: number;
    low_certainty_players: number;
    quality_events: number;
  };
  top_players: DashboardPlayer[];
  best_value: DashboardPlayer[];
  differentials: DashboardPlayer[];
  upside: DashboardPlayer[];
  team_strength: TeamStrength[];
  quality: Record<string, number>;
  market: Record<string, unknown>[];
}

export interface CompareResponse {
  season: number;
  gw: number;
  players: DashboardPlayer[];
  missing: number[];
}

export interface FixtureOutlook {
  season: number;
  start_gw: number;
  horizon: number;
  fixtures: {
    gw: number;
    team_id: number;
    team: string;
    opponent_id: number;
    opponent: string;
    home: boolean;
    difficulty: number;
  }[];
  teams: {
    team_id: number;
    team: string;
    avg_difficulty: number;
    best_fixture: number;
    worst_fixture: number;
    fixtures: number;
  }[];
}

export interface MarketWatch {
  players: {
    player_id: number;
    /**
     * Player identity, resolved server-side. The endpoint joins the
     * snapshots table to `players` and `teams` so these are always
     * present for a player who exists in the current season, matching
     * the shape every other dashboard endpoint returns.
     */
    web_name: string | null;
    position: Position | null;
    short_name: string | null;
    selected_by_percent: number | null;
    transfers_in_event: number | null;
    transfers_out_event: number | null;
    net_transfers: number;
    transfer_ratio: number | null;
    now_cost: number | null;
    form: number | null;
    chance_of_playing_next_round: number | null;
    status: string | null;
    news: string | null;
  }[];
  note?: string;
}

export interface ModelDashboard {
  evaluation_runs: EvaluationRun[];
  quality_events: QualityEvent[];
  version: string;
}