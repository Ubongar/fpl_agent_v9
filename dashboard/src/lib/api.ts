import type {
  AttributionChange,
  BacktestResult,
  CaptainCandidate,
  EvaluationRun,
  PlayerDetail,
  PredictionsPage,
  SetPieceHolder,
  SquadSolution,
  TeamStrength,
} from "./types";

const SERVER_BASE = process.env.FPL_API_URL ?? "http://127.0.0.1:8000";
const BASE = typeof window === "undefined" ? SERVER_BASE : "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string): Promise<T> {
  let res: Response;
  try {
    const headers: Record<string, string> = {};
    if (typeof window === "undefined" && process.env.FPL_AGENT_API_KEY) headers["X-API-Key"] = process.env.FPL_AGENT_API_KEY;
    res = await fetch(`${BASE}${path}`, { cache: "no-store", headers });
  } catch {
    throw new ApiError(
      `Could not reach the API at ${BASE}. Start it with: uvicorn fplagent.api.main:app --port 8000`,
      0,
    );
  }
  if (!res.ok) {
    let detail = `${path} returned ${res.status}`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = body.detail;
    } catch {}
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (typeof window === "undefined" && process.env.FPL_AGENT_API_KEY) headers["X-API-Key"] = process.env.FPL_AGENT_API_KEY;
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = `${path} returned ${res.status}`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = body.detail;
    } catch {}
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export interface PredictionQuery {
  gw?: number;
  position?: string;
  team?: string;
  search?: string;
  minPrice?: number;
  maxPrice?: number;
  sort?: string;
  desc?: boolean;
  offset?: number;
  limit?: number;
}

export function predictions(q: PredictionQuery = {}): Promise<PredictionsPage> {
  const p = new URLSearchParams();
  if (q.gw != null) p.set("gw", String(q.gw));
  if (q.position) p.set("position", q.position);
  if (q.team) p.set("team", q.team);
  if (q.search) p.set("search", q.search);
  if (q.minPrice != null) p.set("min_price", String(q.minPrice));
  if (q.maxPrice != null) p.set("max_price", String(q.maxPrice));
  p.set("sort", q.sort ?? "xp_mean");
  p.set("desc", String(q.desc ?? true));
  p.set("offset", String(q.offset ?? 0));
  p.set("limit", String(q.limit ?? 50));
  return get<PredictionsPage>(`/predictions?${p.toString()}`);
}

export const playerDetail = (id: number) => get<PlayerDetail>(`/predictions/${id}`);

export const currentGameweek = () =>
  get<{ season: number; target_gw: number; deadline: string | null }>(
    "/gameweeks/current",
  );

export const teams = () =>
  get<{ teams: { team_id: number; name: string; short_name: string }[] }>("/teams");

export const teamStrength = () =>
  get<{ model_version: string; teams: TeamStrength[] }>("/team-strength");

export const setPieces = (duty = "penalties") =>
  get<{ duty: string; players: SetPieceHolder[] }>(`/set-pieces?duty=${duty}`);

export const evaluationHistory = () =>
  get<{ runs: EvaluationRun[] }>("/evaluation/history");

export const evaluationLatest = (gwFrom = 2, gwTo?: number) =>
  get<BacktestResult>(
    `/evaluation/latest?gw_from=${gwFrom}${gwTo != null ? `&gw_to=${gwTo}` : ""}`,
  );

export const changes = (gwNow: number, gwPrev: number) =>
  get<{ changes: AttributionChange[] }>(
    `/changes?gw_now=${gwNow}&gw_prev=${gwPrev}`,
  );

export const optimiseSquad = (body: {
  budget: number;
  risk_lambda: number;
  locked_in?: number[];
  excluded?: number[];
}) => post<SquadSolution>("/optimise/squad", body);

export const rankCaptains = (body: { risk_lambda: number; top_n: number }) =>
  post<{ candidates: CaptainCandidate[] }>("/optimise/captaincy", body);

export const dataQuality = (limit = 50) =>
  get<{ events: import("./types").QualityEvent[] }>(`/quality?limit=${limit}`);

export const dashboardOverview = (gw?: number) =>
  get<import("./types").DashboardOverview>(`/dashboard/overview${gw != null ? `?gw=${gw}` : ""}`);

export const comparePlayers = (player_ids: number[], gw?: number) =>
  post<import("./types").CompareResponse>("/players/compare", {
    player_ids,
    ...(gw != null ? { gw } : {}),
  });

export const fixtureOutlook = (startGw?: number, horizon = 6) =>
  get<import("./types").FixtureOutlook>(
    `/fixtures/outlook?horizon=${horizon}${startGw != null ? `&start_gw=${startGw}` : ""}`,
  );

export const marketWatch = (limit = 40) =>
  get<import("./types").MarketWatch>(`/market/watch?limit=${limit}`);

export const modelDashboard = () =>
  get<import("./types").ModelDashboard>("/model/dashboard");

export const managerTeam = (entryId: number, gw?: number) =>
  get<import("./types").ManagerTeam>(`/manager/${entryId}/team${gw != null ? `?gw=${gw}` : ""}`);

export const pipelineStatus = () =>
  get<{ status: string; message: string }>("/pipeline/status");

export const newsBriefing = () =>
  get<{ briefing: string }>("/news/briefing");