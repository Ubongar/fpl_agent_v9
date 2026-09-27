import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const API_BASE = process.env.FPL_API_URL ?? "http://127.0.0.1:8000";
const API_KEY = process.env.FPL_AGENT_API_KEY ?? "";

async function forward(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const suffix = path.map(segment => encodeURIComponent(segment)).join("/");
  const target = `${API_BASE}/${suffix}${request.nextUrl.search}`;

  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("content-length");
  if (API_KEY) headers.set("X-API-Key", API_KEY);

  const init: RequestInit = { method: request.method, headers, cache: "no-store" };
  if (request.method !== "GET" && request.method !== "HEAD") init.body = await request.arrayBuffer();

  try {
    const response = await fetch(target, init);
    const outHeaders = new Headers();
    const contentType = response.headers.get("content-type");
    if (contentType) outHeaders.set("content-type", contentType);
    return new Response(response.body, { status: response.status, headers: outHeaders });
  } catch {
    return Response.json({ detail: `Backend unavailable at ${API_BASE}` }, { status: 502 });
  }
}

export const GET = forward;
export const POST = forward;
export const PUT = forward;
export const PATCH = forward;
export const DELETE = forward;
