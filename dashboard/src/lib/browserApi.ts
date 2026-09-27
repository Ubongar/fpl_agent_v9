"use client";

/**
 * Browser-side counterpart to lib/api.ts.
 *
 * Server Components use api.ts, which talks straight to FPL_API_URL. Client
 * Components can't see that env var (it's a server-only value) and don't
 * need to: they go through the same-origin `/api/*` Next.js route handler, so this file
 * never needs to know where the FastAPI service actually lives.
 */

export class BrowserApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "BrowserApiError";
  }
}

async function handle<T>(res: Response, path: string): Promise<T> {
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body?.detail ? `: ${body.detail}` : "";
    } catch {
      /* body wasn't JSON, ignore */
    }
    throw new BrowserApiError(`${path} returned ${res.status}${detail}`, res.status);
  }
  return (await res.json()) as T;
}

export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`, { cache: "no-store" });
  return handle<T>(res, path);
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  return handle<T>(res, path);
}
