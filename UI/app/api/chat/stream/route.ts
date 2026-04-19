import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function backendBase(): string {
  return (
    process.env.API_PROXY_TARGET ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://127.0.0.1:8000"
  )
    .trim()
    .replace(/\/$/, "");
}

/**
 * Proxies POST /api/chat/stream to FastAPI with a streaming body (SSE; must not go through a generic proxy that buffers).
 */
export async function POST(req: NextRequest) {
  const backend = backendBase();
  let body: string;
  try {
    body = await req.text();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return new Response(JSON.stringify({ detail: `Failed to read request body: ${msg}` }), {
      status: 400,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
  const auth = req.headers.get("authorization");

  let upstream: Response;
  try {
    upstream = await fetch(`${backend}/api/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(auth ? { Authorization: auth } : {}),
      },
      body,
      cache: "no-store",
      signal: req.signal,
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return new Response(
      JSON.stringify({
        detail: `Cannot reach FastAPI at ${backend}. From the NovaAI_v2 repo root run: python api.py (${msg})`,
      }),
      { status: 502, headers: { "Content-Type": "application/json; charset=utf-8" } }
    );
  }

  if (!upstream.ok) {
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "text/plain; charset=utf-8",
      },
    });
  }

  const headers = new Headers();
  const ct = upstream.headers.get("content-type");
  if (ct) headers.set("Content-Type", ct);
  headers.set("Cache-Control", "no-cache, no-transform");
  headers.set("X-Accel-Buffering", "no");

  return new Response(upstream.body, {
    status: upstream.status,
    headers,
  });
}
