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
 * Explicit proxy for `/api/*` routes that do not have their own `route.ts`.
 * Avoids Next.js rewrite "connection failed" surfacing as an opaque 500; returns 502 JSON instead.
 */
async function proxyToFastAPI(req: NextRequest, pathSegments: string[]): Promise<Response> {
  const backend = backendBase();
  const url = `${backend}/api/${pathSegments.join("/")}${req.nextUrl.search}`;

  const headers = new Headers();
  const auth = req.headers.get("authorization");
  if (auth) headers.set("Authorization", auth);
  const contentType = req.headers.get("content-type");
  if (contentType) headers.set("Content-Type", contentType);
  const accept = req.headers.get("accept");
  if (accept) headers.set("Accept", accept);

  const init: RequestInit = {
    method: req.method,
    headers,
    cache: "no-store",
    signal: req.signal,
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    const buf = await req.arrayBuffer();
    if (buf.byteLength) init.body = buf;
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return new Response(
      JSON.stringify({
        detail: `Cannot reach FastAPI at ${backend}. From the NovaAI_v2 repo root run: python api.py (${msg})`,
      }),
      { status: 502, headers: { "Content-Type": "application/json; charset=utf-8" } }
    );
  }

  const outHeaders = new Headers();
  for (const name of ["content-type", "cache-control"]) {
    const v = upstream.headers.get(name);
    if (v) outHeaders.set(name, v);
  }

  return new Response(upstream.body, { status: upstream.status, headers: outHeaders });
}

type RouteCtx = { params: Promise<{ path: string[] }> };

async function handle(req: NextRequest, ctx: RouteCtx): Promise<Response> {
  const { path } = await ctx.params;
  if (!path?.length) {
    return new Response(JSON.stringify({ detail: "Missing API path" }), {
      status: 400,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
  return proxyToFastAPI(req, path);
}

export function GET(req: NextRequest, ctx: RouteCtx) {
  return handle(req, ctx);
}

export function HEAD(req: NextRequest, ctx: RouteCtx) {
  return handle(req, ctx);
}

export function POST(req: NextRequest, ctx: RouteCtx) {
  return handle(req, ctx);
}

export function PATCH(req: NextRequest, ctx: RouteCtx) {
  return handle(req, ctx);
}

export function DELETE(req: NextRequest, ctx: RouteCtx) {
  return handle(req, ctx);
}
