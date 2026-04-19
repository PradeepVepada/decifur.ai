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
 * Proxies multipart POST /api/papers/upload → FastAPI.
 * Rebuilds FormData so Node's fetch sends a correct multipart body (forwarding req.formData() can fail).
 */
export async function POST(req: NextRequest) {
  try {
    const incoming = await req.formData();
    const raw = incoming.get("file");
    if (!raw || typeof raw === "string") {
      return new Response(JSON.stringify({ detail: "Expected multipart field 'file' with a PDF." }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const blob = raw as Blob;
    const filename =
      typeof File !== "undefined" && raw instanceof File && raw.name
        ? raw.name
        : "upload.pdf";

    const out = new FormData();
    out.append("file", blob, filename);

    const backend = backendBase();
    let upstream: Response;
    try {
      upstream = await fetch(`${backend}/api/papers/upload`, {
        method: "POST",
        body: out,
        cache: "no-store",
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return new Response(
        JSON.stringify({
          detail: `Cannot reach FastAPI at ${backend}. From NovaAI_v2 run: python api.py (${msg})`,
        }),
        { status: 502, headers: { "Content-Type": "application/json; charset=utf-8" } }
      );
    }

    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "application/json; charset=utf-8",
      },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return new Response(JSON.stringify({ detail: msg }), {
      status: 500,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
}
