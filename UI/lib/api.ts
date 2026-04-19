/**
 * In the browser, default to same-origin `/api` (Next.js `app/api/[...path]` proxies to FastAPI).
 * Set NEXT_PUBLIC_API_URL only if you need to call the API from a different host without the proxy.
 */
function resolveApiBase(): string {
  const fromEnv = (process.env.NEXT_PUBLIC_API_URL ?? "").trim().replace(/\/$/, "");
  if (typeof window !== "undefined") {
    return fromEnv || "";
  }
  return fromEnv || "http://127.0.0.1:8000";
}

const API_BASE = resolveApiBase();

/**
 * Same base as `apiFetch`. When `NEXT_PUBLIC_API_URL` is empty in the browser, this is `""`
 * (same-origin). Chat SSE uses `/api/chat/stream` → `app/api/chat/stream/route.ts` proxies to
 * FastAPI via this route (no buffering). If NEXT_PUBLIC_API_URL is set, streaming hits that host directly.
 */
export function publicApiBase(): string {
  return API_BASE;
}

export type ChatMode = "corpus" | "web";

export type CorpusGenerationModel = "ollama" | "gpt-4o-mini" | "gpt-5-nano";

export type ChatSource = {
  source: string;
  title?: string;
  text?: string;
  year?: string;
  authors?: string[];
  score?: number;
  is_web?: boolean;
  chunk_index?: number;
};

export type ChatResponsePayload = {
  answer: string;
  conversation_id?: string;
  message_id?: string;
  intent?: string;
  chunks_count?: number;
  memory_info?: Record<string, unknown>;
  sources?: ChatSource[];
};

function getAuthHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = localStorage.getItem("auth_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `Request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

/** Non-streaming fallback (rare); uses same FastAPI routes as the Streamlit stack. */
export async function sendChatMessage(
  query: string,
  mode: ChatMode,
  conversationId?: string,
  corpusGeneration?: CorpusGenerationModel
): Promise<ChatResponsePayload> {
  const body: Record<string, unknown> = {
    query,
    stream: false,
    conversation_id: conversationId ?? null,
    mode,
  };
  if (mode === "corpus" && corpusGeneration && corpusGeneration !== "ollama") {
    body.corpus_generation_model = corpusGeneration;
  }
  return apiFetch<ChatResponsePayload>("/api/chat", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export type StreamEvent =
  | { type: "conversation_id"; data: string }
  | { type: "token"; data: string; replace_full?: boolean }
  | { type: "status"; data: string }
  | { type: "sources"; data: ChatSource[] }
  | { type: "memory_info"; data: Record<string, unknown> }
  | { type: "done"; data: { message_id: string } }
  | { type: "error"; data: string };

function parseSseBlock(block: string): StreamEvent | null {
  const line = block.trim();
  if (!line.startsWith("data:")) return null;
  const json = line.slice(5).trim();
  if (json === "[DONE]") return null;
  try {
    return JSON.parse(json) as StreamEvent;
  } catch {
    return null;
  }
}

export type StreamChatOptions = {
  conversationId?: string;
  mode: ChatMode;
  corpusGeneration?: CorpusGenerationModel;
  /** When aborted, the generator ends after the fetch throws (caller handles partial UI). */
  signal?: AbortSignal;
};

export async function* streamChatMessage(
  query: string,
  options: StreamChatOptions
): AsyncGenerator<StreamEvent> {
  const { conversationId, mode, corpusGeneration, signal } = options;
  const body: Record<string, unknown> = {
    query,
    stream: true,
    conversation_id: conversationId ?? null,
    mode,
  };
  if (mode === "corpus" && corpusGeneration && corpusGeneration !== "ollama") {
    body.corpus_generation_model = corpusGeneration;
  }

  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
    },
    body: JSON.stringify(body),
    cache: "no-store",
    signal,
  });

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `Stream failed: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      if (buffer.trim()) {
        for (const block of buffer.split("\n\n")) {
          const ev = parseSseBlock(block);
          if (ev) {
            yield ev;
            if (ev.type === "error" || ev.type === "done") return;
          }
        }
      }
      return;
    }
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const block of parts) {
      const ev = parseSseBlock(block);
      if (ev) {
        yield ev;
        if (ev.type === "error" || ev.type === "done") return;
      }
    }
  }
}
