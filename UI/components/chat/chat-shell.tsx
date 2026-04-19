"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowUp,
  Bot,
  BrainCircuit,
  ChevronDown,
  Copy,
  FileText,
  Globe,
  LogIn,
  Menu,
  MessageSquarePlus,
  Moon,
  PanelLeft,
  Settings,
  Square,
  Sparkles,
  Sun,
  Trash2,
  Upload,
} from "lucide-react";
import {
  apiFetch,
  publicApiBase,
  streamChatMessage,
  type ChatMode,
  type ChatSource,
  type CorpusGenerationModel,
  type StreamEvent,
} from "@/lib/api";
import { AssistantMessageBody } from "@/components/chat/assistant-message-body";

const CORPUS_MODEL_OPTIONS: { value: CorpusGenerationModel; label: string }[] = [
  { value: "gpt-5-nano", label: "GPT-5 nano" },
  { value: "gpt-4o-mini", label: "GPT-4o mini" },
  { value: "ollama", label: "Syntropy (local)" },
];

function corpusModelLabel(value: CorpusGenerationModel): string {
  return CORPUS_MODEL_OPTIONS.find((o) => o.value === value)?.label ?? value;
}

/** 1-based [S#] indices appearing in the assistant reply */
function extractCitationIndicesFromAnswer(text: string | undefined): number[] {
  if (!text) return [];
  const re = /\[S(\d+)\]/gi;
  const seen = new Set<number>();
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const n = parseInt(m[1] ?? "0", 10);
    if (!Number.isNaN(n) && n > 0) seen.add(n);
  }
  return [...seen].sort((a, b) => a - b);
}

type DisplaySourceRow = {
  citeTags: number[];
  title: string;
  fileLabel: string;
  copyText: string;
};

/** Stable key so multiple chunks from the same PDF collapse in the Sources list. */
function sourcePaperKey(s: ChatSource, idx1Based: number): string {
  return ((s.source || "").trim() || (s.title || "").trim() || `row-${idx1Based}`).toLowerCase();
}

/** When there are no inline [S#], show one row per PDF to reduce noise. */
function dedupeRetrievalRowsByFile(sources: ChatSource[]): DisplaySourceRow[] {
  const byFile = new Map<string, { citeTags: number[]; s: ChatSource }>();
  for (let i = 0; i < sources.length; i++) {
    const idx = i + 1;
    const s = sources[i]!;
    const key = sourcePaperKey(s, idx);
    const ex = byFile.get(key);
    if (!ex) {
      byFile.set(key, { citeTags: [idx], s });
    } else if (!ex.citeTags.includes(idx)) {
      ex.citeTags.push(idx);
    }
  }
  for (const v of byFile.values()) {
    v.citeTags.sort((a, b) => a - b);
  }
  return [...byFile.values()].map(({ citeTags, s }) => ({
    citeTags,
    title: s.title || s.source || "Source",
    fileLabel: s.source || "",
    copyText: `${s.title || s.source}\n\n${s.text || ""}`,
  }));
}

/**
 * With inline [S#]: one row per paper, merging tags (e.g. [S1], [S2], [S3] for the same PDF).
 * Without citations: deduplicated retrieval (one row per file).
 */
function buildAssistantSourceRows(sources: ChatSource[], answerText: string | undefined): DisplaySourceRow[] {
  if (!sources.length) return [];
  const cited = extractCitationIndicesFromAnswer(answerText);
  if (cited.length > 0) {
    const pairs = cited
      .filter((n) => n >= 1 && n <= sources.length)
      .map((n) => ({ idx: n, s: sources[n - 1]! }));
    if (pairs.length === 0) {
      return dedupeRetrievalRowsByFile(sources);
    }
    const byFile = new Map<string, { citeTags: number[]; s: ChatSource; textParts: string[] }>();
    for (const { idx, s } of pairs) {
      const key = sourcePaperKey(s, idx);
      const chunkCopy = `${s.title || s.source || "Source"}\n\n${s.text || ""}`.trim();
      const ex = byFile.get(key);
      if (!ex) {
        byFile.set(key, { citeTags: [idx], s, textParts: [chunkCopy] });
      } else {
        if (!ex.citeTags.includes(idx)) ex.citeTags.push(idx);
        if (chunkCopy && !ex.textParts.includes(chunkCopy)) ex.textParts.push(chunkCopy);
      }
    }
    for (const v of byFile.values()) {
      v.citeTags.sort((a, b) => a - b);
    }
    return [...byFile.values()]
      .sort((a, b) => (a.citeTags[0] ?? 0) - (b.citeTags[0] ?? 0))
      .map(({ citeTags, s, textParts }) => ({
        citeTags,
        title: s.title || s.source || "Source",
        fileLabel: s.source || "",
        copyText: textParts.join("\n\n---\n\n"),
      }));
  }
  return dedupeRetrievalRowsByFile(sources);
}

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  memoryInfo?: Record<string, unknown>;
  isStreaming?: boolean;
};

type RecentChat = {
  id: string;
  title: string;
};

const DEMO_QUESTIONS = [
  "What are the key findings on actin dynamics?",
  "Explain chemotaxis in simpler words.",
  "Which papers discuss PI3K signaling?",
];

function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function shortLabel(text: string | undefined) {
  if (!text) return "Untitled";
  if (text.length <= 42) return text;
  return `${text.slice(0, 39)}...`;
}

export function ChatShell() {
  const router = useRouter();
  const [mode, setMode] = useState<ChatMode>("corpus");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSources, setShowSources] = useState(true);
  const [renderMath, setRenderMath] = useState(true);
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [userName, setUserName] = useState<string>("");
  const [uploadLabel, setUploadLabel] = useState("Upload PDF");
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [ollamaModelLabel, setOllamaModelLabel] = useState("Syntropy");
  const [corpusGeneration, setCorpusGeneration] = useState<CorpusGenerationModel>("gpt-5-nano");
  const [recentChats, setRecentChats] = useState<RecentChat[]>([]);
  const [modelPickerOpen, setModelPickerOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const addPaperInputRef = useRef<HTMLInputElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const modelPickerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const token = localStorage.getItem("auth_token");
    if (!token) {
      router.replace("/signin");
      return;
    }
    try {
      const user = JSON.parse(localStorage.getItem("auth_user") ?? "{}");
      setUserName(user.name ?? user.email ?? "");
    } catch {
      // ignore
    }
  }, [router]);

  useEffect(() => {
    const savedTheme = (localStorage.getItem("ui_theme") as "dark" | "light" | null) ?? "dark";
    setTheme(savedTheme);
    setRenderMath(localStorage.getItem("ui_render_math") !== "0");
  }, []);

  useEffect(() => {
    if (mode === "web") setModelPickerOpen(false);
  }, [mode]);

  useEffect(() => {
    if (!modelPickerOpen) return;
    const onPointerDown = (e: MouseEvent | PointerEvent) => {
      const el = modelPickerRef.current;
      if (el && !el.contains(e.target as Node)) setModelPickerOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setModelPickerOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [modelPickerOpen]);

  useEffect(() => {
    void refreshRecentChats();
    void (async () => {
      try {
        const cfg = await apiFetch<{ ollama_model?: string }>("/api/ui/config");
        if (cfg.ollama_model) setOllamaModelLabel(cfg.ollama_model);
      } catch {
        /* API down */
      }
    })();
  }, []);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const isNearBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight < 120;
    if (isNearBottom) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const onScroll = () => {
      const distFromBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight;
      setShowScrollBtn(distFromBottom > 200);
    };
    container.addEventListener("scroll", onScroll);
    return () => container.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [query]);

  async function refreshRecentChats() {
    try {
      const grouped = await apiFetch<Record<string, Array<{ conversation_id: string; title?: string }>>>(
        "/api/conversations?user_id=default&limit=50"
      );
      const ordered = ["today", "yesterday", "this_week", "older"]
        .flatMap((k) => grouped[k] ?? [])
        .map((c) => ({
          id: c.conversation_id,
          title: c.title?.trim() || "Untitled chat",
        }));
      setRecentChats(ordered);
    } catch {
      // keep UI responsive if recents fail
    }
  }

  async function loadConversation(conversationIdToLoad: string) {
    try {
      const res = await apiFetch<{
        messages: Array<{
          role: "user" | "assistant";
          content: string;
          memory_info?: Record<string, unknown>;
        }>;
      }>(`/api/conversations/${conversationIdToLoad}`);
      const nextMessages: ChatMessage[] = res.messages.map((m) => ({
        role: m.role,
        content: m.content,
        memoryInfo: m.memory_info ?? {},
      }));
      setConversationId(conversationIdToLoad);
      setMessages(nextMessages);
    } catch {
      // ignore load failures
    }
  }

  const handleStopGeneration = useCallback(() => {
    streamAbortRef.current?.abort();
  }, []);

  const handleSend = useCallback(
    async (overrideQuery?: string) => {
      const text = (overrideQuery ?? query).trim();
      if (!text || loading) return;

      streamAbortRef.current?.abort();
      const ac = new AbortController();
      streamAbortRef.current = ac;

      setLoading(true);
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setQuery("");

      setMessages((prev) => [...prev, { role: "assistant", content: "", isStreaming: true }]);

      try {
        let streamedContent = "";
        let streamedSources: ChatSource[] = [];
        let streamedMemory: Record<string, unknown> = {};

        for await (const event of streamChatMessage(text, {
          conversationId,
          mode,
          corpusGeneration: mode === "corpus" ? corpusGeneration : undefined,
          signal: ac.signal,
        })) {
          const e = event as StreamEvent;
          if (e.type === "conversation_id") {
            setConversationId(e.data);
          } else if (e.type === "status") {
            /* SSE heartbeat from RAG prep — keeps connection alive; no UI change required */
          } else if (e.type === "token" && typeof e.data === "string") {
            const rf = Boolean((e as { replace_full?: boolean }).replace_full);
            if (rf) streamedContent = e.data;
            else streamedContent += e.data;
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = { ...last, content: streamedContent, isStreaming: true };
              }
              return next;
            });
          } else if (e.type === "sources") {
            streamedSources = e.data as ChatSource[];
          } else if (e.type === "memory_info") {
            streamedMemory = e.data as Record<string, unknown>;
          } else if (e.type === "done") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  content: streamedContent,
                  sources: streamedSources,
                  memoryInfo: streamedMemory,
                  isStreaming: false,
                };
              }
              return next;
            });
            void refreshRecentChats();
            streamAbortRef.current = null;
          } else if (e.type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  content: `Error: ${e.data}`,
                  isStreaming: false,
                };
              }
              return next;
            });
            streamAbortRef.current = null;
          }
        }
      } catch (err) {
        const name = (err as Error)?.name;
        if (name === "AbortError") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant" && last.isStreaming) {
              const base = (last.content || "").trim();
              next[next.length - 1] = {
                ...last,
                content: base ? `${base}\n\n(Generation stopped.)` : "(Generation stopped.)",
                isStreaming: false,
              };
            }
            return next;
          });
        } else {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              next[next.length - 1] = {
                ...last,
                content: `Error: ${(err as Error).message}`,
                isStreaming: false,
              };
            }
            return next;
          });
        }
      } finally {
        streamAbortRef.current = null;
        setLoading(false);
      }
    },
    [query, loading, mode, conversationId, corpusGeneration]
  );

  async function handleClearMemory() {
    try {
      await apiFetch("/api/memory/clear", { method: "POST", body: JSON.stringify({}) });
    } catch {
      /* optional */
    }
    setMessages([]);
    setConversationId(undefined);
  }

  async function handleNewChat() {
    try {
      await apiFetch("/api/memory/clear", { method: "POST", body: JSON.stringify({}) });
    } catch {
      /* optional */
    }
    try {
      const c = await apiFetch<{ conversation_id: string }>("/api/conversations", {
        method: "POST",
        body: JSON.stringify({ user_id: "default" }),
      });
      setConversationId(c.conversation_id);
    } catch {
      setConversationId(undefined);
    }
    setMessages([]);
    void refreshRecentChats();
  }

  async function handleDeleteChatHistory() {
    const ids = recentChats.map((c) => c.id);
    if (!ids.length) return;
    await Promise.allSettled(ids.map((id) => apiFetch(`/api/conversations/${id}`, { method: "DELETE" })));
    setRecentChats([]);
    setMessages([]);
    setConversationId(undefined);
  }

  function handleSignOut() {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_user");
    router.replace("/signin");
  }

  async function handleUpload(file: File | null) {
    if (!file) return;
    setUploadLabel("Uploading...");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${publicApiBase()}/api/papers/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const t = await res.text();
        let message = t.trim() || `HTTP ${res.status}`;
        try {
          const j = JSON.parse(t) as { detail?: unknown };
          if (typeof j?.detail === "string" && j.detail.trim()) message = j.detail.trim();
        } catch {
          /* not JSON */
        }
        throw new Error(message);
      }
      setUploadLabel("Uploaded!");
      try {
        await apiFetch("/api/ingest/status");
      } catch {
        /* optional */
      }
    } catch (err) {
      setUploadLabel("Upload failed");
      console.error(err);
    } finally {
      setTimeout(() => setUploadLabel("Upload PDF"), 3000);
    }
  }

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("ui_theme", next);
  }

  function scrollToBottom() {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (loading) {
        handleStopGeneration();
        return;
      }
      void handleSend();
    }
  }

  async function handleCopy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // ignore clipboard failures
    }
  }

  const dk = theme === "dark";
  const webEnabled = mode === "web";

  const pageTheme = dk
    ? "bg-[radial-gradient(circle_at_top,#2a2147_0%,#120f1d_36%,#09090b_100%)] text-zinc-100"
    : "bg-[radial-gradient(circle_at_top,#efe8ff_0%,#f7f5fb_30%,#ffffff_100%)] text-zinc-900";
  const panelTheme = dk
    ? "bg-white/5 border-white/10 backdrop-blur-xl"
    : "bg-white/80 border-zinc-200/80 backdrop-blur-xl";
  const softTheme = dk ? "bg-white/5 hover:bg-white/10" : "bg-zinc-100 hover:bg-zinc-200/80";

  return (
    <div className={cn("h-screen w-full overflow-hidden transition-colors duration-300", pageTheme)}>
      <div className="mx-auto flex h-full max-w-[1600px] gap-4 p-4 md:p-6">
        <AnimatePresence initial={false}>
          {sidebarOpen && (
            <motion.aside
              initial={{ x: -16, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -16, opacity: 0 }}
              className={cn("hidden w-[290px] shrink-0 rounded-[28px] border md:flex md:flex-col", panelTheme)}
            >
              <div className="flex items-center justify-between border-b border-inherit p-4">
                  <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-violet-500/90 text-white shadow-lg shadow-violet-500/20">
                    <BrainCircuit className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-lg font-semibold tracking-tight">Research</p>
                    <p className={cn("text-xs", dk ? "text-zinc-400" : "text-zinc-500")}>Corpus chat</p>
                  </div>
                </div>
                <button
                  className={cn("rounded-2xl p-2", softTheme)}
                  onClick={() => setSidebarOpen(false)}
                  type="button"
                >
                  <PanelLeft className="h-4 w-4" />
                </button>
              </div>

              <div className="p-4">
                <button
                  type="button"
                  onClick={() => void handleNewChat()}
                  className={cn(
                    "flex h-12 w-full items-center justify-start gap-3 rounded-2xl border border-white/10 px-4 text-base",
                    softTheme
                  )}
                >
                  <MessageSquarePlus className="h-4 w-4" />
                  New chat
                </button>
                <button
                  type="button"
                  onClick={() => addPaperInputRef.current?.click()}
                  className={cn(
                    "mt-2 flex h-12 w-full items-center justify-start gap-3 rounded-2xl border border-white/10 px-4 text-base",
                    softTheme
                  )}
                >
                  <Upload className="h-4 w-4" />
                  Add paper
                </button>
                <input
                  ref={addPaperInputRef}
                  type="file"
                  accept=".pdf"
                  hidden
                  onChange={(e) => handleUpload(e.target.files?.[0] ?? null)}
                  className="hidden"
                />
              </div>

              <div className="px-4 pb-2">
                <p className={cn("px-2 text-xs font-medium uppercase tracking-[0.18em]", dk ? "text-zinc-400" : "text-zinc-500")}>
                  Recents
                </p>
              </div>

              <div className="flex-1 overflow-y-auto px-3 pb-4">
                <div className="space-y-2">
                  {(recentChats.length
                    ? recentChats.map((c) => ({ key: c.id, label: c.title, conversationId: c.id as string | undefined }))
                    : DEMO_QUESTIONS.map((q) => ({ key: q, label: q, conversationId: undefined }))).map((chat) => (
                    <button
                      key={chat.key}
                      className={cn("w-full rounded-2xl border border-transparent px-3 py-3 text-left text-sm transition", softTheme)}
                      onClick={() => {
                        if (chat.conversationId) {
                          void loadConversation(chat.conversationId);
                          return;
                        }
                        setQuery(chat.label);
                      }}
                      type="button"
                    >
                      <p className="truncate">{shortLabel(chat.label)}</p>
                    </button>
                  ))}
                </div>
              </div>

              <div className="border-t border-inherit p-4">
                <p className={cn("mb-2 text-xs", dk ? "text-zinc-400" : "text-zinc-500")}>
                  {userName ? `Signed in as ${userName}` : "Session active"}
                </p>
                <button
                  type="button"
                  onClick={handleSignOut}
                  className={cn("flex h-11 w-full items-center justify-start gap-3 rounded-2xl px-3 text-sm", softTheme)}
                >
                  <LogIn className="h-4 w-4" />
                  Sign out
                </button>
                <button
                  type="button"
                  onClick={() => void handleDeleteChatHistory()}
                  className={cn("mt-2 flex h-10 w-full items-center justify-start gap-3 rounded-2xl px-3 text-sm", softTheme)}
                >
                  <Trash2 className="h-4 w-4" />
                  Delete chat history
                </button>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        <main className={cn("flex min-h-0 flex-1 flex-col rounded-[32px] border", panelTheme)}>
          <header className="flex items-center justify-between border-b border-inherit px-4 py-4 md:px-6">
            <div className="flex items-center gap-2">
              {!sidebarOpen && (
                <button
                  type="button"
                  className={cn("rounded-2xl p-2", softTheme)}
                  onClick={() => setSidebarOpen(true)}
                >
                  <Menu className="h-4 w-4" />
                </button>
              )}
            </div>

            <details className="relative">
              <summary className={cn("flex cursor-pointer list-none items-center gap-2 rounded-2xl px-3 py-2 text-sm", softTheme)}>
                <Settings className="h-4 w-4" />
                Settings
              </summary>
              <div className={cn("absolute right-0 z-20 mt-2 w-72 rounded-2xl border p-3 shadow-xl", panelTheme)}>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wider">Appearance</p>
                <button
                  type="button"
                  onClick={toggleTheme}
                  className={cn("mb-3 flex w-full items-center justify-between rounded-xl px-2 py-2 text-sm", softTheme)}
                >
                  <span className="flex items-center gap-2">
                    {dk ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                    Theme
                  </span>
                  <span className="text-xs">{dk ? "Dark" : "Light"}</span>
                </button>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wider">Response details</p>
                <label className={cn("flex items-center justify-between rounded-xl px-2 py-2 text-sm", softTheme)}>
                  Show sources
                  <input type="checkbox" checked={showSources} onChange={(e) => setShowSources(e.target.checked)} />
                </label>
                <label className={cn("mt-1 flex items-center justify-between rounded-xl px-2 py-2 text-sm", softTheme)}>
                  Render math (KaTeX)
                  <input
                    type="checkbox"
                    checked={renderMath}
                    onChange={(e) => {
                      const next = e.target.checked;
                      setRenderMath(next);
                      localStorage.setItem("ui_render_math", next ? "1" : "0");
                    }}
                  />
                </label>
              </div>
            </details>
          </header>

          <section className="flex min-h-0 flex-1 flex-col justify-between p-4 md:p-6 lg:p-8">
            <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col overflow-hidden">
              {messages.length === 0 && (
                <div className="mb-6 mt-2 text-center md:mt-4">
                  <h1 className="text-4xl font-semibold tracking-tight md:text-6xl">Densely Research. Explore.</h1>
                  <p className={cn("mx-auto mt-4 max-w-2xl text-sm leading-6 md:text-base", dk ? "text-zinc-400" : "text-zinc-500")}>
                    Minimal research chat for biomedical literature, grounded answers, source-level confidence, and fast model switching.
                  </p>
                  {conversationId && (
                    <p className={cn("mt-2 text-xs", dk ? "text-zinc-500" : "text-zinc-400")}>
                      Conversation: {conversationId.slice(0, 12)}...
                    </p>
                  )}
                </div>
              )}

              {messages.length === 0 ? (
                <div className="grid gap-4 md:grid-cols-3">
                  {DEMO_QUESTIONS.map((prompt, i) => (
                    <button
                      key={prompt}
                      onClick={() => setQuery(prompt)}
                      type="button"
                      className={cn("rounded-[24px] border p-4 text-left transition-all hover:-translate-y-0.5", panelTheme)}
                    >
                      <div className="mb-3 flex items-center gap-2">
                        <div className={cn("flex h-9 w-9 items-center justify-center rounded-2xl", dk ? "bg-white/10" : "bg-zinc-100")}>
                          {i === 0 ? (
                            <Sparkles className="h-4 w-4" />
                          ) : i === 1 ? (
                            <FileText className="h-4 w-4" />
                          ) : (
                            <Bot className="h-4 w-4" />
                          )}
                        </div>
                        <p className="text-sm font-medium">Prompt {i + 1}</p>
                      </div>
                      <p className="text-sm leading-6">{prompt}</p>
                    </button>
                  ))}
                </div>
              ) : (
                <div ref={messagesContainerRef} className="flex-1 space-y-4 overflow-y-auto pr-1 pt-1">
                  {messages.map((m, idx) => {
                    const sourceRows =
                      showSources &&
                      m.role === "assistant" &&
                      m.sources &&
                      m.sources.length > 0 &&
                      !m.isStreaming
                        ? buildAssistantSourceRows(m.sources, m.content)
                        : [];
                    const hasInlineCites =
                      m.role === "assistant" ? extractCitationIndicesFromAnswer(m.content).length > 0 : false;
                    return (
                    <div key={idx} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
                      <div className={cn("max-w-[92%] space-y-2", m.role === "user" ? "" : "w-full")}>
                        <div
                          className={cn(
                            "rounded-2xl px-4 py-3 text-sm leading-relaxed",
                            m.role === "user"
                              ? "bg-violet-600 text-white"
                              : dk
                                ? "border-0 bg-transparent text-white shadow-none"
                                : "border-0 bg-transparent text-zinc-900 shadow-none"
                          )}
                        >
                          {m.role === "assistant" && !m.isStreaming ? (
                            <AssistantMessageBody content={m.content || "..."} dark={dk} renderMath={renderMath} />
                          ) : (
                            <p className="whitespace-pre-wrap break-words">{m.content || "..."}</p>
                          )}
                        </div>
                        {m.isStreaming && <span className="ml-1 inline-block h-4 w-0.5 animate-pulse bg-current align-middle opacity-70" />}
                        {m.role === "assistant" && !m.isStreaming && m.memoryInfo && typeof m.memoryInfo.confidence === "object" && m.memoryInfo.confidence !== null && (
                          <p className={cn("text-xs", dk ? "text-zinc-500" : "text-zinc-600")}>
                            Answer confidence:{" "}
                            <span className="font-medium">
                              {String((m.memoryInfo.confidence as { label?: string }).label ?? "—")}
                            </span>{" "}
                            (
                            {Math.round(
                              Number((m.memoryInfo.confidence as { final_conf?: number }).final_conf ?? 0) * 100
                            )}
                            %)
                          </p>
                        )}

                        {sourceRows.length > 0 && (
                              <details
                                className={cn(
                                  "rounded-2xl border open:pb-1 open:[&_summary_svg:first-child]:rotate-180",
                                  dk ? "border-white/10 bg-white/[0.02]" : "border-zinc-200 bg-white/70"
                                )}
                              >
                                <summary
                                  className={cn(
                                    "flex cursor-pointer list-none items-center gap-2 rounded-2xl px-3 py-2.5 text-sm font-medium marker:content-none [&::-webkit-details-marker]:hidden",
                                    dk ? "text-zinc-200" : "text-zinc-800"
                                  )}
                                >
                                  <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200" />
                                  <span>
                                    Sources ({sourceRows.length})
                                    {hasInlineCites ? "" : " · full retrieval deduped"}
                                  </span>
                                </summary>
                                <p className={cn("px-3 pb-2 text-[11px]", dk ? "text-zinc-500" : "text-zinc-500")}>
                                  {hasInlineCites
                                    ? "Tags are grouped by paper; copy includes all cited excerpts for that paper."
                                    : "Add [S#] in the model reply to list only cited passages."}
                                </p>
                                <div className="max-h-72 space-y-2 overflow-y-auto px-2 pb-2 pr-1">
                                  {sourceRows.map((row) => (
                                    <div
                                      key={`${row.citeTags.join("-")}-${row.fileLabel}`}
                                      className={cn("flex items-center justify-between rounded-xl px-3 py-2", softTheme)}
                                    >
                                      <div className="min-w-0">
                                        <p className="truncate text-sm font-medium">
                                          <span className={cn("mr-2 font-mono text-xs", dk ? "text-violet-300" : "text-violet-700")}>
                                            {row.citeTags.map((n) => `[S${n}]`).join(", ")}
                                          </span>
                                          {shortLabel(row.title)}
                                        </p>
                                        {row.fileLabel ? (
                                          <p className={cn("truncate text-xs", dk ? "text-zinc-400" : "text-zinc-500")}>
                                            {shortLabel(row.fileLabel)}
                                          </p>
                                        ) : null}
                                      </div>
                                      <div className="ml-3 shrink-0">
                                        <button
                                          type="button"
                                          title="Copy excerpt"
                                          aria-label="Copy excerpt"
                                          onClick={() => void handleCopy(row.copyText)}
                                          className={cn(
                                            "flex h-8 w-8 items-center justify-center rounded-lg border text-xs",
                                            dk ? "border-white/15" : "border-zinc-200"
                                          )}
                                        >
                                          <Copy className="h-3.5 w-3.5" />
                                        </button>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </details>
                            )}
                      </div>
                    </div>
                    );
                  })}
                  <div ref={messagesEndRef} />
                </div>
              )}

            </div>

            <div className="mx-auto w-full max-w-5xl pt-6">
              <div
                className={cn(
                  "overflow-visible rounded-[24px] border p-2.5 shadow-2xl shadow-black/10",
                  dk ? "border-white/10 bg-zinc-900/80 backdrop-blur-2xl" : "border-zinc-200 bg-white/95 backdrop-blur-2xl"
                )}
              >
                <div className="px-3 pt-2 pb-3">
                  <textarea
                    ref={textareaRef}
                    rows={1}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    spellCheck={false}
                    placeholder={webEnabled ? "Search the web…" : "Ask about your corpus…"}
                    className={cn(
                      "max-h-[200px] w-full resize-none bg-transparent text-sm outline-none md:text-[15px]",
                      dk ? "text-zinc-100 placeholder:text-zinc-500" : "text-zinc-900 placeholder:text-zinc-400",
                      loading && "cursor-wait"
                    )}
                    readOnly={loading}
                    aria-busy={loading}
                  />
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3 px-1 pb-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="relative" ref={modelPickerRef}>
                      {modelPickerOpen && mode === "corpus" && (
                        <ul
                          role="listbox"
                          className={cn(
                            "absolute bottom-full left-0 z-40 mb-1.5 min-w-[12rem] overflow-hidden rounded-2xl border py-1 shadow-xl",
                            dk
                              ? "border-white/10 bg-zinc-950/95 text-zinc-100 shadow-black/50 backdrop-blur-xl"
                              : "border-zinc-200 bg-white text-zinc-900 shadow-lg"
                          )}
                        >
                          {CORPUS_MODEL_OPTIONS.map((opt) => (
                            <li key={opt.value}>
                              <button
                                type="button"
                                role="option"
                                aria-selected={corpusGeneration === opt.value}
                                className={cn(
                                  "flex w-full items-center px-3 py-2.5 text-left text-xs font-medium transition",
                                  corpusGeneration === opt.value
                                    ? dk
                                      ? "bg-violet-500/30 text-violet-100"
                                      : "bg-violet-100 text-violet-900"
                                    : dk
                                      ? "hover:bg-white/10"
                                      : "hover:bg-zinc-100"
                                )}
                                onClick={() => {
                                  setCorpusGeneration(opt.value);
                                  setModelPickerOpen(false);
                                }}
                              >
                                {opt.label}
                              </button>
                            </li>
                          ))}
                        </ul>
                      )}
                      <button
                        type="button"
                        disabled={mode === "web"}
                        aria-haspopup="listbox"
                        aria-expanded={modelPickerOpen}
                        onClick={() => mode !== "web" && setModelPickerOpen((o) => !o)}
                        className={cn(
                          "flex h-9 min-w-[11rem] max-w-[16rem] items-center justify-between gap-2 rounded-full border px-3 text-xs font-medium outline-none transition focus-visible:ring-2 disabled:cursor-not-allowed disabled:opacity-50",
                          dk
                            ? "border-white/10 bg-white/5 text-zinc-100 hover:bg-white/10 focus-visible:ring-violet-500/40"
                            : "border-zinc-200/90 bg-zinc-100/90 text-zinc-900 hover:bg-zinc-200/90 focus-visible:ring-violet-500/50"
                        )}
                        title={
                          mode === "web"
                            ? "Web mode uses OpenAI for search summary"
                            : corpusGeneration === "ollama"
                              ? `Local Ollama: ${ollamaModelLabel}`
                              : "OpenAI model for corpus answers"
                        }
                      >
                        <span className="min-w-0 truncate">{corpusModelLabel(corpusGeneration)}</span>
                        <ChevronDown
                          className={cn(
                            "h-3.5 w-3.5 shrink-0 transition-transform",
                            modelPickerOpen && "rotate-180",
                            dk ? "text-zinc-400" : "text-zinc-500"
                          )}
                          aria-hidden
                        />
                      </button>
                    </div>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => setMode(webEnabled ? "corpus" : "web")}
                      className={cn(
                        "flex h-9 w-9 items-center justify-center rounded-full border transition",
                        webEnabled
                          ? dk
                            ? "border-violet-400/40 bg-violet-500/10 text-violet-200"
                            : "border-violet-200 bg-violet-50 text-violet-700"
                          : dk
                            ? "border-white/10 bg-white/5"
                            : "border-zinc-200 bg-zinc-50"
                      )}
                      title="Web toggle"
                    >
                      <Globe className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        if (loading) handleStopGeneration();
                        else void handleSend();
                      }}
                      disabled={!loading && !query.trim()}
                      title={loading ? "Stop (Enter)" : "Send (Enter)"}
                      className={cn(
                        "flex h-9 w-9 items-center justify-center rounded-full p-0 transition",
                        loading
                          ? dk
                            ? "border border-rose-400/50 bg-rose-500/15 text-rose-200 hover:bg-rose-500/25"
                            : "border border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100"
                          : cn(
                              "text-white disabled:opacity-40",
                              dk ? "bg-violet-500 hover:bg-violet-400" : "bg-violet-600 hover:bg-violet-500"
                            )
                      )}
                    >
                      {loading ? <Square className="h-4 w-4 fill-current" /> : <ArrowUp className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
              </div>

              <div className={cn("mt-3 flex items-center justify-end gap-2 px-1 text-xs", dk ? "text-zinc-400" : "text-zinc-500")}>
                <button type="button" onClick={handleClearMemory} className={cn("rounded-xl px-2 py-1", softTheme)}>
                  <Trash2 className="mr-1 inline h-3.5 w-3.5" />
                  Clear
                </button>
                {showScrollBtn && (
                  <button type="button" onClick={scrollToBottom} className={cn("rounded-xl px-2 py-1", softTheme)}>
                    <ChevronDown className="mr-1 inline h-3.5 w-3.5" />
                    Latest
                  </button>
                )}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
