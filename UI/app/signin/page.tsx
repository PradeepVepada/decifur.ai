"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch } from "@/lib/api";
import Link from "next/link";
import { FlaskConical, Eye, EyeOff, Loader2 } from "lucide-react";

export default function SignInPage() {
  const router = useRouter();
  // Prefilled for local dev — change defaults here or clear and type your own (any values work with the API stub).
  const [email, setEmail] = useState("researcher@local.dev");
  const [password, setPassword] = useState("local");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await apiFetch<{ token: string; user: { name: string; email: string } }>("/api/auth/signin", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      localStorage.setItem("auth_token", res.token);
      localStorage.setItem("auth_user", JSON.stringify(res.user));
      router.replace("/chat");
    } catch (err) {
      setError((err as Error).message || "Sign in failed. Check your credentials.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-neutral-950 p-6">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="mb-8 flex flex-col items-center gap-3 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-600">
            <FlaskConical size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">Research Explorer</h1>
            <p className="text-sm text-neutral-400">Devreotes Lab · GraphRAG</p>
          </div>
        </div>

        <form
          onSubmit={onSubmit}
          className="space-y-4 rounded-2xl border border-neutral-800 bg-neutral-900/60 p-6 backdrop-blur-sm"
        >
          <h2 className="text-lg font-semibold text-white">Sign in</h2>

          <div className="space-y-3">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-neutral-400">Email</label>
              <input
                type="email"
                required
                autoComplete="email"
                placeholder="researcher@local.dev"
                className="w-full rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2.5 text-sm text-white outline-none transition-colors placeholder:text-neutral-600 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div>
              <label className="mb-1.5 block text-xs font-medium text-neutral-400">Password</label>
              <div className="relative">
                <input
                  type={showPw ? "text" : "password"}
                  required
                  autoComplete="current-password"
                  placeholder="anything"
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2.5 pr-10 text-sm text-white outline-none transition-colors placeholder:text-neutral-600 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500 hover:text-neutral-300"
                  tabIndex={-1}
                >
                  {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-950/40 px-3 py-2 text-sm text-red-400">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 py-2.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-60"
          >
            {loading && <Loader2 size={15} className="animate-spin" />}
            {loading ? "Signing in…" : "Sign in"}
          </button>

          <p className="text-center text-sm text-neutral-500">
            New user?{" "}
            <Link href="/signup" className="font-medium text-indigo-400 hover:text-indigo-300">
              Create an account
            </Link>
          </p>
          <p className="text-center text-xs leading-relaxed text-neutral-600">
            Local dev: start <code className="rounded bg-neutral-800 px-1 py-0.5">python api.py</code> on port 8000.
            The server accepts any email/password — if sign-in fails, check <code className="rounded bg-neutral-800 px-1">NEXT_PUBLIC_API_URL</code> in{" "}
            <code className="rounded bg-neutral-800 px-1">UI/.env.local</code>.
          </p>
        </form>
      </div>
    </main>
  );
}
