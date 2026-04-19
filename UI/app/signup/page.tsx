"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch } from "@/lib/api";
import Link from "next/link";
import { FlaskConical, Eye, EyeOff, Loader2 } from "lucide-react";

export default function SignUpPage() {
  const router = useRouter();
  const [name, setName] = useState("Local Researcher");
  const [email, setEmail] = useState("researcher@local.dev");
  const [password, setPassword] = useState("localdev1");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await apiFetch<{ token: string; user: { name: string; email: string } }>("/api/auth/signup", {
        method: "POST",
        body: JSON.stringify({ name, email, password }),
      });
      localStorage.setItem("auth_token", res.token);
      localStorage.setItem("auth_user", JSON.stringify(res.user));
      router.replace("/chat");
    } catch (err) {
      setError((err as Error).message || "Sign up failed. Try again.");
    } finally {
      setLoading(false);
    }
  }

  const pwStrength =
    password.length === 0
      ? 0
      : password.length < 8
        ? 1
        : password.length < 12
          ? 2
          : 3;

  const pwColors = ["", "bg-red-500", "bg-yellow-400", "bg-green-500"];
  const pwLabels = ["", "Too short", "Good", "Strong"];

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
          <h2 className="text-lg font-semibold text-white">Create account</h2>

          <div className="space-y-3">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-neutral-400">Full name</label>
              <input
                required
                autoComplete="name"
                placeholder="Jane Smith"
                className="w-full rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2.5 text-sm text-white outline-none transition-colors placeholder:text-neutral-600 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            <div>
              <label className="mb-1.5 block text-xs font-medium text-neutral-400">Email</label>
              <input
                type="email"
                required
                autoComplete="email"
                placeholder="you@example.com"
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
                  minLength={8}
                  autoComplete="new-password"
                  placeholder="Min. 8 characters"
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

              {/* Strength bar */}
              {password.length > 0 && (
                <div className="mt-2 space-y-1">
                  <div className="flex gap-1">
                    {[1, 2, 3].map((i) => (
                      <div
                        key={i}
                        className={`h-1 flex-1 rounded-full transition-colors ${
                          i <= pwStrength ? pwColors[pwStrength] : "bg-neutral-700"
                        }`}
                      />
                    ))}
                  </div>
                  <p className={`text-xs ${pwStrength === 1 ? "text-red-400" : pwStrength === 2 ? "text-yellow-400" : "text-green-400"}`}>
                    {pwLabels[pwStrength]}
                  </p>
                </div>
              )}
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
            {loading ? "Creating account…" : "Create account"}
          </button>

          <p className="text-center text-sm text-neutral-500">
            Already have an account?{" "}
            <Link href="/signin" className="font-medium text-indigo-400 hover:text-indigo-300">
              Sign in
            </Link>
          </p>
        </form>
      </div>
    </main>
  );
}
