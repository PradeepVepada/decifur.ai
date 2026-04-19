"use client";

import React, { Component, type ErrorInfo, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { sanitizeAssistantMarkdown } from "@/lib/sanitizeAssistantMarkdown";

/** Passed through to KaTeX (rehype-katex omits throwOnError; plugin retries with throwOnError: false). */
const katexRehypeOptions = {
  strict: "ignore" as const,
  errorColor: "#cc0000",
};

type Props = {
  content: string;
  dark: boolean;
  /** When false, skip Markdown/KaTeX and show plain text. */
  renderMath?: boolean;
};

type BoundaryProps = {
  children: ReactNode;
  fallback: ReactNode;
  resetKey: string;
};

class MarkdownErrorBoundary extends Component<BoundaryProps, { hasError: boolean }> {
  state = { hasError: false };

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidUpdate(prevProps: BoundaryProps) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }

  componentDidCatch(error: Error, _info: ErrorInfo) {
    console.warn("[AssistantMessageBody] markdown render failed, using plain fallback", error);
  }

  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

function PlainTextBody({ text, dark }: { text: string; dark: boolean }) {
  return (
    <pre
      className={
        dark
          ? "whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-100"
          : "whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-900"
      }
    >
      {text}
    </pre>
  );
}

/**
 * Renders assistant prose with Markdown + LaTeX ($...$, $$...$$) via KaTeX.
 * Sanitizes fragile output; on render failure falls back to plain pre-wrap text.
 */
export function AssistantMessageBody({ content, dark, renderMath = true }: Props) {
  const safe = sanitizeAssistantMarkdown(content);

  const proseClass =
    dark
      ? "max-w-none text-sm leading-relaxed text-zinc-100 [&_.katex]:text-zinc-100 [&_.katex-display]:my-3 [&_.katex-display]:overflow-x-auto [&_p]:my-2 [&_ul]:my-2 [&_li]:my-0.5 [&_strong]:font-semibold"
      : "max-w-none text-sm leading-relaxed text-zinc-900 [&_.katex]:text-zinc-900 [&_.katex-display]:my-3 [&_.katex-display]:overflow-x-auto [&_p]:my-2 [&_ul]:my-2 [&_li]:my-0.5 [&_strong]:font-semibold";

  if (!renderMath) {
    return (
      <div className={proseClass}>
        <PlainTextBody text={safe} dark={dark} />
      </div>
    );
  }

  const fallback = <PlainTextBody text={safe} dark={dark} />;

  return (
    <div className={proseClass}>
      <MarkdownErrorBoundary resetKey={safe} fallback={fallback}>
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[[rehypeKatex, katexRehypeOptions]]}
        >
          {safe}
        </ReactMarkdown>
      </MarkdownErrorBoundary>
    </div>
  );
}
