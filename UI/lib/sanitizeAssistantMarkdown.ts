/**
 * Display-side cleanup before react-markdown + KaTeX.
 * Does not alter meaning of well-formed math; reduces broken $$ and PUA tofu.
 */

/** BMP private-use area (often renders as tofu without a glyph). */
const PUA_BMP = /[\uE000-\uF8FF]/g;

export function stripPrivateUseChars(input: string): string {
  return input.replace(PUA_BMP, "");
}

/**
 * If there is an odd number of `$$` delimiters, display math is unclosed — KaTeX/remark can mis-parse.
 * Close with a trailing `$$` on its own line (often empty display, tolerated by KaTeX with throwOnError: false).
 */
export function balanceDisplayMathDelimiters(input: string): string {
  const matches = input.match(/\$\$/g);
  if (!matches || matches.length % 2 === 0) {
    return input;
  }
  return `${input.trimEnd()}\n$$\n`;
}

/**
 * If the model returns one huge line (no blank-line paragraph breaks), split into 2–4
 * Markdown paragraphs at sentence boundaries so ReactMarkdown renders multiple `<p>` blocks.
 */
export function ensureReadableParagraphs(input: string): string {
  const t = (input ?? "").trim();
  if (!t || t.includes("\n\n")) return input ?? "";

  // Avoid splitting inside display math.
  if (t.includes("$$")) return input ?? "";

  const minChars = 480;
  if (t.length < minChars) return input ?? "";

  const sentences = t
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (sentences.length < 4) return input ?? "";

  const targetParas = Math.min(4, Math.max(2, Math.round(sentences.length / 3)));
  const per = Math.ceil(sentences.length / targetParas);
  const paras: string[] = [];
  for (let i = 0; i < sentences.length; i += per) {
    paras.push(sentences.slice(i, i + per).join(" "));
  }
  return paras.join("\n\n");
}

export function sanitizeAssistantMarkdown(input: string): string {
  let s = stripPrivateUseChars(input ?? "");
  s = balanceDisplayMathDelimiters(s);
  s = ensureReadableParagraphs(s);
  return s;
}
