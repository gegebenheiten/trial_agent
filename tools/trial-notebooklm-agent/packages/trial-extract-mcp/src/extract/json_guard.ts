export function tryParseJsonStrict(text: string): unknown {
  const trimmed = stripBom(text.trim());
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
    throw new Error("Not JSON");
  }
  return JSON.parse(trimmed);
}

export function extractJsonCandidate(text: string): string | null {
  const trimmed = stripBom(text.trim());
  if (!trimmed) {
    return null;
  }
  const startIdx = trimmed.search(/[\[{]/);
  if (startIdx === -1) {
    return null;
  }
  const slice = extractBalancedJson(trimmed, startIdx);
  if (slice) {
    return slice;
  }
  const lastCurly = trimmed.lastIndexOf("}");
  const lastSquare = trimmed.lastIndexOf("]");
  const lastClose = Math.max(lastCurly, lastSquare);
  if (lastClose === -1) {
    return null;
  }
  const openCurly = trimmed.lastIndexOf("{", lastClose);
  const openSquare = trimmed.lastIndexOf("[", lastClose);
  const open = Math.max(openCurly, openSquare);
  if (open === -1) {
    return null;
  }
  return trimmed.slice(open, lastClose + 1);
}

export function buildRepairPrompt(badText: string): string {
  const cleaned = sanitizeBadTextForRepair(badText);
  return [
    "You returned invalid JSON.",
    "Fix it to STRICT valid JSON.",
    "Rules:",
    "- Output JSON ONLY (no markdown, no commentary).",
    "- Keep the same keys/structure.",
    "- If you must drop something, drop minimally.",
    "",
    "INVALID_JSON_START",
    cleaned,
    "INVALID_JSON_END"
  ].join("\n");
}

export function containsSystemFailure(text: string): boolean {
  const lowered = text.toLowerCase();
  return (
    lowered.includes("the system was unable to answer") ||
    lowered.includes("系统无法回答") ||
    lowered.includes("无法回答")
  );
}

export function sanitizeBadTextForRepair(text: string): string {
  const uiLine = /^(chat|tune|more_vert|arrow_forward|keyboard_arrow_down|sources?|today\\b|pinpointing the core issue)/i;
  const lines = text.split(/\r?\n/);
  const kept: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      kept.push("");
      continue;
    }
    const looksJsonish = /[\\{\\}\\[\\]":]/.test(trimmed);
    if (!looksJsonish && uiLine.test(trimmed)) {
      continue;
    }
    kept.push(line);
  }
  return kept.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function stripBom(text: string): string {
  return text.replace(/^\uFEFF/, "");
}

function extractBalancedJson(text: string, startIdx: number): string | null {
  const stack: string[] = [];
  let inString = false;
  let escape = false;
  for (let i = startIdx; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escape) {
        escape = false;
        continue;
      }
      if (ch === "\\") {
        escape = true;
        continue;
      }
      if (ch === "\"") {
        inString = false;
      }
      continue;
    }
    if (ch === "\"") {
      inString = true;
      continue;
    }
    if (ch === "{" || ch === "[") {
      stack.push(ch);
    } else if (ch === "}" || ch === "]") {
      if (!stack.length) {
        continue;
      }
      const open = stack.pop();
      if ((open === "{" && ch !== "}") || (open === "[" && ch !== "]")) {
        return null;
      }
      if (stack.length === 0) {
        return text.slice(startIdx, i + 1);
      }
    }
  }
  return null;
}
