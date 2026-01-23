import fs from "node:fs";
import path from "node:path";
import { Page } from "playwright";
import type { StepLogger } from "../logger.js";
import { ask, getLastAnswerText } from "../notebooklm/flows.js";
import { containsSystemFailure, extractJsonCandidate, tryParseJsonStrict } from "./json_guard.js";

export type JsonResult = {
  json: unknown;
  rawText: string;
};

type JsonMeta = {
  outputDir?: string;
  tag?: string;
};

export async function askJsonWithRepair(
  page: Page,
  prompt: string,
  maxRetries = 4,
  meta?: JsonMeta,
  logger?: StepLogger
): Promise<JsonResult> {
  await ask(page, prompt, logger);
  let answer = await getLastAnswerText(page, logger);
  if (containsSystemFailure(answer)) {
    throw new Error("NotebookLM returned: system was unable to answer.");
  }
  const trimmed = answer.trim();
  try {
    const json = tryParseJsonStrict(trimmed);
    return { json, rawText: trimmed };
  } catch {
    const candidate = extractJsonCandidate(answer);
    if (candidate) {
      try {
        const json = tryParseJsonStrict(candidate);
        return { json, rawText: candidate };
      } catch {
        // fall through to error handling
      }
    }
    if (meta?.outputDir) {
      const errorDir = path.join(meta.outputDir, "_errors");
      fs.mkdirSync(errorDir, { recursive: true });
      const name = meta.tag ? meta.tag.replace(/[^a-zA-Z0-9_-]+/g, "_") : "unknown";
      const outPath = path.join(errorDir, `${name}_${Date.now()}.txt`);
      fs.writeFileSync(outPath, answer);
    }
    const error = new Error("JSON parse failed");
    (error as { lastAnswer?: string }).lastAnswer = answer;
    throw error;
  }
}
