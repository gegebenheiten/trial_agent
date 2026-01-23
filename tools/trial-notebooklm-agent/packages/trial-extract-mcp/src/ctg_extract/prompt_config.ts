import fs from "node:fs";
import path from "node:path";

export type PromptConfig = {
  instructions: string;
  notes: Record<string, string>;
  llm_fields: string[];
  text_modules: string[];
  dedupe_fields: string[];
  dedupe_key_fields: string[];
};

export type PromptConfigSet = Record<string, PromptConfig>;

export type PromptConfigIndex = Record<string, PromptConfigSet>;

export function loadPromptConfigs(repoRoot: string): PromptConfigIndex {
  const cfgPath = path.join(repoRoot, "tools", "ctg_extract_v2", "llm", "prompt_configs.json");
  const raw = fs.readFileSync(cfgPath, "utf8");
  return JSON.parse(raw) as PromptConfigIndex;
}

export function resolvePromptConfig(
  configs: PromptConfigIndex,
  table: string,
  group: string
): PromptConfig {
  const tableConfigs = configs[table];
  if (!tableConfigs) {
    throw new Error(`Prompt configs not found for table: ${table}`);
  }
  const cfg = tableConfigs[group] || tableConfigs.all;
  if (!cfg) {
    throw new Error(`Prompt config group '${group}' not found for table: ${table}`);
  }
  return cfg;
}
