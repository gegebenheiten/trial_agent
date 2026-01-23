import fs from "node:fs";
import path from "node:path";

export type PromptField = {
  key: string;
  question: string;
  required?: boolean;
};

export type PromptPack = {
  name: string;
  description?: string;
  fields: PromptField[];
};

export function readPromptPack(packName: string, promptsDir: string): PromptPack {
  const packPath = path.join(promptsDir, "packs", `${packName}.json`);
  const raw = fs.readFileSync(packPath, "utf8");
  return JSON.parse(raw) as PromptPack;
}

export function readSystemRules(promptsDir: string): string {
  const rulesPath = path.join(promptsDir, "templates", "system_rules.md");
  return fs.readFileSync(rulesPath, "utf8");
}

export function buildFinalPrompt(nctid: string, pack: PromptPack, systemRules: string): string {
  const lines: string[] = [];
  lines.push(systemRules.trim());
  lines.push("");
  lines.push(`Task: Extract fields for ${nctid}.`);
  if (pack.description) {
    lines.push(`Pack: ${pack.description}`);
  }
  lines.push("Return JSON with the following keys:");
  for (const field of pack.fields) {
    const req = field.required ? "required" : "optional";
    lines.push(`- ${field.key} (${req}): ${field.question}`);
  }
  lines.push("");
  lines.push("JSON schema:");
  lines.push("{");
  for (const field of pack.fields) {
    lines.push(`  \"${field.key}\": null,`);
  }
  lines.push("}");
  return lines.join("\n");
}
