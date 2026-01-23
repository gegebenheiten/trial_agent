import { CsvRow } from "./csv_io.js";
import { PromptConfig } from "./prompt_config.js";

export type EvidenceMode = "text_only" | "sources";

export function buildPrompt(
  table: string,
  nctid: string,
  row: CsvRow,
  missingFields: string[],
  promptConfig: PromptConfig,
  evidenceMode: EvidenceMode,
  sourceHints: string[] = []
): string {
  const instructions = buildInstructions(promptConfig.instructions, evidenceMode, table);
  const template = outputTemplate(missingFields);
  const contextLines = buildTableContext(table, row);
  const sourcesLine = sourceHints.length
    ? `Sources (use exact source.name from this list): ${sourceHints.join(" | ")}`
    : "Sources: use the exact source title shown in NotebookLM.";

  return [
    instructions,
    ...(contextLines.length ? [contextLines.join("\n")] : []),
    sourcesLine,
    `NCT ID: ${nctid}`,
    `Table: ${table}`,
    "",
    "OUTPUT_TEMPLATE_START",
    template,
    "OUTPUT_TEMPLATE_END"
  ].join("\n");
}

function buildInstructions(base: string, evidenceMode: EvidenceMode, table: string): string {
  const lines: string[] = [
    `Follow instruction.txt and feature_instruction_${table}.txt for this table.`,
    "Return ONLY a valid JSON object (no markdown, no commentary)."
  ];
  const trimmed = String(base || "").trim();
  if (trimmed) {
    lines.push(`Task: ${trimmed}`);
  }
  void evidenceMode;
  return lines.join("\n");
}

function outputTemplate(missingFields: string[]): string {
  const template: Record<string, { value: string; evidence: string; source: { name: string; type: string } }> = {};
  for (const field of missingFields) {
    template[field] = { value: "", evidence: "", source: { name: "", type: "" } };
  }
  return JSON.stringify(template);
}

function buildTableContext(table: string, row: CsvRow): string[] {
  if (table !== "R_Arm_Study" && table !== "R_Arm") {
    return [];
  }
  const armId = pickRowValue(row, ["Arm_ID", "ArmID", "Arm_Id"]);
  const description = pickRowValue(row, ["Arm_Description", "Baseline_Group_Desc", "Group_Description"]);
  const lines: string[] = ["Group context:"];
  if (armId) {
    lines.push(`- Arm_ID: ${armId}`);
  }
  if (description) {
    lines.push(`- Description: ${truncate(description, 400)}`);
  }
  return lines.length > 1 ? lines : [];
}

function pickRowValue(row: CsvRow, keys: string[]): string {
  for (const key of keys) {
    const raw = row[key];
    if (typeof raw === "string" && raw.trim()) {
      return raw.trim();
    }
  }
  return "";
}

function truncate(text: string, maxLen: number): string {
  const trimmed = text.trim();
  if (trimmed.length <= maxLen) {
    return trimmed;
  }
  return `${trimmed.slice(0, maxLen - 3)}...`;
}
