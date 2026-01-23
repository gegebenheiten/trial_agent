import fs from "node:fs";
import path from "node:path";

import { CsvRow } from "./csv_io.js";

export type TextBlockRecord = Record<string, unknown>;

const STUDY_INFO_FIELDS: Array<[string, string]> = [
  ["Brief Title", "brief_title"],
  ["Official Title", "official_title"],
  ["Brief Summary", "brief_summary"],
  ["Detailed Description", "detailed_description"]
];

const DEFAULT_TEXT_MODULES_GENERIC = [
  "study_info",
  "eligibility",
  "participant_flow",
  "baseline_results",
  "baseline_measures",
  "results_outcomes",
  "reported_events",
  "arm_groups",
  "interventions",
  "primary_outcomes",
  "secondary_outcomes",
  "keywords",
  "conditions",
  "location_countries"
];

const DEFAULT_TEXT_MODULES: Record<string, string[]> = {
  D_Design: ["design_info", "study_info", "eligibility", "arm_groups", "interventions"],
  D_Pop: ["study_info", "eligibility", "conditions"],
  D_Drug: ["interventions", "arm_groups", "study_info"],
  R_Study: [
    "endpoint_target",
    "endpoint_matches",
    "participant_flow",
    "design_info",
    "study_info",
    "keywords",
    "conditions"
  ],
  R_Arm_Study: [
    "group_target",
    "participant_flow",
    "baseline_results",
    "baseline_measures",
    "reported_events",
    "arm_groups",
    "interventions"
  ],
  R_Arm: [
    "group_target",
    "participant_flow",
    "baseline_results",
    "baseline_measures",
    "reported_events",
    "arm_groups",
    "interventions"
  ]
};

export function loadTextBlocks(repoRoot: string, nctid: string): TextBlockRecord {
  const baseDir = path.join(repoRoot, "data", "ctg_extract_v2", nctid);
  const jsonPath = path.join(baseDir, "ctg_text_blocks.json");
  if (fs.existsSync(jsonPath)) {
    const payload = JSON.parse(fs.readFileSync(jsonPath, "utf8")) as TextBlockRecord[];
    const found = payload.find((entry) => (entry as Record<string, unknown>).nct_id === nctid);
    if (found) {
      return found;
    }
  }

  const jsonlPath = path.join(baseDir, "ctg_text_blocks.jsonl");
  if (fs.existsSync(jsonlPath)) {
    const lines = fs.readFileSync(jsonlPath, "utf8").split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      const entry = JSON.parse(line) as TextBlockRecord;
      if ((entry as Record<string, unknown>).nct_id === nctid) {
        return entry;
      }
    }
  }

  throw new Error(`Missing ctg_text_blocks for ${nctid}`);
}

export function formatTextBlocks(
  record: TextBlockRecord,
  table: string,
  row: CsvRow,
  textModules: string[] | undefined,
  maxChars = 120_000
): string {
  const modules = resolveTextModules(table, textModules);
  const sections: string[] = [];

  for (const module of modules) {
    switch (module) {
      case "design_info": {
        const value = record.design_info;
        if (value) {
          sections.push("Structured Design:\n" + JSON.stringify(value));
        }
        break;
      }
      case "study_info": {
        for (const [label, key] of STUDY_INFO_FIELDS) {
          const value = normalizeWhitespace(String((record as Record<string, unknown>)[key] || ""));
          if (value) {
            sections.push(`${label}: ${value}`);
          }
        }
        break;
      }
      case "eligibility": {
        const value = normalizeWhitespace(String(record.eligibility_criteria || ""));
        if (value) {
          sections.push(`Eligibility Criteria: ${value}`);
        }
        break;
      }
      case "arm_groups": {
        const value = record.arm_groups;
        if (value) {
          sections.push("Arm Groups:\n" + JSON.stringify(value));
        }
        break;
      }
      case "interventions": {
        const value = record.interventions;
        if (value) {
          sections.push("Interventions:\n" + JSON.stringify(value));
        }
        break;
      }
      case "primary_outcomes": {
        const value = record.primary_outcomes;
        if (value) {
          sections.push("Primary Outcomes:\n" + formatOutcomeDefinitions(value));
        }
        break;
      }
      case "secondary_outcomes": {
        const value = record.secondary_outcomes;
        if (value) {
          sections.push("Secondary Outcomes:\n" + formatOutcomeDefinitions(value));
        }
        break;
      }
      case "participant_flow": {
        const value = record.participant_flow;
        if (value) {
          sections.push("Participant Flow: " + JSON.stringify(value));
        }
        break;
      }
      case "baseline_results": {
        const value = record.baseline_results;
        if (value) {
          sections.push("Baseline Results: " + JSON.stringify(value));
        }
        break;
      }
      case "baseline_measures": {
        const value = record.baseline_measures;
        if (value) {
          sections.push("Baseline Measures: " + JSON.stringify(value));
        }
        break;
      }
      case "results_outcomes": {
        const value = record.results_outcomes;
        if (value) {
          sections.push("Results Outcomes: " + JSON.stringify(value));
        }
        break;
      }
      case "reported_events": {
        const value = record.reported_events;
        if (value) {
          sections.push("Reported Events: " + JSON.stringify(value));
        }
        break;
      }
      case "keywords": {
        const value = record.keywords;
        if (value) {
          sections.push("Keywords: " + JSON.stringify(value));
        }
        break;
      }
      case "conditions": {
        const value = record.conditions;
        if (value) {
          sections.push("Conditions: " + JSON.stringify(value));
        }
        break;
      }
      case "location_countries": {
        const value = record.location_countries;
        if (value) {
          sections.push("Location Countries: " + JSON.stringify(value));
        }
        break;
      }
      case "group_target": {
        const targetTitle = normalizeWhitespace(String(row.group_title_raw || ""));
        const targetDesc = normalizeWhitespace(String(row.group_desc_raw || ""));
        const targetId = normalizeWhitespace(String(row.group_id_raw || ""));
        const targetArm = normalizeWhitespace(String(row.Arm_ID || ""));
        if (targetArm) {
          sections.push(`Target Arm_ID: ${targetArm}`);
        }
        if (targetTitle) {
          sections.push(`Target Group Title: ${targetTitle}`);
        }
        if (targetDesc) {
          sections.push(`Target Group Description: ${targetDesc}`);
        }
        if (targetId) {
          sections.push(`Target Group ID: ${targetId}`);
        }
        break;
      }
      case "endpoint_target": {
        const endpoint = normalizeWhitespace(
          String(row.Endpoint_Name || row.EP_Name || row.Outcome || "")
        );
        if (endpoint) {
          sections.push(`Target Endpoint: ${endpoint}`);
        }
        break;
      }
      case "endpoint_matches": {
        const endpoint = normalizeWhitespace(
          String(row.Endpoint_Name || row.EP_Name || row.Outcome || "")
        );
        const matches = selectOutcomes(record.results_outcomes, endpoint, 1, 80);
        if (matches.length) {
          sections.push("Matched Outcomes:");
          for (const match of matches) {
            sections.push(JSON.stringify(match.outcome));
          }
        } else {
          sections.push("No matching outcome found for Target Endpoint; return empty for all fields.");
        }
        break;
      }
      default:
        break;
    }
  }

  let text = sections.join("\n");
  if (maxChars && text.length > maxChars) {
    text = text.slice(0, maxChars).trim() + "\n[TRUNCATED]";
  }
  return text;
}

function resolveTextModules(table: string, textModules?: string[]): string[] {
  const modules = textModules && textModules.length ? textModules : DEFAULT_TEXT_MODULES[table] || DEFAULT_TEXT_MODULES_GENERIC;
  return modules;
}

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function formatOutcomeDefinitions(outcomes: unknown): string {
  if (!outcomes) {
    return "";
  }
  if (!Array.isArray(outcomes)) {
    return normalizeWhitespace(String(outcomes));
  }
  const lines: string[] = [];
  for (const outcome of outcomes) {
    if (!outcome || typeof outcome !== "object") {
      const text = normalizeWhitespace(String(outcome));
      if (text) {
        lines.push(`- Outcome: ${text}`);
      }
      continue;
    }
    const entry = outcome as Record<string, unknown>;
    const title = normalizeWhitespace(String(entry.title || ""));
    lines.push(title ? `- Outcome: ${title}` : "- Outcome: (missing)");
    const timeFrame = normalizeWhitespace(String(entry.time_frame || ""));
    if (timeFrame) {
      lines.push(`  Time frame: ${timeFrame}`);
    }
    const desc = normalizeWhitespace(String(entry.description || ""));
    if (desc) {
      lines.push(`  Description: ${desc}`);
    }
  }
  return lines.join("\n");
}

function selectOutcomes(
  outcomes: unknown,
  endpointName: string,
  maxCandidates: number,
  minScore: number
): Array<{ outcome: Record<string, unknown>; score: number }> {
  if (!endpointName || !Array.isArray(outcomes)) {
    return [];
  }
  const normalizedTarget = normalizeMatchText(endpointName);
  if (!normalizedTarget) {
    return [];
  }
  const scored: Array<{ outcome: Record<string, unknown>; score: number }> = [];
  for (const outcome of outcomes) {
    if (!outcome || typeof outcome !== "object") {
      continue;
    }
    const entry = outcome as Record<string, unknown>;
    const title = normalizeMatchText(String(entry.title || ""));
    const score = matchScore(normalizedTarget, title);
    if (score >= minScore) {
      scored.push({ outcome: entry, score });
    }
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, maxCandidates);
}

function normalizeMatchText(text: string): string {
  return normalizeWhitespace(text.toLowerCase().replace(/[^a-z0-9]+/g, " "));
}

function matchScore(a: string, b: string): number {
  if (!a || !b) {
    return 0;
  }
  const tokensA = new Set(a.split(" "));
  const tokensB = new Set(b.split(" "));
  const overlap = [...tokensA].filter((token) => tokensB.has(token));
  const tokenScore = Math.round((overlap.length / Math.max(tokensA.size, tokensB.size)) * 100);
  const diceScore = Math.round(diceCoefficient(a, b) * 100);
  return Math.max(tokenScore, diceScore);
}

function diceCoefficient(a: string, b: string): number {
  if (a.length < 2 || b.length < 2) {
    return 0;
  }
  const bigrams = (text: string) => {
    const pairs: string[] = [];
    for (let i = 0; i < text.length - 1; i += 1) {
      pairs.push(text.slice(i, i + 2));
    }
    return pairs;
  };
  const pairsA = bigrams(a);
  const pairsB = bigrams(b);
  let matches = 0;
  const counts = new Map<string, number>();
  for (const pair of pairsA) {
    counts.set(pair, (counts.get(pair) || 0) + 1);
  }
  for (const pair of pairsB) {
    const count = counts.get(pair) || 0;
    if (count > 0) {
      matches += 1;
      counts.set(pair, count - 1);
    }
  }
  return (2 * matches) / (pairsA.length + pairsB.length);
}
