import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { Page } from "playwright";

import { askJsonWithRepair } from "../extract/pack_runner.js";
import type { StepLogger } from "../logger.js";
import { loadPromptConfigs, resolvePromptConfig } from "./prompt_config.js";
import { readCsv, writeCsv, CsvRow } from "./csv_io.js";
import { buildPrompt, EvidenceMode } from "./prompt_builder.js";
import { findXmlFile, listPdfFiles, resolveTableCsv } from "./sources.js";
import { uploadSources, setNotebookTitle, setActiveSourcesForTable } from "../notebooklm/flows.js";

export type CtgRunOptions = {
  repoRoot: string;
  nctid: string;
  tables: string[];
  group: string;
  maxChars: number;
  outputRoot?: string;
  logger?: StepLogger;
};

type SourceInfo = {
  name: string;
  type: string;
  path: string;
};

type TableState = {
  table: string;
  header: string[];
  rows: CsvRow[];
  evidenceRecords: Record<string, unknown>[];
  responseRecords: Record<string, unknown>[];
};

const SKIP_FIELDS = new Set(["StudyID"]);
const DEFAULT_MAX_FIELDS = 12;
const TABLE_MAX_FIELDS: Record<string, number> = {
  D_Design: 12,
  D_Pop: 12,
  D_Drug: 12,
  R_Study: 12,
  R_Arm_Study: 12
};
const MAX_PROMPT_CHARS = 3000;
const NO_LLM_TABLES = new Set(["R_Study_Endpoint", "R_Arm_Study_Endpoint"]);

export async function runCtgExtract(page: Page, opts: CtgRunOptions): Promise<string> {
  const configs = loadPromptConfigs(opts.repoRoot);
  const logger = opts.logger;
  const outputRoot = opts.outputRoot
    ? path.isAbsolute(opts.outputRoot)
      ? opts.outputRoot
      : path.join(opts.repoRoot, opts.outputRoot)
    : path.join(opts.repoRoot, "data", "ctg_extract_v2");

  const xmlPath = findXmlFile(opts.repoRoot, opts.nctid);
  const pdfPathsRaw = listPdfFiles(opts.repoRoot, opts.nctid);
  const { valid: pdfPaths, invalid: invalidPdfs } = filterValidPdfFiles(pdfPathsRaw);
  if (invalidPdfs.length) {
    logInfo(logger, "pdf.validation_failed", { invalid: invalidPdfs }, page);
  }

  const tables = normalizeTables(opts.tables);
  await ensureTablesBuilt(opts.repoRoot, opts.nctid, tables, outputRoot, logger, page);
  const outputDir = path.join(outputRoot, opts.nctid, "notebooklm");
  fs.mkdirSync(outputDir, { recursive: true });
  const tableStates = prepareTableStates(opts.repoRoot, opts.nctid, tables, outputRoot);

  const xmlSource = prepareXmlSource(xmlPath, outputDir, opts.nctid);
  const sharedDir = ensureSharedSourcesDir(opts.repoRoot);
  const instructionSource = prepareInstructionSource(sharedDir);
  const llmTables = tables.filter((table) => !NO_LLM_TABLES.has(table));
  const noteSources = prepareFeatureNoteSources(sharedDir, configs, llmTables, opts.group);
  const sources = [xmlSource, ...pdfPaths, instructionSource, ...noteSources];
  if (logger) {
    await logger.step("uploadSources", async () => {
      await uploadSources(page, sources, logger);
    }, page);
    await logger.snap(page, "upload");
  } else {
    await uploadSources(page, sources);
  }
  await setNotebookTitle(page, opts.nctid);
  const sourceLookup = buildSourceLookup(sources);
  const sourceBundle: SourceBundle = {
    allPaths: sources,
    xmlPath: xmlSource,
    pdfPaths,
    instructionPath: instructionSource,
    notePaths: noteSources
  };
  await processTables(
    page,
    { ...opts, tables },
    configs,
    tableStates,
    "sources",
    outputDir,
    sourceBundle,
    sourceLookup,
    logger
  );

  writeOutputs(tableStates, outputDir);
  return outputDir;
}

type SourceBundle = {
  allPaths: string[];
  xmlPath: string;
  pdfPaths: string[];
  instructionPath: string;
  notePaths: string[];
};

function ensureSharedSourcesDir(repoRoot: string): string {
  const sharedDir = path.join(repoRoot, "data", "ctg_extract_v2", "_shared", "notebooklm_sources");
  fs.mkdirSync(sharedDir, { recursive: true });
  return sharedDir;
}

function filterValidPdfFiles(paths: string[]): { valid: string[]; invalid: { path: string; reason: string }[] } {
  const valid: string[] = [];
  const invalid: { path: string; reason: string }[] = [];
  for (const filePath of paths) {
    try {
      const stat = fs.statSync(filePath);
      if (!stat.isFile() || stat.size < 8) {
        invalid.push({ path: filePath, reason: "too_small_or_not_file" });
        continue;
      }
      const fd = fs.openSync(filePath, "r");
      try {
        const header = Buffer.alloc(5);
        fs.readSync(fd, header, 0, header.length, 0);
        if (!header.toString("utf8").startsWith("%PDF-")) {
          invalid.push({ path: filePath, reason: "missing_pdf_header" });
          continue;
        }
        const tailSize = Math.min(2048, stat.size);
        const tail = Buffer.alloc(tailSize);
        fs.readSync(fd, tail, 0, tailSize, Math.max(0, stat.size - tailSize));
        if (!tail.toString("utf8").includes("%%EOF")) {
          invalid.push({ path: filePath, reason: "missing_eof_marker" });
          continue;
        }
        valid.push(filePath);
      } finally {
        fs.closeSync(fd);
      }
    } catch (err) {
      invalid.push({ path: filePath, reason: err instanceof Error ? err.message : String(err) });
    }
  }
  return { valid, invalid };
}

function prepareInstructionSource(sharedDir: string): string {
  const filePath = path.join(sharedDir, "instruction.txt");
  if (fs.existsSync(filePath)) {
    return filePath;
  }
  const lines = [
    "NotebookLM extraction instructions.",
    "",
    "Rules:",
    "- Return ONLY a valid JSON object (no markdown, no commentary).",
    "- If the text is ambiguous or conflicts with the field meaning, return empty.",
    "- If a field is not explicitly stated, return empty.",
    "- Keys MUST match exactly and MUST include ALL keys listed in the template.",
    "- Output must match this schema for every key: {\"value\":\"...\",\"evidence\":\"...\"}.",
    "- Each field MUST include source: {\"name\":\"...\",\"type\":\"...\"}.",
    "- source.name must match the exact NotebookLM source title.",
    "- source.type must be one of: XML, PDF, Feature notes, Instructions.",
    "- Do NOT add extra keys.",
    "- Evidence can be multiple substrings separated by ' | '.",
    "- Keep evidence short (<=240 chars per segment). No ellipses.",
    "- Evidence MUST be exact quotes copied from NotebookLM sources.",
    "- Do NOT cite feature notes or instruction.txt as evidence for values.",
    "- Do NOT use any context outside of NotebookLM sources."
  ];
  fs.writeFileSync(filePath, lines.join("\n") + "\n");
  return filePath;
}

function prepareXmlSource(xmlPath: string, outputDir: string, nctid: string): string {
  const ext = path.extname(xmlPath).toLowerCase();
  if (ext !== ".xml") {
    return xmlPath;
  }
  const sourcesDir = path.join(outputDir, "_sources");
  fs.mkdirSync(sourcesDir, { recursive: true });
  const destPath = path.join(sourcesDir, `${nctid}.txt`);
  const xmlContent = fs.readFileSync(xmlPath, "utf8");
  const textContent = xmlToPlainText(xmlContent);
  fs.writeFileSync(destPath, textContent);
  return destPath;
}

function xmlToPlainText(xml: string): string {
  const decoded = xml
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, "\"")
    .replace(/&apos;/g, "'");
  const stripped = decoded.replace(/<[^>]+>/g, "\n");
  return stripped
    .replace(/\r/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim() + "\n";
}

async function processTables(
  page: Page,
  opts: CtgRunOptions,
  configs: ReturnType<typeof loadPromptConfigs>,
  tableStates: Map<string, TableState>,
  evidenceMode: EvidenceMode,
  outputDir: string,
  sourceBundle: SourceBundle,
  sourceLookup: Map<string, SourceInfo>,
  logger?: StepLogger
): Promise<void> {
  for (const [table, state] of tableStates.entries()) {
    if (NO_LLM_TABLES.has(table)) {
      logInfo(logger, "table.skip_no_llm", { table }, page);
      continue;
    }
    resetStreamingFiles(outputDir, table);
    await setActiveSourcesForTable(page, buildTableSourceSelection(sourceBundle, table), logger);
    const config = applyRTableOverrides(resolvePromptConfig(configs, table, normalizeGroup(opts.group)), table);
    const sourceHints = buildSourceHintsForTable(sourceBundle.allPaths, table);
    const work = async () =>
      processTableRows({
        page,
        opts,
        table,
        rows: state.rows,
        header: state.header,
        config,
        evidenceMode,
        outputDir,
        sourceHints,
        sourceLookup,
        logger
      });
    const tableState = logger ? await logger.step(`table:${table}`, work, page) : await work();
    if (logger) {
      await logger.snap(page, `table_${table}`);
    }
    state.rows = tableState.rows;
    state.evidenceRecords.push(...tableState.evidenceRecords);
    state.responseRecords.push(...tableState.responseRecords);
  }
}

function buildTableSourceSelection(bundle: SourceBundle, table: string): { include: string[]; all: string[] } {
  const all = bundle.allPaths.map((filePath) => path.basename(filePath)).filter(Boolean);
  const includePaths: string[] = [bundle.xmlPath, ...bundle.pdfPaths, bundle.instructionPath];
  const tableNote = bundle.notePaths.find(
    (notePath) => path.basename(notePath).toLowerCase() === `feature_instruction_${table}.txt`.toLowerCase()
  );
  if (tableNote) {
    includePaths.push(tableNote);
  }
  const include = includePaths.map((filePath) => path.basename(filePath)).filter(Boolean);
  return { include, all };
}

function normalizeGroup(group: string): string {
  const key = (group || "").trim().toLowerCase();
  if (!key) {
    return "all";
  }
  const aliases: Record<string, string> = {
    operational: "ops",
    design: "ops",
    stat_reg: "stat",
    reg: "stat",
    statistics: "stat",
    endpoint: "endpoint",
    ep: "endpoint",
    type: "type",
    arm: "type",
    group_type: "type",
    flow: "flow",
    disposition: "flow",
    missing: "flow",
    baseline: "baseline",
    demo: "baseline",
    demographics: "baseline",
    ae: "ae",
    safety: "ae",
    regimen: "regimen",
    dose: "regimen",
    dosing: "regimen",
    all: "all"
  };
  return aliases[key] || key;
}

function applyRTableOverrides(
  config: ReturnType<typeof resolvePromptConfig>,
  table: string
): ReturnType<typeof resolvePromptConfig> {
  if (table === "R_Study") {
    return {
      ...config,
      dedupe_key_fields: ["StudyID"],
      dedupe_fields: config.llm_fields?.length ? [...config.llm_fields] : config.dedupe_fields
    };
  }
  if (table === "R_Arm_Study") {
    return {
      ...config,
      dedupe_key_fields: ["StudyID", "Arm_ID"],
      dedupe_fields: config.llm_fields?.length ? [...config.llm_fields] : config.dedupe_fields
    };
  }
  return config;
}

function prepareTableStates(
  repoRoot: string,
  nctid: string,
  tables: string[],
  outputRoot: string
): Map<string, TableState> {
  const states = new Map<string, TableState>();
  for (const table of tables) {
    const csvPath = resolveTableCsv(repoRoot, nctid, table, outputRoot);
    if (!fs.existsSync(csvPath)) {
      continue;
    }
    const { rows, header } = readCsv(csvPath);
    states.set(table, {
      table,
      header,
      rows,
      evidenceRecords: [],
      responseRecords: []
    });
  }
  return states;
}

function normalizeTables(tables: string[]): string[] {
  const aliases: Record<string, string> = {
    R_Arm: "R_Arm_Study"
  };
  const normalized: string[] = [];
  for (const table of tables) {
    const mapped = aliases[table] || table;
    if (!normalized.includes(mapped)) {
      normalized.push(mapped);
    }
  }
  return normalized;
}

async function ensureTablesBuilt(
  repoRoot: string,
  nctid: string,
  tables: string[],
  outputRoot: string,
  logger?: StepLogger,
  page?: Page
): Promise<void> {
  const missing = tables.filter((table) => !fs.existsSync(resolveTableCsv(repoRoot, nctid, table, outputRoot)));
  if (!missing.length) {
    logInfo(logger, "buildTables.skip", { nctid, tables }, page);
    return;
  }

  const python = process.env.PYTHON || "python";
  const args = [
    "tools/ctg_extract_v2/build_tables.py",
    "--nct-id",
    nctid,
    "--tables",
    missing.join(","),
    "--output-root",
    outputRoot
  ];
  const run = () => {
    logInfo(logger, "buildTables.start", { nctid, tables: missing, cmd: [python, ...args] }, page);
    const result = spawnSync(python, args, {
      cwd: repoRoot,
      encoding: "utf8"
    });
    if (result.status !== 0) {
      const stderr = String(result.stderr || "").trim();
      const stdout = String(result.stdout || "").trim();
      const message = stderr || stdout || `build_tables.py exited ${result.status}`;
      const err = new Error(message);
      (err as { code?: string }).code = "BUILD_TABLES_FAILED";
      throw err;
    }
  };
  if (logger) {
    await logger.step("buildTables", async () => run(), page);
  } else {
    run();
  }
}

function prepareFeatureNoteSources(
  sharedDir: string,
  configs: ReturnType<typeof loadPromptConfigs>,
  tables: string[],
  group: string
): string[] {
  const files: string[] = [];
  for (const table of tables) {
    const config = resolvePromptConfig(configs, table, normalizeGroup(group));
    const notes = config.notes || {};
    const lines: string[] = [
      `Table: ${table}`,
      "",
      "Feature notes (definitions):"
    ];
    if (Object.keys(notes).length === 0) {
      lines.push("- (none)");
    } else {
      for (const [field, note] of Object.entries(notes)) {
        const cleaned = String(note || "").trim();
        if (cleaned) {
          lines.push(`- ${field}: ${cleaned}`);
        } else {
          lines.push(`- ${field}`);
        }
      }
    }
    const filePath = path.join(sharedDir, `feature_instruction_${table}.txt`);
    if (!fs.existsSync(filePath)) {
      fs.writeFileSync(filePath, lines.join("\n") + "\n");
    }
    files.push(filePath);
  }
  return files;
}

function writeOutputs(tableStates: Map<string, TableState>, outputDir: string): void {
  for (const [table, state] of tableStates.entries()) {
    const outCsv = path.join(outputDir, `${table}_notebooklm.csv`);
    writeCsv(outCsv, state.header, state.rows);

    if (NO_LLM_TABLES.has(table)) {
      continue;
    }

    const evidenceJsonl = path.join(outputDir, `${table}_notebooklm_evidence.jsonl`);
    const responseJsonl = path.join(outputDir, `${table}_notebooklm_responses.jsonl`);
    writeJsonl(evidenceJsonl, state.evidenceRecords);
    writeJsonl(responseJsonl, state.responseRecords);

    const evidenceJson = path.join(outputDir, `${table}_notebooklm_evidence.json`);
    fs.writeFileSync(evidenceJson, JSON.stringify(state.evidenceRecords, null, 2) + "\n");
  }
}

function isMissing(value: unknown): boolean {
  if (value === null || value === undefined) {
    return true;
  }
  return !String(value).trim();
}

function stringifyValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean).join("; ");
  }
  return String(value).trim();
}

function emptySource(): SourceInfo {
  return { name: "", type: "", path: "" };
}

function buildSourceHintsForTable(paths: string[], table: string, limit = 20): string[] {
  const tableNote = `feature_instruction_${table}.txt`;
  const names = paths
    .map((filePath) => path.basename(filePath))
    .filter((name) => {
      const lower = name.toLowerCase();
      if (lower.endsWith(".pdf")) {
        return true;
      }
      if (lower === tableNote.toLowerCase()) {
        return true;
      }
      if (lower.endsWith(".txt") && !lower.startsWith("feature_instruction_")) {
        return true;
      }
      return false;
    });
  const unique: string[] = [];
  for (const name of names) {
    if (!unique.includes(name)) {
      unique.push(name);
    }
    if (unique.length >= limit) {
      break;
    }
  }
  return unique;
}

function buildSourceLookup(paths: string[]): Map<string, SourceInfo> {
  const lookup = new Map<string, SourceInfo>();
  for (const filePath of paths) {
    const name = path.basename(filePath);
    const type = inferSourceType(name);
    lookup.set(name.toLowerCase(), { name, type, path: filePath });
  }
  return lookup;
}

function inferSourceType(name: string): string {
  const lower = name.toLowerCase();
  if (lower.endsWith(".pdf")) {
    return "PDF";
  }
  if (lower === "instruction.txt") {
    return "Instructions";
  }
  if (lower.includes("feature_instruction")) {
    return "Feature notes";
  }
  if (lower.endsWith(".txt")) {
    return "XML";
  }
  return "Source";
}

function normalizeSource(raw: unknown, lookup: Map<string, SourceInfo>): SourceInfo {
  let name = "";
  let type = "";
  let pathValue = "";
  if (typeof raw === "string") {
    name = raw.trim();
  } else if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    const record = raw as Record<string, unknown>;
    name = stringifyValue(record.name || record.file || record.title || record.source);
    type = stringifyValue(record.type);
    pathValue = stringifyValue(record.path);
  }
  const mapped = name ? lookup.get(name.toLowerCase()) : undefined;
  if (mapped) {
    return {
      name: name || mapped.name,
      type: type || mapped.type,
      path: pathValue || mapped.path
    };
  }
  return {
    name,
    type,
    path: pathValue
  };
}

async function processTableRows(params: {
  page: Page;
  opts: CtgRunOptions;
  table: string;
  rows: CsvRow[];
  header: string[];
  config: ReturnType<typeof resolvePromptConfig>;
  evidenceMode: EvidenceMode;
  outputDir: string;
  sourceHints: string[];
  sourceLookup: Map<string, SourceInfo>;
  logger?: StepLogger;
}): Promise<{
  rows: CsvRow[];
  evidenceRecords: Record<string, unknown>[];
  responseRecords: Record<string, unknown>[];
}> {
  const { page, opts, table, rows, header, config, evidenceMode, outputDir, sourceHints, sourceLookup, logger } = params;

  const evidenceRecords: Record<string, unknown>[] = [];
  const responseRecords: Record<string, unknown>[] = [];
  const responsePath = path.join(outputDir, `${table}_notebooklm_responses.jsonl`);
  const evidencePath = path.join(outputDir, `${table}_notebooklm_evidence.jsonl`);

  const dedupeCache = new Map<string, Record<string, { value: string; evidence: string; source: SourceInfo }>>();
  const processedKeys = new Set<string>();

  const maxFieldsPerCall = TABLE_MAX_FIELDS[table] ?? DEFAULT_MAX_FIELDS;
  const llmFieldSet = config.llm_fields?.length ? new Set(config.llm_fields) : null;
  const dedupeFields = new Set(config.dedupe_fields || []);

  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    const nctId = String(row.StudyID || row.NCT_No || "").trim();
    if (nctId && nctId !== opts.nctid) {
      continue;
    }

    const dedupeKey = buildDedupeKey(row, opts.nctid, config.dedupe_key_fields || []);
    if (dedupeKey && dedupeFields.size) {
      const bucket = dedupeCache.get(dedupeKey) || {};
      for (const field of dedupeFields) {
        const existingValue = stringifyValue(row[field]);
        if (existingValue && !bucket[field]) {
          bucket[field] = { value: existingValue, evidence: "", source: emptySource() };
        }
      }
      dedupeCache.set(dedupeKey, bucket);
    }

    const missingFields: string[] = [];
    for (const field of header) {
      if (SKIP_FIELDS.has(field)) {
        continue;
      }
      if (llmFieldSet && !llmFieldSet.has(field)) {
        continue;
      }
      if (isMissing(row[field])) {
        missingFields.push(field);
      }
    }

    if (dedupeKey && dedupeFields.size) {
      const cached = dedupeCache.get(dedupeKey) || {};
      const remaining: string[] = [];
      for (const field of missingFields) {
        const hasCached = Object.prototype.hasOwnProperty.call(cached, field);
        if (dedupeFields.has(field) && hasCached) {
          row[field] = cached[field].value;
        } else {
          remaining.push(field);
        }
      }
      missingFields.length = 0;
      missingFields.push(...remaining);
    }

    if (dedupeKey && processedKeys.has(dedupeKey)) {
      if (missingFields.length) {
        logInfo(logger, "prompt.skip_dedupe", { table, row_index: rowIndex, dedupe_key: dedupeKey, fields: missingFields }, page);
        responseRecords.push({
          table,
          nct_id: opts.nctid,
          row_index: rowIndex,
          missing_fields: missingFields,
          evidence_mode: evidenceMode,
          skipped: "dedupe_key_processed"
        });
      }
      continue;
    }

    if (!missingFields.length) {
      continue;
    }

    const batches = splitFields(missingFields, maxFieldsPerCall);

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
      const batchFields = batches[batchIndex];
      const limitedBatches = splitFieldsByPromptLimit(
        batchFields,
        opts.maxChars || MAX_PROMPT_CHARS,
        (fields) => buildPrompt(table, opts.nctid, row, fields, config, evidenceMode, sourceHints)
      );

      for (let subIndex = 0; subIndex < limitedBatches.length; subIndex += 1) {
        const subFields = limitedBatches[subIndex];
        const prompt = buildPrompt(table, opts.nctid, row, subFields, config, evidenceMode, sourceHints);
        logInfo(
          logger,
          "prompt.plan",
          {
            table,
            row_index: rowIndex,
            dedupe_key: dedupeKey,
            batch_index: batchIndex + 1,
            batch_total: batches.length,
            sub_batch_index: subIndex + 1,
            sub_batch_total: limitedBatches.length,
            field_count: subFields.length,
            fields: subFields,
            prompt_len: prompt.length
          },
          page
        );
        let parsed: Record<string, unknown> | null = null;
        let rawText = "";
        try {
          const result = await askJsonWithRepair(
            page,
            prompt,
            4,
            {
              outputDir,
              tag: `${table}_row${rowIndex + 1}_batch${batchIndex + 1}_${subIndex + 1}_${evidenceMode}`
            },
            logger
          );
          parsed = result.json as Record<string, unknown> | null;
          rawText = result.rawText;
        } catch (err) {
          const lastAnswer = (err as { lastAnswer?: string }).lastAnswer;
          const errorRecord = {
            table,
            nct_id: opts.nctid,
            row_index: rowIndex,
            missing_fields: subFields,
            evidence_mode: evidenceMode,
            error: err instanceof Error ? err.message : String(err),
            last_answer: lastAnswer || ""
          };
          responseRecords.push(errorRecord);
          appendJsonl(responsePath, errorRecord);
          continue;
        }

        const responseRecord = {
          table,
          nct_id: opts.nctid,
          row_index: rowIndex,
          missing_fields: subFields,
          evidence_mode: evidenceMode,
          response: rawText
        };
        responseRecords.push(responseRecord);
        appendJsonl(responsePath, responseRecord);

        const fieldOutputs: Record<string, { value: string; evidence: string; source: SourceInfo }> = {};

        for (const field of subFields) {
          const payload = parsed?.[field];
          let value = "";
          let evidence = "";
          let source = emptySource();
          if (payload && typeof payload === "object" && !Array.isArray(payload)) {
            const record = payload as Record<string, unknown>;
            value = stringifyValue(record.value);
            evidence = stringifyValue(record.evidence);
            source = normalizeSource(record.source, sourceLookup);
          } else {
            value = stringifyValue(payload);
          }
          if (value) {
            row[field] = value;
          }
          fieldOutputs[field] = { value, evidence, source };
          if (dedupeKey && dedupeFields.has(field)) {
            const cache = dedupeCache.get(dedupeKey) || {};
            cache[field] = { value, evidence, source };
            dedupeCache.set(dedupeKey, cache);
          }
        }

        const evidenceRecord = {
          table,
          nct_id: opts.nctid,
          row_index: rowIndex,
          fields: fieldOutputs,
          evidence_mode: evidenceMode,
          batch_index: batchIndex + 1,
          batch_total: batches.length,
          sub_batch_index: subIndex + 1,
          sub_batch_total: limitedBatches.length
        };
        evidenceRecords.push(evidenceRecord);
        appendJsonl(evidencePath, evidenceRecord);
      }
    }

    if (dedupeKey) {
      processedKeys.add(dedupeKey);
    }
  }

  return { rows, evidenceRecords, responseRecords };
}

function splitFields(fields: string[], maxFields: number): string[][] {
  if (!maxFields || fields.length <= maxFields) {
    return [fields];
  }
  const out: string[][] = [];
  for (let i = 0; i < fields.length; i += maxFields) {
    out.push(fields.slice(i, i + maxFields));
  }
  return out;
}

function splitFieldsByPromptLimit(
  fields: string[],
  maxChars: number,
  build: (fields: string[]) => string
): string[][] {
  if (!fields.length) {
    return [];
  }
  const batches: string[][] = [];
  let current: string[] = [];
  for (const field of fields) {
    const candidate = [...current, field];
    const prompt = build(candidate);
    if (prompt.length > maxChars && current.length > 0) {
      batches.push(current);
      current = [field];
    } else {
      current = candidate;
    }
  }
  if (current.length) {
    batches.push(current);
  }
  return batches;
}

function buildDedupeKey(row: CsvRow, nctid: string, keyFields: string[]): string | null {
  if (!keyFields || !keyFields.length) {
    return null;
  }
  const values: string[] = [];
  for (const field of keyFields) {
    const normalized = field.toLowerCase();
    if (["nct_id", "study_id", "studyid"].includes(normalized)) {
      values.push(nctid);
      continue;
    }
    const value = stringifyValue(row[field]);
    values.push(value);
  }
  if (!values.some((value) => value)) {
    return null;
  }
  return values.join("|");
}

function logInfo(
  logger: StepLogger | undefined,
  step: string,
  data: Record<string, unknown>,
  page?: Page
): void {
  if (!logger) {
    return;
  }
  logger.log({
    ts: new Date().toISOString(),
    step,
    phase: "info",
    url: page?.url(),
    data
  });
}

function writeJsonl(filePath: string, entries: Record<string, unknown>[]): void {
  if (!entries.length) {
    fs.writeFileSync(filePath, "");
    return;
  }
  const lines = entries.map((entry) => JSON.stringify(entry, null, 0));
  fs.writeFileSync(filePath, lines.join("\n") + "\n");
}

function appendJsonl(filePath: string, entry: Record<string, unknown>): void {
  fs.appendFileSync(filePath, JSON.stringify(entry, null, 0) + "\n");
}

function resetStreamingFiles(outputDir: string, table: string): void {
  fs.writeFileSync(path.join(outputDir, `${table}_notebooklm_responses.jsonl`), "");
  fs.writeFileSync(path.join(outputDir, `${table}_notebooklm_evidence.jsonl`), "");
}
