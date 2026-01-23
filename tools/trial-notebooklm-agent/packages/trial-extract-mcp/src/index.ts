import fs from "node:fs";
import path from "node:path";
import { parse } from "csv-parse/sync";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

import { cfgFromEnv, resolvePath } from "./config.js";
import { launchContext } from "./notebooklm/browser.js";
import {
  gotoNotebookLM,
  ensureAuth,
  openOrCreateNotebook,
  uploadSources,
  setNotebookTitle,
  deleteNotebook
} from "./notebooklm/flows.js";
import { readPromptPack, readSystemRules, buildFinalPrompt } from "./extract/build_prompt.js";
import { askJsonWithRepair } from "./extract/pack_runner.js";
import { storeAll } from "./io/store.js";
import { fetchRemoteIfNeeded } from "./io/fetch_remote.js";
import { RunArgs } from "./types.js";
import { runCtgExtract } from "./ctg_extract/pipeline.js";
import { StepLogger, normalizeError } from "./logger.js";

const cfg = cfgFromEnv();
const promptsDir = path.join(cfg.agentRoot, "prompts");

const server = new McpServer({ name: "trial-extract-mcp", version: "0.1.0" });

type BatchResult = {
  nctid: string;
  status: "success" | "error" | "skipped";
  output_dir?: string;
  error_code?: string;
  error_message?: string;
  logs_dir?: string;
};

function normalizeNctId(value: unknown): string {
  const match = String(value || "").match(/NCT\d+/i);
  return match ? match[0].toUpperCase() : "";
}

function detectNctIdColumn(headers: string[]): string {
  const normalized = headers.map((h) => h.trim().toLowerCase());
  const candidates = ["nctid", "nct_id", "nct", "trial_id"];
  for (const candidate of candidates) {
    const idx = normalized.indexOf(candidate);
    if (idx >= 0) {
      return headers[idx];
    }
  }
  if (headers.length === 1) {
    return headers[0];
  }
  throw new Error(`NCT ID column not found. Headers: ${headers.join(", ")}`);
}

function extractNctIdsFromRaw(raw: string, limit = 0): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const match of raw.matchAll(/NCT\d+/gi)) {
    const id = match[0].toUpperCase();
    if (seen.has(id)) {
      continue;
    }
    seen.add(id);
    ids.push(id);
    if (limit > 0 && ids.length >= limit) {
      break;
    }
  }
  return ids;
}

function loadNctIdsFromCsv(csvPath: string, nctIdCol?: string, limit = 0): string[] {
  const raw = fs.readFileSync(csvPath, "utf8");
  let records: Record<string, string>[] = [];
  try {
    records = parse(raw, { columns: true, skip_empty_lines: true, trim: true });
  } catch {
    return extractNctIdsFromRaw(raw, limit);
  }
  if (!records.length) {
    return extractNctIdsFromRaw(raw, limit);
  }
  const headers = Object.keys(records[0]);
  const col = nctIdCol || detectNctIdColumn(headers);
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const record of records) {
    let id = normalizeNctId(record[col]);
    if (!id) {
      for (const value of Object.values(record)) {
        id = normalizeNctId(value);
        if (id) {
          break;
        }
      }
    }
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    ids.push(id);
    if (limit > 0 && ids.length >= limit) {
      break;
    }
  }
  if (!ids.length) {
    return extractNctIdsFromRaw(raw, limit);
  }
  return ids;
}

function resolveMissingTables(
  repoRoot: string,
  nctid: string,
  runDir: string,
  tables: string[],
  resumeRoot?: string
): string[] {
  const roots: string[] = [];
  const resolvedResume = resumeRoot ? resolvePath(repoRoot, resumeRoot) : "";
  if (resolvedResume) {
    const altDir = path.join(resolvedResume, nctid, "notebooklm");
    if (fs.existsSync(altDir)) {
      roots.push(altDir);
    }
  }
  roots.push(runDir);
  const missing: string[] = [];
  for (const table of tables) {
    let done = false;
    for (const root of roots) {
      const outCsv = path.join(root, `${table}_notebooklm.csv`);
      if (!fs.existsSync(outCsv)) {
        continue;
      }
      const stat = fs.statSync(outCsv);
      if (stat.size >= 10) {
        done = true;
        break;
      }
    }
    if (!done) {
      missing.push(table);
    }
  }
  return missing;
}

server.tool(
  "setup_auth",
  {
    profileDir: z.string().optional().describe("Chrome user data dir for persistent auth")
  },
  async ({ profileDir }) => {
    const resolvedProfile = resolvePath(cfg.repoRoot, profileDir || cfg.defaultProfileDir);
    const { ctx, page } = await launchContext(resolvedProfile, { headless: false });
    await gotoNotebookLM(page);
    return {
      content: [
        {
          type: "text",
          text: "Opened NotebookLM. Complete Google login in the browser window, then close it."
        }
      ]
    };
  }
);

server.tool(
  "run",
  {
    nctid: z.string(),
    remote: z
      .object({
        enabled: z.boolean().default(false),
        sshHost: z.string().optional(),
        sshUser: z.string().optional(),
        remoteDir: z.string().optional()
      })
      .optional(),
    localProfileDir: z.string().default("data/.chrome-profile"),
    localStagingDir: z.string().default("data/staging"),
    outputDir: z.string().default("data/outputs"),
    promptPacks: z.array(z.string()).default(["stat_reg", "design", "endpoints"])
  },
  async (args) => {
    const resolvedArgs: RunArgs = {
      ...args,
      localProfileDir: resolvePath(cfg.repoRoot, args.localProfileDir),
      localStagingDir: resolvePath(cfg.repoRoot, args.localStagingDir),
      outputDir: resolvePath(cfg.repoRoot, args.outputDir)
    };

    const runDir = path.join(resolvedArgs.outputDir, resolvedArgs.nctid);
    const logger = new StepLogger(runDir);
    const { ctx, page } = await launchContext(resolvedArgs.localProfileDir, { headless: cfg.headless });
    logger.attachPage(page);
    const traceEnabled = (process.env.TRIAL_TRACE || "false").toLowerCase() === "true";
    if (traceEnabled) {
      const tracePath = path.join(logger.rawDir, "playwright_trace.zip");
      await logger.startTrace(ctx, tracePath);
    }

    let error: unknown = null;
    let meta;
    let results: Record<string, unknown> = {};
    let raw: Record<string, string> = {};
    let sources: Awaited<ReturnType<typeof fetchRemoteIfNeeded>> | null = null;

    try {
      await logger.step("gotoNotebookLM", async () => {
        await gotoNotebookLM(page);
      }, page);
      await logger.snap(page, "goto");

      await logger.step("ensureAuth", async () => {
        await ensureAuth(page);
      }, page);
      await logger.snap(page, "auth");

      sources = await logger.step("fetchRemoteIfNeeded", async () => fetchRemoteIfNeeded(resolvedArgs, cfg), page);
      logger.log({
        ts: new Date().toISOString(),
        step: "fetchRemoteIfNeeded",
        phase: "info",
        data: { file_count: sources.filePaths.length }
      });

      await logger.step("openOrCreateNotebook", async () => {
        await openOrCreateNotebook(page, resolvedArgs.nctid);
      }, page);
      await logger.snap(page, "notebook");

      await logger.step("uploadSources", async () => {
        await uploadSources(page, sources?.filePaths || [], logger);
      }, page);
      await logger.snap(page, "upload");

      await logger.step("setNotebookTitle", async () => {
        await setNotebookTitle(page, resolvedArgs.nctid);
      }, page);
      await logger.snap(page, "title");

      const systemRules = readSystemRules(promptsDir);
      for (const packName of resolvedArgs.promptPacks) {
        await logger.step(`pack:${packName}`, async () => {
          const pack = readPromptPack(packName, promptsDir);
          const prompt = buildFinalPrompt(resolvedArgs.nctid, pack, systemRules);
          const { json, rawText } = await askJsonWithRepair(
            page,
            prompt,
            4,
            {
              outputDir: resolvedArgs.outputDir,
              tag: `pack_${packName}`
            },
            logger
          );
          results[packName] = json;
          raw[packName] = rawText;
        }, page);
        await logger.snap(page, `answer_${packName}`);
      }

      await logger.step("storeAll", async () => {
        storeAll(resolvedArgs.outputDir, resolvedArgs.nctid, results, raw, { files: sources?.filePaths || [] });
      }, page);
    } catch (err) {
      error = err;
      await logger.snap(page, "fail", { html: true });
    } finally {
      if (traceEnabled) {
        await logger.stopTrace(ctx).catch(() => {});
      }
      const keepOpen = process.env.TRIAL_KEEP_OPEN === "true";
      if (keepOpen) {
        const keepMs = Number(process.env.TRIAL_KEEP_OPEN_MS || 300_000);
        await page.waitForTimeout(keepMs);
      } else {
        await ctx.close();
      }
      meta = logger.finalize(error ? "error" : "success", error || undefined);
    }

    if (error) {
      const errMeta = normalizeError(error);
      return {
        content: [
          {
            type: "text",
            text: `Failed: ${resolvedArgs.nctid}. ${errMeta.code}: ${errMeta.message}`
          },
          {
            type: "text",
            text: JSON.stringify(meta, null, 2)
          }
        ]
      };
    }

    return {
      content: [
        {
          type: "text",
          text: `Done: ${resolvedArgs.nctid}. Packs: ${resolvedArgs.promptPacks.join(", ")}`
        },
        {
          type: "text",
          text: JSON.stringify(meta, null, 2)
        }
      ]
    };
  }
);

server.tool(
  "run_ctg_extract_v2",
  {
    nctid: z.string(),
    tables: z
      .array(z.string())
      .default([
        "D_Design",
        "D_Pop",
        "D_Drug",
        "R_Study",
        "R_Arm_Study"
      ]),
    group: z.string().default("all"),
    maxChars: z.number().default(3000),
    localProfileDir: z.string().default("data/.chrome-profile")
  },
  async (args) => {
    const resolvedProfile = resolvePath(cfg.repoRoot, args.localProfileDir);
    const runDir = path.join(cfg.repoRoot, "data", "ctg_extract_v2", args.nctid, "notebooklm");
    const logger = new StepLogger(runDir);
    const { ctx, page } = await launchContext(resolvedProfile, { headless: cfg.headless });
    logger.attachPage(page);
    const traceEnabled = (process.env.TRIAL_TRACE || "false").toLowerCase() === "true";
    if (traceEnabled) {
      const tracePath = path.join(logger.rawDir, "playwright_trace.zip");
      await logger.startTrace(ctx, tracePath);
    }

    let error: unknown = null;
    let meta;
    let outputDir = "";
    try {
      await logger.step("gotoNotebookLM", async () => {
        await gotoNotebookLM(page);
      }, page);
      await logger.snap(page, "goto");

      await logger.step("ensureAuth", async () => {
        await ensureAuth(page);
      }, page);
      await logger.snap(page, "auth");

      await logger.step("openOrCreateNotebook", async () => {
        await openOrCreateNotebook(page, args.nctid);
      }, page);
      await logger.snap(page, "notebook");

      outputDir = await runCtgExtract(page, {
        repoRoot: cfg.repoRoot,
        nctid: args.nctid,
        tables: args.tables,
        group: args.group,
        maxChars: args.maxChars,
        logger
      });
    } catch (err) {
      error = err;
      await logger.snap(page, "fail", { html: true });
    } finally {
      if (traceEnabled) {
        await logger.stopTrace(ctx).catch(() => {});
      }
      if (process.env.TRIAL_KEEP_OPEN === "true") {
        const keepMs = Number(process.env.TRIAL_KEEP_OPEN_MS || 300_000);
        await page.waitForTimeout(keepMs);
      } else {
        await ctx.close();
      }
      meta = logger.finalize(error ? "error" : "success", error || undefined);
    }

    if (error) {
      const errMeta = normalizeError(error);
      return {
        content: [
          {
            type: "text",
            text: `CTG extract failed for ${args.nctid}. ${errMeta.code}: ${errMeta.message}`
          },
          {
            type: "text",
            text: JSON.stringify(meta, null, 2)
          }
        ]
      };
    }

    return {
      content: [
        {
          type: "text",
          text: `CTG extract complete for ${args.nctid}. Output: ${outputDir}`
        },
        {
          type: "text",
          text: JSON.stringify(meta, null, 2)
        }
      ]
    };
  }
);

server.tool(
  "run_ctg_extract_v2_batch",
  {
    nctCsv: z.string(),
    nctIdCol: z.string().optional(),
    limit: z.number().default(2),
    parallel: z.number().int().min(1).max(4).default(1),
    resume: z.boolean().default(true),
    resumeRoot: z.string().optional(),
    outputRoot: z.string().optional(),
    tables: z
      .array(z.string())
      .default([
        "D_Design",
        "D_Pop",
        "D_Drug",
        "R_Study",
        "R_Arm_Study"
      ]),
    group: z.string().default("all"),
    maxChars: z.number().default(3000),
    localProfileDir: z.string().default("data/.chrome-profile"),
    deleteNotebook: z.boolean().default(true)
  },
  async (args) => {
    const resolvedProfile = resolvePath(cfg.repoRoot, args.localProfileDir);
    const csvPath = resolvePath(cfg.repoRoot, args.nctCsv);
    const outputRoot = args.outputRoot ? resolvePath(cfg.repoRoot, args.outputRoot) : path.join(cfg.repoRoot, "data", "ctg_extract_v2");
    const nctIds = loadNctIdsFromCsv(csvPath, args.nctIdCol, args.limit);
    const results: BatchResult[] = [];
    const noLlmTables = new Set(["R_Study_Endpoint", "R_Arm_Study_Endpoint"]);
    const batchDir = path.join(
      cfg.repoRoot,
      "data",
      "ctg_extract_v2",
      "_batch",
      new Date().toISOString().replace(/[:.]/g, "-")
    );
    fs.mkdirSync(batchDir, { recursive: true });
    const batchLogPath = path.join(batchDir, "batch_results.jsonl");
    fs.writeFileSync(
      batchLogPath,
      `${JSON.stringify({
        ts: new Date().toISOString(),
        type: "batch_start",
        csv: csvPath,
        nct_count: nctIds.length,
        sample: nctIds.slice(0, 5),
        parallel: args.parallel,
        resume: args.resume,
        resume_root: args.resumeRoot || null,
        output_root: outputRoot
      })}\n`
    );
    const { ctx } = await launchContext(resolvedProfile, { headless: cfg.headless });
    const traceEnabled = (process.env.TRIAL_TRACE || "false").toLowerCase() === "true";

    try {
      if (!nctIds.length) {
        const raw = fs.readFileSync(csvPath, "utf8");
        const preview = raw.slice(0, 200);
        const matches = Array.from(raw.matchAll(/NCT\\d+/gi))
          .slice(0, 5)
          .map((m) => m[0]);
        fs.appendFileSync(
          batchLogPath,
          `${JSON.stringify({
            ts: new Date().toISOString(),
            type: "batch_debug",
            csv: csvPath,
            raw_len: raw.length,
            preview,
            matches
          })}\n`
        );
        return {
          content: [
            {
              type: "text",
              text: `CTG batch aborted: no NCT IDs found in ${csvPath}.`
            },
            {
              type: "text",
              text: JSON.stringify({ batch_log: batchLogPath, results: [] }, null, 2)
            }
          ]
        };
      }
      const total = nctIds.length;
      let cursor = 0;
      const parallel = Math.max(1, Math.min(4, Math.floor(args.parallel || 1)));
      const workers = Array.from({ length: parallel }, (_, workerId) =>
        (async () => {
          while (true) {
            const index = cursor;
            cursor += 1;
            if (index >= total) {
              break;
            }
            const nctid = nctIds[index];
            const runDir = path.join(outputRoot, nctid, "notebooklm");
            const existingTables = args.resume
              ? resolveMissingTables(cfg.repoRoot, nctid, runDir, args.tables, args.resumeRoot)
              : args.tables;
            const tablesToRun = existingTables.filter((table) => !noLlmTables.has(table));
            if (!tablesToRun.length) {
              const entry: BatchResult = {
                nctid,
                status: "skipped",
                output_dir: runDir,
                logs_dir: path.join(runDir, "logs")
              };
              results.push(entry);
              fs.appendFileSync(batchLogPath, `${JSON.stringify(entry)}\n`);
              continue;
            }

            const logger = new StepLogger(runDir);
            const page = await ctx.newPage();
            logger.attachPage(page);
            if (traceEnabled) {
              const tracePath = path.join(logger.rawDir, "playwright_trace.zip");
              await logger.startTrace(ctx, tracePath);
            }

            let error: unknown = null;
            let outputDir = "";
            let meta;
            try {
              await logger.step("gotoNotebookLM", async () => {
                await gotoNotebookLM(page);
              }, page);
              await logger.step("ensureAuth", async () => {
                await ensureAuth(page);
              }, page);
              await logger.step("openOrCreateNotebook", async () => {
                await openOrCreateNotebook(page, nctid);
              }, page);
              outputDir = await runCtgExtract(page, {
                repoRoot: cfg.repoRoot,
                nctid,
                tables: tablesToRun,
                group: args.group,
                maxChars: args.maxChars,
                outputRoot,
                logger
              });
              if (args.deleteNotebook) {
                await logger.step("deleteNotebook", async () => {
                  await deleteNotebook(page, nctid, logger);
                }, page);
              }
            } catch (err) {
              error = err;
              await logger.snap(page, "fail", { html: true });
            } finally {
              if (traceEnabled) {
                await logger.stopTrace(ctx).catch(() => {});
              }
              await page.close().catch(() => {});
              meta = logger.finalize(error ? "error" : "success", error || undefined);
            }

            const entry: BatchResult = {
              nctid,
              status: error ? "error" : "success",
              output_dir: outputDir || undefined,
              logs_dir: meta.logs_dir
            };
            if (error) {
              const errMeta = normalizeError(error);
              entry.error_code = errMeta.code;
              entry.error_message = errMeta.message;
            }
            results.push(entry);
            fs.appendFileSync(batchLogPath, `${JSON.stringify(entry)}\n`);
          }
        })()
      );
      await Promise.all(workers);
    } finally {
      if (process.env.TRIAL_KEEP_OPEN === "true") {
        const keepMs = Number(process.env.TRIAL_KEEP_OPEN_MS || 300_000);
        const page = await ctx.newPage();
        await page.waitForTimeout(keepMs);
        await page.close().catch(() => {});
      }
      await ctx.close().catch(() => {});
    }

    return {
      content: [
        {
          type: "text",
          text: `CTG batch complete. Count: ${results.length}/${nctIds.length}. Log: ${batchLogPath}`
        },
        {
          type: "text",
          text: JSON.stringify({ batch_log: batchLogPath, results }, null, 2)
        }
      ]
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
