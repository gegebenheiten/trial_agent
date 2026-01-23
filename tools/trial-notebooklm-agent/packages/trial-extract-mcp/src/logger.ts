import fs from "node:fs";
import path from "node:path";
import type { BrowserContext, Page } from "playwright";

type StepPhase = "start" | "end" | "error" | "snap" | "info";

export type StepRecord = {
  ts: string;
  step: string;
  phase: StepPhase;
  ms?: number;
  url?: string;
  error_code?: string;
  error_message?: string;
  data?: Record<string, unknown>;
};

export type SnapOptions = {
  html?: boolean;
};

export type RunMeta = {
  status: "running" | "success" | "error";
  started_at: string;
  ended_at: string;
  duration_ms: number;
  output_dir: string;
  logs_dir: string;
  steps_log_path: string;
  console_log_path: string;
  network_log_path: string;
  trace_path?: string;
  fail_html_path?: string;
  error_code?: string;
  error_message?: string;
};

export class StepLogger {
  readonly outputDir: string;
  readonly logsDir: string;
  readonly rawDir: string;
  readonly stepsPath: string;
  readonly metaPath: string;
  readonly consolePath: string;
  readonly networkPath: string;
  tracePath?: string;
  failHtmlPath?: string;
  private snapIndex = 0;
  private readonly startedAt: number;
  private readonly startedAtIso: string;
  private finalized = false;
  private lastMeta: RunMeta | null = null;
  private readonly stepsOnly: boolean;

  constructor(outputDir: string, logDirName = "logs") {
    this.outputDir = outputDir;
    this.logsDir = path.join(outputDir, logDirName);
    this.rawDir = path.join(this.logsDir, "raw");
    this.stepsPath = path.join(this.logsDir, "steps.jsonl");
    this.metaPath = path.join(this.logsDir, "run_meta.json");
    this.consolePath = path.join(this.rawDir, "console.log");
    this.networkPath = path.join(this.rawDir, "network.log");
    this.stepsOnly = (process.env.TRIAL_LOG_STEPS_ONLY || "true").toLowerCase() === "true";
    fs.mkdirSync(this.logsDir, { recursive: true });
    fs.writeFileSync(this.stepsPath, "");
    this.startedAt = Date.now();
    this.startedAtIso = new Date(this.startedAt).toISOString();
    if (!this.stepsOnly) {
      fs.mkdirSync(this.rawDir, { recursive: true });
      this.writeRunningMeta();
    }
  }

  log(record: StepRecord): void {
    const line = JSON.stringify(record);
    fs.appendFileSync(this.stepsPath, `${line}\n`);
  }

  async step<T>(name: string, fn: () => Promise<T>, page?: Page): Promise<T> {
    const started = Date.now();
    this.log({ ts: new Date(started).toISOString(), step: name, phase: "start", url: page?.url() });
    try {
      const result = await fn();
      const ended = Date.now();
      this.log({
        ts: new Date(ended).toISOString(),
        step: name,
        phase: "end",
        ms: ended - started,
        url: page?.url()
      });
      return result;
    } catch (err) {
      const ended = Date.now();
      const { code, message } = normalizeError(err);
      this.log({
        ts: new Date(ended).toISOString(),
        step: name,
        phase: "error",
        ms: ended - started,
        url: page?.url(),
        error_code: code,
        error_message: message
      });
      throw err;
    }
  }

  async snap(page: Page, label: string, options?: SnapOptions): Promise<{ htmlPath?: string }> {
    if (this.stepsOnly) {
      this.log({
        ts: new Date().toISOString(),
        step: label,
        phase: "snap",
        url: page.url(),
        data: { screenshot: false }
      });
      return {};
    }
    const safe = label.replace(/[^a-zA-Z0-9_-]+/g, "_").slice(0, 80);
    const index = String(++this.snapIndex).padStart(2, "0");
    let htmlPath: string | undefined;
    let htmlError: string | undefined;
    if (options?.html) {
      htmlPath = path.join(this.rawDir, `page_${index}_${safe}.html`);
      try {
        const html = await page.content();
        fs.writeFileSync(htmlPath, html);
      } catch (err) {
        htmlError = err instanceof Error ? err.message : String(err);
        htmlPath = undefined;
      }
    }
    if (label === "fail") {
      this.failHtmlPath = htmlPath;
    }
    this.log({
      ts: new Date().toISOString(),
      step: label,
      phase: "snap",
      url: page.url(),
      data: { html_path: htmlPath, html_error: htmlError, screenshot: false }
    });
    return { htmlPath };
  }

  attachPage(page: Page): void {
    if (this.stepsOnly) {
      return;
    }
    page.on("console", (msg) => {
      const line = `[${new Date().toISOString()}] ${msg.type().toUpperCase()} ${msg.text()}`;
      fs.appendFileSync(this.consolePath, `${line}\n`);
    });
    page.on("pageerror", (err) => {
      const line = `[${new Date().toISOString()}] PAGEERROR ${err.message}`;
      fs.appendFileSync(this.consolePath, `${line}\n`);
    });
    page.on("requestfailed", (req) => {
      const failure = req.failure();
      const line = `[${new Date().toISOString()}] ${req.method()} ${req.url()} ${failure?.errorText || ""}`;
      fs.appendFileSync(this.networkPath, `${line}\n`);
    });
  }

  async startTrace(ctx: BrowserContext, tracePath: string): Promise<void> {
    if (this.stepsOnly) {
      return;
    }
    this.tracePath = tracePath;
    await ctx.tracing.start({ screenshots: true, snapshots: true, sources: true });
  }

  async stopTrace(ctx: BrowserContext): Promise<void> {
    if (this.stepsOnly || !this.tracePath) {
      return;
    }
    await ctx.tracing.stop({ path: this.tracePath });
  }

  finalize(status: "success" | "error", err?: unknown): RunMeta {
    if (this.finalized && this.lastMeta) {
      return this.lastMeta;
    }
    const ended = Date.now();
    const meta: RunMeta = {
      status,
      started_at: this.startedAtIso,
      ended_at: new Date(ended).toISOString(),
      duration_ms: ended - this.startedAt,
      output_dir: this.outputDir,
      logs_dir: this.logsDir,
      steps_log_path: this.stepsPath,
      console_log_path: this.consolePath,
      network_log_path: this.networkPath,
      trace_path: this.tracePath,
      fail_html_path: this.failHtmlPath
    };
    if (err) {
      const { code, message } = normalizeError(err);
      meta.error_code = code;
      meta.error_message = message;
    }
    if (!this.stepsOnly) {
      fs.writeFileSync(this.metaPath, `${JSON.stringify(meta, null, 2)}\n`);
    }
    this.finalized = true;
    this.lastMeta = meta;
    return meta;
  }

  private writeRunningMeta(): void {
    if (this.stepsOnly) {
      return;
    }
    const meta: RunMeta = {
      status: "running",
      started_at: this.startedAtIso,
      ended_at: this.startedAtIso,
      duration_ms: 0,
      output_dir: this.outputDir,
      logs_dir: this.logsDir,
      steps_log_path: this.stepsPath,
      console_log_path: this.consolePath,
      network_log_path: this.networkPath,
      trace_path: this.tracePath
    };
    fs.writeFileSync(this.metaPath, `${JSON.stringify(meta, null, 2)}\n`);
  }
}

export function normalizeError(err: unknown): { code: string; message: string } {
  if (err && typeof err === "object") {
    const record = err as Record<string, unknown>;
    const code = typeof record.code === "string" ? record.code : typeof record.name === "string" ? record.name : "ERROR";
    const message = typeof record.message === "string" ? record.message : String(err);
    return { code, message };
  }
  return { code: "ERROR", message: String(err) };
}
