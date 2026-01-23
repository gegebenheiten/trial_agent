import fs from "node:fs";
import path from "node:path";
import { buildManifest } from "./file_manifest.js";

export type StoreMeta = {
  files?: string[];
  extra?: Record<string, unknown>;
};

export function storeAll(
  outputDir: string,
  nctid: string,
  jsonByPack: Record<string, unknown>,
  rawByPack: Record<string, string>,
  meta: StoreMeta
): void {
  const baseDir = path.join(outputDir, nctid);
  const rawDir = path.join(baseDir, "raw");
  const jsonDir = path.join(baseDir, "json");

  fs.mkdirSync(rawDir, { recursive: true });
  fs.mkdirSync(jsonDir, { recursive: true });

  for (const [pack, rawText] of Object.entries(rawByPack)) {
    const rawPath = path.join(rawDir, `notebooklm_answer_${pack}.txt`);
    fs.writeFileSync(rawPath, rawText, "utf8");
  }

  for (const [pack, json] of Object.entries(jsonByPack)) {
    const jsonPath = path.join(jsonDir, `${pack}.json`);
    fs.writeFileSync(jsonPath, JSON.stringify(json, null, 2) + "\n", "utf8");
  }

  const manifest = meta.files ? buildManifest(meta.files) : undefined;
  const metaOut = {
    nctid,
    timestamp: new Date().toISOString(),
    packs: Object.keys(jsonByPack),
    manifest,
    extra: meta.extra || {}
  };

  fs.writeFileSync(path.join(baseDir, "meta.json"), JSON.stringify(metaOut, null, 2) + "\n", "utf8");
}
