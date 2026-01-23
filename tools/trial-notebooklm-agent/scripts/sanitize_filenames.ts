#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

function sanitizeName(name: string): string {
  return name
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function sanitizeDir(dir: string): void {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const oldPath = path.join(dir, entry.name);
    const cleanName = sanitizeName(entry.name);
    const newPath = path.join(dir, cleanName || entry.name);
    if (oldPath !== newPath) {
      fs.renameSync(oldPath, newPath);
    }
    if (entry.isDirectory()) {
      sanitizeDir(newPath);
    }
  }
}

function main(): number {
  const target = process.argv[2];
  if (!target) {
    console.error("Usage: sanitize_filenames.ts <dir>");
    return 1;
  }
  const abs = path.resolve(target);
  if (!fs.existsSync(abs) || !fs.statSync(abs).isDirectory()) {
    console.error("Target is not a directory: " + abs);
    return 1;
  }
  sanitizeDir(abs);
  return 0;
}

process.exit(main());
