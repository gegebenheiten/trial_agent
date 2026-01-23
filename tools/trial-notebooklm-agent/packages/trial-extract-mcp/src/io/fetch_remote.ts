import fs from "node:fs";
import path from "node:path";
import { RunArgs, FileList } from "../types.js";
import { AppConfig } from "../config.js";

function walkFiles(dir: string, out: string[]): void {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkFiles(full, out);
    } else if (entry.isFile()) {
      out.push(full);
    }
  }
}

export async function fetchRemoteIfNeeded(args: RunArgs, cfg: AppConfig): Promise<FileList> {
  if (args.remote?.enabled) {
    throw new Error("Remote fetch is not implemented in this scaffold. Add SSH/rsync logic here.");
  }

  const stagingRoot = args.localStagingDir;
  const nctidRoot = path.join(stagingRoot, args.nctid);
  const sourcesRoot = fs.existsSync(path.join(nctidRoot, "sources"))
    ? path.join(nctidRoot, "sources")
    : nctidRoot;

  if (!fs.existsSync(sourcesRoot)) {
    throw new Error(`Sources directory not found: ${sourcesRoot}`);
  }

  const filePaths: string[] = [];
  walkFiles(sourcesRoot, filePaths);

  if (filePaths.length === 0) {
    throw new Error(`No files found under: ${sourcesRoot}`);
  }

  return { root: sourcesRoot, filePaths };
}
