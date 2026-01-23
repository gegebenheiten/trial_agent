import fs from "node:fs";
import path from "node:path";

export function findXmlFile(repoRoot: string, nctid: string): string {
  const root = path.join(repoRoot, "data", "raw_data");
  const target = `${nctid}.xml`;
  const found = findFileRecursive(root, target);
  if (!found) {
    throw new Error(`XML not found for ${nctid} under ${root}`);
  }
  return found;
}

export function listPdfFiles(repoRoot: string, nctid: string): string[] {
  const dir = path.join(repoRoot, "data", "ctgov_documents", nctid);
  if (!fs.existsSync(dir)) {
    return [];
  }
  return fs
    .readdirSync(dir)
    .filter((name) => name.toLowerCase().endsWith(".pdf"))
    .map((name) => path.join(dir, name));
}

export function resolveTableCsv(repoRoot: string, nctid: string, table: string, outputRoot?: string): string {
  const root = outputRoot
    ? path.isAbsolute(outputRoot)
      ? outputRoot
      : path.join(repoRoot, outputRoot)
    : path.join(repoRoot, "data", "ctg_extract_v2");
  return path.join(root, nctid, `${table}.csv`);
}

function findFileRecursive(root: string, filename: string): string | null {
  const entries = fs.readdirSync(root, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) {
      const found = findFileRecursive(full, filename);
      if (found) {
        return found;
      }
    } else if (entry.isFile() && entry.name === filename) {
      return full;
    }
  }
  return null;
}
