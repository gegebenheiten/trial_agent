import fs from "node:fs";
import { parse } from "csv-parse/sync";
import { stringify } from "csv-stringify/sync";

export type CsvRow = Record<string, string>;

export function readCsv(filePath: string): { rows: CsvRow[]; header: string[] } {
  const raw = fs.readFileSync(filePath, "utf8");
  const records = parse(raw, {
    columns: true,
    skip_empty_lines: true
  }) as CsvRow[];
  const header = records.length ? Object.keys(records[0]) : extractHeader(raw);
  return { rows: records, header };
}

export function writeCsv(filePath: string, header: string[], rows: CsvRow[]): void {
  const output = stringify(rows, {
    header: true,
    columns: header
  });
  fs.writeFileSync(filePath, output);
}

function extractHeader(raw: string): string[] {
  const firstLine = raw.split(/\r?\n/)[0] || "";
  return firstLine.split(",").map((value) => value.trim());
}
