import fs from "node:fs";
import crypto from "node:crypto";

export type FileManifestEntry = {
  path: string;
  size: number;
  sha256: string;
};

export function buildManifest(filePaths: string[]): FileManifestEntry[] {
  return filePaths.map((filePath) => {
    const buf = fs.readFileSync(filePath);
    const hash = crypto.createHash("sha256").update(buf).digest("hex");
    return {
      path: filePath,
      size: buf.length,
      sha256: hash
    };
  });
}
