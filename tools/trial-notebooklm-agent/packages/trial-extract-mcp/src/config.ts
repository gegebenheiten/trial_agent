import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";

dotenv.config();

export type AppConfig = {
  repoRoot: string;
  agentRoot: string;
  defaultProfileDir: string;
  defaultStagingDir: string;
  defaultOutputDir: string;
  headless: boolean;
  sshHost?: string;
  sshUser?: string;
  sshKeyPath?: string;
  sshPort?: number;
  remoteDir?: string;
};

export function cfgFromEnv(): AppConfig {
  const repoRoot = resolveRepoRoot();
  const headless = (process.env.TRIAL_HEADLESS || "true").toLowerCase() === "true";
  const agentRoot = path.join(repoRoot, "tools", "trial-notebooklm-agent");

  return {
    repoRoot,
    agentRoot,
    defaultProfileDir: process.env.TRIAL_PROFILE_DIR || "data/.chrome-profile",
    defaultStagingDir: "data/staging",
    defaultOutputDir: "data/outputs",
    headless,
    sshHost: process.env.TRIAL_SSH_HOST,
    sshUser: process.env.TRIAL_SSH_USER,
    sshKeyPath: process.env.TRIAL_SSH_KEY_PATH,
    sshPort: process.env.TRIAL_SSH_PORT ? Number(process.env.TRIAL_SSH_PORT) : undefined,
    remoteDir: process.env.TRIAL_REMOTE_DIR
  };
}

export function resolveRepoRoot(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  return path.resolve(here, "..", "..", "..", "..", "..");
}

export function resolvePath(repoRoot: string, maybeRelative: string): string {
  if (path.isAbsolute(maybeRelative)) {
    return maybeRelative;
  }
  return path.resolve(repoRoot, maybeRelative);
}
