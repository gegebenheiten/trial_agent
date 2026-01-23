import path from "node:path";
import { chromium, type BrowserContext, type Page } from "playwright";

export type LaunchOptions = {
  headless: boolean;
  profileDir: string;
  viewport?: { width: number; height: number };
};

export async function launchContext(profileDir: string, opts?: Partial<LaunchOptions>): Promise<{ ctx: BrowserContext; page: Page }> {
  const userDataDir = path.resolve(profileDir);
  const launchOptions: Parameters<typeof chromium.launchPersistentContext>[1] = {
    headless: opts?.headless ?? false,
    viewport: opts?.viewport ?? { width: 1400, height: 900 },
    args: ["--disable-blink-features=AutomationControlled"]
  };

  const channel = process.env.TRIAL_CHROME_CHANNEL;
  const executablePath = process.env.TRIAL_CHROME_EXECUTABLE;
  if (channel) {
    launchOptions.channel = channel as "chrome" | "chrome-beta" | "chrome-dev" | "chrome-canary";
  } else if (executablePath) {
    launchOptions.executablePath = executablePath;
  }

  const ctx = await chromium.launchPersistentContext(userDataDir, launchOptions);
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  return { ctx, page };
}
