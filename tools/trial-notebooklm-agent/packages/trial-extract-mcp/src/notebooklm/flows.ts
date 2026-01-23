import fs from "node:fs";
import path from "node:path";
import { ElementHandle, Locator, Page } from "playwright";
import { S } from "./selectors.js";
import type { StepLogger } from "../logger.js";

const CHAT_INPUT_SELECTOR = 'textarea.query-box-input, textarea, [contenteditable="true"], [role="textbox"], input[type="text"]';
const MCP_RESPONSE_SELECTORS = [
  ".to-user-container .message-text-content",
  "[data-message-author='bot']",
  "[data-message-author='assistant']",
  "[data-message-role='assistant']",
  "[data-author='assistant']",
  "[data-renderer*='assistant']",
  "[data-automation-id='response-text']",
  "[data-automation-id='assistant-response']",
  "[data-automation-id='chat-response']",
  "[data-testid*='assistant']",
  "[data-testid*='response']",
  "[aria-live='polite']",
  "[role='listitem'][data-message-author]"
];
const lastAskContext = new WeakMap<Page, { question: string; existingResponses: string[] }>();

export async function gotoNotebookLM(page: Page): Promise<void> {
  await page.goto("https://notebooklm.google.com/?hl=en", { waitUntil: "domcontentloaded" });
}

export async function ensureAuth(page: Page): Promise<void> {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    const createVisible = await page.locator(S.createNotebookButton).first().isVisible().catch(() => false);
    const chatVisible = await page.locator(S.chatInput).first().isVisible().catch(() => false);
    if (createVisible || chatVisible) {
      return;
    }
    const loginVisible = await page.locator(S.loginButton).first().isVisible().catch(() => false);
    if (loginVisible && page.url().includes("accounts.google.com")) {
      await page.waitForTimeout(500);
      continue;
    }
    await page.waitForTimeout(500);
  }
  if (page.url().includes("accounts.google.com")) {
    throw new Error("NotebookLM not authenticated. Run setup_auth to complete login.");
  }
  throw new Error("NotebookLM UI not recognized. Check selectors or language settings.");
}

export async function openOrCreateNotebook(page: Page, notebookTitle: string): Promise<void> {
  if (!page.url().includes("/notebook/")) {
    await page.locator(S.createNotebookButton).first().click();
    const titleInput = page.locator(S.notebookTitleInput).first();
    if (await titleInput.isVisible().catch(() => false)) {
      await titleInput.fill(notebookTitle);
      await page.keyboard.press("Enter");
    }
  }
  await page.waitForLoadState("domcontentloaded");
  await waitForNotebookShellReady(page, 60_000);
  await setNotebookTitle(page, notebookTitle);
}

export async function uploadSources(page: Page, filePaths: string[], logger?: StepLogger): Promise<void> {
  const fileStats = filePaths.map((filePath) => {
    try {
      const stat = fs.statSync(filePath);
      return { path: filePath, name: path.basename(filePath), size: stat.size };
    } catch (err) {
      return { path: filePath, name: path.basename(filePath), error: String(err) };
    }
  });
  const missing = fileStats.filter((item) => "error" in item);
  logInfo(
    logger,
    "uploadSources.start",
    {
      file_count: filePaths.length,
      names: filePaths.map((p) => path.basename(p)),
      files: fileStats
    },
    page
  );
  if (missing.length) {
    throw new Error(`Source files missing: ${missing.map((item) => item.name).join(", ")}`);
  }
  await runStep(logger, "uploadSources.uploadFilePaths", () => uploadFilePaths(page, filePaths, logger), page);
  let initialOk = false;
  try {
    await runStep(logger, "uploadSources.waitForSourcesVisible", () => waitForSourcesVisible(page, filePaths, logger), page);
    await runStep(logger, "uploadSources.waitForSourcesReady", () => waitForSourcesReady(page, filePaths, logger), page);
    await runStep(logger, "uploadSources.waitForIndexingReady", () => waitForIndexingReady(page, logger), page);
    initialOk = true;
  } catch (err) {
    logInfo(logger, "uploadSources.initial_wait_failed", { error: String(err) }, page);
  }

  const retryResult = await runStep(logger, "uploadSources.retryFailedUploads", () => retryFailedUploads(page, filePaths, logger), page);
  if (retryResult.reuploaded.length) {
    await runStep(logger, "uploadSources.waitForSourcesVisible.retry", () => waitForSourcesVisible(page, retryResult.reuploaded, logger), page);
    await runStep(logger, "uploadSources.waitForSourcesReady.retry", () => waitForSourcesReady(page, retryResult.reuploaded, logger), page);
    await runStep(logger, "uploadSources.waitForIndexingReady.retry", () => waitForIndexingReady(page, logger), page);
  }
  if (!initialOk && !retryResult.reuploaded.length) {
    throw new Error("Sources failed to upload and no retry candidates found.");
  }

  const selectAll = page.locator(S.selectAllSourcesButton).first();
  if (await selectAll.isVisible().catch(() => false)) {
    await selectAll.click().catch(() => {});
    logInfo(logger, "uploadSources.selectAllSources", { clicked: true }, page);
  }
}

export async function setActiveSourcesForTable(
  page: Page,
  selection: { include: string[]; all: string[] },
  logger?: StepLogger
): Promise<void> {
  const includeSet = new Set(selection.include.map((name) => name.toLowerCase()));
  const allNames = selection.all;
  const summary = { include: selection.include, total: allNames.length, checked: 0, unchecked: 0, missing: 0 };
  logInfo(logger, "sourceSelection.start", { include: selection.include, total: allNames.length }, page);
  for (const name of allNames) {
    const desired = includeSet.has(name.toLowerCase());
    const result = await setSourceChecked(page, name, desired);
    if (!result.present) {
      summary.missing += 1;
      logInfo(logger, "sourceSelection.missing", { name }, page);
      continue;
    }
    if (result.checked) {
      summary.checked += 1;
    } else {
      summary.unchecked += 1;
    }
    if (result.changed) {
      logInfo(logger, "sourceSelection.toggle", { name, desired, checked: result.checked }, page);
    }
  }
  logInfo(logger, "sourceSelection.done", summary, page);
}

export async function waitForIndexingReady(page: Page, logger?: StepLogger, timeoutMs = 15 * 60 * 1000): Promise<void> {
  const spinner = page.locator(S.sourceProcessingSpinner);
  const processingText = page.locator(S.sourceProcessingText);
  const minWaitMs = 20_000;
  const quietWindowMs = 12_000;
  const start = Date.now();
  let lastProcessingSeen = Date.now();
  let processingSeen = false;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hasProcessing = (await anyVisible(spinner)) || (await anyVisible(processingText));
    if (hasProcessing) {
      if (!processingSeen) {
        logInfo(logger, "waitForIndexingReady.processing_seen", {}, page);
      }
      processingSeen = true;
      lastProcessingSeen = Date.now();
    }
    const quietFor = Date.now() - lastProcessingSeen;
    const waitedLongEnough = Date.now() - start >= minWaitMs;
    if ((processingSeen && quietFor >= quietWindowMs) || (!processingSeen && waitedLongEnough && quietFor >= quietWindowMs)) {
      break;
    }
    await page.waitForTimeout(1000);
  }
  const chat = await waitForChatInput(page, logger, 60_000);
  const enabledDeadline = Date.now() + timeoutMs;
  while (Date.now() < enabledDeadline) {
    if (await chat.isEnabled().catch(() => false)) {
      logInfo(logger, "waitForIndexingReady.chat_enabled", {}, page);
      return;
    }
    await page.waitForTimeout(500);
  }
  throw new Error("Chat input did not become enabled after indexing.");
}

export async function ask(page: Page, prompt: string, logger?: StepLogger): Promise<void> {
  const existingResponses = await snapshotAllResponses(page, logger);
  lastAskContext.set(page, { question: prompt, existingResponses });
  const chat = await waitForChatInput(page, logger, 30_000);
  logInfo(logger, "chatInput.send", { prompt_len: prompt.length }, page);
  await chat.scrollIntoViewIfNeeded().catch(() => {});
  await chat.click();
  await chat.fill(prompt);
  await page.keyboard.press("Enter");
  logInfo(logger, "chatInput.send_enter", { via: "enter" }, page);
  const generationStarted = await waitForGenerationStart(page, chat, logger, 8000);
  if (generationStarted) {
    return;
  }
  const inputValue = await readEditableValue(chat);
  const stillHasPrompt = looksLikePromptInInput(inputValue, prompt);
  logInfo(
    logger,
    "chatInput.send_not_confirmed",
    { input_len: inputValue.length, still_has_prompt: stillHasPrompt },
    page
  );
  if (!stillHasPrompt) {
    return;
  }
  const afterResponses = await snapshotAllResponses(page, logger);
  const responseCountSame = afterResponses.length === existingResponses.length;
  logInfo(logger, "chatInput.resend_check", { responseCountSame }, page);
  if (!responseCountSame) {
    return;
  }
  await chat.click().catch(() => {});
  await page.keyboard.press("Enter");
  logInfo(logger, "chatInput.resend_enter", { via: "enter_retry" }, page);
  await waitForGenerationStart(page, chat, logger, 8000);
}

export async function getLastAnswerText(page: Page, logger?: StepLogger): Promise<string> {
  const ctx = lastAskContext.get(page) || { question: "", existingResponses: [] };
  const answer = await waitForLatestAnswer(page, {
    question: ctx.question,
    ignoreTexts: ctx.existingResponses,
    timeoutMs: 120_000,
    pollIntervalMs: 1000
  });
  lastAskContext.delete(page);
  if (!answer) {
    throw new Error("Timeout waiting for answer (notebooklm-mcp logic).");
  }
  const cleaned = answer.trim();
  if (looksLikeUiChrome(cleaned)) {
    throw new Error("Answer text looks like UI chrome. Update selectors.");
  }
  return cleaned;
}

export async function getShareLink(page: Page): Promise<string> {
  const share = page.locator(S.shareButton).first();
  if (await share.isVisible().catch(() => false)) {
    await share.click();
    const input = page.locator('input[type="text"]').first();
    if (await input.isVisible().catch(() => false)) {
      return await input.inputValue();
    }
  }
  return "";
}

type WaitAnswerOptions = {
  question?: string;
  timeoutMs?: number;
  pollIntervalMs?: number;
  ignoreTexts?: string[];
};

async function snapshotAllResponses(page: Page, logger?: StepLogger): Promise<string[]> {
  const allTexts: string[] = [];
  const primarySelector = ".to-user-container";
  try {
    const containers = await page.$$(primarySelector);
    if (containers.length > 0) {
      for (const container of containers) {
        try {
          const textElement = await container.$(".message-text-content");
          if (textElement) {
            const text = await textElement.innerText();
            if (text && text.trim()) {
              allTexts.push(text.trim());
            }
          }
        } catch {
          continue;
        }
      }
    }
  } catch {
    // ignore
  }
  logInfo(logger, "answer.snapshot", { count: allTexts.length }, page);
  return allTexts;
}

function hashString(text: string): number {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    const chr = text.charCodeAt(i);
    hash = (hash << 5) - hash + chr;
    hash |= 0;
  }
  return hash;
}

async function waitForLatestAnswer(page: Page, options: WaitAnswerOptions = {}): Promise<string | null> {
  const {
    question = "",
    timeoutMs = 120_000,
    pollIntervalMs = 1000,
    ignoreTexts = []
  } = options;
  const deadline = Date.now() + timeoutMs;
  const sanitizedQuestion = question.trim().toLowerCase();
  const knownHashes = new Set<number>();
  for (const text of ignoreTexts) {
    if (typeof text === "string" && text.trim()) {
      knownHashes.add(hashString(text.trim()));
    }
  }

  let lastCandidate: string | null = null;
  let stableCount = 0;
  const requiredStablePolls = 3;

  while (Date.now() < deadline) {
    try {
      const thinkingElement = await page.$("div.thinking-message");
      if (thinkingElement) {
        const isVisible = await thinkingElement.isVisible();
        if (isVisible) {
          await page.waitForTimeout(pollIntervalMs);
          continue;
        }
      }
    } catch {
      // ignore
    }

    const candidate = await extractLatestText(page, knownHashes);
    if (candidate) {
      const normalized = candidate.trim();
      if (normalized) {
        if (normalized.toLowerCase() === sanitizedQuestion) {
          knownHashes.add(hashString(normalized));
          await page.waitForTimeout(pollIntervalMs);
          continue;
        }
        if (normalized === lastCandidate) {
          stableCount += 1;
        } else {
          stableCount = 1;
          lastCandidate = normalized;
        }
        if (stableCount >= requiredStablePolls) {
          return normalized;
        }
      }
    }
    await page.waitForTimeout(pollIntervalMs);
  }
  return null;
}

async function extractLatestText(page: Page, knownHashes: Set<number>): Promise<string | null> {
  const primarySelector = ".to-user-container";
  try {
    const containers = await page.$$(primarySelector);
    const totalContainers = containers.length;
    if (totalContainers <= knownHashes.size) {
      return null;
    }
    if (containers.length > 0) {
      for (let idx = 0; idx < containers.length; idx += 1) {
        const container = containers[idx];
        try {
          const textElement = await container.$(".message-text-content");
          if (textElement) {
            const text = await textElement.innerText();
            if (text && text.trim()) {
              const textHash = hashString(text.trim());
              if (!knownHashes.has(textHash)) {
                return text.trim();
              }
            }
          }
        } catch {
          continue;
        }
      }
      return null;
    }
  } catch {
    // ignore
  }

  for (const selector of MCP_RESPONSE_SELECTORS) {
    try {
      const elements = await page.$$(selector);
      if (!elements.length) {
        continue;
      }
      for (const element of elements) {
        try {
          let container = element;
          try {
            const closest = await element.evaluateHandle((el) => {
              return el.closest(
                "[data-message-author], [data-message-role], [data-author], " +
                  "[data-testid*='assistant'], [data-automation-id*='response'], article, section"
              );
            });
            if (closest) {
              const asElement = closest.asElement() as ElementHandle<SVGElement | HTMLElement> | null;
              if (asElement) {
                container = asElement;
              }
            }
          } catch {
            container = element;
          }
          const text = await container.innerText();
          if (text && text.trim() && !knownHashes.has(hashString(text.trim()))) {
            return text.trim();
          }
        } catch {
          continue;
        }
      }
    } catch {
      continue;
    }
  }

  try {
    const fallbackText = await page.evaluate(() => {
      const unique = new Set<Element>();
      const isVisible = (el: Element | null) => {
        if (!el || !(el as HTMLElement).isConnected) return false;
        const rect = (el as HTMLElement).getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return false;
        const style = window.getComputedStyle(el as Element);
        if (style.visibility === "hidden" || style.display === "none" || parseFloat(style.opacity || "1") === 0) {
          return false;
        }
        return true;
      };
      const selectors = [
        "[data-message-author]",
        "[data-message-role]",
        "[data-author]",
        "[data-renderer*='assistant']",
        "[data-testid*='assistant']",
        "[data-automation-id*='response']"
      ];
      const candidates: string[] = [];
      for (const selector of selectors) {
        for (const el of document.querySelectorAll(selector)) {
          if (!isVisible(el)) continue;
          if (unique.has(el)) continue;
          unique.add(el);
          const text = (el as HTMLElement).innerText || (el.textContent || "");
          if (!text.trim()) continue;
          candidates.push(text.trim());
        }
      }
      if (candidates.length > 0) {
        return candidates[candidates.length - 1];
      }
      return null;
    });
    if (typeof fallbackText === "string" && fallbackText.trim()) {
      return fallbackText.trim();
    }
  } catch {
    // ignore
  }
  return null;
}

async function waitForStableText(page: Page, locator: Locator, stableCount: number, intervalMs: number): Promise<string> {
  let lastText = await locator.innerText();
  let stable = 0;
  while (stable < stableCount) {
    await page.waitForTimeout(intervalMs);
    const nextText = await locator.innerText();
    if (nextText === lastText) {
      stable += 1;
    } else {
      lastText = nextText;
      stable = 0;
    }
  }
  return lastText;
}

async function waitForAnswerComplete(
  page: Page,
  answer: Locator,
  logger?: StepLogger,
  timeoutMs = 5 * 60 * 1000
): Promise<void> {
  const stop = page.locator(S.stopGeneratingButton).first();
  const regen = answer.locator(S.regenerateButton).first();
  const copy = answer.locator(S.copyAnswerButton).first();
  const regenGlobal = page.locator(S.regenerateButton).last();
  const copyGlobal = page.locator(S.copyAnswerButton).last();
  const thinking = page.locator(S.thinkingIndicator).first();
  const start = Date.now();
  let stableText = "";
  let stableCount = 0;
  const intervalMs = 1500;
  let stopSeen = false;

  while (Date.now() - start < timeoutMs) {
    const thinkingVisible = await thinking.isVisible().catch(() => false);
    if (thinkingVisible) {
      await page.waitForTimeout(intervalMs);
      continue;
    }
    const stopVisible = await stop.isVisible().catch(() => false);
    if (stopVisible) {
      stopSeen = true;
    }
    const regenVisible = (await regen.isVisible().catch(() => false)) || (await regenGlobal.isVisible().catch(() => false));
    const copyVisible = (await copy.isVisible().catch(() => false)) || (await copyGlobal.isVisible().catch(() => false));
    const copyTarget = (await copy.isVisible().catch(() => false)) ? copy : copyGlobal;
    const copyInView = copyVisible ? await isLocatorInViewport(page, copyTarget) : false;
    const text = await answer.innerText().catch(() => "");
    if (text && text === stableText) {
      stableCount += 1;
    } else {
      stableText = text;
      stableCount = 0;
    }
    const chat = await waitForChatInput(page, undefined, 3000).catch(() => null);
    const chatEnabled = chat ? await chat.isEnabled().catch(() => false) : false;

    if (stopSeen && !stopVisible && copyVisible && copyInView && stableCount >= 2) {
      logInfo(logger, "answer.complete", { stableCount, regenVisible, copyVisible, copyInView, chatEnabled }, page);
      return;
    }
    if (stopSeen && !stopVisible && regenVisible && stableCount >= 3) {
      logInfo(logger, "answer.complete_regen", { stableCount, regenVisible, copyVisible, copyInView, chatEnabled }, page);
      return;
    }
    if (!stopVisible && stableCount >= 6) {
      logInfo(logger, "answer.complete_stable", { stableCount, chatEnabled }, page);
      return;
    }
    await page.waitForTimeout(intervalMs);
  }
  logInfo(logger, "answer.wait_timeout", { timeout_ms: timeoutMs }, page);
}

async function waitForNotebookShellReady(page: Page, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  const candidates = [
    page.locator(S.addSourcesButton).first(),
    page.locator(S.uploadEmptyStateButton).first(),
    page.locator(S.chatInput).first()
  ];
  while (Date.now() < deadline) {
    for (const locator of candidates) {
      if (await locator.isVisible().catch(() => false)) {
        return;
      }
    }
    await page.waitForTimeout(500);
  }
  throw new Error("Notebook shell not ready (no Add sources / Upload a source / chat input).");
}

export async function setNotebookTitle(page: Page, notebookTitle: string): Promise<void> {
  await closeOverlays(page);
  const editableTitle = page.locator(S.notebookTitleEditable).first();
  if (await editableTitle.isVisible().catch(() => false)) {
    await editableTitle.click();
    try {
      await editableTitle.fill(notebookTitle);
    } catch {
      await page.keyboard.press("Control+A").catch(() => {});
      await page.keyboard.type(notebookTitle);
    }
    await page.keyboard.press("Enter").catch(() => {});
    return;
  }
  const titleInput = page.locator(S.notebookTitleInput).first();
  if (await titleInput.isVisible().catch(() => false)) {
    await titleInput.fill(notebookTitle);
    await page.keyboard.press("Enter");
    return;
  }

  const configBtn = page.locator(S.configureNotebookButton).first();
  if (await configBtn.isVisible().catch(() => false)) {
    await closeOverlays(page);
    try {
      await configBtn.click();
    } catch {
      await closeOverlays(page);
      await configBtn.click().catch(() => {});
    }
    const configInput = page.locator(S.notebookTitleInput).first();
    if (await configInput.isVisible().catch(() => false)) {
      await configInput.fill(notebookTitle);
      await page.keyboard.press("Enter");
      await page.keyboard.press("Escape").catch(() => {});
      return;
    }
    await page.keyboard.press("Escape").catch(() => {});
  }

  const titleText = page.locator(S.notebookTitleText).first();
  if (await titleText.isVisible().catch(() => false)) {
    await titleText.click();
    const inlineInput = page.locator(S.notebookTitleInput).first();
    if (await inlineInput.isVisible().catch(() => false)) {
      await inlineInput.fill(notebookTitle);
      await page.keyboard.press("Enter");
    }
  }
}

export async function deleteNotebook(page: Page, notebookTitle: string, logger?: StepLogger): Promise<boolean> {
  await closeOverlays(page);
  logInfo(logger, "deleteNotebook.start", { title: notebookTitle }, page);
  const deletedFromSettings = await tryDeleteFromSettings(page, logger);
  if (deletedFromSettings) {
    logInfo(logger, "deleteNotebook.done", { title: notebookTitle, method: "settings" }, page);
    return true;
  }
  const deletedFromMenu = await tryDeleteFromHeaderMenu(page, logger);
  if (deletedFromMenu) {
    logInfo(logger, "deleteNotebook.done", { title: notebookTitle, method: "header_menu" }, page);
    return true;
  }
  const deletedFromHome = await tryDeleteFromHome(page, notebookTitle, logger);
  logInfo(logger, "deleteNotebook.done", { title: notebookTitle, method: deletedFromHome ? "home" : "none" }, page);
  return deletedFromHome;
}

async function closeOverlays(page: Page, attempts = 3): Promise<void> {
  const overlay = page.locator(".cdk-overlay-backdrop");
  for (let i = 0; i < attempts; i += 1) {
    const count = await overlay.count().catch(() => 0);
    if (!count) {
      return;
    }
    await page.keyboard.press("Escape").catch(() => {});
    await page.waitForTimeout(300);
  }
}

async function tryDeleteFromSettings(page: Page, logger?: StepLogger): Promise<boolean> {
  const header = page.locator("header").first();
  const settingsCandidates = [
    header.locator(S.notebookSettingsButton).first(),
    page.locator(S.notebookSettingsButton).first(),
    header.locator('text=/Settings|设置/i').first()
  ];
  let settings: Locator | null = null;
  for (const candidate of settingsCandidates) {
    if (await candidate.isVisible().catch(() => false)) {
      settings = candidate;
      break;
    }
  }
  if (!settings) {
    logInfo(logger, "deleteNotebook.settings_missing", {}, page);
    return false;
  }
  await settings.click().catch(() => {});
  await page.waitForTimeout(500);
  const dialog = page.locator("dialog, [role='dialog']").first();
  const deleteBtn = dialog.locator(S.deleteNotebookButton).first();
  const deleteGlobal = page.locator(S.deleteNotebookButton).first();
  const deleteVisible =
    (await deleteBtn.isVisible().catch(() => false)) || (await deleteGlobal.isVisible().catch(() => false));
  const activeDelete = (await deleteBtn.isVisible().catch(() => false)) ? deleteBtn : deleteGlobal;
  if (!deleteVisible) {
    logInfo(logger, "deleteNotebook.settings_delete_missing", {}, page);
    await page.keyboard.press("Escape").catch(() => {});
    return false;
  }
  await activeDelete.click().catch(() => {});
  logInfo(logger, "deleteNotebook.settings_click", {}, page);
  await page.waitForTimeout(300);
  const confirmDialog = page.locator("dialog, [role='dialog']").first();
  const confirm = confirmDialog.locator(S.confirmDeleteButton).first();
  if (await confirm.isVisible().catch(() => false)) {
    await confirm.click().catch(() => {});
    logInfo(logger, "deleteNotebook.settings_confirm", {}, page);
  }
  await page.waitForTimeout(1000);
  return true;
}

async function tryDeleteFromHeaderMenu(page: Page, logger?: StepLogger): Promise<boolean> {
  await closeOverlays(page);
  const header = page.locator("header").first();
  const menuCandidates = [
    header.locator(S.notebookHeaderMenuButton).first(),
    page.locator(S.notebookHeaderMenuButton).first()
  ];
  let menu: Locator | null = null;
  for (const candidate of menuCandidates) {
    if (await candidate.isVisible().catch(() => false)) {
      menu = candidate;
      break;
    }
  }
  if (!menu) {
    logInfo(logger, "deleteNotebook.header_menu_missing", {}, page);
    return false;
  }
  await menu.click().catch(() => {});
  await page.waitForTimeout(200);
  const deleteBtn = page.locator(S.deleteNotebookButton).first();
  if (!(await deleteBtn.isVisible().catch(() => false))) {
    logInfo(logger, "deleteNotebook.header_delete_missing", {}, page);
    await page.keyboard.press("Escape").catch(() => {});
    return false;
  }
  await deleteBtn.click().catch(() => {});
  logInfo(logger, "deleteNotebook.header_delete_click", {}, page);
  await page.waitForTimeout(300);
  const confirm = page.locator(S.confirmDeleteButton).first();
  if (await confirm.isVisible().catch(() => false)) {
    await confirm.click().catch(() => {});
    logInfo(logger, "deleteNotebook.header_confirm", {}, page);
  }
  await page.waitForTimeout(1000);
  return true;
}

async function tryDeleteFromHome(page: Page, notebookTitle: string, logger?: StepLogger): Promise<boolean> {
  await gotoNotebookLM(page);
  await ensureAuth(page);
  const title = page.locator(`text=${JSON.stringify(notebookTitle)}`).first();
  if (!(await title.isVisible().catch(() => false))) {
    logInfo(logger, "deleteNotebook.home_missing", { title: notebookTitle }, page);
    return false;
  }
  const card = title.locator("xpath=ancestor::*[self::article or self::div][1]").first();
  const menu = card.locator(S.notebookCardMenuButton).first();
  if (!(await menu.isVisible().catch(() => false))) {
    logInfo(logger, "deleteNotebook.home_menu_missing", { title: notebookTitle }, page);
    return false;
  }
  await menu.click().catch(() => {});
  await page.waitForTimeout(200);
  const deleteBtn = page.locator(S.deleteNotebookButton).first();
  if (!(await deleteBtn.isVisible().catch(() => false))) {
    logInfo(logger, "deleteNotebook.home_delete_missing", { title: notebookTitle }, page);
    await page.keyboard.press("Escape").catch(() => {});
    return false;
  }
  await deleteBtn.click().catch(() => {});
  logInfo(logger, "deleteNotebook.home_delete_click", { title: notebookTitle }, page);
  await page.waitForTimeout(300);
  const confirm = page.locator(S.confirmDeleteButton).first();
  if (await confirm.isVisible().catch(() => false)) {
    await confirm.click().catch(() => {});
    logInfo(logger, "deleteNotebook.home_confirm", { title: notebookTitle }, page);
  }
  await page.waitForTimeout(1000);
  return true;
}

function looksLikeUiChrome(text: string): boolean {
  const lowered = text.toLowerCase();
  const uiTokens = ["tune", "more_vert", "arrow_forward", "keyboard_arrow_down"];
  if (uiTokens.some((token) => lowered.includes(token))) {
    return true;
  }
  const uiPhrases = ["the system was unable to answer", "pinpointing the core issue"];
  return uiPhrases.some((phrase) => lowered.includes(phrase));
}

async function anyVisible(locator: Locator): Promise<boolean> {
  try {
    return await locator.evaluateAll((nodes) =>
      nodes.some((node) => {
        const el = node as { getClientRects?: () => { length: number; 0?: { width: number; height: number } } } | null;
        if (!el || typeof el.getClientRects !== "function") {
          return false;
        }
        const rects = el.getClientRects();
        return rects.length > 0 && rects[0] && rects[0].width > 0 && rects[0].height > 0;
      })
    );
  } catch {
    return false;
  }
}

async function waitForSourcesVisible(
  page: Page,
  filePaths: string[],
  logger?: StepLogger,
  timeoutMs = 2 * 60 * 1000
): Promise<void> {
  const names = Array.from(new Set(filePaths.map((filePath) => path.basename(filePath)).filter(Boolean)));
  if (!names.length) {
    return;
  }
  const candidates = new Map<string, string[]>();
  for (const name of names) {
    const noExt = name.replace(/\.[^.]+$/, "");
    const list = [name];
    if (noExt && noExt !== name) {
      list.push(noExt);
    }
    candidates.set(name, list);
  }
  await page.keyboard.press("Escape").catch(() => {});
  const remaining = new Set(names);
  logInfo(logger, "waitForSourcesVisible.start", { remaining: Array.from(remaining) }, page);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline && remaining.size) {
    for (const name of Array.from(remaining)) {
      const variants = candidates.get(name) || [name];
      let matched: string | null = null;
      for (const variant of variants) {
        const visible = await page.getByText(variant, { exact: false }).first().isVisible().catch(() => false);
        if (visible) {
          matched = variant;
          break;
        }
      }
      if (matched) {
        remaining.delete(name);
        logInfo(logger, "waitForSourcesVisible.found", { name, matched }, page);
      }
    }
    if (!remaining.size) {
      return;
    }
    await page.waitForTimeout(1000);
  }
  if (remaining.size) {
    logInfo(logger, "waitForSourcesVisible.timeout", { remaining: Array.from(remaining) }, page);
    throw new Error(`Sources did not appear after upload: ${Array.from(remaining).join(", ")}`);
  }
}

async function waitForSourcesReady(
  page: Page,
  filePaths: string[],
  logger?: StepLogger,
  timeoutMs = 15 * 60 * 1000
): Promise<void> {
  const names = Array.from(new Set(filePaths.map((filePath) => path.basename(filePath)).filter(Boolean)));
  if (!names.length) {
    return;
  }
  const stallTimeoutMs = 60_000;
  const stalledSince = new Map<string, number>();
  let lastSelectAttempt = 0;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const statuses = await collectSourceStatuses(page, names, logger);
    const now = Date.now();
    for (const status of statuses) {
      if (status.present && !status.processing && !status.checked && !status.failed) {
        if (!stalledSince.has(status.name)) {
          stalledSince.set(status.name, now);
        } else if (now - (stalledSince.get(status.name) || now) > stallTimeoutMs) {
          status.stalled = true;
          status.failed = true;
        }
      } else {
        stalledSince.delete(status.name);
      }
    }
    const processing = statuses.filter((item) => item.processing);
    const failed = statuses.filter((item) => item.failed);
    const missing = statuses.filter((item) => !item.present);
    const unchecked = statuses.filter((item) => !item.checked);
    const stalled = statuses.filter((item) => item.stalled);
    const checkedAll = statuses.length > 0 && statuses.every((item) => item.checked);

    if (!processing.length && !missing.length && unchecked.length) {
      const shouldAttempt = now - lastSelectAttempt > 5_000;
      if (shouldAttempt) {
        lastSelectAttempt = now;
        const selected = await trySelectAllSources(page, logger);
        if (!selected) {
          await trySelectUncheckedSources(page, unchecked.map((u) => u.name), logger);
        }
        await page.waitForTimeout(1000);
        continue;
      }
    }

    if (failed.length) {
      logInfo(
        logger,
        "waitForSourcesReady.failed",
        { failed: failed.map((f) => f.name), stalled: stalled.map((s) => s.name) },
        page
      );
      throw new Error(`Sources failed during upload: ${failed.map((f) => f.name).join(", ")}`);
    }

    if (!processing.length && !missing.length && !unchecked.length) {
      logInfo(logger, "waitForSourcesReady.done", { total: statuses.length, checked_all: checkedAll }, page);
      return;
    }

    logInfo(
      logger,
      "waitForSourcesReady.pending",
      {
        processing: processing.map((p) => p.name),
        missing: missing.map((m) => m.name),
        unchecked: unchecked.map((u) => u.name),
        stalled: stalled.map((s) => s.name),
        total: statuses.length,
        checked_all: checkedAll
      },
      page
    );
    await page.waitForTimeout(1000);
  }

  logInfo(logger, "waitForSourcesReady.timeout", { names }, page);
  throw new Error(`Sources still processing after timeout: ${names.join(", ")}`);
}

async function detectUploadIndicator(page: Page, logger?: StepLogger, timeoutMs = 6_000): Promise<boolean> {
  const spinner = page.locator(S.sourceProcessingSpinner);
  const processingText = page.locator(S.sourceProcessingText);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const visible = (await anyVisible(spinner)) || (await anyVisible(processingText));
    if (visible) {
      logInfo(logger, "uploadFilePaths.processing_seen", { timeout_ms: timeoutMs }, page);
      return true;
    }
    await page.waitForTimeout(250);
  }
  logInfo(logger, "uploadFilePaths.processing_not_seen", { timeout_ms: timeoutMs }, page);
  return false;
}

type ChatInputCandidate = {
  index: number;
  score: number;
  rect: { x: number; y: number; w: number; h: number };
  placeholder: string;
  aria: string;
  is_search: boolean;
  is_chat_hint: boolean;
};

async function pickChatInputCandidate(page: Page): Promise<{ index: number; candidates: ChatInputCandidate[] }> {
  const locator = page.locator(CHAT_INPUT_SELECTOR);
  const result = await locator.evaluateAll((nodes) => {
    const candidates: {
      index: number;
      score: number;
      rect: { x: number; y: number; w: number; h: number };
      placeholder: string;
      aria: string;
      is_search: boolean;
      is_chat_hint: boolean;
    }[] = [];

    nodes.forEach((node, index) => {
      const el = node as {
        getBoundingClientRect: () => { x: number; y: number; width: number; height: number };
        getAttribute: (name: string) => string | null;
        ownerDocument?: { defaultView?: { getComputedStyle: (target: unknown) => { display: string; visibility: string } } };
      };
      const rect = el.getBoundingClientRect();
      const style =
        el.ownerDocument?.defaultView?.getComputedStyle(el) ?? { display: "block", visibility: "visible" };
      if (style.display === "none" || style.visibility === "hidden" || rect.width < 80 || rect.height < 20) {
        return;
      }
      const placeholder = (el.getAttribute("placeholder") || "").toLowerCase();
      const aria = (el.getAttribute("aria-label") || "").toLowerCase();
      const isSearch =
        placeholder.includes("search") ||
        aria.includes("search") ||
        placeholder.includes("research") ||
        aria.includes("research") ||
        placeholder.includes("web") ||
        aria.includes("web") ||
        placeholder.includes("搜索") ||
        aria.includes("搜索") ||
        placeholder.includes("来源") ||
        aria.includes("来源");
      const isChatHint =
        placeholder.includes("start typing") ||
        placeholder.includes("开始输入") ||
        placeholder.includes("ask") ||
        placeholder.includes("提问") ||
        aria.includes("query") ||
        aria.includes("ask") ||
        aria.includes("start typing") ||
        aria.includes("提问") ||
        aria.includes("问题") ||
        aria.includes("输入");
      const bottom = rect.y + rect.height;
      let score = bottom + rect.width * 0.2;
      if (isChatHint) {
        score += 1000;
      }
      if (isSearch) {
        score -= 600;
      }
      if (rect.x < 200) {
        score -= 200;
      }
      candidates.push({
        index,
        score,
        rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
        placeholder,
        aria,
        is_search: isSearch,
        is_chat_hint: isChatHint
      });
    });

    candidates.sort((a, b) => b.score - a.score);
    return {
      index: candidates.length ? candidates[0].index : -1,
      candidates: candidates.slice(0, 5)
    };
  });

  return result;
}

async function waitForChatInput(page: Page, logger?: StepLogger, timeoutMs = 60_000): Promise<Locator> {
  const selectors = Array.isArray(S.skillQueryInputSelectors) ? S.skillQueryInputSelectors : [];
  const perSelectorTimeout = Math.min(10_000, timeoutMs);
  const deadline = Date.now() + timeoutMs;
  for (const selector of selectors) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      break;
    }
    try {
      const candidate = page.locator(selector).first();
      await candidate.waitFor({ state: "visible", timeout: Math.min(perSelectorTimeout, remaining) });
      logInfo(logger, "chatInput.direct", { selector }, page);
      return candidate;
    } catch {
      // try next selector
    }
  }
  throw new Error("Chat input not found with notebooklm-skill selectors.");
}

async function readEditableValue(locator: Locator): Promise<string> {
  try {
    return await locator.evaluate((el) => {
      const anyEl = el as {
        value?: string;
        textContent?: string | null;
        isContentEditable?: boolean;
        innerText?: string;
      };
      if (typeof anyEl.value === "string") {
        return anyEl.value;
      }
      if (anyEl.isContentEditable) {
        return anyEl.textContent || "";
      }
      return anyEl.textContent || anyEl.innerText || "";
    });
  } catch {
    return "";
  }
}

function normalizeForCompare(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function looksLikePromptInInput(inputValue: string, prompt: string): boolean {
  const normalizedInput = normalizeForCompare(inputValue);
  const normalizedPrompt = normalizeForCompare(prompt);
  if (!normalizedInput || !normalizedPrompt) {
    return false;
  }
  const signature = normalizedPrompt.slice(0, 80);
  if (!signature) {
    return false;
  }
  return normalizedInput.includes(signature);
}

async function waitForGenerationStart(
  page: Page,
  chat: Locator,
  logger?: StepLogger,
  timeoutMs = 12_000
): Promise<boolean> {
  const stop = page.locator(S.stopGeneratingButton).first();
  const thinking = page.locator(S.thinkingIndicator).first();
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const stopVisible = await stop.isVisible().catch(() => false);
    if (stopVisible) {
      logInfo(logger, "chatInput.generation_started", { via: "stop_button" }, page);
      return true;
    }
    const thinkingVisible = await thinking.isVisible().catch(() => false);
    if (thinkingVisible) {
      logInfo(logger, "chatInput.generation_started", { via: "thinking_indicator" }, page);
      return true;
    }
    const value = await readEditableValue(chat);
    if (!value.trim()) {
      logInfo(logger, "chatInput.generation_started", { via: "input_cleared" }, page);
      return true;
    }
    await page.waitForTimeout(500);
  }
  logInfo(logger, "chatInput.generation_timeout", { timeout_ms: timeoutMs }, page);
  return false;
}

async function clickSendButton(page: Page, logger?: StepLogger): Promise<boolean> {
  const scope = (await page.locator(S.chatPanel).count()) > 0 ? page.locator(S.chatPanel).first() : page;
  const sendButton = scope.locator(S.sendButton).first();
  const sendVisible = await sendButton.isVisible().catch(() => false);
  const sendEnabled = await sendButton.isEnabled().catch(() => false);
  if (sendVisible && sendEnabled) {
    await sendButton.scrollIntoViewIfNeeded().catch(() => {});
    await sendButton.click().catch(() => {});
    logInfo(logger, "chatInput.send_click", { via: "button", sendVisible, sendEnabled }, page);
    return true;
  }

  const sendIcon = scope.locator(S.sendIcon).first();
  if ((await sendIcon.count()) > 0) {
    const iconButton = sendIcon.locator("xpath=ancestor::*[self::button or @role='button'][1]").first();
    const iconButtonVisible = await iconButton.isVisible().catch(() => false);
    const iconButtonEnabled = await iconButton.isEnabled().catch(() => false);
    if (iconButtonVisible && iconButtonEnabled) {
      await iconButton.scrollIntoViewIfNeeded().catch(() => {});
      await iconButton.click().catch(() => {});
      logInfo(logger, "chatInput.send_click", { via: "icon_button", iconButtonVisible, iconButtonEnabled }, page);
      return true;
    }
    const iconVisible = await sendIcon.isVisible().catch(() => false);
    if (iconVisible) {
      await sendIcon.scrollIntoViewIfNeeded().catch(() => {});
      await sendIcon.click().catch(() => {});
      logInfo(logger, "chatInput.send_click", { via: "icon_direct", iconVisible }, page);
      return true;
    }
  }

  return false;
}

async function findLatestAnswerLocator(page: Page, logger?: StepLogger): Promise<Locator> {
  const selectors = Array.isArray(S.responseSelectors) ? S.responseSelectors : [];
  for (const selector of selectors) {
    const locator = page.locator(selector);
    const count = await locator.count().catch(() => 0);
    if (count > 0) {
      const candidate = locator.nth(count - 1);
      if (await candidate.isVisible().catch(() => false)) {
        logInfo(logger, "answer.selector", { selector, count }, page);
        return candidate;
      }
    }
  }

  const answerBlocks = page.locator(S.answerBlocks);
  if ((await answerBlocks.count()) > 0) {
    return answerBlocks.last();
  }
  const fallbackBlocks = page.locator(S.answerFallback);
  if ((await fallbackBlocks.count()) > 0) {
    return fallbackBlocks.last();
  }
  throw new Error("No answer blocks found. Update selectors.");
}

async function waitForSkillAnswerText(
  page: Page,
  logger?: StepLogger,
  timeoutMs = 120_000
): Promise<{ text: string; selector: string } | null> {
  const selectors = Array.isArray(S.skillResponseSelectors) ? S.skillResponseSelectors : [];
  const thinking = page.locator(S.thinkingIndicator).first();
  const deadline = Date.now() + timeoutMs;
  let lastText = "";
  let stableCount = 0;

  while (Date.now() < deadline) {
    const thinkingVisible = await thinking.isVisible().catch(() => false);
    if (thinkingVisible) {
      await page.waitForTimeout(1000);
      continue;
    }
    for (const selector of selectors) {
      try {
        const loc = page.locator(selector);
        const count = await loc.count();
        if (!count) {
          continue;
        }
        const latest = loc.nth(count - 1);
        const text = (await latest.innerText().catch(() => "")).trim();
        if (!text) {
          continue;
        }
        if (text === lastText) {
          stableCount += 1;
        } else {
          stableCount = 0;
          lastText = text;
        }
        if (stableCount >= 3) {
          logInfo(logger, "answer.selector", { selector, count }, page);
          return { text, selector };
        }
      } catch {
        // ignore
      }
    }
    await page.waitForTimeout(1000);
  }
  return null;
}

function escapeRegex(input: string): string {
  return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function escapeCssAttrValue(input: string): string {
  return input.replace(/["\\\\]/g, "\\$&");
}

async function isLocatorInViewport(page: Page, locator: Locator): Promise<boolean> {
  const box = await locator.boundingBox().catch(() => null);
  if (!box) {
    return false;
  }
  const viewport = page.viewportSize();
  if (!viewport) {
    return true;
  }
  const withinX = box.x + box.width > 0 && box.x < viewport.width;
  const withinY = box.y + box.height > 0 && box.y < viewport.height;
  return withinX && withinY;
}

async function findSourceCheckbox(page: Page, name: string, container: Locator | null): Promise<Locator | null> {
  const variants = [name];
  const noExt = name.replace(/\.[^.]+$/, "");
  if (noExt && noExt !== name) {
    variants.push(noExt);
  }
  for (const variant of variants) {
    const byRole = page.getByRole("checkbox", { name: new RegExp(escapeRegex(variant), "i") }).first();
    if ((await byRole.count()) > 0) {
      return byRole;
    }
    const attrValue = escapeCssAttrValue(variant);
    const byLabel = page.locator(`input[type="checkbox"][aria-label*="${attrValue}"]`).first();
    if ((await byLabel.count()) > 0) {
      return byLabel;
    }
  }
  if (container) {
    const inner = container.locator('input[type="checkbox"], [role="checkbox"]').first();
    if ((await inner.count()) > 0) {
      return inner;
    }
  }
  return null;
}

async function isCheckboxChecked(locator: Locator): Promise<boolean> {
  try {
    return await locator.evaluate((el) => {
      const anyEl = el as { tagName?: string; checked?: boolean; querySelector?: (selector: string) => unknown; getAttribute?: (name: string) => string | null };
      if (anyEl.tagName === "INPUT") {
        return Boolean(anyEl.checked);
      }
      const input = anyEl.querySelector ? (anyEl.querySelector('input[type="checkbox"]') as { checked?: boolean } | null) : null;
      if (input && typeof input.checked === "boolean") {
        return Boolean(input.checked);
      }
      const aria = anyEl.getAttribute ? anyEl.getAttribute("aria-checked") : null;
      return aria === "true";
    });
  } catch {
    return false;
  }
}

async function setSourceChecked(
  page: Page,
  name: string,
  desired: boolean
): Promise<{ present: boolean; checked: boolean; changed: boolean }> {
  const label = page.getByText(name, { exact: false }).first();
  const present = await label.isVisible().catch(() => false);
  if (!present) {
    return { present: false, checked: false, changed: false };
  }
  await label.scrollIntoViewIfNeeded().catch(() => {});
  const container = await findSourceContainer(label);
  const checkbox = await findSourceCheckbox(page, name, container);
  if (!checkbox) {
    return { present: true, checked: false, changed: false };
  }
  const current = await isCheckboxChecked(checkbox);
  if (current !== desired) {
    await checkbox.click({ force: true }).catch(() => {});
    await page.waitForTimeout(200);
  }
  const checked = await isCheckboxChecked(checkbox);
  return { present: true, checked, changed: current !== checked };
}

async function trySelectAllSources(page: Page, logger?: StepLogger): Promise<boolean> {
  const selectAll = page.locator(S.selectAllSourcesButton).first();
  const visible = await selectAll.isVisible().catch(() => false);
  if (!visible) {
    logInfo(logger, "selectAllSources.not_visible", {}, page);
    return false;
  }
  await selectAll.click({ timeout: 10_000 }).catch(() => {});
  logInfo(logger, "selectAllSources.clicked", {}, page);
  return true;
}

async function trySelectUncheckedSources(page: Page, names: string[], logger?: StepLogger): Promise<void> {
  let changed = 0;
  for (const name of names) {
    const result = await setSourceChecked(page, name, true);
    if (result.changed) {
      changed += 1;
    }
  }
  logInfo(logger, "selectUncheckedSources.done", { count: names.length, changed }, page);
}

async function uploadFilePaths(page: Page, filePaths: string[], logger?: StepLogger): Promise<void> {
  if (!filePaths.length) {
    return;
  }
  const openSourcesButton = page.locator(S.addSourcesButton).first();
  const uploadFilesButton = page.locator(S.uploadFilesButton).first();
  const uploadSourceButton = page.locator(S.uploadSourceButton).first();
  const genericUploadText = page.getByText(/Upload a source|Upload source|Upload files|Upload|上传来源|上传文件|上传/i).first();

  let uploaded = false;
  for (let attempt = 0; attempt < 3 && !uploaded; attempt += 1) {
    const chooserPromise = page.waitForEvent("filechooser", { timeout: 10_000 }).catch(() => null);
    try {
      const openVisible = await openSourcesButton.isVisible().catch(() => false);
      const filesVisible = await uploadFilesButton.isVisible().catch(() => false);
      const sourceVisible = await uploadSourceButton.isVisible().catch(() => false);
      const genericVisible = await genericUploadText.isVisible().catch(() => false);
      logInfo(
        logger,
        "uploadFilePaths.attempt",
        { attempt: attempt + 1, openVisible, filesVisible, sourceVisible, genericVisible },
        page
      );
      if (await openSourcesButton.isVisible().catch(() => false)) {
        await openSourcesButton.click({ timeout: 10_000 });
        await page.waitForTimeout(500);
      }
      if (await uploadFilesButton.isVisible().catch(() => false)) {
        await uploadFilesButton.click({ timeout: 10_000 });
      } else if (await uploadSourceButton.isVisible().catch(() => false)) {
        await uploadSourceButton.click({ timeout: 10_000 });
      } else if (await genericUploadText.isVisible().catch(() => false)) {
        await genericUploadText.click({ timeout: 10_000 });
      }
    } catch {
      // Ignore click failures and fall back to input detection below.
    }

    const chooser = await chooserPromise;
    if (chooser) {
      logInfo(logger, "uploadFilePaths.chooser", { attempt: attempt + 1 }, page);
      await chooser.setFiles(filePaths);
      logInfo(logger, "uploadFilePaths.setFiles", { attempt: attempt + 1, file_count: filePaths.length }, page);
      await detectUploadIndicator(page, logger);
      uploaded = true;
      break;
    }

    const input = page.locator('input[type="file"]').first();
    if ((await input.count()) > 0) {
      logInfo(logger, "uploadFilePaths.input", { attempt: attempt + 1 }, page);
      await input.setInputFiles(filePaths);
      logInfo(logger, "uploadFilePaths.setInputFiles", { attempt: attempt + 1, file_count: filePaths.length }, page);
      await detectUploadIndicator(page, logger);
      uploaded = true;
      break;
    }

    logInfo(logger, "uploadFilePaths.no_input", { attempt: attempt + 1 }, page);
    await page.keyboard.press("Escape").catch(() => {});
    await page.waitForTimeout(500);
  }

  if (!uploaded) {
    throw new Error("Failed to open upload dialog for sources.");
  }
}

type SourceCheck = {
  name: string;
  path: string;
  failed: boolean;
  present: boolean;
};

type SourceStatus = {
  name: string;
  present: boolean;
  failed: boolean;
  stalled: boolean;
  processing: boolean;
  checked: boolean;
};

async function retryFailedUploads(
  page: Page,
  filePaths: string[],
  logger?: StepLogger,
  maxRetries = 2
): Promise<{ reuploaded: string[] }> {
  const nameToPath = new Map<string, string>();
  for (const filePath of filePaths) {
    nameToPath.set(path.basename(filePath), filePath);
  }

  const reuploaded: string[] = [];
  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    const statuses = await collectSourceStatuses(page, Array.from(nameToPath.keys()), logger);
    const failed = statuses.filter((item) => item.failed).map((item) => item.name);
    const missing = statuses.filter((item) => !item.present).map((item) => item.name);
    let unchecked = statuses.filter((item) => item.present && !item.checked && !item.processing).map((item) => item.name);
    const stalled = statuses.filter((item) => item.stalled).map((item) => item.name);
    logInfo(
      logger,
      "retryFailedUploads.scan",
      {
        attempt: attempt + 1,
        failed,
        missing,
        stalled,
        unchecked,
        present: statuses.filter((s) => s.present).length
      },
      page
    );
    if (!failed.length && !missing.length && !stalled.length && !unchecked.length) {
      return { reuploaded };
    }

    if (unchecked.length && !failed.length && !missing.length && !stalled.length) {
      const selectedAll = await trySelectAllSources(page, logger);
      if (!selectedAll) {
        await trySelectUncheckedSources(page, unchecked, logger);
      }
      await page.waitForTimeout(1000);
      const after = await collectSourceStatuses(page, Array.from(nameToPath.keys()), logger);
      unchecked = after.filter((item) => item.present && !item.checked && !item.processing).map((item) => item.name);
      if (!unchecked.length) {
        return { reuploaded };
      }
    }

    const retryNames = Array.from(new Set([...failed, ...missing, ...stalled, ...unchecked]));
    if (retryNames.length) {
      await removeFailedSources(page, retryNames, logger);
    }

    const retryPaths = retryNames.map((name) => nameToPath.get(name)).filter(Boolean) as string[];
    if (!retryPaths.length) {
      break;
    }
    await uploadFilePaths(page, retryPaths, logger);
    reuploaded.push(...retryPaths);
    await waitForSourcesVisible(page, retryPaths, logger);
    await waitForIndexingReady(page, logger);
  }

  const finalChecks = await collectSourceChecks(page, Array.from(nameToPath.keys()), logger);
  const stillMissing = finalChecks.filter((item) => !item.present || item.failed).map((item) => item.name);
  if (stillMissing.length) {
    logInfo(logger, "retryFailedUploads.failed", { stillMissing }, page);
    throw new Error(`Sources failed after retry: ${stillMissing.join(", ")}`);
  }
  return { reuploaded };
}

async function collectSourceChecks(page: Page, names: string[], logger?: StepLogger): Promise<SourceCheck[]> {
  await page.keyboard.press("Escape").catch(() => {});
  const checks: SourceCheck[] = [];
  for (const name of names) {
    const label = page.getByText(name, { exact: false }).first();
    const present = await label.isVisible().catch(() => false);
    let failed = false;
    if (present) {
      const container = await findSourceContainer(label);
      if (container) {
        failed = await isSourceFailed(container, label);
      }
    }
    checks.push({ name, path: name, failed, present });
  }
  logInfo(
    logger,
    "collectSourceChecks",
    { total: checks.length, present: checks.filter((c) => c.present).length, failed: checks.filter((c) => c.failed).length },
    page
  );
  return checks;
}

async function findSourceContainer(label: Locator): Promise<Locator | null> {
  try {
    const container = label.locator(
      "xpath=ancestor::*[self::div or self::li or self::mat-list-item][1]"
    );
    if ((await container.count()) > 0) {
      return container.first();
    }
  } catch {
    return null;
  }
  return null;
}

async function collectSourceStatuses(page: Page, names: string[], logger?: StepLogger): Promise<SourceStatus[]> {
  await page.keyboard.press("Escape").catch(() => {});
  const statuses: SourceStatus[] = [];
  for (const name of names) {
    const label = page.getByText(name, { exact: false }).first();
    const present = await label.isVisible().catch(() => false);
    let failed = false;
    let processing = false;
    let checked = false;
    let stalled = false;
    if (present) {
      const container = await findSourceContainer(label);
      const checkbox = await findSourceCheckbox(page, name, container);
      if (checkbox) {
        checked = await isCheckboxChecked(checkbox);
      }
      if (container) {
        failed = await isSourceFailed(container, label);
        processing =
          (await container.locator(S.sourceProcessingSpinner).first().isVisible().catch(() => false)) ||
          (await container.locator(S.sourceProcessingText).first().isVisible().catch(() => false));
      }
    } else {
      processing = true;
    }
    statuses.push({ name, present, failed, stalled, processing, checked });
  }
  const globalProcessing =
    (await anyVisible(page.locator(S.sourceProcessingSpinner))) || (await anyVisible(page.locator(S.sourceProcessingText)));
  if (globalProcessing) {
    for (const item of statuses) {
      if (item.present && !item.failed) {
        item.processing = true;
      }
    }
  }
  logInfo(
    logger,
    "collectSourceStatuses",
    {
      total: statuses.length,
      present: statuses.filter((s) => s.present).length,
      processing: statuses.filter((s) => s.processing).length,
      failed: statuses.filter((s) => s.failed).length,
      checked: statuses.filter((s) => s.checked).length,
      checked_all: statuses.length > 0 && statuses.every((s) => s.checked),
      items: statuses.map((s) => ({
        name: s.name,
        present: s.present,
        checked: s.checked,
        processing: s.processing,
        failed: s.failed
      }))
    },
    page
  );
  return statuses;
}

async function removeFailedSources(page: Page, names: string[], logger?: StepLogger): Promise<void> {
  if (!names.length) {
    return;
  }
  const removeAll = await tryRemoveAllFailedSources(page, names, logger);
  if (removeAll) {
    return;
  }
  for (const name of names) {
    const label = page.getByText(name, { exact: false }).first();
    const present = await label.isVisible().catch(() => false);
    if (!present) {
      continue;
    }
    const container = await findSourceContainer(label);
    try {
      await (container ?? label).click();
    } catch {
      // Ignore
    }
    let removed = false;
    const removeBtn = page.locator(S.sourceRemoveButton).first();
    if (await removeBtn.isVisible().catch(() => false)) {
      await removeBtn.click().catch(() => {});
      removed = true;
    } else if (container) {
      const menuBtn = container.locator(S.sourceMenuButton).first();
      if (await menuBtn.isVisible().catch(() => false)) {
        await menuBtn.click().catch(() => {});
        const menuRemove = page.locator(S.sourceRemoveButton).first();
        if (await menuRemove.isVisible().catch(() => false)) {
          await menuRemove.click().catch(() => {});
          removed = true;
        }
      }
    }
    if (removed) {
      const confirm = page.locator(S.sourceRemoveButton).first();
      if (await confirm.isVisible().catch(() => false)) {
        await confirm.click().catch(() => {});
      }
    } else {
      await page.keyboard.press("Delete").catch(() => {});
    }
    await label.waitFor({ state: "hidden", timeout: 10_000 }).catch(() => {});
    logInfo(logger, "removeFailedSources", { name, removed }, page);
  }
}

async function isSourceFailed(container: Locator, label?: Locator): Promise<boolean> {
  const failedText = await container.locator(S.sourceErrorText).first().isVisible().catch(() => false);
  if (failedText) {
    return true;
  }
  const failedIcon = await container.locator(S.sourceFailedIcon).first().isVisible().catch(() => false);
  if (failedIcon) {
    return true;
  }
  const attr = await container.getAttribute("aria-label").catch(() => null);
  if (attr && /failed|error|unsupported|无法|失败|错误|不支持/i.test(attr)) {
    return true;
  }
  if (label) {
    const labelAttr = await label.getAttribute("aria-label").catch(() => null);
    if (labelAttr && /failed|error|unsupported|无法|失败|错误|不支持/i.test(labelAttr)) {
      return true;
    }
  }
  return false;
}

async function tryRemoveAllFailedSources(page: Page, names: string[], logger?: StepLogger): Promise<boolean> {
  for (const name of names) {
    const label = page.getByText(name, { exact: false }).first();
    const present = await label.isVisible().catch(() => false);
    if (!present) {
      continue;
    }
    const container = await findSourceContainer(label);
    const menuBtn = container ? container.locator(S.sourceMenuButton).first() : page.locator(S.sourceMenuButton).first();
    if (await menuBtn.isVisible().catch(() => false)) {
      await menuBtn.click().catch(() => {});
      const removeAllBtn = page.locator(S.removeAllFailedSourcesButton).first();
      if (await removeAllBtn.isVisible().catch(() => false)) {
        await removeAllBtn.click().catch(() => {});
        logInfo(logger, "removeFailedSources.all", { names }, page);
        await page.waitForTimeout(1000);
        return true;
      }
      await page.keyboard.press("Escape").catch(() => {});
    }
  }
  return false;
}

function logInfo(logger: StepLogger | undefined, step: string, data: Record<string, unknown>, page?: Page): void {
  if (!logger) {
    return;
  }
  logger.log({
    ts: new Date().toISOString(),
    step,
    phase: "info",
    url: page?.url(),
    data
  });
}

async function runStep<T>(
  logger: StepLogger | undefined,
  name: string,
  fn: () => Promise<T>,
  page?: Page
): Promise<T> {
  if (!logger) {
    return fn();
  }
  return logger.step(name, fn, page);
}
