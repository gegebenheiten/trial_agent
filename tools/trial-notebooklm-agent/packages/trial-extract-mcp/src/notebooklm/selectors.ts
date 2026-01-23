export const S = {
  createNotebookButton:
    'button[aria-label="Create new notebook"], button[aria-label="Create notebook"], button:has-text("Create new"), button:has-text("Create"), button:has-text("创建笔记本"), button:has-text("新建笔记本")',
  notebookTitleInput:
    'input[aria-label="Notebook name"], input[aria-label*="Notebook" i], input[aria-label*="title" i], input[aria-label*="笔记本" i], input[aria-label*="标题" i], input[placeholder*="Notebook" i], input[placeholder*="title" i], input[placeholder*="笔记本" i], input[placeholder*="标题" i]',
  notebookTitleEditable: 'header [contenteditable="true"], header [role="textbox"]',
  notebookTitleText: 'text=/Untitled notebook/i, text=/未命名笔记本/i, header h1, header h2, header [role="heading"]',
  addSourcesButton:
    'button[aria-label="Add source"], button[aria-label="Add sources"], button:has-text("Add sources"), button:has-text("Add source"), button:has-text("添加来源"), button:has-text("新增来源")',
  uploadEmptyStateButton:
    'button:has-text("Upload a source"), button:has-text("Upload source"), button:has-text("Upload"), button:has-text("上传来源"), button:has-text("上传文件"), button:has-text("上传")',
  uploadFilesButton: 'button:has-text("Upload files"), button:has-text("上传文件")',
  uploadSourceButton:
    'button:has-text("Upload a source"), button:has-text("Upload source"), button:has-text("Upload files"), button[aria-label*="upload source" i], button:has-text("Upload"), button:has-text("上传"), button:has-text("上传来源"), button:has-text("上传文件"), [role="button"]:has-text("Upload"), [role="button"]:has-text("上传"), [role="menuitem"]:has-text("Upload"), [role="menuitem"]:has-text("上传"), a:has-text("Upload"), label:has-text("Upload"), label:has-text("上传"), label:has-text("上传来源"), label:has-text("Upload a source")',
  sourceProcessingSpinner: '[aria-label*="Processing" i]',
  sourceProcessingText: 'text=/Processing|Indexing|Uploading|处理中|索引中|正在处理|上传中/i',
  sourceErrorText: 'text=/Upload failed|Failed to upload|Processing failed|Unsupported|Error|失败|无法|错误|不支持/i',
  sourceFailedIcon:
    'mat-icon[data-mat-icon-name*="error" i], mat-icon[data-mat-icon-name*="warning" i], mat-icon[data-mat-icon-name*="report" i], mat-icon[data-mat-icon-name*="cancel" i], mat-icon[data-mat-icon-name*="block" i], [aria-label*="error" i], [aria-label*="failed" i], [aria-label*="warning" i], [data-testid*="error" i], [data-icon*="error" i]',
  sourceRemoveButton:
    'button:has-text("Remove"), button:has-text("Delete"), button:has-text("移除"), button:has-text("删除"), button[aria-label*="remove" i], button[aria-label*="delete" i], button[aria-label*="移除" i], button[aria-label*="删除" i]',
  removeAllFailedSourcesButton:
    'button:has-text("Remove all failed sources"), button:has-text("移除所有失败来源"), button:has-text("删除所有失败来源"), button:has-text("移除全部失败"), button:has-text("删除全部失败")',
  sourceMenuButton:
    'button[aria-label*="more" i], button[aria-label*="options" i], button:has-text("More"), button:has-text("更多"), button[aria-label*="更多" i], button:has(mat-icon[data-mat-icon-name*="more" i]), button:has(mat-icon[data-mat-icon-name*="more_vert" i]), button:has(mat-icon:has-text("more_vert"))',
  chatInput:
    'textarea.query-box-input, textarea[aria-label="Input for queries"], textarea[aria-label="Feld für Anfragen"], textarea[aria-label="Query box"], textarea[aria-label*="query" i], textarea[aria-label*="ask" i], textarea[aria-label*="提问" i], textarea[aria-label*="问题" i], textarea[aria-label*="输入" i]:not([aria-label*="discover sources" i]), textarea[placeholder*="ask" i], textarea[placeholder*="提问" i], textarea[placeholder*="输入" i]:not([placeholder*="search the web" i])',
  queryInputSelectors: [
    "textarea.query-box-input",
    'textarea[aria-label="Input for queries"]',
    'textarea[aria-label="Feld für Anfragen"]',
    'textarea[aria-label="Query box"]',
    'textarea[aria-label*="query" i]',
    'textarea[aria-label*="ask" i]',
    'textarea[aria-label*="提问" i]',
    'textarea[aria-label*="问题" i]',
    'textarea[aria-label*="输入" i]:not([aria-label*="discover sources" i])',
    'textarea[placeholder*="ask" i]',
    'textarea[placeholder*="提问" i]',
    'textarea[placeholder*="输入" i]:not([placeholder*="search the web" i])'
  ],
  skillQueryInputSelectors: [
    "textarea.query-box-input",
    'textarea[aria-label="Feld für Anfragen"]',
    'textarea[aria-label="Input for queries"]'
  ],
  chatPanel: 'section.chat-panel',
  answerBlocks: '[data-testid="answer"], div[data-message-author="assistant"], div[role="article"]',
  answerFallback: 'div[aria-live="polite"]',
  responseSelectors: [
    ".to-user-container .message-text-content",
    "[data-message-author='bot']",
    "[data-message-author='assistant']",
    "[data-testid='answer']",
    "div[role='article']",
    "div[aria-live='polite']"
  ],
  skillResponseSelectors: [
    ".to-user-container .message-text-content",
    "[data-message-author='bot']",
    "[data-message-author='assistant']"
  ],
  thinkingIndicator: "div.thinking-message",
  stopGeneratingButton: 'button:has-text("Stop"), button[aria-label*="Stop" i]',
  regenerateButton:
    'button:has-text("Regenerate"), button[aria-label*="Regenerate" i], button:has-text("重新生成"), button:has-text("重新回答")',
  copyAnswerButton:
    'button:has-text("Copy"), button[aria-label*="Copy" i], button:has-text("复制"), button:has(mat-icon[data-mat-icon-name*="copy" i]), button:has(mat-icon[svgicon*="copy" i]), button:has([data-mat-icon-name*="copy" i])',
  sendButton:
    'button[aria-label*="Send" i], button[aria-label*="发送" i], button:has-text("Send"), button:has-text("发送"), button:has(mat-icon[data-mat-icon-name*="send" i]), button:has([data-mat-icon-name*="send" i]), button:has(mat-icon[svgicon*="send" i]), button:has(mat-icon[data-mat-icon-name*="arrow_forward" i])',
  sendIcon:
    'mat-icon[data-mat-icon-name*="arrow_forward" i], mat-icon:has-text("arrow_forward"), mat-icon[aria-label*="Send" i], mat-icon[aria-label*="发送" i]',
  loginButton: 'a:has-text("Sign in"), button:has-text("Sign in"), a:has-text("Log in"), button:has-text("Log in")',
  shareButton: 'button[aria-label="Share notebook"], button:has-text("Share")',
  configureNotebookButton:
    'button[aria-label="Configure notebook"], button:has-text("Configure"), button:has-text("配置"), button:has-text("设置"), button:has-text("重命名")',
  notebookSettingsButton:
    'button[aria-label*="Settings" i], button:has-text("Settings"), button:has-text("设置"), button[aria-label*="设置" i], a:has-text("Settings"), a:has-text("设置"), [role="button"]:has-text("Settings"), [role="button"]:has-text("设置")',
  notebookHeaderMenuButton:
    'button[aria-label*="More" i], button[aria-label*="Options" i], button[aria-label*="Menu" i], button:has-text("More"), button:has-text("更多"), button:has-text("更多选项")',
  notebookCardMenuButton:
    'button[aria-label*="More" i], button[aria-label*="Options" i], button[aria-label*="Menu" i], button:has-text("More"), button:has-text("更多")',
  deleteNotebookButton:
    'button:has-text("Delete notebook"), button:has-text("Move to trash"), button:has-text("Delete"), button:has-text("删除笔记本"), button:has-text("移到回收站"), button:has-text("删除")',
  confirmDeleteButton:
    'button:has-text("Delete"), button:has-text("Confirm"), button:has-text("OK"), button:has-text("确定"), button:has-text("删除"), button:has-text("确认")',
  selectAllSourcesButton: 'button:has-text("Select all sources"), button:has-text("选择所有来源")'
};
