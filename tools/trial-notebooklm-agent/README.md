# trial-notebooklm-agent

This repo provides a two-server setup:

- `notebooklm` (existing MCP server): interactive NotebookLM QA and debugging.
- `trial_extract` (new MCP server in this repo): automated end-to-end extraction
  (upload sources -> ask prompt packs -> parse JSON -> store outputs).

## Why two MCP servers

- Keep a stable, interactive NotebookLM tool for manual work.
- Run long, structured, repeatable extraction jobs via a dedicated server.

## Directory layout

- `prompts/`: prompt packs and system templates.
- `scripts/`: helper scripts (optional).
- `data/`: local staging and outputs.
- `packages/trial-extract-mcp/`: the MCP server implementation.

## Quick start (trial_extract)

1) Install deps (from the package directory):

```
cd packages/trial-extract-mcp
npm install
```

2) Run in dev:

```
npm run dev
```

3) Configure Codex MCP:

```
[mcp_servers.trial_extract]
cwd = "/abs/path/to/tools/trial-notebooklm-agent/packages/trial-extract-mcp"
command = "npx"
args = ["-y", "tsx", "src/index.ts"]
startup_timeout_sec = 30
tool_timeout_sec = 1800
```

## Environment

Copy `.env.example` to `.env` and fill values if you use remote fetching.

## Output

Results are stored under:

```
data/outputs/<NCTID>/
  raw/
  json/
  meta.json
```

## CTG extract v2 (NotebookLM)

This repo also supports a CTG extraction flow that reuses the existing
`tools/ctg_extract_v2/llm` prompt configs and fills missing CSV fields using
NotebookLM (with value + evidence), producing new `*_notebooklm.csv` outputs.

Expected inputs:

- CT.gov XML: `data/raw_data/**/<NCTID>.xml`
- CT.gov documents: `data/ctgov_documents/<NCTID>/*.pdf` (optional)
- CTG tables: `data/ctg_extract_v2/<NCTID>/{D_Design,D_Pop,D_Drug,R_Study,R_Arm}.csv`
- CTG text blocks: `data/ctg_extract_v2/<NCTID>/ctg_text_blocks.json` (already generated)

Run via MCP tool:

```
trial_extract.run_ctg_extract_v2({
  nctid: "NCT03693612",
  tables: ["D_Design", "D_Pop", "D_Drug", "R_Study", "R_Arm"],
  group: "all"
})
```

Outputs:

```
data/ctg_extract_v2/<NCTID>/notebooklm/
  D_Design_notebooklm.csv
  D_Design_notebooklm_evidence.jsonl
  D_Design_notebooklm_evidence.json
  ...
```

## Notes

- UI selectors may require updates as NotebookLM changes.
- Long tool runs need generous `tool_timeout_sec`.
