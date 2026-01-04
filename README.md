# Trial Agent MVP

Minimal end-to-end demo for clinical trial outcome risk estimation and protocol optimization suggestions. The focus is an auditable, explainable baseline: structured schema in, lightweight retrieval + heuristics out, with evidence snippets attached to every suggestion.

## Repository layout
- `data/processed/trials_ctgov_phase2_oncology.jsonl` — parsed CT.gov corpus built from provided Phase 2 oncology NCT IDs.
- `data/processed/trials.jsonl` — tiny demo corpus in the unified schema (kept for quick smoke tests).
- `data/processed/sample_input.json` — example trial to run through the pipeline.
- `src/trial_agent` — code (ingest, retrieval, features, prediction, agent, api).
- `eval/golden_cases.json` — hand-crafted cases for quick regression checks.
- `requirements.txt` — optional deps for the FastAPI wrapper.

## Build processed corpus from CT.gov XML
The repo includes `trial_agent/data/raw/Phase_2_filtered_icd_C00_D48.csv` with oncology Phase 2 NCT IDs and raw XML under `/opt/workspace/dataset/research/ML2/Trialbench/data_curation/raw_data`. Run:
```bash
cd /opt/workspace/dataset/research/ML2/Trialbench
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/preprocess/build_ctgov_corpus.py \
  --csv trial_agent/data/raw/Phase_2_filtered_icd_C00_D48.csv \
  --xml-root /opt/workspace/dataset/research/ML2/Trialbench/data_curation/raw_data \
  --output trial_agent/data/processed/trials_ctgov_phase2_oncology.jsonl
```
If you are already inside the `trial_agent` directory, use:
```bash
cd /opt/workspace/dataset/research/ML2/Trialbench/trial_agent
PYTHONPATH=src \
python -m trial_agent.preprocess.build_ctgov_corpus \
  --csv data/raw/Phase_2_filtered_icd_C00_D48.csv \
  --xml-root ../data_curation/raw_data \
  --output data/processed/trials_ctgov_phase2_oncology.jsonl
```
Defaults already point to these paths. The script parses XML into the unified schema and writes JSONL into `data/processed/`.

## DrugBank (optional, offline drug knowledge)
If you have DrugBank full XML at `data/drugbank/full_database.xml`, build a compact index for fast lookup:
```bash
cd /opt/workspace/dataset/research/ML2/Trialbench
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/preprocess/build_drugbank_minimal.py \
  --xml trial_agent/data/drugbank/full_database.xml \
  --output trial_agent/data/processed/drugbank_minimal.jsonl
```
The multi-agent ReAct pipeline exposes a `drugbank_lookup` tool that uses this file to ground target/gene hypotheses.

## TrialPanorama (optional, large-scale retrieval corpus)
Download the dataset locally (large ~4.7GB) and upload it to the server:
```bash
hf download TrialPanorama/TrialPanorama-database \
  --repo-type dataset \
  --local-dir TrialPanorama_raw
```
Build a retrieval-friendly JSONL from the parquet tables:
```bash
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/preprocess/build_trialpanorama_corpus.py \
  --raw-dir trial_agent/data/trialpanorama/raw \
  --output trial_agent/data/processed/trialpanorama_trials.jsonl
```
Then point the pipeline to it by updating `trial_agent/src/trial_agent/config.py` (`trialpanorama_processed`) or passing it to your loader.
For fast relation-based retrieval, build a JSONL offset index once:
```bash
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/preprocess/build_jsonl_index.py \
  --jsonl trial_agent/data/processed/trialpanorama_trials.jsonl
```
If relations are missing for some trials, build a SimHash index for fast fallback retrieval (no external deps):
```bash
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/preprocess/build_simhash_index.py \
  --jsonl trial_agent/data/processed/trialpanorama_trials.jsonl \
  --output trial_agent/data/processed/trialpanorama_trials.simhash.sqlite
```
When the SimHash index exists, the agent will automatically use it instead of loading the full corpus into memory.
If you want true semantic retrieval, build a FAISS vector index (requires `sentence-transformers`, `faiss`):
```bash
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/preprocess/build_vector_index.py \
  --jsonl trial_agent/data/processed/trialpanorama_trials.jsonl \
  --index-out trial_agent/data/indexes/vectors/trialpanorama_trials.faiss \
  --ids-out trial_agent/data/indexes/vectors/trialpanorama_trials.vector_ids.txt \
  --model BAAI/bge-m3 \
  --trust-remote-code \
  --normalize \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --batch-size 4 \
  --max-length 512 \
  --max-chars 8000 \
  --checkpoint-every 200000 \
  --fields full,condition,drug,biomarker,study,design,endpoint,outcome
```
The `full` focus is now structured into multiple chunks (one per field group); other focuses remain single-text embeddings capped by `--max-chars`.
Use `--resume` to continue from existing `.faiss` + `.vector_ids.txt` files (skips already indexed chunks).
When the FAISS index exists, the agent will prefer it over SimHash and the in-memory lexical index.

## Optional: call Dify for LLM suggestions
- Set your key (do not commit it): `export DIFY_API_KEY=...` and optionally `export DIFY_BASE_URL=https://ai-playground.trialos.com.cn/v1`.
- Or create a `.env` in the repo root with `DIFY_API_KEY=...` (auto-loaded).
The prompt payload uses a compacted view of each trial (key features only) to avoid redundant fields.
- Run pipeline with LLM:  
```bash
cd /opt/workspace/dataset/research/ML2/Trialbench
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/agent/orchestrator.py --use-dify --pretty
```
or from inside `trial_agent`:
```bash
cd /opt/workspace/dataset/research/ML2/Trialbench/trial_agent
PYTHONPATH=src \
python -m trial_agent.agent.orchestrator --use-dify --pretty
```
- The output field `dify_response` contains the raw answer from the Dify app, grounded with the current trial, retrieved snippets, and baseline prediction.

## ReAct single-agent (Python orchestrator + Dify)
This mode lets the LLM decide which tools to call (retrieve, summarize) and in what order, then produce a final JSON analysis (including an LLM heuristic `success_probability` if you want).
```bash
cd /opt/workspace/dataset/research/ML2/Trialbench
export DIFY_API_KEY="your-key"
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/agent/react_orchestrator.py --pretty
```
You can override tool depth and retrieval:
```bash
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/agent/react_orchestrator.py --top-k 5 --max-steps 6 --pretty
```
To skip lexical retrieval and only use the relation graph:
```bash
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/agent/react_orchestrator.py --relation-only --pretty
```
The final output is returned directly from the LLM. If it fails to finish within `max-steps`, the response includes a `partial_state`.

## ReAct multi-agent (Drug/Biomarker -> Design -> Outcome)
This mode runs 3 specialized agents sequentially, all backed by the same local tools/corpus:
- `DrugBiomarkerAgent`: drug/target/biomarker hypotheses + uncertainties
- `DesignAgent`: design risk assessment conditioned on drug/biomarker context
- `OutcomeSummaryAgent`: final success probability estimate (LLM heuristic) + drivers
```bash
cd /opt/workspace/dataset/research/ML2/Trialbench
export DIFY_API_KEY="your-key"
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/agent/multi_react_orchestrator.py --pretty
```
You can tune:
```bash
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/agent/multi_react_orchestrator.py --top-k 5 --max-steps 5 --pretty
```
To skip lexical retrieval and only use the relation graph:
```bash
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/agent/multi_react_orchestrator.py --relation-only --pretty
```

## Unified schema (abridged)
Each trial record follows:
```json
{
  "trial_id": "NCTxxxx",
  "condition": ["..."],
  "phase": "Phase 2",
  "interventions": [{"type": "Drug", "name": "..."}],
  "design": {"allocation": "...", "intervention_model": "...", "masking": "...", "primary_purpose": "...", "arms": [{"name": "...", "description": "..."}], "dose": "..."},
  "criteria": {"inclusion_text": "...", "exclusion_text": "...", "parsed": {"age_min": null, "age_max": null, "ecog_max": null, "prior_lines_max": null, "key_flags": []}},
  "endpoints": {"primary": [{"name": "...", "time_frame": "...", "description": "..."}], "secondary": [{"name": "..."}], "parsed": {"primary_type": "ORR"}},
  "outcome_label": {"status": "success|fail|unknown", "source": "synthetic", "notes": ""}
}
```

## Run the demo (CLI)
```bash
cd /opt/workspace/dataset/research/ML2/Trialbench
PYTHONPATH=trial_agent/src \
python trial_agent/src/trial_agent/agent/orchestrator.py --pretty
```
- Input defaults to `data/processed/sample_input.json`; override with `--input your_trial.json`.
- Retrieval defaults to `data/processed/trials_ctgov_phase2_oncology.jsonl` (set in `src/trial_agent/config.py`).
- Output includes `prediction`, `recommendations` (design/criteria/endpoints), retrieved evidence, pattern summary, and `open_questions_for_medical_review`.

## FastAPI wrapper (optional)
```bash
pip install -r trial_agent/requirements.txt
PYTHONPATH=trial_agent/src uvicorn trial_agent.api.app:get_app --reload --port 8080
# Then POST /run with JSON: {"trial": {...}, "top_k": 5}
```
The API simply calls the same orchestrator pipeline.

## How the MVP works
- **Normalize**: `ingest/parse_ctgov.py` fills missing fields and cleans text.
- **Retrieve**: relation graph first; if missing, FAISS vector index (if built) is used; otherwise SimHash or in-memory lexical ranking as fallback.
- Vector focus: `retrieve_similar_trials` accepts `focus` (full/condition/drug/biomarker/design/endpoint/outcome) so different tools can emphasize different facets.
- **Compare**: `agent/compare.py` surfaces common design/criteria/endpoint patterns across TopK.
- **Predict**: `prediction/baseline_rules.py` uses transparent heuristics (phase, control/randomization, endpoint type, criteria strictness) to output a probability and factor list.
- **Suggest**: `agent/suggest.py` emits design/criteria/endpoint recommendations with evidence snippets and risk notes; forces open questions for medical review when risk is high.

## Next steps
- Swap in a trained model in `prediction/train_xgb.py` when labeled data is available.
- Expand schema parsing (e.g., richer endpoint/criteria extraction) and retrieval (vector search on cleaned snippets).
- Add more golden cases and tighten prompt templates if connecting to an LLM for suggestion phrasing.
