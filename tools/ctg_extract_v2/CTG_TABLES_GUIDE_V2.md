# CTG Tables Extraction Guide (CSR-Vars 2026-01-12.xlsx)

This is a scaffold for the new CTG extraction pipeline (v2).
Field lists are frozen in `tools/ctg_extract_v2/schema_fields.py`.

Tables:
- D_Design (study-level design)
- D_Pop (study-level population)
- D_Drug (one row per drug per NCT)
- R_Study (study-level results)
- R_Arm (one row per group)

Legend:
- CTG direct: direct ClinicalTrials.gov XML fields
- Derived: deterministic rules from CTG XML
- LLM: extracted from CTG text blocks
- External/TBD: not in CTG XML
- *_DB suffix: values sourced from DrugBank (kept separate from CTG/LLM fields)

Next steps:
- Fill in per-table field mappings in `tools/ctg_extract_v2/extract/*.py`.
- Add LLM fields and notes in `tools/ctg_extract_v2/llm/fill_*_with_llm.py`.
