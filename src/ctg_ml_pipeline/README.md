CTG NotebookLM ML Pipeline (Framework)

This folder provides a lightweight, end-to-end scaffold for:
1) checking extraction completeness
2) merging NotebookLM tables into a single dataset
3) feature visualization (missing rates + ranges)
4) feature selection (filter + embedded)
5) data imputation
6) model training + evaluation

Most steps run with `polars`. Feature selection and modeling require `scikit-learn`.

Quick start

1) Status check (which NCTs are complete)
   python -m ctg_ml_pipeline.cli status \
     --group-dir data/ctg_extract_v2/NSCLC_Trialpanorama_pd1 \
     --output-json data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/status.json

2) Merge (study-level)
   python -m ctg_ml_pipeline.cli merge \
     --group-dir data/ctg_extract_v2/NSCLC_Trialpanorama_pd1 \
     --mode study \
     --output-csv data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/merged_study.csv \
     --manifest-json data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/manifest.json

2b) Merge by table (keeps 7 separate tables)
   python -m ctg_ml_pipeline.cli merge-tables \
     --group-dir data/ctg_extract_v2/NSCLC_Trialpanorama_pd1 \
     --tables D_Design,D_Pop,D_Drug,R_Study,R_Study_Endpoint,R_Arm_Study,R_Arm_Study_Endpoint \
     --source notebooklm \
     --output-dir data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/merged_tables_notebooklm \
     --manifest-json data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/manifest_tables_notebooklm.json

3) Feature visualization
   python -m ctg_ml_pipeline.cli viz \
     --input-csv data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/merged_study.csv \
     --output-dir data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/feature_stats

4) Simple imputation
   python -m ctg_ml_pipeline.cli impute \
     --input-csv data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/merged_study.csv \
     --output-csv data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/merged_study_imputed.csv

5) Feature selection (requires scikit-learn)
   pip install scikit-learn

   # Filter stage (e.g., mutual information)
   python -m ctg_ml_pipeline.cli select \
     --input-csv data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/merged_study_imputed.csv \
     --target <TARGET_COLUMN> \
     --stage filter \
     --method mutual_info \
     --top-ratio 0.2 \
     --output-json data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/filter_selection.json

   # Embedded stage (e.g., L1 logistic)
   python -m ctg_ml_pipeline.cli select \
     --input-csv data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/merged_study_imputed.csv \
     --target <TARGET_COLUMN> \
     --stage embedded \
     --method l1 \
     --top-ratio 0.2 \
     --output-json data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/embedded_selection.json

6) Baseline modeling (requires scikit-learn)
   python -m ctg_ml_pipeline.cli train \
     --input-csv data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/merged_study_imputed.csv \
     --target <TARGET_COLUMN> \
     --model logistic \
     --output-json data/ctg_extract_v2/NSCLC_Trialpanorama_pd1/_ml_pipeline/metrics.json

Notes
- `merge` uses NotebookLM CSVs when available; pass --no-notebooklm to fall back to base CSVs.
- `mode=study` aggregates multi-row tables (D_Drug and R_Arm_Study) by StudyID.
- `mode=arm` keeps each arm row and joins study-level tables onto it.
- Selection + modeling assumes a classification target; adapt if you need regression.
