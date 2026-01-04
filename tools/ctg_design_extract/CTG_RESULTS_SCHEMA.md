# CTG Results Extraction Schema (v1)

This document defines the results tables for ClinicalTrials.gov XML extraction.
Design goals: keep measure and analysis separate, keep outcomes stable, and
avoid LLM unless needed for unstructured dose parsing.

## Overview (3 tables)

1) ctg_results_outcomes
2) ctg_results_measurements
3) ctg_results_analyses

## Table: ctg_results_outcomes

Purpose: one row per outcome in the results section.

Columns (CTG paths in parentheses):
- nct_id (id_info/nct_id)
- outcome_id (derived: outcome index or generated id)
- outcome_type (clinical_results/outcome_list/outcome/type)
- outcome_title (clinical_results/outcome_list/outcome/title)
- outcome_description (clinical_results/outcome_list/outcome/description)
- outcome_time_frame (clinical_results/outcome_list/outcome/time_frame)
- outcome_population (clinical_results/outcome_list/outcome/population)
- is_key_secondary (derived rule; not in CTG)

## Table: ctg_results_measurements

Purpose: one row per measurement value per group per class/category.

Columns (CTG paths in parentheses):
- nct_id (id_info/nct_id)
- outcome_id (derived: link to outcomes table)
- measure_id (derived: measure index within outcome)
- measure_title (clinical_results/outcome_list/outcome/measure/title)
- measure_description (clinical_results/outcome_list/outcome/measure/description)
- unit (clinical_results/outcome_list/outcome/measure/units)
- param_type (clinical_results/outcome_list/outcome/measure/param)
- dispersion_type (clinical_results/outcome_list/outcome/measure/dispersion)
- dispersion_value (derived from measurement spread or CI)
- lower_limit (clinical_results/.../measurement/lower_limit)
- upper_limit (clinical_results/.../measurement/upper_limit)
- n_analyzed (clinical_results/.../measure/analyzed_list/analyzed/count_list/count)
- group_id (clinical_results/outcome_list/outcome/group_list/group[@group_id])
- group_title (clinical_results/outcome_list/outcome/group_list/group/title)
- group_description (clinical_results/outcome_list/outcome/group_list/group/description)
- category_title (clinical_results/outcome_list/outcome/measure/class_list/class/category_list/category/title)
- class_title (clinical_results/outcome_list/outcome/measure/class_list/class/title)
- value (clinical_results/outcome_list/outcome/measure/class_list/class/category_list/category/measurement_list/measurement/@value)
- value_text (derived: textual form of value + spread)
- measure_time_frame (optional: if per-measure time_frame exists)
- measure_population (optional: if per-measure population exists)

Optional group-dose fields (from CTG main section, not results):
- arm_label (arm_group/arm_group_label)
- arm_description (arm_group/description)
- intervention_names (intervention/intervention_name linked by arm_group_label)
- intervention_types (intervention/intervention_type)
- dose_texts (intervention/description and/or arm_group/description)
- dose_value, dose_unit, dose_frequency, dose_route (derived from dose_texts)
- dose_source (arm_group.description or intervention.description)

## Table: ctg_results_analyses

Purpose: one row per analysis entry.

Columns (CTG paths in parentheses):
- nct_id (id_info/nct_id)
- outcome_id (derived: link to outcomes table)
- analysis_id (derived: analysis index within outcome)
- non_inferiority_type (clinical_results/outcome_list/outcome/analysis_list/analysis/non_inferiority_type)
- method (clinical_results/outcome_list/outcome/analysis_list/analysis/method)
- param_type (clinical_results/outcome_list/outcome/analysis_list/analysis/param_type)
- groups_desc (clinical_results/outcome_list/outcome/analysis_list/analysis/groups_desc)
- method_desc (clinical_results/outcome_list/outcome/analysis_list/analysis/method_desc)
- estimate_desc (clinical_results/outcome_list/outcome/analysis_list/analysis/estimate_desc)
- p_value (clinical_results/outcome_list/outcome/analysis_list/analysis/p_value)
- p_value_desc (clinical_results/outcome_list/outcome/analysis_list/analysis/p_value_desc)

## Source Classification

CTG direct (no LLM needed):
- All outcome fields (type/title/description/time_frame/population).
- All measure fields (title/units/param/dispersion/value/spread/limits).
- Group fields (group_id/title/description).
- Analysis fields (method/p_value/non_inferiority_type and *_desc fields).
- intervention_names/types and dose_texts (from arm_group/intervention description).

Derived (deterministic, no LLM needed):
- outcome_id, measure_id, analysis_id (sequence indices).
- value_text (format value + spread/limits into one string).
- is_key_secondary (rule-based mapping from secondary outcomes or Design file).

LLM optional:
- dose_value, dose_unit, dose_frequency, dose_route if dose_texts are complex
  and regex parsing is insufficient.

## Notes

- CTG does not label "key secondary" in results; if needed, define a rule
  (e.g., first secondary outcome or match Design EP_Key_Second).
- group_id is scoped to each outcome; use (nct_id, outcome_id, group_id) as a key.
- Keep measure and analysis separate to avoid mixing incompatible layers.
