SYSTEM_PROMPT = """You are a clinical trial protocol optimizer. 
You never make unreferenced medical claims. 
Each suggestion must cite at least one similar trial snippet and mark items that need medical review."""

SUGGESTION_TEMPLATE = """Given the current trial and top similar trials:
- Return JSON with `recommendations` (design/criteria/endpoints) and `open_questions_for_medical_review`.
- Each recommendation: action, rationale, evidence_trials[], risk_note, change_impact (small/medium/large).
- Evidence must include trial_id and short snippet.
"""

COMPARE_TEMPLATE = """Compare the retrieved trials and summarize common successful patterns and pitfalls.
Highlight design, criteria, and endpoint themes separately."""

FOCUS_GUIDE = """Focus options for retrieve_keyword_trials (which fields provide keywords):
- full: conditions + phase + interventions/drugs + endpoints + biomarkers + drug_moa targets/genes + outcomes/results + disposition + adverse_events + criteria (if present)
- condition: condition names only (CT.gov condition[]; TrialPanorama conditions.condition_name)
- drug: CT.gov interventions (name/type); TrialPanorama drugs (drug_name/rx_normalized_name/drugbank_name) + drug_moa targets/genes
- biomarker: biomarker_name + biomarker_genes
- endpoint: CT.gov primary endpoint names/parsed type; TrialPanorama primary_endpoint/domain/subdomain
- study: TrialPanorama studies fields (trial_type/recruitment_status/phase/sex/actual_accrual/target_accrual)
"""
