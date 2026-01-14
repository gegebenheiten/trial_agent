#!/usr/bin/env python3
"""
Fill D_Design table using the shared CTG LLM interface (v2 schema).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fill_table_with_llm import PromptConfig, main as fill_main

OPS_INSTRUCTIONS = (
    "You extract D_Design operational fields from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Follow each field's definition exactly; return empty unless explicitly supported."
)

STAT_INSTRUCTIONS = (
    "You extract D_Design statistical/regulatory fields from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Be conservative; return empty unless the text explicitly supports the field."
)

EP_INSTRUCTIONS = (
    "You extract D_Design endpoint assessment fields from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Only fill when explicitly stated; otherwise return empty."
)

ALL_INSTRUCTIONS = (
    "You extract D_Design variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Follow each field's annotation as the definition; be conservative and evidence-based."
)

OPS_FIELDS = [
    "Inv_Prod",
    "GCP_Compliance",
    "Enroll_Duration_Plan",
    "FU_Duration_Plan",
    "Central_Lab",
    "Run_in",
]

STAT_FIELDS = [
    "Central_Random",
    "Rand_Ratio",
    "Random_Cluster",
    "Stratification",
    "No_Stratification",
    "Primary_Obj",
    "Success_Criteria_Text",
    "IRC",
    "Power",
    "Alpha",
    "Sided",
    "Subgroup",
    "Interim",
    "Timing_IA",
    "Alpha_Spend_Func",
    "Adaptive_Design",
    "Intercurrent_Events",
    "Gatekeeping_Strategy",
    "Reg_Alignment",
    "Fast_Track",
    "Breakthrough",
    "Priority_Review",
    "Accelerated_App",
    "Orphan_Drug",
    "Pediatric",
    "Rare_Disease",
]

EP_FIELDS = [
    "No_Key_Second_EP",
    "Name_Key_Second_EP",
    "Assess_Primary_EP",
    "Assess_Key_Second_EP",
    "Assess_Second_EP",
]

ALL_FIELDS = OPS_FIELDS + STAT_FIELDS + EP_FIELDS

OPS_MODULES = [
    "study_info",
    "eligibility",
    "arm_groups",
    "interventions",
]

STAT_MODULES = [
    "design_info",
    "study_info",
    "interventions",
    "keywords",
    "conditions",
]

EP_MODULES = [
    "primary_outcomes",
    "secondary_outcomes",
    "study_info",
]

ALL_MODULES = [
    "design_info",
    "study_info",
    "eligibility",
    "arm_groups",
    "interventions",
    "primary_outcomes",
    "secondary_outcomes",
    "keywords",
    "conditions",
]

DESIGN_NOTES = {
    'Add_On_Treat': "Definition: Whether the treatment is a combination therapy (two or more drugs/interventions used together).\nReturn 'yes' only if the text explicitly indicates a combination regimen (e.g., 'in combination with', 'plus', 'added to', 'co-administered with', 'with/and' another drug, multi-drug regimen). Require >=2 distinct drugs/interventions mentioned; otherwise empty. Do NOT infer from arm count.",
    'Adherence_Treat': "Definition: Adherence/compliance strategy or assessment for treatment (monitoring/management of adherence).\nReturn 'yes' only if explicitly describes adherence/compliance assessment or enforcement (e.g., pill count, diary, electronic monitoring, compliance assessed); else empty.\nNOTE: your earlier 'crossover/switch' interpretation is likely not aligned with CSR-Vars here.",
    'Central_Lab': "Definition: Whether a central laboratory is used (centralized lab testing).\nReturn 'yes' only if explicitly mentions central lab/central laboratory; return 'no' only if explicitly says local labs only / no central lab; else empty.",
    'Data_Cutoff_Date': "Definition: Data cutoff / database cutoff / database lock date used for analysis.\nReturn a date only if text explicitly mentions 'data cutoff', 'cut-off', 'database lock', 'DCO'. Normalize to YYYY-MM-DD if possible; if only month/year, keep literal (e.g., 'June 2023'); else empty. Do NOT use posted/update dates unless explicitly called data cutoff.",
    'Enroll_Duration_Plan': "Definition: Planned enrollment/recruitment duration.\nReturn only if explicitly stated (e.g., 'enrollment will last 18 months'); must include unit; do NOT compute from Start/Completion dates; else empty.",
    'FU_Duration_Plan': "Definition: Planned follow-up duration.\nReturn only if explicitly stated (e.g., 'followed for 12 months'); must include unit; do NOT infer; else empty.",
    'GCP_Compliance': "Definition: Whether the trial is conducted in compliance with Good Clinical Practice (GCP/ICH-GCP).\nReturn 'yes' only if explicitly states GCP/ICH-GCP (e.g., 'Good Clinical Practice', 'ICH E6'); return 'no' only if explicitly states not compliant; else empty.",
    'Inv_Prod': "Definition: investigational product(s) under study (drug-like interventions in experimental arms). Output: drug name(s) separated by '; '. Extract: use explicit labels like 'investigational product/drug' or experimental arm descriptions; exclude placebo/sham/vehicle/SOC/BSC comparators. If unclear -> empty.",
    'Route_Admin': "Definition: Route(s) of administration of the study intervention(s).\nChoose from allowed tokens; multiple use '; '. Map: intravenous/IV/infusion->iv; subcutaneous/SC->sc; intramuscular/IM->im; oral/by mouth->oral; inhaled/nebulized->inhalation; transdermal/patch->transdermal; topical->topical; otherwise 'other'.",
    'Run_in': "Definition: Whether there is a safety run-in / lead-in period (including placebo run-in).\nReturn 'yes' only if explicitly mentions run-in/lead-in period; return 'no' only if explicitly says no run-in; else empty.",
    'Treat_Duration': "Definition: Planned treatment duration / length of regimen (e.g., multiple cycles) or stopping rule.\nReturn literal duration/regimen phrase (e.g., '6 cycles', '24 weeks', 'until disease progression'); do not infer/convert; cycles allowed.",
}

STAT_REG_NOTES = {
    'Accelerated_App': "Meaning: Accelerated Approval pathway is explicitly stated (FDA accelerated approval or equivalent). Output: 'yes' only; otherwise empty. Extract: require explicit 'Accelerated Approval'. Do NOT infer from surrogate endpoint usage.",
    'Adaptive_Design': "Meaning: protocol includes a pre-planned adaptive feature that can modify design based on interim data (beyond ordinary randomization). Output: either 'yes' (if only generic adaptive is stated) OR short element keywords separated by ';' when explicit (e.g., 'sample size re-estimation; adaptive randomization; seamless II/III; drop-the-loser'). Extract: fill ONLY if TEXT explicitly uses 'adaptive' language or names a recognized adaptive element (SSR, blinded/unblinded sample size re-estimation, response-adaptive randomization, arm dropping, seamless phase II/III, enrichment, platform/master protocol adaptation). If TEXT says only 'adaptive design' with no element -> output 'yes'. Do NOT label as adaptive just because there is interim analysis or stratification.",
    'Alpha': "Meaning: planned Type I error / significance level for primary testing (alpha). Output: numeric string like '0.05','0.025','0.10'; otherwise empty. Extract: fill ONLY if TEXT explicitly states alpha/significance level (e.g., 'two-sided alpha=0.05', 'one-sided 2.5%'). Normalize: '5%' -> '0.05', '2.5%' -> '0.025' ONLY if explicitly tied to alpha/significance. Do NOT infer alpha from 'statistically significant'. If alpha differs by side (e.g., one-sided 0.025) still output the numeric alpha.",
    'Alpha_Spend_Func': "Meaning: named alpha-spending function / group-sequential boundary family used to control Type I error. Output: one short token only: prefer 'OBF','Pocock','Lan-DeMets','Shi-Hwang' (else use the named family as-is). Extract: fill ONLY if TEXT explicitly names the function/boundary, e.g., 'O'Brien-Fleming', 'Pocock', 'Lan-DeMets alpha spending', 'Hwang-Shih-DeCani'. Normalize allowed: 'O'Brien-Fleming'/'OBrien Fleming' -> 'OBF'; 'Hwang-Shih-DeCani' -> 'Shi-Hwang' (if you want one bucket). Do NOT infer from 'group sequential design' without naming the function.",
    'Breakthrough': "Meaning: drug has Breakthrough Therapy designation. Output: 'yes' only; otherwise empty. Extract: require explicit 'Breakthrough Therapy designation'.",
    'Central_Random': "Meaning: randomization assignment is performed via a centralized system (not local envelopes/site lists). Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly names a central randomization mechanism, e.g., 'IWRS', 'IVRS', 'IRT', 'interactive response technology/system', 'central randomization system', 'central randomization service'. Do NOT infer from 'randomized', Allocation='Randomized', or masking alone. Positive cues: 'IWRS will assign', 'randomization via IVRS', 'IRT-generated allocation'. Negative cues (leave empty): 'randomized', 'computer-generated randomization list' (without stating centralized system), 'sealed envelopes'/'site randomization list' (this is typically NOT central).",
    'Consistency_MRCT': "Meaning: in a multi-regional clinical trial (MRCT), TEXT explicitly states consistency of efficacy/safety across regions/ethnicities. Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly mentions MRCT/regional consistency, e.g., 'consistent across regions', 'no heterogeneity by region', 'treatment effect consistent in Asia and non-Asia'. Do NOT infer from multinational recruitment alone.",
    'Consistency_Sens_Ana_PE': "Meaning: sensitivity analyses for the primary endpoint are explicitly reported as consistent/robust vs primary analysis. Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly states consistency/robustness, e.g., 'results were consistent across sensitivity analyses', 'robust to sensitivity analyses'. Do NOT infer consistency from similar numbers without an explicit statement.",
    'Consistency_Sens_Ana_SE': "Meaning: sensitivity analyses for the key secondary endpoint are explicitly reported as consistent/robust. Output: 'yes' only; otherwise empty. Extract: same as Consistency_Sens_Ana_PE but tied to key secondary endpoint.",
    'Fast_Track': "Meaning: drug has FDA Fast Track designation (or explicitly says fast track designation). Output: 'yes' only; otherwise empty. Extract: require explicit 'Fast Track' / 'fast track designation'. Do NOT infer from unmet need/accelerated pathways.",
    'Gatekeeping_Strategy': "Meaning: explicit multiplicity control / multiple-endpoint testing procedure. Output: one short keyword/token (or short method name) such as 'hierarchical', 'fixed-sequence', 'gatekeeping', 'Bonferroni', 'Hochberg', 'Holm', 'fallback'. Extract: fill ONLY if TEXT explicitly states a multiplicity strategy/procedure. Do NOT infer from having multiple endpoints. If TEXT names multiple procedures, prefer the one described as primary multiplicity control for primary+key secondary.",
    'IRC': "Meaning: independent committee performs central/blinded endpoint assessment/adjudication (e.g., BICR/IRC/CEC). Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly mentions an independent review/adjudication body for endpoints, e.g., 'IRC', 'independent review committee', 'blinded independent central review (BICR/BIRC)', 'clinical events committee (CEC)', 'independent adjudication committee', 'central imaging review'. Do NOT infer from 'blinded study' alone. Positive cues: 'BICR will assess tumor response', 'CEC adjudicates MACE'.",
    'Intercurrent_Events': "Meaning: explicitly defined intercurrent events (ICH E9(R1) estimand context) and how handled. Output: short semicolon-separated list of event tokens (e.g., 'treatment discontinuation; rescue medication; death'). Extract: fill ONLY if TEXT explicitly uses estimand/intercurrent-event language and names events/handling (e.g., 'discontinuation handled by treatment policy strategy', 'rescue meds handled by hypothetical strategy'). If TEXT only mentions withdrawals/missing data generally without estimand/ICE framing -> empty.",
    'Interim': "Meaning: any formal interim analysis/look for efficacy/futility/safety with planned decision/boundaries. Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly states 'interim analysis', 'interim look', 'group sequential', 'futility analysis', 'efficacy stopping boundary', 'IDMC/DMC interim review for efficacy/futility'. Do NOT infer from 'DMC will review safety periodically' unless it explicitly says interim analysis/look.",
    'Key_Second_EP_SC': "Meaning: whether the KEY SECONDARY endpoint is clinical (C) or surrogate/intermediate (S). Output: 'C' or 'S' only; otherwise empty. Extract: same rule as Primary_EP_SC but apply to the key secondary endpoint title/definition. If multiple key secondary endpoints are listed and you cannot tell which is 'key' -> empty.",
    'No_Stratification': "Meaning: count of distinct stratification factors used in the randomization scheme. Output: integer count only (e.g., '1','2','3'); otherwise empty. Extract: fill ONLY when TEXT explicitly lists stratification factors; count factors separated by commas/and/by. Counting rule: each concept = 1 factor even if it has levels (e.g., 'ECOG 0-1 vs 2' counts as 1). Examples: 'stratified by age, sex, region' -> 3; 'stratified by ECOG and PD-L1 status' -> 2. Negative cues: only says 'stratified' or 'stratified randomization' but no factors listed -> empty.",
    'Obj_Primary': "Meaning: statistical objective framework of the primary hypothesis (superiority vs non-inferiority). Output: 'superiority' OR 'non-inferiority' only. Extract: fill ONLY if TEXT explicitly states 'superiority', 'non-inferiority/noninferiority', or defines a non-inferiority margin (NI margin) / NI bound. Do NOT guess from endpoint type or comparator. If both are mentioned (e.g., NI then test superiority), output the one stated as primary objective; if unclear -> empty.",
    'Orphan_Drug': "Meaning: Orphan Drug designation is explicitly stated. Output: 'yes' only; otherwise empty. Extract: require explicit 'Orphan Drug designation'/'orphan designation'. Do NOT infer from rare disease alone.",
    'Pediatric': "Meaning: study is explicitly positioned as pediatric development/indication (children/adolescents as pediatric program). Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly says 'pediatric', 'children', 'adolescent pediatric program', or references pediatric regulatory programs (PIP, PREA) / pediatric formulation, etc. Do NOT fill merely because eligibility includes age 16-17 unless explicitly framed as pediatric.",
    'Post_Hoc_Ana': "Meaning: any analysis explicitly described as post hoc. Output: 'yes' only; otherwise empty. Extract: fill ONLY when TEXT uses the phrase 'post hoc' (case-insensitive). Do NOT treat 'exploratory' as post hoc unless it literally says post hoc.",
    'Power': "Meaning: planned statistical power for the primary hypothesis. Output: percent string like '80%','90%'; otherwise empty. Extract: fill ONLY if TEXT explicitly states power (e.g., '80% power', 'power=0.9'). Normalize: '90 percent' -> '90%'; '0.8 power' -> '80%' ONLY if explicitly labeled as power. If multiple powers are given (different endpoints/scenarios), prefer the primary endpoint/power statement; if unclear -> empty.",
    'Primary_EP_SC': "Meaning: whether the PRIMARY endpoint is a clinical endpoint (C) or a surrogate/intermediate endpoint (S). Output: 'C' or 'S' only; otherwise empty. Extract: you MAY classify using the primary endpoint title/definition in TEXT when classification is clear. C (clinical) = how patient feels/functions/survives: overall survival, all-cause mortality, hospitalization, MACE, stroke/MI, symptom scales, functional outcomes, QoL/PRO, exacerbations, clinical remission/relapse requiring treatment. S (surrogate/intermediate) = biomarker/response/proxy: PFS/EFS/DFS, ORR, pCR, tumor response by RECIST, viral load, antibody titers, imaging measures, lab values, tumor size, MRD negativity. Rule: if endpoint is clearly a biomarker/response/time-to-progression proxy -> S; if it is survival/mortality or direct clinical events/PRO -> C. If endpoint text is missing/too vague -> empty. Evidence can be the endpoint phrase itself.",
    'Primary_Obj': "Meaning: statistical objective framework of the primary hypothesis (superiority vs non-inferiority). Output: 'superiority' OR 'non-inferiority' only. Extract: fill ONLY if TEXT explicitly states 'superiority', 'non-inferiority/noninferiority', or defines a non-inferiority margin (NI margin) / NI bound. Do NOT guess from endpoint type or comparator. If both are mentioned (e.g., NI then test superiority), output the one stated as primary objective; if unclear -> empty.",
    'Priority_Review': "Meaning: Priority Review designation/intent is explicitly stated. Output: 'yes' only; otherwise empty. Extract: require explicit phrase 'Priority Review'. Do NOT infer from expedited review language.",
    'Rand_Ratio': "Meaning: planned allocation ratio between treatment arms at randomization. Output: ratio string only, formatted 'x:y' (e.g., '1:1','2:1','3:2'); no extra words. Extract: fill ONLY if TEXT explicitly states the ratio (e.g., '2:1', '2 to 1', '2-1', 'two-to-one'). Normalize: convert '2 to 1'/'2-1'/'two to one' -> '2:1'. If multiple stages/parts report different ratios, prefer the main/randomization ratio for the primary phase; if ambiguous -> empty. Negative cues: 'randomized' without ratio -> empty.",
    'Random_Cluster': "Meaning: cluster randomized trial where the unit of randomization is a group/cluster (site, clinic, school, village, household, ward), not the individual participant. Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly says 'cluster randomized/randomised', 'group randomized', 'randomized by site/clinic/school', 'clusters were randomized', 'unit of randomization is ...'. Do NOT infer from multi-center recruitment alone. If TEXT says 'stratified by site' that is NOT cluster randomization (leave empty unless it says randomize-by-site).",
    'Random_Crossover': "Meaning: randomized crossover intervention model (participants receive sequences of treatments, e.g., AB/BA) with possible washout/periods. Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly contains 'crossover/cross-over', 'randomized crossover', 'treatment sequence', 'period 1/period 2', 'AB/BA', 'washout period', or CT.gov structured says 'Intervention Model: Crossover Assignment'. Do NOT infer from 'switch' or 'open-label extension' unless clearly described as crossover design.",
    'Random_Fact': "Meaning: randomized factorial design (simultaneous randomization to >1 intervention factors, e.g., 2x2). Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly says 'factorial', '2x2 factorial', '2-by-2', 'factorial assignment', or CT.gov structured says 'Intervention Model: Factorial Assignment'. Do NOT infer from having many arms unless factorial is explicitly described.",
    'Random_Parallel': "Meaning: randomized trial uses a parallel-group intervention model (participants stay in assigned arm; no crossover by design). Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly states 'parallel-group', 'parallel assignment', 'parallel design', or CT.gov structured Study Design explicitly says 'Intervention Model: Parallel Assignment'. Do NOT infer from 'two arms'/'randomized' alone. Negative cues: explicit crossover/factorial/cluster wording -> do not set this unless parallel is explicitly stated.",
    'Rare_Disease': "Meaning: condition or program is explicitly labeled as a rare disease/rare disorder (or rare disease program language). Output: 'yes' only; otherwise empty. Extract: require explicit 'rare disease'/'rare disorder' wording or equivalent program statement. Do NOT infer from prevalence knowledge.",
    'Reg_Alignment': "Meaning: explicit prior alignment/interaction with regulator(s) about protocol/endpoints/design (not mere GCP compliance). Output: short token(s) like 'FDA','EMA','PMDA','ICH E9','Scientific Advice','SPA'; otherwise empty. Extract: fill ONLY if TEXT explicitly mentions regulator meeting/advice/agreement, e.g., 'FDA End-of-Phase 2 meeting', 'EMA Scientific Advice', 'Special Protocol Assessment (SPA)', 'protocol agreed with FDA/EMA', 'aligned with ICH E9(R1) estimand guidance' (only if stated as alignment target). Do NOT fill for generic boilerplate 'conducted per ICH-GCP' unless it claims alignment/endpoint agreement.",
    'Reg_Audit': "Meaning: explicit report of major issues/findings from regulatory inspection/audit (e.g., GCP inspection findings). Output: short issue phrase only (e.g., 'data integrity findings', 'protocol deviations cited'); otherwise empty. Extract: fill ONLY if TEXT explicitly mentions inspection/audit and describes findings/observations (483, warning letter, critical/major findings, inspection deficiencies). If inspection/audit is not mentioned -> empty.",
    'Sided': "Meaning: sidedness of the primary hypothesis test (one-sided vs two-sided). Output: '1' for one-sided, '2' for two-sided; otherwise empty. Extract: fill ONLY if TEXT explicitly says 'one-sided/one tailed' or 'two-sided/two tailed'. Heuristic allowed ONLY with explicit alpha wording: if TEXT says 'one-sided 2.5%' -> '1'; if says 'two-sided 5%' -> '2'. Otherwise empty.",
    'Stratification': "Meaning: stratified randomization is used (randomization performed within strata defined by baseline factors). Output: 'yes' only; otherwise empty. Extract: fill if TEXT explicitly says 'stratified randomization' / 'randomization stratified by ...' / 'stratification factor(s)'. Also fill if TEXT lists stratification factors in the context of randomization (e.g., 'randomization will be stratified by age, region'). If only says 'stratified' without clarifying it is for randomization -> usually empty unless clearly tied to allocation. Consistency rule: if No_Stratification is filled (>=1) then Stratification should be 'yes' (because factors imply stratified randomization).",
    'Subgroup': "Meaning: the primary objective/hypothesis is targeted to a predefined subgroup (not just exploratory subgroup analyses). Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly states primary analysis/primary endpoint evaluation is in a subgroup (e.g., 'primary analysis in PD-L1 TPS >=50%', 'primary endpoint tested in biomarker-positive population'). Do NOT fill for generic phrases like 'subgroup analyses will be performed' or 'exploratory subgroup analyses'.",
    'Success_Criteria_Text': "Meaning: explicit statistical success criterion/decision rule for declaring primary success. Output: short verbatim criterion phrase (compact, no long prose). Extract: fill ONLY if TEXT defines a rule like: 'p<0.05', 'two-sided p<0.05', 'lower bound of 95% CI > 1', 'upper bound of 95% CI < 1', 'non-inferiority if lower bound > -10%', 'success if HR < 0.8 and p<0.025'. Prefer the main primary endpoint decision rule; if multiple gates exist, capture the top-level rule succinctly. Do NOT invent criteria from generic 'statistically significant'.",
    'Timing_IA': "Meaning: when the interim analysis is planned (events/information fraction/timepoint). Output: short timing phrase only (no extra explanation), e.g., 'after 50% events', '60% information fraction', 'after 100 events', 'at week 24'. Extract: fill ONLY if TEXT states a concrete interim timing trigger. If Interim='yes' but timing not specified -> empty. Prefer: information fraction/events over vague time (unless time is the only explicit trigger).",
}

EP_NOTES = {
    "No_Key_Second_EP": (
        "Meaning: count of key secondary endpoints explicitly labeled as key secondary. "
        "Output: integer count only. "
        "Extract: fill ONLY if TEXT explicitly uses 'key secondary endpoint(s)'."
    ),
    "Name_Key_Second_EP": (
        "Meaning: name(s) of key secondary endpoint(s). "
        "Output: semicolon-separated names. "
        "Extract: fill ONLY if TEXT explicitly labels key secondary endpoints."
    ),
    "Assess_Primary_EP": (
        "Meaning: assessment method/criteria used for primary endpoint (e.g., RECIST v1.1, central imaging review, questionnaire). "
        "Output: method/criteria phrase only. "
        "Extract: fill ONLY if explicitly stated."
    ),
    "Assess_Key_Second_EP": (
        "Meaning: assessment method/criteria used for key secondary endpoint(s). "
        "Output: method/criteria phrase only. "
        "Extract: fill ONLY if explicitly stated."
    ),
    "Assess_Second_EP": (
        "Meaning: assessment method/criteria used for secondary endpoint(s). "
        "Output: method/criteria phrase only. "
        "Extract: fill ONLY if explicitly stated."
    ),
}

NOTES = {}
NOTES.update(DESIGN_NOTES)
NOTES.update(STAT_REG_NOTES)
NOTES.update(EP_NOTES)

PROMPT_CONFIGS = {
    "ops": PromptConfig(
        instructions=OPS_INSTRUCTIONS,
        notes=NOTES,
        llm_fields=OPS_FIELDS,
        text_modules=OPS_MODULES,
    ),
    "stat": PromptConfig(
        instructions=STAT_INSTRUCTIONS,
        notes=NOTES,
        llm_fields=STAT_FIELDS,
        text_modules=STAT_MODULES,
    ),
    "endpoint": PromptConfig(
        instructions=EP_INSTRUCTIONS,
        notes=NOTES,
        llm_fields=EP_FIELDS,
        text_modules=EP_MODULES,
    ),
    "all": PromptConfig(
        instructions=ALL_INSTRUCTIONS,
        notes=NOTES,
        llm_fields=ALL_FIELDS,
        text_modules=ALL_MODULES,
    ),
}


def main() -> None:
    def pop_arg(flag: str) -> str:
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                value = sys.argv[idx + 1]
                del sys.argv[idx : idx + 2]
                return value
        return ""

    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "D_Design"])
    group = pop_arg("--group") or pop_arg("--pass")
    group = (group or "all").strip().lower()
    aliases = {
        "operational": "ops",
        "design": "ops",
        "stat_reg": "stat",
        "reg": "stat",
        "statistics": "stat",
        "endpoint": "endpoint",
        "ep": "endpoint",
        "all": "all",
    }
    group = aliases.get(group, group)
    prompt_config = PROMPT_CONFIGS.get(group, PROMPT_CONFIGS["all"])
    fill_main(prompt_config=prompt_config)


if __name__ == "__main__":
    main()
