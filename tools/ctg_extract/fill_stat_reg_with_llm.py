#!/usr/bin/env python3
"""
Fill Stat_Reg table using the shared CTG LLM interface.
"""

import sys
from typing import List

from fill_ctg_table_with_llm import PromptConfig, main as fill_main

# NOTE:
# - Field definitions come from CSR-Vars Stat_Reg sheet "Annotations" (shown in FIELDS_START).
# - This file adds field-wise decision rules + output constraints + examples (STAT_REG_NOTES),
#   and global normalization permissions (extra rules) to reduce ambiguity.

INSTRUCTIONS = (
    "You extract Stat_Reg (statistical/regulatory) variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Read each field's definition in FIELDS_START (from CSR-Vars 'Annotations'), then apply the field's Note rules.\n"
    "Output must be conservative, evidence-grounded, and schema-compliant.\n"
    "\n"
    "Scope focus:\n"
    "- Statistics: randomization logistics/ratio/stratification, objective type, endpoint type (surrogate vs clinical),\n"
    "  IRC/BIRC/adjudication, power/alpha/sidedness, interim & alpha spending, adaptive design, multiplicity/gatekeeping,\n"
    "  sensitivity consistency, post-hoc, success criteria, intercurrent events/estimand.\n"
    "- Regulatory: protocol alignment with agencies/guidelines, expedited programs (Fast Track, Breakthrough, etc.),\n"
    "  orphan/pediatric/rare disease flags, audit/inspection issues, MRCT regional consistency.\n"
    "\n"
    "Important:\n"
    "- CT.gov often lacks SAP-level details. Do NOT infer; return empty unless TEXT explicitly supports it.\n"
    "- Prefer structured design text: Structured Design, Arm Groups, Interventions, Outcome definitions; ignore result tables.\n"
)

LLM_FIELDS = [
    "Central_Random",
    "Rand_Ratio",
    "Random_Parallel",
    "Random_Crossover",
    "Random_Fact",
    "Random_Cluster",
    "Stratification",
    "No_Stratification",
    "Obj_Primary",
    "Primary_EP_SC",
    "Key_Second_EP_SC",
    "IRC",
    "Power",
    "Alpha",
    "Sided",
    "Subgroup",
    "Interim",
    "Timing_IA",
    "Alpha_Spend_Func",
    "Adaptive_Design",
    "Consistency_Sens_Ana_PE",
    "Consistency_Sens_Ana_SE",
    "Post_Hoc_Ana",
    "Success_Criteria_Text",
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
    "Reg_Audit",
    "Consistency_MRCT",
]

# Field-wise rules:
# - Each note is appended after the Excel annotation line in FIELDS_START.
# - Keep outputs short and normalized only where Extra rules allow.
STAT_REG_NOTES = {
    # ---------------------------
    # Statistics
    # ---------------------------
    "Central_Random": (
        "Meaning: randomization assignment is performed via a centralized system (not local envelopes/site lists). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly names a central randomization mechanism, e.g., 'IWRS', 'IVRS', 'IRT', "
        "'interactive response technology/system', 'central randomization system', 'central randomization service'. "
        "Do NOT infer from 'randomized', Allocation='Randomized', or masking alone. "
        "Positive cues: 'IWRS will assign', 'randomization via IVRS', 'IRT-generated allocation'. "
        "Negative cues (leave empty): 'randomized', 'computer-generated randomization list' (without stating centralized system), "
        "'sealed envelopes'/'site randomization list' (this is typically NOT central)."
    ),

    "Rand_Ratio": (
        "Meaning: planned allocation ratio between treatment arms at randomization. "
        "Output: ratio string only, formatted 'x:y' (e.g., '1:1','2:1','3:2'); no extra words. "
        "Extract: fill ONLY if TEXT explicitly states the ratio (e.g., '2:1', '2 to 1', '2-1', 'two-to-one'). "
        "Normalize: convert '2 to 1'/'2-1'/'two to one' -> '2:1'. "
        "If multiple stages/parts report different ratios, prefer the main/randomization ratio for the primary phase; "
        "if ambiguous -> empty. "
        "Negative cues: 'randomized' without ratio -> empty."
    ),
    "Random_Parallel": (
        "Meaning: randomized trial uses a parallel-group intervention model (participants stay in assigned arm; no crossover by design). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly states 'parallel-group', 'parallel assignment', 'parallel design', "
        "or CT.gov structured Study Design explicitly says 'Intervention Model: Parallel Assignment'. "
        "Do NOT infer from 'two arms'/'randomized' alone. "
        "Negative cues: explicit crossover/factorial/cluster wording -> do not set this unless parallel is explicitly stated."
    ),

    "Random_Crossover": (
        "Meaning: randomized crossover intervention model (participants receive sequences of treatments, e.g., AB/BA) with possible washout/periods. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly contains 'crossover/cross-over', 'randomized crossover', 'treatment sequence', "
        "'period 1/period 2', 'AB/BA', 'washout period', or CT.gov structured says 'Intervention Model: Crossover Assignment'. "
        "Do NOT infer from 'switch' or 'open-label extension' unless clearly described as crossover design."
    ),

    "Random_Fact": (
        "Meaning: randomized factorial design (simultaneous randomization to >1 intervention factors, e.g., 2x2). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly says 'factorial', '2x2 factorial', '2-by-2', 'factorial assignment', "
        "or CT.gov structured says 'Intervention Model: Factorial Assignment'. "
        "Do NOT infer from having many arms unless factorial is explicitly described."
    ),

    "Random_Cluster": (
        "Meaning: cluster randomized trial where the unit of randomization is a group/cluster (site, clinic, school, village, household, ward), "
        "not the individual participant. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly says 'cluster randomized/randomised', 'group randomized', 'randomized by site/clinic/school', "
        "'clusters were randomized', 'unit of randomization is ...'. "
        "Do NOT infer from multi-center recruitment alone. "
        "If TEXT says 'stratified by site' that is NOT cluster randomization (leave empty unless it says randomize-by-site)."
    ),

    "Stratification": (
        "Meaning: stratified randomization is used (randomization performed within strata defined by baseline factors). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill if TEXT explicitly says 'stratified randomization' / 'randomization stratified by ...' / 'stratification factor(s)'. "
        "Also fill if TEXT lists stratification factors in the context of randomization (e.g., 'randomization will be stratified by age, region'). "
        "If only says 'stratified' without clarifying it is for randomization -> usually empty unless clearly tied to allocation. "
        "Consistency rule: if No_Stratification is filled (>=1) then Stratification should be 'yes' (because factors imply stratified randomization)."
    ),
    
    "No_Stratification": (
        "Meaning: count of distinct stratification factors used in the randomization scheme. "
        "Output: integer count only (e.g., '1','2','3'); otherwise empty. "
        "Extract: fill ONLY when TEXT explicitly lists stratification factors; count factors separated by commas/and/by. "
        "Counting rule: each concept = 1 factor even if it has levels (e.g., 'ECOG 0-1 vs 2' counts as 1). "
        "Examples: 'stratified by age, sex, region' -> 3; 'stratified by ECOG and PD-L1 status' -> 2. "
        "Negative cues: only says 'stratified' or 'stratified randomization' but no factors listed -> empty."
    ),

    "Obj_Primary": (
        "Meaning: statistical objective framework of the primary hypothesis (superiority vs non-inferiority). "
        "Output: 'superiority' OR 'non-inferiority' only. "
        "Extract: fill ONLY if TEXT explicitly states 'superiority', 'non-inferiority/noninferiority', "
        "or defines a non-inferiority margin (NI margin) / NI bound. "
        "Do NOT guess from endpoint type or comparator. "
        "If both are mentioned (e.g., NI then test superiority), output the one stated as primary objective; "
        "if unclear -> empty."
    ),

    "Primary_EP_SC": (
        "Meaning: whether the PRIMARY endpoint is a clinical endpoint (C) or a surrogate/intermediate endpoint (S). "
        "Output: 'C' or 'S' only; otherwise empty. "
        "Extract: you MAY classify using the primary endpoint title/definition in TEXT when classification is clear. "
        "C (clinical) = how patient feels/functions/survives: overall survival, all-cause mortality, hospitalization, MACE, "
        "stroke/MI, symptom scales, functional outcomes, QoL/PRO, exacerbations, clinical remission/relapse requiring treatment. "
        "S (surrogate/intermediate) = biomarker/response/proxy: PFS/EFS/DFS, ORR, pCR, tumor response by RECIST, viral load, "
        "antibody titers, imaging measures, lab values, tumor size, MRD negativity. "
        "Rule: if endpoint is clearly a biomarker/response/time-to-progression proxy -> S; if it is survival/mortality or "
        "direct clinical events/PRO -> C. "
        "If endpoint text is missing/too vague -> empty. Evidence can be the endpoint phrase itself."
    ),

    "Key_Second_EP_SC": (
        "Meaning: whether the KEY SECONDARY endpoint is clinical (C) or surrogate/intermediate (S). "
        "Output: 'C' or 'S' only; otherwise empty. "
        "Extract: same rule as Primary_EP_SC but apply to the key secondary endpoint title/definition. "
        "If multiple key secondary endpoints are listed and you cannot tell which is 'key' -> empty."
    ),

    "IRC": (
        "Meaning: independent committee performs central/blinded endpoint assessment/adjudication (e.g., BICR/IRC/CEC). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly mentions an independent review/adjudication body for endpoints, e.g., "
        "'IRC', 'independent review committee', 'blinded independent central review (BICR/BIRC)', "
        "'clinical events committee (CEC)', 'independent adjudication committee', 'central imaging review'. "
        "Do NOT infer from 'blinded study' alone. "
        "Positive cues: 'BICR will assess tumor response', 'CEC adjudicates MACE'."
    ),

    "Power": (
        "Meaning: planned statistical power for the primary hypothesis. "
        "Output: percent string like '80%','90%'; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly states power (e.g., '80% power', 'power=0.9'). "
        "Normalize: '90 percent' -> '90%'; '0.8 power' -> '80%' ONLY if explicitly labeled as power. "
        "If multiple powers are given (different endpoints/scenarios), prefer the primary endpoint/power statement; "
        "if unclear -> empty."
    ),

    "Alpha": (
        "Meaning: planned Type I error / significance level for primary testing (alpha). "
        "Output: numeric string like '0.05','0.025','0.10'; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly states alpha/significance level (e.g., 'two-sided alpha=0.05', 'one-sided 2.5%'). "
        "Normalize: '5%' -> '0.05', '2.5%' -> '0.025' ONLY if explicitly tied to alpha/significance. "
        "Do NOT infer alpha from 'statistically significant'. "
        "If alpha differs by side (e.g., one-sided 0.025) still output the numeric alpha."
    ),

    "Sided": (
        "Meaning: sidedness of the primary hypothesis test (one-sided vs two-sided). "
        "Output: '1' for one-sided, '2' for two-sided; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly says 'one-sided/one tailed' or 'two-sided/two tailed'. "
        "Heuristic allowed ONLY with explicit alpha wording: if TEXT says 'one-sided 2.5%' -> '1'; "
        "if says 'two-sided 5%' -> '2'. Otherwise empty."
    ),

    "Subgroup": (
        "Meaning: the primary objective/hypothesis is targeted to a predefined subgroup (not just exploratory subgroup analyses). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly states primary analysis/primary endpoint evaluation is in a subgroup "
        "(e.g., 'primary analysis in PD-L1 TPS ≥50%', 'primary endpoint tested in biomarker-positive population'). "
        "Do NOT fill for generic phrases like 'subgroup analyses will be performed' or 'exploratory subgroup analyses'."
    ),

    "Interim": (
        "Meaning: any formal interim analysis/look for efficacy/futility/safety with planned decision/boundaries. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly states 'interim analysis', 'interim look', 'group sequential', "
        "'futility analysis', 'efficacy stopping boundary', 'IDMC/DMC interim review for efficacy/futility'. "
        "Do NOT infer from 'DMC will review safety periodically' unless it explicitly says interim analysis/look."
    ),

    "Timing_IA": (
        "Meaning: when the interim analysis is planned (events/information fraction/timepoint). "
        "Output: short timing phrase only (no extra explanation), e.g., 'after 50% events', '60% information fraction', "
        "'after 100 events', 'at week 24'. "
        "Extract: fill ONLY if TEXT states a concrete interim timing trigger. "
        "If Interim='yes' but timing not specified -> empty. "
        "Prefer: information fraction/events over vague time (unless time is the only explicit trigger)."
    ),

    "Alpha_Spend_Func": (
        "Meaning: named alpha-spending function / group-sequential boundary family used to control Type I error. "
        "Output: one short token only: prefer 'OBF','Pocock','Lan-DeMets','Shi-Hwang' (else use the named family as-is). "
        "Extract: fill ONLY if TEXT explicitly names the function/boundary, e.g., 'O'Brien-Fleming', 'Pocock', "
        "'Lan-DeMets alpha spending', 'Hwang-Shih-DeCani'. "
        "Normalize allowed: 'O’Brien-Fleming'/'OBrien Fleming' -> 'OBF'; "
        "'Hwang-Shih-DeCani' -> 'Shi-Hwang' (if you want one bucket). "
        "Do NOT infer from 'group sequential design' without naming the function."
    ),

    "Adaptive_Design": (
        "Meaning: protocol includes a pre-planned adaptive feature that can modify design based on interim data "
        "(beyond ordinary randomization). "
        "Output: either 'yes' (if only generic adaptive is stated) OR short element keywords separated by ';' "
        "when explicit (e.g., 'sample size re-estimation; adaptive randomization; seamless II/III; drop-the-loser'). "
        "Extract: fill ONLY if TEXT explicitly uses 'adaptive' language or names a recognized adaptive element "
        "(SSR, blinded/unblinded sample size re-estimation, response-adaptive randomization, arm dropping, "
        "seamless phase II/III, enrichment, platform/master protocol adaptation). "
        "If TEXT says only 'adaptive design' with no element -> output 'yes'. "
        "Do NOT label as adaptive just because there is interim analysis or stratification."
    ),

    "Consistency_Sens_Ana_PE": (
        "Meaning: sensitivity analyses for the primary endpoint are explicitly reported as consistent/robust vs primary analysis. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly states consistency/robustness, e.g., "
        "'results were consistent across sensitivity analyses', 'robust to sensitivity analyses'. "
        "Do NOT infer consistency from similar numbers without an explicit statement."
    ),

    "Consistency_Sens_Ana_SE": (
        "Meaning: sensitivity analyses for the key secondary endpoint are explicitly reported as consistent/robust. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: same as Consistency_Sens_Ana_PE but tied to key secondary endpoint."
    ),

    "Post_Hoc_Ana": (
        "Meaning: any analysis explicitly described as post hoc. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY when TEXT uses the phrase 'post hoc' (case-insensitive). "
        "Do NOT treat 'exploratory' as post hoc unless it literally says post hoc."
    ),

    "Success_Criteria_Text": (
        "Meaning: explicit statistical success criterion/decision rule for declaring primary success. "
        "Output: short verbatim criterion phrase (compact, no long prose). "
        "Extract: fill ONLY if TEXT defines a rule like: 'p<0.05', 'two-sided p<0.05', "
        "'lower bound of 95% CI > 1', 'upper bound of 95% CI < 1', "
        "'non-inferiority if lower bound > -10%', 'success if HR < 0.8 and p<0.025'. "
        "Prefer the main primary endpoint decision rule; if multiple gates exist, capture the top-level rule succinctly. "
        "Do NOT invent criteria from generic 'statistically significant'."
    ),

    "Intercurrent_Events": (
        "Meaning: explicitly defined intercurrent events (ICH E9(R1) estimand context) and how handled. "
        "Output: short semicolon-separated list of event tokens (e.g., 'treatment discontinuation; rescue medication; death'). "
        "Extract: fill ONLY if TEXT explicitly uses estimand/intercurrent-event language and names events/handling "
        "(e.g., 'discontinuation handled by treatment policy strategy', 'rescue meds handled by hypothetical strategy'). "
        "If TEXT only mentions withdrawals/missing data generally without estimand/ICE framing -> empty."
    ),

    "Gatekeeping_Strategy": (
        "Meaning: explicit multiplicity control / multiple-endpoint testing procedure. "
        "Output: one short keyword/token (or short method name) such as 'hierarchical', 'fixed-sequence', 'gatekeeping', "
        "'Bonferroni', 'Hochberg', 'Holm', 'fallback'. "
        "Extract: fill ONLY if TEXT explicitly states a multiplicity strategy/procedure. "
        "Do NOT infer from having multiple endpoints. "
        "If TEXT names multiple procedures, prefer the one described as primary multiplicity control for primary+key secondary."
    ),

    # ---------------------------
    # Regulatory
    # ---------------------------
    "Reg_Alignment": (
        "Meaning: explicit prior alignment/interaction with regulator(s) about protocol/endpoints/design (not mere GCP compliance). "
        "Output: short token(s) like 'FDA','EMA','PMDA','ICH E9','Scientific Advice','SPA'; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly mentions regulator meeting/advice/agreement, e.g., "
        "'FDA End-of-Phase 2 meeting', 'EMA Scientific Advice', 'Special Protocol Assessment (SPA)', "
        "'protocol agreed with FDA/EMA', 'aligned with ICH E9(R1) estimand guidance' (only if stated as alignment target). "
        "Do NOT fill for generic boilerplate 'conducted per ICH-GCP' unless it claims alignment/endpoint agreement."
    ),

    "Fast_Track": (
        "Meaning: drug has FDA Fast Track designation (or explicitly says fast track designation). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: require explicit 'Fast Track' / 'fast track designation'. Do NOT infer from unmet need/accelerated pathways."
    ),

    "Breakthrough": (
        "Meaning: drug has Breakthrough Therapy designation. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: require explicit 'Breakthrough Therapy designation'."
    ),

    "Priority_Review": (
        "Meaning: Priority Review designation/intent is explicitly stated. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: require explicit phrase 'Priority Review'. Do NOT infer from expedited review language."
    ),

    "Accelerated_App": (
        "Meaning: Accelerated Approval pathway is explicitly stated (FDA accelerated approval or equivalent). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: require explicit 'Accelerated Approval'. Do NOT infer from surrogate endpoint usage."
    ),

    "Orphan_Drug": (
        "Meaning: Orphan Drug designation is explicitly stated. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: require explicit 'Orphan Drug designation'/'orphan designation'. Do NOT infer from rare disease alone."
    ),

    "Pediatric": (
        "Meaning: study is explicitly positioned as pediatric development/indication (children/adolescents as pediatric program). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly says 'pediatric', 'children', 'adolescent pediatric program', "
        "or references pediatric regulatory programs (PIP, PREA) / pediatric formulation, etc. "
        "Do NOT fill merely because eligibility includes age 16-17 unless explicitly framed as pediatric."
    ),

    "Rare_Disease": (
        "Meaning: condition or program is explicitly labeled as a rare disease/rare disorder (or rare disease program language). "
        "Output: 'yes' only; otherwise empty. "
        "Extract: require explicit 'rare disease'/'rare disorder' wording or equivalent program statement. "
        "Do NOT infer from prevalence knowledge."
    ),

    "Reg_Audit": (
        "Meaning: explicit report of major issues/findings from regulatory inspection/audit (e.g., GCP inspection findings). "
        "Output: short issue phrase only (e.g., 'data integrity findings', 'protocol deviations cited'); otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly mentions inspection/audit and describes findings/observations (483, warning letter, "
        "critical/major findings, inspection deficiencies). "
        "If inspection/audit is not mentioned -> empty."
    ),

    "Consistency_MRCT": (
        "Meaning: in a multi-regional clinical trial (MRCT), TEXT explicitly states consistency of efficacy/safety across regions/ethnicities. "
        "Output: 'yes' only; otherwise empty. "
        "Extract: fill ONLY if TEXT explicitly mentions MRCT/regional consistency, e.g., "
        "'consistent across regions', 'no heterogeneity by region', 'treatment effect consistent in Asia and non-Asia'. "
        "Do NOT infer from multinational recruitment alone."
    ),
}


def stat_reg_extra_rules(missing_fields: List[str]) -> List[str]:
    _ = missing_fields
    return [
        # General conservatism
        "- Many SAP details are not in CT.gov. Return empty unless explicitly stated verbatim in TEXT.",
        "- Do NOT infer anything from the CSV CONTEXT fields (Allocation/Masking etc.). Use TEXT only.",
        # Allowed normalizations (value only; evidence must quote original phrase)
        "- You MAY normalize ratios: '2 to 1', '2-1', '2/1' -> output '2:1'.",
        "- You MAY normalize power percent: '80 percent'/'80%' -> output '80%'.",
        "- You MAY normalize sidedness: if TEXT says 'one-sided' -> output '1'; 'two-sided' -> output '2'.",
        "- You MAY normalize alpha if explicitly equivalent: keep as decimal like 0.05; do not derive from p-values.",
        "- Alpha_Spend_Func mapping allowed: 'O'Brien-Fleming'/'O’Brien Fleming' -> 'OBF'; keep 'Pocock','Lan-DeMets','Shi-Hwang' as-is.",
        # Endpoint S/C classification guidance
        "- For Primary_EP_SC / Key_Second_EP_SC: classify as C when endpoint is direct patient benefit (survival, mortality, hospitalization, MACE, symptoms/function, QoL/PRO); "
        "classify as S when endpoint is intermediate/surrogate (PFS/ORR/pCR/DOR, biomarkers, imaging response, viral load, lab measures). If not clear -> empty.",
        # Output strictness reminders
        "- For yes/no style fields, output ONLY 'yes' (never output 'no'); use empty when not supported.",
        "- Keep outputs short; lists should be semicolon-separated (value) and evidence can be multiple segments separated by ' | '.",
    ]


PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    notes=STAT_REG_NOTES,
    extra_rules_fn=stat_reg_extra_rules,
    llm_fields=LLM_FIELDS,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "Stat_Reg"])
    if "--max-fields-per-call" not in sys.argv:
        sys.argv.extend(["--max-fields-per-call", "8"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
