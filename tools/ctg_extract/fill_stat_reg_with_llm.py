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
    # Statistics
    "Central_Random": (
        "Definition: central randomization used. "
        "Output: 'yes' only (else empty). "
        "Fill when TEXT explicitly mentions central randomization / IVRS / IWRS / IRT / interactive response system. "
        "Do NOT infer from 'Randomized' or Allocation/Masking alone. "
        "Example: 'IWRS will assign treatment' -> yes."
    ),
    "Rand_Ratio": (
        "Definition: randomization ratio. "
        "Output: ratio only like '1:1', '2:1', '3:1'. "
        "Fill when TEXT states an allocation ratio (e.g., '2:1', '2 to 1', '2-1'). "
        "If only says 'randomized' without ratio -> empty. "
        "Example: 'randomized 2 to 1' -> 2:1."
    ),
    "No_Stratification": (
        "Definition: number of stratification variables. "
        "Output: integer only (e.g., '1','2','3'). "
        "Fill when TEXT explicitly lists stratification factors; count distinct factors. "
        "If TEXT only says 'stratified' but does not list factors -> empty. "
        "Example: 'stratified by age, sex, region' -> 3."
    ),
    "Obj_Primary": (
        "Definition: superiority vs non-inferiority. "
        "Output: 'superiority' or 'non-inferiority' only. "
        "Fill only if TEXT explicitly states superiority/non-inferiority (or equivalent wording like 'noninferiority margin'). "
        "If unclear -> empty. "
        "Example: 'non-inferiority trial with margin ...' -> non-inferiority."
    ),
    "Primary_EP_SC": (
        "Definition: primary endpoint is surrogate (S) vs clinical (C). "
        "Output: 'S' or 'C' only. "
        "You MAY classify using the endpoint title/description in TEXT when it is clearly a clinical outcome vs intermediate/surrogate. "
        "Clinical (C) examples: overall survival, mortality, hospitalization, MACE, symptom/function endpoints, QoL/PRO. "
        "Surrogate (S) examples: PFS, ORR, pCR, biomarkers/labs, imaging response, viral load, tumor size. "
        "If not clearly classifiable -> empty. Evidence can be the endpoint name phrase."
    ),
    "Key_Second_EP_SC": (
        "Definition: key secondary endpoint surrogate (S) vs clinical (C). "
        "Output: 'S' or 'C' only. "
        "Same classification rule as Primary_EP_SC; use endpoint title/description only when clear; otherwise empty."
    ),
    "IRC": (
        "Definition: independent (central/blinded) review committee used for endpoint assessment. "
        "Output: 'yes' only (else empty). "
        "Fill when TEXT explicitly mentions IRC/BIRC/independent central review/adjudication committee/clinical events committee "
        "or similar independent blinded assessment. "
        "Example: 'BIRC will assess tumor response' -> yes."
    ),
    "Power": (
        "Definition: statistical power. "
        "Output: percent only like '80%','85%','90%'. "
        "Fill only if TEXT explicitly states power. "
        "Example: '80% power' / 'power of 90 percent' -> 80% / 90%."
    ),
    "Alpha": (
        "Definition: significance level alpha. "
        "Output: numeric only like '0.05','0.025','0.10'. "
        "Fill only if TEXT explicitly states alpha/significance level. "
        "If only says 'statistically significant' without alpha -> empty. "
        "Example: 'two-sided alpha=0.05' -> 0.05."
    ),
    "Sided": (
        "Definition: one-sided vs two-sided test. "
        "Output: '1' (one-sided) or '2' (two-sided) only. "
        "Fill only if TEXT explicitly says one-sided/two-sided. "
        "Example: 'one-sided 2.5%' -> 1; 'two-sided' -> 2."
    ),
    "Subgroup": (
        "Definition: study focuses on a specific subgroup (not merely exploratory subgroup analyses). "
        "Output: 'yes' only (else empty). "
        "Fill when TEXT explicitly states the primary objective/endpoint evaluation focuses on a subgroup "
        "(e.g., primary analysis in PD-L1 high subgroup) even if eligibility is broader. "
        "If only says 'subgroup analyses will be performed' -> empty."
    ),
    "Interim": (
        "Definition: interim analysis performed. "
        "Output: 'yes' only (else empty). "
        "Fill when TEXT explicitly states interim analysis/interim look/group sequential interim. "
        "Example: 'an interim analysis after 50% events' -> yes."
    ),
    "Timing_IA": (
        "Definition: interim timing (e.g., information fraction). "
        "Output: timing phrase only (short). "
        "Fill only if TEXT states timing; prefer compact phrases like 'after 50% events' or '60% information fraction'. "
        "If interim is mentioned but timing not stated -> empty."
    ),
    "Alpha_Spend_Func": (
        "Definition: alpha spending function (e.g., OBF, Pocock, Shi-Hwang, Lan-DeMets). "
        "Output: one short name only (prefer: 'OBF','Pocock','Shi-Hwang','Lan-DeMets'). "
        "Fill only if TEXT explicitly names a spending function/boundary family. "
        "You MAY map common variants (see Extra rules). "
        "Example: 'O'Brien-Fleming boundary' -> OBF."
    ),
    "Adaptive_Design": (
        "Definition: adaptive design elements used. "
        "Output: 'yes' OR short element keywords (semicolon-separated) if explicitly named. "
        "Fill when TEXT mentions adaptive design and/or specific elements (e.g., sample size re-estimation, seamless phase II/III, "
        "adaptive randomization, drop-the-loser/keep-the-winner). "
        "If only generic 'adaptive design' without details -> 'yes'. "
        "If unclear -> empty."
    ),
    "Consistency_Sens_Ana_PE": (
        "Definition: sensitivity analysis results for primary endpoint consistent with primary analysis. "
        "Output: 'yes' only (else empty). "
        "Fill only if TEXT explicitly states consistency/robustness of sensitivity analyses for the primary endpoint."
    ),
    "Consistency_Sens_Ana_SE": (
        "Definition: sensitivity analysis results for key secondary endpoint consistent with primary analysis. "
        "Output: 'yes' only (else empty). "
        "Fill only if TEXT explicitly states consistency/robustness of sensitivity analyses for the key secondary endpoint."
    ),
    "Post_Hoc_Ana": (
        "Definition: post hoc analysis performed. "
        "Output: 'yes' only (else empty). "
        "Fill only if TEXT explicitly says 'post hoc'."
    ),
    "Success_Criteria_Text": (
        "Definition: statistical success criterion (e.g., CI bound threshold). "
        "Output: short verbatim criterion phrase. "
        "Fill only if TEXT explicitly defines success (e.g., 'lower bound of 95% CI > 1', 'p<0.05'). "
        "If not explicitly defined -> empty."
    ),
    "Intercurrent_Events": (
        "Definition: list of intercurrent events handled (ICH E9(R1) estimand-style). "
        "Output: short list (semicolon-separated) such as 'discontinuation; rescue medication'. "
        "Fill only if TEXT explicitly names intercurrent events and how handled; otherwise empty."
    ),
    "Gatekeeping_Strategy": (
        "Definition: multiple endpoint testing strategy (e.g., Hierarchical). "
        "Output: short keyword only (e.g., 'hierarchical', 'fixed-sequence', 'gatekeeping', 'Hochberg', 'Bonferroni'). "
        "Fill only if TEXT explicitly states a multiplicity strategy; otherwise empty."
    ),

    # Regulatory
    "Reg_Alignment": (
        "Definition: protocol pre-aligned with a regulatory agency/guideline. "
        "Output: short agency/guideline token(s) if explicit (e.g., 'FDA','EMA','ICH E9'). "
        "Fill only if TEXT explicitly mentions prior alignment/meeting/advice/endpoint agreement with regulator or explicit compliance target. "
        "If only generic 'conducted per ICH-GCP' without alignment claim -> usually empty."
    ),
    "Fast_Track": (
        "Definition: fast track approval/designation. "
        "Output: 'yes' only (else empty). "
        "Require explicit phrase like 'Fast Track designation' or 'fast track'."
    ),
    "Breakthrough": (
        "Definition: breakthrough therapy. "
        "Output: 'yes' only (else empty). "
        "Require explicit phrase like 'Breakthrough Therapy designation'."
    ),
    "Priority_Review": (
        "Definition: priority review. "
        "Output: 'yes' only (else empty). "
        "Require explicit phrase like 'Priority Review'."
    ),
    "Accelerated_App": (
        "Definition: accelerated approval. "
        "Output: 'yes' only (else empty). "
        "Require explicit phrase like 'Accelerated Approval'."
    ),
    "Orphan_Drug": (
        "Definition: orphan drug designation. "
        "Output: 'yes' only (else empty). "
        "Require explicit phrase like 'Orphan Drug designation'."
    ),
    "Pediatric": (
        "Definition: pediatric indication/design. "
        "Output: 'yes' only (else empty). "
        "Fill only if TEXT explicitly states pediatric indication/program (not merely includes adolescents unless clearly framed as pediatric)."
    ),
    "Rare_Disease": (
        "Definition: rare disease. "
        "Output: 'yes' only (else empty). "
        "Fill only if TEXT explicitly calls the condition 'rare disease/rare disorder' or equivalent program language."
    ),
    "Reg_Audit": (
        "Definition: major issues detected during regulatory inspection/audit. "
        "Output: short issue phrase only if explicit. "
        "If TEXT does not mention inspection/audit findings -> empty."
    ),
    "Consistency_MRCT": (
        "Definition: consistency across regions in an MRCT. "
        "Output: 'yes' only (else empty). "
        "Fill only if TEXT explicitly states consistency across regions/ethnicities/subpopulations in MRCT context."
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
