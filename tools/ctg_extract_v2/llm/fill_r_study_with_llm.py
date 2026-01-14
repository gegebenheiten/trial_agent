#!/usr/bin/env python3
"""
Fill R_Study table using the shared CTG LLM interface (v2 schema).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fill_table_with_llm import PromptConfig, main as fill_main

ANALYSIS_INSTRUCTIONS = (
    "You extract R_Study endpoint analysis variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Use the matched outcome for the current EP_Name only; return empty unless explicitly stated."
)

FLOW_INSTRUCTIONS = (
    "You extract R_Study screening/enrollment variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Return empty unless explicitly stated."
)

OPS_INSTRUCTIONS = (
    "You extract R_Study regulatory/operational variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Return empty unless explicitly stated."
)

ALL_INSTRUCTIONS = (
    "You extract R_Study variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Return empty unless explicitly stated."
)

ANALYSIS_FIELDS = [
    "Strategy",
    "Analysis_Set",
    "Missing_Imput",
    "Covariate_Adjust",
    "MCP",
    "Subgroup_Ana",
    "Post_Hoc_Ana",
    "Consistency_Sens_Ana_PE",
    "Consistency_Sens_Ana_SE",
]

FLOW_FIELDS = [
    "No_Amendment",
    "No_Substan_Amend",
    "No_Sub_Screen",
    "No_Sub_Enroll",
    "Screen_Failure",
    "Reasons_Screen_Fail",
    "Enroll_Rate",
    "Time_FPI",
    "Prop_Center_Enroll_Target",
]

OPS_FIELDS = [
    "Reg_Audit",
    "Consistency_MRCT",
    "Sponsor_Type",
    "COVID_Impact",
    "No_Competing_Trial",
    "No_SOC",
    "CRO",
    "CRO_Oper",
    "CRO_Stat",
    "CRO_DM",
    "RBM",
]

ALL_FIELDS = ANALYSIS_FIELDS + FLOW_FIELDS + OPS_FIELDS

ANALYSIS_MODULES = [
    "endpoint_target",
    "endpoint_matches",
    "study_info",
]

FLOW_MODULES = [
    "participant_flow",
    "design_info",
    "study_info",
]

OPS_MODULES = [
    "study_info",
    "design_info",
    "keywords",
    "conditions",
]

ALL_MODULES = [
    "endpoint_target",
    "endpoint_matches",
    "participant_flow",
    "design_info",
    "study_info",
    "keywords",
    "conditions",
]

R_STUDY_NOTES = {
    "Strategy": "Definition: Strategy used in the analysis of this endpoint, e.g., (1) treatment policy strategy, (2) hypothetical strategy, (3) composite variable strategy, (4) principal strata strategy, (5) while-on-treatment strategy.\nHow to decide: In the selected (first, highest-score) outcome block, extract ONLY if one or more of the above strategy terms are explicitly stated. If none are explicitly stated, or wording is ambiguous, return empty.\nOutput example: 'treatment policy strategy' or 'hypothetical strategy' or 'composite variable strategy' or 'principal strata strategy' or 'while-on-treatment strategy'.",
    "Analysis_Set": "Definition: analysis population/set used for this endpoint (e.g., ITT, mITT, per-protocol, safety set). Return the explicit label only if stated.",
    "Missing_Imput": "Definition: Methods for missing data imputation (CSR-Vars/Endpoints annotation), e.g., MAR/MCAR/MNAR, multiple imputation, LOCF, etc.\nHow to decide: Extract only if the selected outcome block explicitly mentions missingness assumptions (MAR/MCAR/MNAR) or a specific imputation/handling method (e.g., 'multiple imputation', 'LOCF', 'non-responder imputation', 'worst-case imputation').\nOutput example: 'MI' or 'multiple imputation' or 'LOCF' or 'MAR'.",
    "Covariate_Adjust": "Definition: Whether covariates are adjusted using modeling approaches (CSR-Vars/Endpoints annotation), e.g., ANCOVA or Cox regression model.\nHow to decide: Extract only if the selected outcome block explicitly states covariate adjustment or an adjusted model/estimate (e.g., 'adjusted for baseline', 'ANCOVA', 'multivariable Cox', 'covariate-adjusted').\nOutput example: 'ANCOVA (baseline-adjusted)' or 'Cox (adjusted)' or 'covariate-adjusted'.",
    "MCP": "Definition: Whether multiple comparison / multiple testing procedures are used to adjust multiplicity (CSR-Vars/Endpoints annotation).\nHow to decide: Extract only if the selected outcome block explicitly names a multiplicity method (e.g., Bonferroni/Holm/Hochberg/FDR/Dunnett/gatekeeping/hierarchical testing) or states 'multiplicity adjusted'.\nOutput example: 'Bonferroni' or 'Holm' or 'gatekeeping'.",
    "Subgroup_Ana": "Definition: Whether subgroup analysis is performed (CSR-Vars/Endpoints annotation).\nHow to decide: Extract only if the selected outcome block explicitly reports subgroup analyses or mentions 'subgroup', 'pre-specified subgroup', 'interaction test', or subgroup-specific estimates.\nOutput example: 'pre-specified subgroup; interaction test' or 'subgroup analysis'.",
    "Post_Hoc_Ana": "Meaning: any analysis explicitly described as post hoc. Output: 'yes' only; otherwise empty. Extract: fill ONLY when TEXT uses the phrase 'post hoc' (case-insensitive). Do NOT treat 'exploratory' as post hoc unless it literally says post hoc.",
    "Consistency_Sens_Ana_PE": "Meaning: sensitivity analyses for the primary endpoint are explicitly reported as consistent/robust vs primary analysis. Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly states consistency/robustness, e.g., 'results were consistent across sensitivity analyses', 'robust to sensitivity analyses'. Do NOT infer consistency from similar numbers without an explicit statement.",
    "Consistency_Sens_Ana_SE": "Meaning: sensitivity analyses for the key secondary endpoint are explicitly reported as consistent/robust. Output: 'yes' only; otherwise empty. Extract: same as Consistency_Sens_Ana_PE but tied to key secondary endpoint.",
    "No_Amendment": "Definition: Number of protocol amendment before the study is completed, e.g., 1, 2, 3, ...\nDecision rules:\n  - Return an integer count ONLY when explicitly stated in the text.\n  - If not explicitly stated, return empty (do not infer).\nOutput example:\n  - 'Number of protocol amendments: 2' -> 2\n",
    "No_Substan_Amend": "Definition: Number of substantial amendment before the study is completed, e.g., 1, 2, 3, ...\nDecision rules:\n  - Return an integer count ONLY when explicitly stated in the text.\n  - If not explicitly stated, return empty (do not infer).\nOutput example:\n  - 'Number of substantial amendments: 1' -> 1\n",
    "No_Sub_Screen": "Definition: Number of participants screened\nDecision rules:\n  - Return an integer count ONLY when explicitly stated in the text.\n  - If not explicitly stated, return empty (do not infer).\nOutput example:\n  - 'Participants screened: 120' -> 120\n",
    "No_Sub_Enroll": "Definition: number of participants enrolled in the study. Return an integer only if explicitly stated.",
    "Screen_Failure": "Definition: Screening failure rate of the study, e.g., 20%, 30%, ...\nDecision rules:\n  - Return screening failure rate as a percentage string with '%'.\n  - If explicitly reported, copy it.\n  - Otherwise, you may compute ONLY if Screened and Enrolled/Randomized/Started counts are explicitly given: (Screened - Enrolled_or_Randomized) / Screened * 100.\n  - If required counts are missing, return empty.\nOutput example:\n  - Screened=120, Enrolled=90 -> '25%'\n",
    "Reasons_Screen_Fail": "Definition: Reasons for screen failure, e.g., (1) inclusion criteria not met, (2) exclusion criteria met\nDecision rules:\n  - Return the reasons exactly as stated in the text (raw text is preferred).\n  - Do not invent reasons or re-classify; keep original phrasing.\n  - If multiple reasons are listed, join with '; '.\n  - If no reasons are explicitly stated, return empty.\nOutput example:\n  - 'inclusion criteria not met; withdrew consent' -> 'inclusion criteria not met; withdrew consent'\n",
    "Enroll_Rate": "Definition: Average number of participants enrolled per month\nDecision rules:\n  - Return the average number of participants enrolled per month.\n  - Only fill when explicitly stated (e.g., '10 participants per month'); otherwise empty.\n  - Output a numeric value only (no unit text).\nOutput example:\n  - 'average 10 participants per month' -> 10\n",
    "Time_FPI": "Definition: Time in days from trial start to first patient in\nDecision rules:\n  - Return an integer number of days from trial start to first patient in.\n  - Only fill when explicitly stated (e.g., 'first patient in after 10 days') or when BOTH dates are explicitly given and the difference in days can be computed.\n  - If dates or linkage are unclear, return empty.\nOutput example:\n  - trial start 2020-01-01; first patient in 2020-01-11 -> 10\n",
    "Prop_Center_Enroll_Target": "Definition: Proportion of centers / site meeting the enrollment targets\nDecision rules:\n  - Return as a percentage string with '%'.\n  - Only fill if explicitly reported, or if both numerator and denominator are explicitly stated (centers meeting target / total centers).\n  - If not attributable, return empty.\nOutput example:\n  - '6/10 centers met target' -> '60%'\n",
    "Reg_Audit": "Meaning: explicit report of major issues/findings from regulatory inspection/audit (e.g., GCP inspection findings). Output: short issue phrase only (e.g., 'data integrity findings', 'protocol deviations cited'); otherwise empty. Extract: fill ONLY if TEXT explicitly mentions inspection/audit and describes findings/observations (483, warning letter, critical/major findings, inspection deficiencies). If inspection/audit is not mentioned -> empty.",
    "Consistency_MRCT": "Meaning: in a multi-regional clinical trial (MRCT), TEXT explicitly states consistency of efficacy/safety across regions/ethnicities. Output: 'yes' only; otherwise empty. Extract: fill ONLY if TEXT explicitly mentions MRCT/regional consistency, e.g., 'consistent across regions', 'no heterogeneity by region', 'treatment effect consistent in Asia and non-Asia'. Do NOT infer from multinational recruitment alone.",
    "Sponsor_Type": "Meaning: Type of trial sponsor or lead organization, categorized by institutional type.\nOutput: One of {Pharma, Biotech, Academic, Government}.\nExtract: Identify the entity listed in <lead_sponsor> or <collaborator> and infer its type.\nPositive cues: Pharma: 'Pfizer', 'Novartis', 'AstraZeneca'; Biotech: 'Moderna', 'BioNTech'; Academic: 'University', 'Hospital', 'Medical Center'; Government: 'NIH', 'NCI', 'CDC', 'U.S. Army'.\nNegative cues (leave empty): Generic names without identifiable type.",
    "COVID_Impact": "Meaning: Whether the trial was affected by the COVID-19 pandemic (delays, deviations, missing data).\nOutput: Boolean (Yes / No).\nExtract: Look for status updates, 'COVID-19 impact', or pandemic-related protocol deviations.\nPositive cues: 'Trial delayed due to COVID-19', 'Enrollment suspended during pandemic', 'COVID-related deviations'.\nNegative cues: 'COVID-19' as the disease under study (not an impact factor).",
    "No_Competing_Trial": "Meaning: Number of other active trials investigating the same disease during the same time period.\nOutput: Integer (e.g., '5') or 'Unknown'.\nExtract: May be inferred from <condition> and <study_start_date> metadata cross-referenced with similar trials.\nPositive cues: 'At least 5 other ongoing trials', 'No competing studies known'.\nNegative cues: No reference to other trials.",
    "No_SOC": "Meaning: Number of standard-of-care (SOC) treatments available for the disease at trial start.\nOutput: Integer (e.g., '0', '1', '2').\nExtract: Check <background> or <eligibility> for mentions of 'standard therapy', 'current treatment options'.\nPositive cues: 'No standard of care available', 'After failure of one SOC regimen', 'Multiple SOC options exist'.\nNegative cues: General disease descriptions without therapy references.",
    "CRO": "Meaning: Whether Contract Research Organization (CRO) services are used in the study.\nOutput: Boolean (Yes / No).\nExtract: Identify mentions of CRO involvement in trial management, operations, or analytics.\nPositive cues: 'Conducted in collaboration with [CRO name]', 'CRO responsible for data collection'.\nNegative cues: No mention of third-party operational or statistical support.",
    "CRO_Oper": "Meaning: Whether CRO operational services (e.g., site / center management, patient enrollment) are used.\nOutput: Boolean (Yes / No).\nExtract: Evidence of CRO handling operational or logistical aspects.\nPositive cues: 'CRO oversaw site monitoring', 'Outsourced patient recruitment to [CRO]'.\nNegative cues: Only internal sponsor staff mentioned.",
    "CRO_Stat": "Meaning: Whether CRO statistical services (sample size calculation, data analysis) are used.\nOutput: Boolean (Yes / No).\nExtract: Look for mention of biostatistics support or outsourced data analysis.\nPositive cues: 'CRO performed statistical analysis', 'External vendor handled sample size estimation'.\nNegative cues: In-house statistics only.",
    "CRO_DM": "Meaning: Whether CRO data management services (CRF design, data collection, data management) are used.\nOutput: Boolean (Yes / No).\nExtract: Identify CRO involvement in EDC systems, CRF design, or database management.\nPositive cues: 'CRO designed case report forms', 'Data managed by [CRO]'.\nNegative cues: Data management handled internally.",
    "RBM": "Meaning: Whether Risk-Based Monitoring (RBM) is used in the study.\nOutput: Boolean (Yes / No).\nExtract: Check for mentions of 'RBM', 'centralized monitoring', or 'adaptive risk-based strategy'.\nPositive cues: 'Implemented risk-based monitoring approach', 'RBM methodology applied'.\nNegative cues: Traditional on-site monitoring only.",
}

PROMPT_CONFIGS = {
    "analysis": PromptConfig(
        instructions=ANALYSIS_INSTRUCTIONS,
        notes=R_STUDY_NOTES,
        llm_fields=ANALYSIS_FIELDS,
        text_modules=ANALYSIS_MODULES,
    ),
    "flow": PromptConfig(
        instructions=FLOW_INSTRUCTIONS,
        notes=R_STUDY_NOTES,
        llm_fields=FLOW_FIELDS,
        text_modules=FLOW_MODULES,
    ),
    "ops": PromptConfig(
        instructions=OPS_INSTRUCTIONS,
        notes=R_STUDY_NOTES,
        llm_fields=OPS_FIELDS,
        text_modules=OPS_MODULES,
    ),
    "all": PromptConfig(
        instructions=ALL_INSTRUCTIONS,
        notes=R_STUDY_NOTES,
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
        sys.argv.extend(["--table", "R_Study"])
    group = pop_arg("--group") or pop_arg("--pass")
    group = (group or "all").strip().lower()
    aliases = {
        "analysis": "analysis",
        "endpoint": "analysis",
        "stats": "analysis",
        "stat": "analysis",
        "flow": "flow",
        "enroll": "flow",
        "screening": "flow",
        "ops": "ops",
        "other": "ops",
        "reg": "ops",
        "all": "all",
    }
    group = aliases.get(group, group)
    prompt_config = PROMPT_CONFIGS.get(group, PROMPT_CONFIGS["all"])
    fill_main(prompt_config=prompt_config)


if __name__ == "__main__":
    main()
