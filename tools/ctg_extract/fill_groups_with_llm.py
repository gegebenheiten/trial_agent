#!/usr/bin/env python3
"""
Fill Groups results table using the shared CTG LLM interface.
"""

import sys

from fill_ctg_table_with_llm import PromptConfig, main as fill_main

INSTRUCTIONS = (
    "You extract GROUP-LEVEL results variables from ClinicalTrials.gov results text using CSR-Vars annotations.\n"
    "Focus on Participant Flow (disposition), Baseline Characteristics (demographics/clinical status), Missingness, "
    "Treatment regimen, and Adverse Events.\n"
    "When multiple groups are present, ALWAYS target the CURRENT row's Arm_ID / group_title_raw / group_desc_raw.\n"
    "Do NOT infer missing information. Only fill a field when it is explicitly stated for the current group, "
    "or when a percentage can be computed from explicit numerator and denominator for the current group.\n"
    "Formatting rules: counts -> integers; percentages/proportions -> string with '%' (e.g., '25%'); "
    "durations -> keep the value and unit exactly as stated (e.g., '16.9 months'); do NOT convert units."
)

LLM_FIELDS = [
    "Arm_Group_Type",
    "No_Amendment",
    "No_Substan_Amend",
    "No_Sub_Screen",
    "Screen_Failure",
    "Reasons_Screen_Fail",
    "Enroll_Rate",
    "Time_FPI",
    "Prop_Center_Enroll_Target",
    "Completer",
    "Withdrawer",
    "Discont_AE",
    "Discont_LE",
    "Loss_FU",
    "Protocol_Viol",
    "Treat_Adherence",
    "Visit_Adherence",
    "Assess_Adherence",
    "Med_Treat",
    "Med_Follow_Up",
    "Discont",
    "Med_Discont",
    "Missing_All",
    "Missing_PE",
    "Missing_Key_SE",
    "Prop_Renal",
    "Prop_Hepatic",
    "Prop_HighRisk",
    "Line0",
    "Line1",
    "Line2",
    "Line3",
    "Lapsed",
    "Refractory",
    "Prop_ECOG0",
    "Prop_ECOG1",
    "Prop_ECOG2",
    "Prop_ECOG3",
    "TEAEs",
    "TRAEs",
    "SAEs",
    "Grade34_TEAEs",
    "Grade34_TRAEs",
    "AE_Spec",
    "Inc_TEAES",
    "Inc_Grade34_TEAEs",
    "Grade34_Lab",
    "Death_rate",
    "No_Dosage",
    "Dosage_Level",
    "Dosing_Frequency",
    "Cycle_Length",
    "Formulation",
]

TEXT_MODULES = [
    "study_info",
    "eligibility",
    "participant_flow",
    "baseline_results",
    "baseline_measures",
    "results_outcomes",
    "reported_events",
    "arm_groups",
    "interventions",
    "primary_outcomes",
    "secondary_outcomes",
    "keywords",
    "conditions",
    "location_countries",
]

GROUPS_NOTES = {
    "Arm_Group_Type": (
        "Definition: A single label describing the role/type of the current arm/group.\n"
        "Decision rules:\n"
        "  - Output MUST be exactly one of: Experimental, Active Comparator, Placebo Comparator, Sham Comparator, No Intervention, Other.\n"
        "  - Prefer a structured arm_group_type value if it is present in the provided text block.\n"
        "  - Otherwise decide ONLY from group_title_raw / group_desc_raw when there are explicit or unambiguous keywords:\n"
        "    * Sham Comparator: contains 'sham' or 'simulated procedure'\n"
        "    * Placebo Comparator: contains 'placebo' / 'matched placebo' / 'saline placebo'\n"
        "    * No Intervention: contains 'no intervention' / 'observation only' / 'watchful waiting' / 'no treatment' / 'natural history'\n"
        "    * Active Comparator: contains 'active comparator' / 'standard of care' / 'usual care' / 'standard therapy' "
        "or clearly states an approved active treatment used as comparator\n"
        "    * Experimental: contains 'investigational' / 'experimental' / 'study drug' "
        "or clearly states this arm receives the investigational intervention\n"
        "  - If unsupported, return empty (do NOT infer).\n"
        "Output example:\n"
        "  - group_title_raw='Placebo' -> Placebo Comparator\n"
    ),
    "No_Amendment": (
        "Definition: Number of protocol amendment before the study is completed, e.g., 1, 2, 3, ...\n"
        "Decision rules:\n"
        "  - Return an integer count ONLY when explicitly stated in the text.\n"
        "  - If not explicitly stated, return empty (do not infer).\n"
        "Output example:\n"
        "  - 'Number of protocol amendments: 2' -> 2\n"
    ),
    "No_Substan_Amend": (
        "Definition: Number of substantial amendment before the study is completed, e.g., 1, 2, 3, ...\n"
        "Decision rules:\n"
        "  - Return an integer count ONLY when explicitly stated in the text.\n"
        "  - If not explicitly stated, return empty (do not infer).\n"
        "Output example:\n"
        "  - 'Number of substantial amendments: 1' -> 1\n"
    ),
    "No_Sub_Screen": (
        "Definition: Number of participants screened\n"
        "Decision rules:\n"
        "  - Return an integer count ONLY when explicitly stated in the text.\n"
        "  - If not explicitly stated, return empty (do not infer).\n"
        "Output example:\n"
        "  - 'Participants screened: 120' -> 120\n"
    ),
    "Screen_Failure": (
        "Definition: Screening failure rate of the study, e.g., 20%, 30%, ...\n"
        "Decision rules:\n"
        "  - Return screening failure rate as a percentage string with '%'.\n"
        "  - If explicitly reported, copy it.\n"
        "  - Otherwise, you may compute ONLY if Screened and Enrolled/Randomized/Started counts are explicitly given: "
        "(Screened - Enrolled_or_Randomized) / Screened * 100.\n"
        "  - If required counts are missing, return empty.\n"
        "Output example:\n"
        "  - Screened=120, Enrolled=90 -> '25%'\n"
    ),
    "Reasons_Screen_Fail": (
        "Definition: Reasons for screen failure, e.g., (1) inclusion criteria not met, (2) exclusion criteria met\n"
        "Decision rules:\n"
        "  - Return the reasons exactly as stated in the text (raw text is preferred).\n"
        "  - Do not invent reasons or re-classify; keep original phrasing.\n"
        "  - If multiple reasons are listed, join with '; '.\n"
        "  - If no reasons are explicitly stated, return empty.\n"
        "Output example:\n"
        "  - 'inclusion criteria not met; withdrew consent' -> 'inclusion criteria not met; withdrew consent'\n"
    ),
    "Enroll_Rate": (
        "Definition: Average number of participants enrolled per month\n"
        "Decision rules:\n"
        "  - Return the average number of participants enrolled per month.\n"
        "  - Only fill when explicitly stated (e.g., '10 participants per month'); otherwise empty.\n"
        "  - Output a numeric value only (no unit text).\n"
        "Output example:\n"
        "  - 'average 10 participants per month' -> 10\n"
    ),
    "Time_FPI": (
        "Definition: Time in days from trial start to first patient in\n"
        "Decision rules:\n"
        "  - Return an integer number of days from trial start to first patient in.\n"
        "  - Only fill when explicitly stated (e.g., 'first patient in after 10 days') "
        "or when BOTH dates are explicitly given and the difference in days can be computed.\n"
        "  - If dates or linkage are unclear, return empty.\n"
        "Output example:\n"
        "  - trial start 2020-01-01; first patient in 2020-01-11 -> 10\n"
    ),
    "Prop_Center_Enroll_Target": (
        "Definition: Proportion of centers / site meeting the enrollment targets\n"
        "Decision rules:\n"
        "  - Return as a percentage string with '%'.\n"
        "  - Only fill if explicitly reported, or if both numerator and denominator are explicitly stated "
        "(centers meeting target / total centers).\n"
        "  - If not attributable, return empty.\n"
        "Output example:\n"
        "  - '6/10 centers met target' -> '60%'\n"
    ),
    "Completer": (
        "Definition: Proportion of participants who complete the study, i.e., complete the trial, either with or without observed / measured primary endpoint\n"
        "Decision rules:\n"
        "  - Return the proportion of participants who completed the study as a percentage string with '%'.\n"
        "  - Prefer an explicitly reported completion percentage.\n"
        "  - Otherwise you may compute ONLY if both Completed and Started counts for the current group are explicitly given: "
        "Completed/Started*100.\n"
        "  - If denominator is unclear, return empty.\n"
        "Output example:\n"
        "  - Started=100, Completed=85 -> '85%'\n"
    ),
    "Withdrawer": (
        "Definition: Proportion of withdrawing Inform Consent form\n"
        "Decision rules:\n"
        "  - Return the proportion for the current group as a percentage string with '%'.\n"
        "  - Only fill if the Participant Flow/Disposition explicitly reports this reason (withdrew consent) for the current group.\n"
        "  - If a count is given and Started (or the relevant denominator in the same table) is explicit, you may compute count/denominator*100.\n"
        "  - If not explicit or denominator unclear, return empty.\n"
        "Output example:\n"
        "  - Started=80, withdrew consent=8 -> '10%'\n"
    ),
    "Discont_AE": (
        "Definition: Proportion of participants who discontinued assigned treatment due to intolerability, e.g., SAE\n"
        "Decision rules:\n"
        "  - Return the proportion for the current group as a percentage string with '%'.\n"
        "  - Only fill if the Participant Flow/Disposition explicitly reports this reason (discontinued due to adverse event / SAE) for the current group.\n"
        "  - If a count is given and Started (or the relevant denominator in the same table) is explicit, you may compute count/denominator*100.\n"
        "  - If not explicit or denominator unclear, return empty.\n"
        "Output example:\n"
        "  - Started=80, discontinued due to AE=8 -> '10%'\n"
    ),
    "Discont_LE": (
        "Definition: Proportion of participants who discontinued assigned treatment due to lack of efficacy, e.g., disease progression\n"
        "Decision rules:\n"
        "  - Return the proportion for the current group as a percentage string with '%'.\n"
        "  - Only fill if the Participant Flow/Disposition explicitly reports this reason (discontinued due to lack of efficacy / disease progression) for the current group.\n"
        "  - If a count is given and Started (or the relevant denominator in the same table) is explicit, you may compute count/denominator*100.\n"
        "  - If not explicit or denominator unclear, return empty.\n"
        "Output example:\n"
        "  - Started=80, discontinued due to progression=8 -> '10%'\n"
    ),
    "Loss_FU": (
        "Definition: Proportion of participants who are lost to follow-up\n"
        "Decision rules:\n"
        "  - Return the proportion for the current group as a percentage string with '%'.\n"
        "  - Only fill if the Participant Flow/Disposition explicitly reports this reason (lost to follow-up) for the current group.\n"
        "  - If a count is given and Started (or the relevant denominator in the same table) is explicit, you may compute count/denominator*100.\n"
        "  - If not explicit or denominator unclear, return empty.\n"
        "Output example:\n"
        "  - Started=80, lost to follow-up=8 -> '10%'\n"
    ),
    "Protocol_Viol": (
        "Definition: Proportion of participants who experienced major protocol violations that are pre-specified in study protocol\n"
        "Decision rules:\n"
        "  - Return the proportion for the current group as a percentage string with '%'.\n"
        "  - Only fill if the Participant Flow/Disposition explicitly reports this reason (protocol violation) for the current group.\n"
        "  - If a count is given and Started (or the relevant denominator in the same table) is explicit, you may compute count/denominator*100.\n"
        "  - If not explicit or denominator unclear, return empty.\n"
        "Output example:\n"
        "  - Started=80, protocol violation=8 -> '10%'\n"
    ),
    "Treat_Adherence": (
        "Definition: Proportion of participants who take the study drug as prescribed\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when the percentage is explicitly stated for the current group, or when both numerator and denominator "
        "for the current group are explicitly given so the percentage can be computed.\n"
        "  - Do NOT infer values.\n"
        "Output example:\n"
        "  - 'Treat adherence: 60%' -> '60%'\n"
    ),
    "Visit_Adherence": (
        "Definition: Proportion of participants who attend scheduled visits within pre-specified time windows\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when the percentage is explicitly stated for the current group, or when both numerator and denominator "
        "for the current group are explicitly given so the percentage can be computed.\n"
        "  - Do NOT infer values.\n"
        "Output example:\n"
        "  - 'Visit adherence: 60%' -> '60%'\n"
    ),
    "Assess_Adherence": (
        "Definition: Proportion of participants who complete assessment of key proceudres and/or endpoints\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when the percentage is explicitly stated for the current group, or when both numerator and denominator "
        "for the current group are explicitly given so the percentage can be computed.\n"
        "  - Do NOT infer values.\n"
        "Output example:\n"
        "  - 'Assessment adherence: 60%' -> '60%'\n"
    ),
    "Med_Treat": (
        "Definition: Median treatment duration (e.g., 16.9 months)\n"
        "Decision rules:\n"
        "  - Return the median duration value EXACTLY as stated (keep units such as days/months/years).\n"
        "  - Only fill when 'median' is explicitly stated for the current group; otherwise empty.\n"
        "  - Do not convert units.\n"
        "Output example:\n"
        "  - 'median treatment duration 16.9 months' -> '16.9 months'\n"
    ),
    "Med_Follow_Up": (
        "Definition: Median follow-up duration\n"
        "Decision rules:\n"
        "  - Return the median duration value EXACTLY as stated (keep units such as days/months/years).\n"
        "  - Only fill when 'median' is explicitly stated for the current group; otherwise empty.\n"
        "  - Do not convert units.\n"
        "Output example:\n"
        "  - 'median follow-up 16.9 months' -> '16.9 months'\n"
    ),
    "Discont": (
        "Definition: Proportion of participants who discontinued the assigned treatment\n"
        "Decision rules:\n"
        "  - Return the proportion of participants who discontinued assigned treatment (any reason) as a percentage string with '%'.\n"
        "  - Use explicitly reported percent if available.\n"
        "  - Otherwise you may compute ONLY if discontinued count and the relevant denominator (usually Started) are explicitly given.\n"
        "  - If unclear, return empty.\n"
        "Output example:\n"
        "  - Started=80, discontinued=12 -> '15%'\n"
    ),
    "Med_Discont": (
        "Definition: Median time to discontinuation of assigned treatment, e.g., 6.3 months\n"
        "Decision rules:\n"
        "  - Return the median duration value EXACTLY as stated (keep units such as days/months/years).\n"
        "  - Only fill when 'median' is explicitly stated for the current group; otherwise empty.\n"
        "  - Do not convert units.\n"
        "Output example:\n"
        "  - 'median time to discontinuation 6.3 months' -> '6.3 months'\n"
    ),
    "Missing_All": (
        "Definition: Proportion of missing values for all variables\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when the percentage is explicitly stated for the current group, or when both numerator and denominator "
        "for the current group are explicitly given so the percentage can be computed.\n"
        "  - Do NOT infer values.\n"
        "Output example:\n"
        "  - 'Missing overall: 10%' -> '10%'\n"
    ),
    "Missing_PE": (
        "Definition: Proportion of missing value for primary endpoint\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when the percentage is explicitly stated for the current group, or when both numerator and denominator "
        "for the current group are explicitly given so the percentage can be computed.\n"
        "  - Do NOT infer values.\n"
        "Output example:\n"
        "  - 'Missing primary endpoint: 10%' -> '10%'\n"
    ),
    "Missing_Key_SE": (
        "Definition: Proportion of missing values for key secondary endpoint\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when the percentage is explicitly stated for the current group, or when both numerator and denominator "
        "for the current group are explicitly given so the percentage can be computed.\n"
        "  - Do NOT infer values.\n"
        "Output example:\n"
        "  - 'Missing key secondary endpoint: 10%' -> '10%'\n"
    ),
    "Prop_Renal": (
        "Definition: Proportion of participants who have adequate renal function\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group (often in baseline characteristics).\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Do NOT infer adequacy thresholds.\n"
        "Output example:\n"
        "  - 'Adequate renal function: 60%' -> '60%'\n"
    ),
    "Prop_Hepatic": (
        "Definition: Proportion of participants who have adequate hepatic function\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group (often in baseline characteristics).\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Do NOT infer adequacy thresholds.\n"
        "Output example:\n"
        "  - 'Adequate hepatic function: 60%' -> '60%'\n"
    ),
    "Prop_HighRisk": (
        "Definition: Proportion of participants who are at high risk of developing metastases of prostate cancer\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Do NOT infer the 'high risk' definition.\n"
        "Output example:\n"
        "  - 'High risk: 60%' -> '60%'\n"
    ),
    "Line0": (
        "Definition: Proportion of participants who are treatment naive\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - 'Treatment-naive: 30%' -> '30%'\n"
    ),
    "Line1": (
        "Definition: Proportion of participants who have experienced first line of therapy\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - '1 prior line: 50%' -> '50%'\n"
    ),
    "Line2": (
        "Definition: Proportion of participants who have experienced second line of therapy\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - '2 prior lines: 40%' -> '40%'\n"
    ),
    "Line3": (
        "Definition: Proportion of participants who have experienced third line of therapy\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - '>=3 prior lines: 20%' -> '20%'\n"
    ),
    "Lapsed": (
        "Definition: Proportion of participants whose disease recurs\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - 'Lapsed: 20%' -> '20%'\n"
    ),
    "Refractory": (
        "Definition: Proportion of participants whose disease does not respond to prior therapies\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - 'Refractory: 20%' -> '20%'\n"
    ),
    "Prop_ECOG0": (
        "Definition: Proportion of ECOG score equal to 0\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - 'ECOG 0: 40%' -> '40%'\n"
    ),
    "Prop_ECOG1": (
        "Definition: Proportion of ECOG score equal to 1\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - 'ECOG 1: 40%' -> '40%'\n"
    ),
    "Prop_ECOG2": (
        "Definition: Proportion of ECOG score equal to 2\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - 'ECOG 2: 15%' -> '15%'\n"
    ),
    "Prop_ECOG3": (
        "Definition: Proportion of ECOG score equal to 3\n"
        "Decision rules:\n"
        "  - Return a percentage string with '%'.\n"
        "  - Only fill when explicitly reported for the current group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "Output example:\n"
        "  - 'ECOG 3: 5%' -> '5%'\n"
    ),
    "TEAEs": (
        "Definition: Percentage of treatment-emerging AEs (e.g., 40% for TEAES > 15%)\n"
        "Decision rules:\n"
        "  - Capture the percentage of participants with treatment-emergent adverse events (TEAEs) for the current group.\n"
        "  - Only fill if TEAEs (or 'treatment-emergent') are explicitly mentioned for this group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Return as a percentage string with '%'.\n"
        "Output example:\n"
        "  - 'TEAEs: 40%' -> '40%'\n"
    ),
    "TRAEs": (
        "Definition: Percentage of Treatment-Related AEs\n"
        "Decision rules:\n"
        "  - Capture the percentage of participants with treatment-related adverse events (TRAEs) for the current group.\n"
        "  - Only fill if events are explicitly described as 'treatment-related', 'drug-related', or 'related to study treatment/drug'.\n"
        "  - Do NOT substitute TEAEs for TRAEs.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Return as a percentage string with '%'.\n"
        "Output example:\n"
        "  - 'Treatment-related AEs: 12%' -> '12%'\n"
    ),
    "SAEs": (
        "Definition: Percentage of serious adverse events, e.g., 0.5%\n"
        "Decision rules:\n"
        "  - Capture the percentage of participants with serious adverse events (SAEs) for the current group.\n"
        "  - Only fill if 'serious adverse event' / 'SAE' is explicitly reported for this group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Return as a percentage string with '%'.\n"
        "Output example:\n"
        "  - 'SAEs: 5%' -> '5%'\n"
    ),
    "Grade34_TEAEs": (
        "Definition: Percentage of grade 3 or 4 of TEAEs (e.g., 5%)\n"
        "Decision rules:\n"
        "  - Capture the percentage of participants with Grade 3 or 4 treatment-emergent AEs for the current group.\n"
        "  - Only fill if Grade 3/4 (or '>=3') AND TEAE/treatment-emergent are explicitly stated for this group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Return as a percentage string with '%'.\n"
        "Output example:\n"
        "  - 'Grade 3-4 TEAEs: 7%' -> '7%'\n"
    ),
    "Grade34_TRAEs": (
        "Definition: Percentage of grade 3 or 4 of TRAEs\n"
        "Decision rules:\n"
        "  - Capture the percentage of participants with Grade 3 or 4 treatment-related AEs for the current group.\n"
        "  - Only fill if Grade 3/4 (or '>=3') AND treatment-related/drug-related are explicitly stated for this group.\n"
        "  - Do NOT substitute Grade34_TEAEs for Grade34_TRAEs.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Return as a percentage string with '%'.\n"
        "Output example:\n"
        "  - 'Grade 3-4 treatment-related AEs: 2%' -> '2%'\n"
    ),
    "AE_Spec": (
        "Definition: Percentage of AEs of special interest (e.g., 1%)\n"
        "Decision rules:\n"
        "  - Capture the percentage of participants with adverse events of special interest for the current group.\n"
        "  - Only fill if the text explicitly labels events as 'AEs of special interest' (or similar).\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Return as a percentage string with '%'.\n"
        "Output example:\n"
        "  - 'AEs of special interest: 1%' -> '1%'\n"
    ),
    "Inc_TEAES": (
        "Definition: Incidence rate (number of AEs divided by the total exposure time X 100) of treatment-emerging AEs  (e.g., 3% for TEAES > 15%)\n"
        "Decision rules:\n"
        "  - Only fill when an incidence rate is explicitly reported (e.g., '3.2 per 100 patient-years', '3.2/100 PY').\n"
        "  - Prefer returning the rate expression as in text (keep the unit).\n"
        "  - Do NOT invent exposure time. If incidence is not explicitly stated, return empty.\n"
        "Output example:\n"
        "  - '3.2 per 100 patient-years' -> '3.2 per 100 PY'\n"
    ),
    "Inc_Grade34_TEAEs": (
        "Definition: Incidence rate  (number of AEs divided by the total exposure time X 100) of grade 3 or 4 of TEAEs  (e.g., 0.5%)\n"
        "Decision rules:\n"
        "  - Only fill when an incidence rate is explicitly reported (e.g., '3.2 per 100 patient-years', '3.2/100 PY').\n"
        "  - Prefer returning the rate expression as in text (keep the unit).\n"
        "  - Do NOT invent exposure time. If incidence is not explicitly stated, return empty.\n"
        "Output example:\n"
        "  - '0.8 per 100 patient-years' -> '0.8 per 100 PY'\n"
    ),
    "Grade34_Lab": (
        "Definition: Percentage of grade 3 or 4 laboratory abnormality (e.g., 2%)\n"
        "Decision rules:\n"
        "  - Capture the percentage of participants with Grade 3 or 4 laboratory abnormalities for the current group.\n"
        "  - Only fill if Grade 3/4 (or '>=3') lab abnormality is explicitly reported for this group.\n"
        "  - If both count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Return as a percentage string with '%'.\n"
        "Output example:\n"
        "  - 'Grade 3-4 laboratory abnormalities: 2%' -> '2%'\n"
    ),
    "Death_rate": (
        "Definition: Percentage of deaths, e.g., 0.3%\n"
        "Decision rules:\n"
        "  - Capture the percentage of deaths for the current group.\n"
        "  - Only fill if deaths are explicitly reported for this group.\n"
        "  - If both death count and group N are explicitly given, you may compute count/N*100.\n"
        "  - Return as a percentage string with '%'.\n"
        "Output example:\n"
        "  - 'Deaths: 0.3%' -> '0.3%'\n"
    ),
    "No_Dosage": (
        "Definition: Number of dosage levels\n"
        "Decision rules:\n"
        "  - Return an integer count of DISTINCT dosage levels.\n"
        "  - Only fill if you can explicitly list the distinct dose strings from text.\n"
        "  - If Dosage_Level is returned with k distinct entries (after de-dup), No_Dosage MUST be k.\n"
        "  - If dose is unclear/implicit, return empty.\n"
        "Output example:\n"
        "  - Dosage_Level='DrugA: 10 mg; DrugA: 20 mg' -> No_Dosage=2\n"
    ),
    "Dosage_Level": (
        "Definition: Dosage level, either new test drug or control drug\n"
        "Decision rules:\n"
        "  - Return dose strings EXACTLY as in text (keep units; do not convert).\n"
        "  - De-duplicate distinct dose strings; join multiple with '; '.\n"
        "  - If multiple interventions are present, prefix each dose with the intervention name (e.g., 'DrugA: 10 mg; DrugB: 5 mg').\n"
        "  - If dose is not explicitly stated, return empty.\n"
        "Output example:\n"
        "  - 'DrugA 10 mg/kg IV' -> 'DrugA: 10 mg/kg'\n"
    ),
    "Dosing_Frequency": (
        "Definition: Frequency of administration (e.g., QD, BID, Q3W)\n"
        "Decision rules:\n"
        "  - Only fill when the schedule is explicitly indicated.\n"
        "  - Normalize to tokens: QD, BID, TID, QW, Q2W, Q3W, Q4W.\n"
        "  - Map: once daily/daily->QD; twice daily->BID; three times daily->TID; weekly->QW; "
        "every 2 weeks->Q2W; every 3 weeks->Q3W; every 4 weeks/28 days->Q4W.\n"
        "  - If multiple interventions are present, prefix each frequency with the intervention name (e.g., 'DrugA: QD; DrugB: Q3W').\n"
        "  - If unclear, return empty.\n"
        "Output example:\n"
        "  - 'DrugA once daily' -> 'DrugA: QD'\n"
    ),
    "Cycle_Length": (
        "Definition: Length of treatment cycle in days (e.g., 21)\n"
        "Decision rules:\n"
        "  - Return cycle length as an integer number of days.\n"
        "  - If text states 'X-week cycle', convert to 7*X days; if text states '28-day cycle', return 28.\n"
        "  - Only fill when explicitly stated; otherwise empty.\n"
        "Output example:\n"
        "  - '3-week cycle' -> 21\n"
    ),
    "Formulation": (
        "Definition: Immediate or extended release\n"
        "Decision rules:\n"
        "  - Only fill when immediate-release / extended-release (or IR/ER/SR/CR/delayed-release) is explicitly stated.\n"
        "  - Return the wording as in text (e.g., 'immediate-release', 'extended-release', 'ER').\n"
        "  - If multiple are mentioned, join with '; '.\n"
        "  - If not mentioned, return empty.\n"
        "Output example:\n"
        "  - 'extended-release (ER) tablet' -> 'extended-release'\n"
    ),
}

PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    notes=GROUPS_NOTES,
    llm_fields=LLM_FIELDS,
    text_modules=TEXT_MODULES,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "Groups"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
