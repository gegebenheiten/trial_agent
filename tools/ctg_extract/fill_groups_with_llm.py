#!/usr/bin/env python3
"""
Fill Groups results table using the shared CTG LLM interface.
"""

import sys
from typing import List

from fill_ctg_table_with_llm import PromptConfig, main as fill_main

INSTRUCTIONS = (
    "You extract group-level results variables from ClinicalTrials.gov results text using CSR-Vars annotations.\n"
    "Categories: Overall, Compliance (by group), Dosing.\n"
    "Focus on group dosage/regimen details, adherence/compliance, demographics, missingness, and safety outcomes.\n"
    "When multiple groups are present, target the current row's Arm_ID/group_title_raw/group_desc_raw."
)

LLM_FIELDS = [
    "Outcome",
    "Termination",
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

GROUPS_NOTES = {
    "No_Dosage": (
        "Return an integer count of DISTINCT dosage levels only if you can explicitly list the distinct dose strings from text; "
        "otherwise empty. If Dosage_Level is provided, No_Dosage should equal the number of distinct entries in Dosage_Level "
        "(including schedule if it is part of the dose string)."
    ),
    "Dosage_Level": (
        "Return dose strings EXACTLY as in text (include units like mg/m2, mg/kg, IU, mcg). "
        "If multiple distinct doses exist, join with '; '. De-duplicate. "
        "If multiple interventions are present, prefix each dose with the intervention name (e.g., 'DrugA: 10 mg; DrugB: 5 mg')."
    ),
    "Dosing_Frequency": (
        "Normalize to tokens like QD, BID, TID, QW, Q2W, Q3W, Q4W. "
        "Map: once daily/daily -> QD; twice daily -> BID; three times daily -> TID; weekly -> QW; every 2 weeks -> Q2W; every 3 weeks -> Q3W; every 4 weeks/28 days -> Q4W. "
        "If multiple interventions are present, prefix each frequency with the intervention name (e.g., 'DrugA: QD; DrugB: Q3W'). "
        "If unclear, empty."
    ),
    "Cycle_Length": (
        "Return number of days only (e.g., 21). "
        "If text states 'X-week cycle' -> 7*X days (e.g., 3-week -> 21). "
        "If text states '28-day cycle' -> 28. If unclear, empty."
    ),
    "Formulation": (
        "Return formulation words exactly as in text if present (e.g., tablet, capsule, solution, suspension, infusion). "
        "Multiple use '; '. Otherwise empty."
    ),
}


def groups_extra_rules(missing_fields: List[str]) -> List[str]:
    rules: List[str] = []
    fields = {field.lower(): field for field in missing_fields}
    if "no_dosage" in fields:
        rules.append(
            "- No_Dosage: return an integer count of DISTINCT dosage levels only if you can explicitly list the distinct dose strings from text; otherwise empty. "
            "If Dosage_Level is returned with k distinct entries, No_Dosage must be k."
        )
    if "dosage_level" in fields:
        rules.append(
            "- Dosage_Level: return dose strings exactly as in text (keep units; do not convert). "
            "De-duplicate. Multiple use '; '. If multiple interventions are present, prefix each dose with the intervention name."
        )
    if "dosing_frequency" in fields:
        rules.append(
            "- Dosing_Frequency: normalize to tokens like QD, BID, TID, QW, Q2W, Q3W, Q4W only when the text explicitly indicates the schedule."
        )
    if "cycle_length" in fields:
        rules.append(
            "- Cycle_Length: return number of days only. 'X-week cycle' -> 7*X. '28-day cycle' -> 28. If unclear, empty."
        )
    if "formulation" in fields:
        rules.append(
            "- Formulation: return formulation words exactly as in text (e.g., tablet, capsule, solution). Multiple use '; '."
        )
    return rules

PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    notes=GROUPS_NOTES,
    extra_rules_fn=groups_extra_rules,
    llm_fields=LLM_FIELDS,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "Groups"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
