#!/usr/bin/env python3
"""
Fill TargetPop table using the shared CTG LLM interface.
"""

import sys
from typing import List

from fill_ctg_table_with_llm import PromptConfig, main as fill_main

INSTRUCTIONS = (
    "You extract target population eligibility variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Categories: Target Population, Overall.\n"
    "Focus on disease/stage, prior lines, relapse/refractory status, biomarkers, performance status, organ function, comorbidities, and concomitant meds."
)

LLM_FIELDS = [
    "Disease_Stage",
    "Prior_Line",
    "Relapsed",
    "Refractory",
    "Geno_Biomarker",
    "Comp_Biomarker",
    "Other_Biomarker",
    "ECOG",
    "Karnofsky",
    "Kidney_Func",
    "Hepatic_Func",
    "Hemo_Func",
    "Cardio_Func",
    "Comorb_Prohib",
    "Concom_Prohib",
    "Baseline_Severity_Score",
    "Gender_Criteria",
    "Washout_Period",
    "Pregnancy_Lactation",
]


def targetpop_extra_rules(missing_fields: List[str]) -> List[str]:
    rules: List[str] = []
    fields = {field.lower(): field for field in missing_fields}

    if "disease_stage" in fields:
        rules.append(
            "- Disease_Stage: include the disease name with the stage (e.g., 'NSCLC Stage I-IV')."
        )
    biomarker_fields = [f for f in missing_fields if "biomarker" in f.lower()]
    if biomarker_fields:
        rules.append(
            "- Biomarker fields: list biomarker names/criteria; separate multiple with '; '."
        )
    return rules


PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    extra_rules_fn=targetpop_extra_rules,
    llm_fields=LLM_FIELDS,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "TargetPop"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
