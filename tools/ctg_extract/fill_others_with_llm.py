#!/usr/bin/env python3
"""
Fill Others table using the shared CTG LLM interface.
"""

import sys

from fill_ctg_table_with_llm import PromptConfig, main as fill_main

INSTRUCTIONS = (
    "You extract operational/other trial metadata from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Category: Overall.\n"
    "Focus on competing trials, standard-of-care counts, CRO usage/roles, and RBM."
)

LLM_FIELDS = [
    "No_Competing_Trial",
    "No_SOC",
    "CRO",
    "CRO_Oper",
    "CRO_Stat",
    "CRO_DM",
    "RBM",
]
# study_info, eligibility, design_info, arm_groups, interventions, primary_outcomes, secondary_outcomes
# participant_flow, baseline_results, baseline_measures, results_outcomes, reported_events
# keywords, conditions, location_countries
# endpoint_target, endpoint_matches

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

PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    llm_fields=LLM_FIELDS,
    text_modules=TEXT_MODULES,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "Others"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
