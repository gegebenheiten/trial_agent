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

PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    llm_fields=LLM_FIELDS,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "Others"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
