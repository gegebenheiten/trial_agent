#!/usr/bin/env python3
"""
Fill Drug table using the shared CTG LLM interface.
"""

import sys

from fill_ctg_table_with_llm import PromptConfig, main as fill_main

INSTRUCTIONS = (
    "You extract drug property variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Categories: Molecular structure, Absorption, Distribution, Metabolism, Excretion, Mechanism of action, Overall.\n"
    "Return a value only when it is explicitly stated in the text."
)

LLM_FIELDS = []

PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    llm_fields=LLM_FIELDS,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "Drug"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
