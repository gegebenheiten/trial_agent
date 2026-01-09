#!/usr/bin/env python3
"""
Fill Endpoints results table using the shared CTG LLM interface.
"""

import sys
from typing import List

from fill_ctg_table_with_llm import PromptConfig, main as fill_main

INSTRUCTIONS = (
    "You extract endpoint-level results variables from ClinicalTrials.gov results text using CSR-Vars annotations.\n"
    "Category: Overall.\n"
    "Focus on endpoint name/type, analysis strategy, CI, OS/PFS/ORR/pCR, PRO/QoL, and p-values.\n"
    "Target endpoint is the row's Endpoint_Name and Endpoint_Type; extract values only for that endpoint.\n"
    "If multiple matched outcomes are shown, use ONLY the first (highest match score) outcome block.\n"
    "Evidence must come from within that selected outcome block; if unsure, return empty."
)

LLM_FIELDS = [
    "Strategy",
    "Missing_Imput",
    "Covariate_Adjust",
    "MCP",
    "Subgroup_Ana",
    "EP_Value",
    "EP_Unit",
    "EP_Point",
    "ARR",
    "NNT",
    "EP_95CI",
    "Med_OS",
    "OS_YrX",
    "Med_PFS",
    "ORR",
    "pCR",
    "Med_DOR",
    "RMST",
    "PRO",
    "QoL",
]

ENDPOINT_NOTES = {
    "EP_Value": "Return value only (numbers or %). Do not include explanatory text.",
    "EP_Unit": "Return unit only (e.g., %, months, days, events/100 PY, HR, OR).",
    "EP_Point": "Return timepoint only (e.g., 'Week 24', 'Month 6', 'Year 2').",
    "EP_95CI": "Return as 'lower, upper' only (no '95% CI' prefix).",
    "Strategy": "Return keywords only (e.g., ITT, per-protocol, log-rank, Cox, MMRM).",
    "Missing_Imput": "Return keywords only (e.g., MI, LOCF) or empty.",
    "Covariate_Adjust": "Return keywords only (e.g., covariate-adjusted, ANCOVA, Cox).",
    "MCP": "Return multiplicity correction keywords only (e.g., Bonferroni, Holm, Hochberg).",
    "Subgroup_Ana": "Return subgroup analysis keywords only (e.g., pre-specified subgroup, interaction test).",
}


def endpoint_extra_rules(missing_fields: List[str]) -> List[str]:
    _ = missing_fields
    return [
        "- Use ONLY the first outcome listed under 'Matched Outcomes' (highest match score). Ignore any other outcomes.",
        "- For EP_Value/EP_Unit/EP_Point/EP_95CI/ARR/NNT/Med_OS/OS_YrX/Med_PFS/ORR/pCR/Med_DOR/RMST/PRO/QoL, prefer the Analyses section inside the selected outcome. If Analyses is absent, return empty unless the value is explicitly stated elsewhere in the selected outcome block.",
    ]


PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    notes=ENDPOINT_NOTES,
    extra_rules_fn=endpoint_extra_rules,
    llm_fields=LLM_FIELDS,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "Endpoints"])
    if "--max-fields-per-call" not in sys.argv:
        sys.argv.extend(["--max-fields-per-call", "8"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
