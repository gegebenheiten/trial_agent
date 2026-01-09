#!/usr/bin/env python3
"""
Fill Design table using the shared CTG LLM interface.
"""

import sys
from typing import List

from fill_ctg_table_with_llm import PromptConfig, main as fill_main

INSTRUCTIONS = (
    "You extract clinical trial design variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Categories: Administrative, CSR, Enrollment, Operational, Overall, Treatment.\n"
    "Follow each field's annotation as the definition."
)

LLM_FIELDS = [
    "GCP_Compliance",
    "Data_Cutoff_Date",
    "Route_Admin",
    "Treat_Duration",
    "Add_On_Treat",
    "Adherence_Treat",
    "Enroll_Duration_Plan",
    "FU_Duration_Plan",
    "Central_Lab",
    "Run_in",
]

MODE_ADMIN_CHOICES = [
    "oral",
    "iv",
    "im",
    "sc",
    "inhalation",
    "topical",
    "transdermal",
    "intrathecal",
    "intranasal",
    "intravitreal",
    "rectal",
    "vaginal",
    "other",
]
TIME_UNIT_TOKENS = ("duration", "time", "period")


def has_time_unit_field(fields: List[str]) -> bool:
    for field in fields:
        lower = field.lower()
        if any(token in lower for token in TIME_UNIT_TOKENS):
            return True
    return False


def design_extra_rules(missing_fields: List[str]) -> List[str]:
    rules: List[str] = []
    fields = {field.lower(): field for field in missing_fields}

    if "route_admin" in fields:
        choices = ", ".join(MODE_ADMIN_CHOICES)
        rules.append(
            f"- Route_Admin: choose one or more from [{choices}] and normalize to these tokens; use '; ' to separate multiple."
        )

    # only apply unit rule to planned duration fields (avoid conflicting with Treat_Duration like "6 cycles")
    if "enroll_duration_plan" in fields or "fu_duration_plan" in fields:
        rules.append(
            "- Enroll_Duration_Plan / FU_Duration_Plan: return the literal planned duration phrase WITH unit (days/weeks/months/years). "
            "Do NOT compute from Start/Completion dates. If unit not explicit, return empty."
        )
    return rules

DESIGN_NOTES = {
    "GCP_Compliance": (
        "Definition: Whether the trial is conducted in compliance with Good Clinical Practice (GCP/ICH-GCP).\n"
        "Return 'yes' only if explicitly states GCP/ICH-GCP (e.g., 'Good Clinical Practice', 'ICH E6'); "
        "return 'no' only if explicitly states not compliant; else empty."
    ),
    "Data_Cutoff_Date": (
        "Definition: Data cutoff / database cutoff / database lock date used for analysis.\n"
        "Return a date only if text explicitly mentions 'data cutoff', 'cut-off', 'database lock', 'DCO'. "
        "Normalize to YYYY-MM-DD if possible; if only month/year, keep literal (e.g., 'June 2023'); else empty. "
        "Do NOT use posted/update dates unless explicitly called data cutoff."
    ),
    "Route_Admin": (
        "Definition: Route(s) of administration of the study intervention(s).\n"
        "Choose from allowed tokens; multiple use '; '. "
        "Map: intravenous/IV/infusion->iv; subcutaneous/SC->sc; intramuscular/IM->im; oral/by mouth->oral; "
        "inhaled/nebulized->inhalation; transdermal/patch->transdermal; topical->topical; otherwise 'other'."
    ),
    "Treat_Duration": (
        "Definition: Planned treatment duration / length of regimen (e.g., multiple cycles) or stopping rule.\n"
        "Return literal duration/regimen phrase (e.g., '6 cycles', '24 weeks', 'until disease progression'); "
        "do not infer/convert; cycles allowed."
    ),
    "Add_On_Treat": (
        "Definition: Whether the treatment is a combination therapy (two or more drugs/interventions used together).\n"
        "Return 'yes' only if the text explicitly indicates a combination regimen (e.g., 'in combination with', 'plus', 'added to', "
        "'co-administered with', 'with/and' another drug, multi-drug regimen). "
        "Require >=2 distinct drugs/interventions mentioned; otherwise empty. Do NOT infer from arm count."
    ),
    "Adherence_Treat": (
        "Definition: Adherence/compliance strategy or assessment for treatment (monitoring/management of adherence).\n"
        "Return 'yes' only if explicitly describes adherence/compliance assessment or enforcement "
        "(e.g., pill count, diary, electronic monitoring, compliance assessed); else empty.\n"
        "NOTE: your earlier 'crossover/switch' interpretation is likely not aligned with CSR-Vars here."
    ),
    "Enroll_Duration_Plan": (
        "Definition: Planned enrollment/recruitment duration.\n"
        "Return only if explicitly stated (e.g., 'enrollment will last 18 months'); must include unit; "
        "do NOT compute from Start/Completion dates; else empty."
    ),
    "FU_Duration_Plan": (
        "Definition: Planned follow-up duration.\n"
        "Return only if explicitly stated (e.g., 'followed for 12 months'); must include unit; "
        "do NOT infer; else empty."
    ),
    "Central_Lab": (
        "Definition: Whether a central laboratory is used (centralized lab testing).\n"
        "Return 'yes' only if explicitly mentions central lab/central laboratory; "
        "return 'no' only if explicitly says local labs only / no central lab; else empty."
    ),
    "Run_in": (
        "Definition: Whether there is a safety run-in / lead-in period (including placebo run-in).\n"
        "Return 'yes' only if explicitly mentions run-in/lead-in period; "
        "return 'no' only if explicitly says no run-in; else empty."
    ),
}


PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    notes=DESIGN_NOTES,
    extra_rules_fn=design_extra_rules,
    llm_fields=LLM_FIELDS,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "Design"])
    if "--max-fields-per-call" not in sys.argv:
        sys.argv.extend(["--max-fields-per-call", "6"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
