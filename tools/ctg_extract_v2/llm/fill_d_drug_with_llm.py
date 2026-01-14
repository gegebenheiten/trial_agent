#!/usr/bin/env python3
"""
Fill D_Drug table using the shared CTG LLM interface (v2 schema).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fill_table_with_llm import PromptConfig, main as fill_main

MECH_INSTRUCTIONS = (
    "You extract D_Drug mechanism/target/biomarker variables from ClinicalTrials.gov text using CSR-Vars "
    "annotations.\n"
    "Focus on the drug in CONTEXT (Generic_Name/Brand_Name). Return empty unless explicitly tied to this drug."
)

FORM_INSTRUCTIONS = (
    "You extract D_Drug formulation/identity variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Focus on the drug in CONTEXT (Generic_Name/Brand_Name). Return empty unless explicitly tied to this drug."
)

ALL_INSTRUCTIONS = (
    "You extract D_Drug variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Focus on the drug in CONTEXT (Generic_Name/Brand_Name). Return empty unless explicitly tied to this drug."
)

NONE_INSTRUCTIONS = "No extraction for this group."

MECH_FIELDS = [
    "Target_Engage",
    "Biochem_Change",
    "Physio_Change",
    "Imaging_Biomarker",
    "Molecular_EP",
]

FORM_FIELDS = [
    "Is_Biosimilar",
    "Excipients",
]

ALL_FIELDS = MECH_FIELDS + FORM_FIELDS

MECH_MODULES = [
    "interventions",
    "arm_groups",
    "study_info",
    "primary_outcomes",
    "secondary_outcomes",
    "results_outcomes",
]

FORM_MODULES = [
    "interventions",
    "arm_groups",
    "study_info",
]

ALL_MODULES = [
    "interventions",
    "arm_groups",
    "study_info",
    "primary_outcomes",
    "secondary_outcomes",
    "results_outcomes",
]

DRUG_NOTES = {
    "Target_Engage": "Targets or pathways explicitly engaged by the drug.",
    "Biochem_Change": "Mechanism of action or biochemical change explicitly attributed to the drug.",
    "Physio_Change": "Physiological effect explicitly attributed to the drug.",
    "Imaging_Biomarker": "Imaging biomarker used to assess drug effect.",
    "Molecular_EP": "Molecular endpoint or biomarker used to assess the drug.",
    "Is_Biosimilar": "Return 'yes' only if explicitly described as biosimilar/biosimilarity.",
    "Excipients": "Inactive ingredients/excipients; exclude active drug names.",
}

PROMPT_CONFIGS = {
    "mechanism": PromptConfig(
        instructions=MECH_INSTRUCTIONS,
        notes=DRUG_NOTES,
        llm_fields=MECH_FIELDS,
        text_modules=MECH_MODULES,
    ),
    "formulation": PromptConfig(
        instructions=FORM_INSTRUCTIONS,
        notes=DRUG_NOTES,
        llm_fields=FORM_FIELDS,
        text_modules=FORM_MODULES,
    ),
    "none": PromptConfig(
        instructions=NONE_INSTRUCTIONS,
        notes=DRUG_NOTES,
        llm_fields=[],
        text_modules=[],
    ),
    "all": PromptConfig(
        instructions=ALL_INSTRUCTIONS,
        notes=DRUG_NOTES,
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
        sys.argv.extend(["--table", "D_Drug"])
    group = pop_arg("--group") or pop_arg("--pass")
    group = (group or "all").strip().lower()
    aliases = {
        "mech": "mechanism",
        "mechanism": "mechanism",
        "target": "mechanism",
        "biomarker": "mechanism",
        "safety": "none",
        "tox": "none",
        "toxicity": "none",
        "pk": "none",
        "pd": "none",
        "pkpd": "none",
        "pharmacokinetics": "none",
        "pharmacodynamics": "none",
        "form": "formulation",
        "formulation": "formulation",
        "identity": "formulation",
        "excipients": "formulation",
        "biosimilar": "formulation",
        "none": "none",
        "all": "all",
    }
    group = aliases.get(group, group)
    prompt_config = PROMPT_CONFIGS.get(group, PROMPT_CONFIGS["all"])
    fill_main(prompt_config=prompt_config)


if __name__ == "__main__":
    main()
