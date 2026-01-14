#!/usr/bin/env python3
"""
Fill D_Pop table using the shared CTG LLM interface (v2 schema).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fill_table_with_llm import PromptConfig, main as fill_main

DISEASE_INSTRUCTIONS = (
    "You extract D_Pop disease/diagnosis variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Follow each field's definition exactly; return empty unless explicitly stated."
)

ASSESS_INSTRUCTIONS = (
    "You extract D_Pop performance/assessment variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Be conservative; return empty unless explicitly supported."
)

ORGAN_INSTRUCTIONS = (
    "You extract D_Pop organ/safety/exclusion variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Be conservative; return empty unless explicitly supported."
)

PROG_INSTRUCTIONS = (
    "You extract D_Pop prognostic/predictive/digital variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Be conservative; return empty unless explicitly supported."
)

ALL_INSTRUCTIONS = (
    "You extract D_Pop variables from ClinicalTrials.gov text using CSR-Vars annotations.\n"
    "Follow each field's definition exactly; return empty unless explicitly stated."
)

DISEASE_FIELDS = [
    "Disease_Stage",
    "Prevalence",
    "Relapsed",
    "Refractory",
    "Prior_Line",
    "GIST",
    "Diag_Dis_Detect",
    "Diag_Dis_Subtyp",
    "Diag_Molecular",
    "Diag_Histologic",
    "Diag_Radiographic",
    "Diag_Physiologic",
]

ASSESS_FIELDS = [
    "Anat_Physio_Measures",
    "Clinical_Symptom_Score",
    "ECOG",
    "Karnofsky",
    "Cognitive_Scale",
    "Vital_Signs",
    "Imaging",
    "Baseline_Severity_Score",
]

ORGAN_FIELDS = [
    "Kidney_Func",
    "Hepatic_Func",
    "Hemo_Func",
    "Cardio_Func",
    "Comorb_Prohib",
    "Concom_Prohib",
    "Washout_Period",
    "Pregnancy_Lactation",
    "Safety_Hema",
    "Safety_Hepa",
    "Safety_Renal",
    "Safety_Cardiac",
    "Safety_Immune",
]

PROG_FIELDS = [
    "Prog_High_Risk",
    "Prog_Strat",
    "Prog_Molecular",
    "Prog_Inflamm",
    "Prog_Histologic",
    "Prog_Circulat",
    "Pred_Enrich",
    "Pred_Oncogenic",
    "Pred_Response",
    "Pred_Others",
    "Digit_Actig",
    "Digit_Physio",
    "Digit_Vocal",
    "Digit_Cognit",
]

ALL_FIELDS = DISEASE_FIELDS + ASSESS_FIELDS + ORGAN_FIELDS + PROG_FIELDS

DISEASE_MODULES = [
    "eligibility",
    "conditions",
    "keywords",
    "study_info",
]

ASSESS_MODULES = [
    "eligibility",
    "primary_outcomes",
    "secondary_outcomes",
    "study_info",
]

ORGAN_MODULES = [
    "eligibility",
    "study_info",
]

PROG_MODULES = [
    "eligibility",
    "study_info",
    "keywords",
]

ALL_MODULES = [
    "study_info",
    "eligibility",
    "conditions",
    "keywords",
    "primary_outcomes",
    "secondary_outcomes",
]

TARGETPOP_NOTES = {
    'Baseline_Severity_Score': "Meaning: Specific baseline severity scores required for eligibility (e.g., HAM-D > 20, PASI > 12).\nOutput: Copy the numeric threshold and the name of the scale (e.g., 'HAM-D > 20', 'PASI >= 12').\nExtract: Fill ONLY if explicit numeric thresholds or scale-based inclusion criteria are stated.\nPositive cues: 'HAM-D >=20', 'PASI >12', 'CGI-S >=4', 'baseline severity score above moderate'.\nNegative cues (leave empty): 'severe disease' without specific scale or number.",
    'Cardio_Func': "Meaning: Whether cardiac (heart) function criteria are specified.\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if eligibility includes cardiac function thresholds or mentions heart disease or cardiovascular conditions.\nPositive cues: 'LVEF >=50%', 'no uncontrolled heart failure', 'adequate cardiac function', 'no history of myocardial infarction', 'patients with arrhythmia excluded'.\nNegative cues (leave empty): 'organ function' without heart or cardiovascular reference.",
    'Comorb_Prohib': "Meaning: Whether specific comorbidities disqualify participants.\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if text explicitly prohibits participants with certain comorbid diseases (e.g., cardiac, renal, hepatic, metabolic).\nPositive cues: 'patients with uncontrolled hypertension are excluded', 'no history of HIV infection', 'no significant cardiac disease'.\nNegative cues (leave empty): vague 'no serious comorbidities' statements.",
    'Comp_Biomarker': "Meaning: Whether composite biomarkers are used (combined indices such as TMB, CPS, or MSI).\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if composite or integrated biomarker scores are specified.\nPositive cues: 'tumor mutational burden (TMB)', 'combined positive score (CPS)', 'microsatellite instability (MSI-high)'.\nNegative cues (leave empty): single-molecule biomarkers only.",
    'Concom_Prohib': "Meaning: prohibited concomitant medications/therapies. Output: 'yes' only; otherwise empty. Extract: fill if text explicitly prohibits concomitant treatments.",
    'Disease_Stage': "Meaning: Disease definition and stage(s) under investigation, based on (1) pathophysiological classification (e.g., severity, diagnostic findings), (2) clinical progression (e.g., early-stage, advanced, symptomatic/asymptomatic), (3) biomarker/imaging evidence (e.g., tumor size, molecular markers), or (4) trial-specific criteria (e.g., treatment-naive, relapsed, refractory).\nOutput: Descriptive text summarizing disease stage(s) explicitly mentioned.\nExtract: Fill ONLY if text explicitly defines or implies disease stage(s) or disease state criteria in eligibility or study design sections.\nPositive cues: 'early-stage breast cancer', 'advanced solid tumor', 'treatment-naive patients', 'relapsed multiple myeloma', 'refractory lymphoma', 'Stage II-III NSCLC'.\nNegative cues (leave empty): general disease names without stage indication (e.g., 'breast cancer', 'lung cancer').",
    'ECOG': "Meaning: Minimum ECOG performance status allowed for participation.\nOutput: Numeric value (0-5 scale).if mentions multiple values,list them all,such as 0 or 1\nExtract: Fill if ECOG requirement is specified.\nPositive cues: 'ECOG 0-1', 'ECOG <=2', 'ECOG performance status of 1 or lower'.\nNegative cues (leave empty): Karnofsky-only mention.",
    'Gender_Criteria': "Meaning: Sex eligibility criteria (Male / Female / Both).\nOutput: 'Male', 'Female', or 'Both'.\nExtract: Fill if eligibility criteria explicitly restrict or allow both sexes.\nPositive cues: 'male participants only', 'females only', 'both sexes eligible'.\nNegative cues (leave empty): epidemiologic mentions (e.g., 'disease more common in women').",
    'Geno_Biomarker': "Meaning: Whether genomic biomarkers are used in identifying participants (e.g., mutations, expression levels).\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if genomic or molecular markers are inclusion criteria.\nPositive cues: 'BRCA1/2 mutation-positive', 'EGFR-mutant NSCLC', 'TP53 mutation'.\nNegative cues (leave empty): general biomarker mentions not genomic (e.g., 'TSH', 'LDL').",
    'Hemo_Func': "Meaning: Whether hematologic (blood) function criteria are specified.\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if eligibility includes blood cell parameters or mentions hematologic disorders.\nPositive cues: 'ANC >=1500/uL', 'platelets >=100,000/uL', 'hemoglobin >=9 g/dL', 'no history of anemia', 'no hematologic malignancy'.\nNegative cues (leave empty): 'adequate function' without blood or hematology reference.",
    'Hepatic_Func': "Meaning: Whether hepatic (liver) function criteria are specified.\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if eligibility includes liver enzyme thresholds or mentions hepatic/liver disease or dysfunction or Hepatitis.\nPositive cues: 'AST/ALT <=2.5x ULN', 'bilirubin <=1.5x ULN', 'adequate hepatic function', 'no hepatic impairment', 'patients with liver disease excluded','Active Hepatitis B or C'\nNegative cues (leave empty): generic 'organ function' without liver reference.",
    'Karnofsky': "Meaning: Minimum Karnofsky performance score required.\nOutput: Numeric value (0-100 scale).\nExtract: Fill if Karnofsky score threshold is given.\nPositive cues: 'Karnofsky >=70', 'Karnofsky performance status of 80 or higher'.\nNegative cues (leave empty): ECOG-only mention.",
    'Kidney_Func': "Meaning: Whether kidney (renal) function criteria are specified.\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if eligibility requires renal function levels or mentions kidney-related disease or dysfunction.\nPositive cues: 'creatinine clearance >=60 mL/min', 'adequate renal function', 'BUN <=1.5x ULN', 'no renal impairment', 'history of kidney disease excluded'.\nNegative cues (leave empty): 'organ function' without renal or kidney reference.",
    'Other_Biomarker': "Meaning: Whether non-genomic biomarkers are used (e.g., biochemical, serologic, or imaging-based).\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if other biomarker-based inclusion criteria are present.\nPositive cues: 'BNP > 100 pg/mL', 'HbA1c > 6.5%', 'LDL-C', 'Amyloid-beta42/40 ratio'.\nNegative cues (leave empty): purely clinical or demographic eligibility.",
    'Pregnancy_Lactation': "Meaning: Exclusion criteria regarding pregnancy or breastfeeding.\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if text explicitly excludes pregnant or lactating women.\nPositive cues: 'pregnant or breastfeeding women are excluded', 'negative pregnancy test required', 'must use contraception'.\nNegative cues (leave empty): 'female participants' without pregnancy-related restriction.",
    'Prior_Line': "Meaning: Minimum number of prior lines of therapy required for eligibility.\nOutput: Integer (e.g., '1', '2').\nExtract: Fill ONLY if text specifies 'at least X prior lines' or equivalent.\nPositive cues: '>=1 prior line of therapy', 'previously treated with at least two regimens'.\nNegative cues (leave empty): 'previously treated' (without specifying number).",
    'Refractory': "Meaning: Indicates whether the disease population includes refractory cases (disease is resistant to treatment).\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if text explicitly mentions refractory disease.\nPositive cues: 'refractory', 'not responsive to prior therapy', 'treatment-resistant'.\nNegative cues (leave empty): 'relapsed' (without mention of refractory).",
    'Relapsed': "Meaning: Indicates whether the disease population includes relapsed cases (disease has returned after initial response).\nOutput: 'yes' only; otherwise empty.\nExtract: Fill if text explicitly describes relapsed/recurrent disease.\nPositive cues: 'relapsed', 'recurrent', 'disease progression after remission'.\nNegative cues (leave empty): 'advanced' or 'late-stage' without mention of relapse.",
    'Washout_Period': "Meaning: Required time interval since last therapy before enrollment.\nOutput: Numeric value plus time unit (e.g., '14 days', '4 weeks').\nExtract: Fill ONLY if text specifies a minimum washout period before screening or randomization.\nPositive cues: 'washout period of at least 14 days', 'no prior therapy within 30 days'.\nNegative cues (leave empty): 'previously treated' without time restriction.",
}

PROMPT_CONFIGS = {
    "disease": PromptConfig(
        instructions=DISEASE_INSTRUCTIONS,
        notes=TARGETPOP_NOTES,
        llm_fields=DISEASE_FIELDS,
        text_modules=DISEASE_MODULES,
    ),
    "assessment": PromptConfig(
        instructions=ASSESS_INSTRUCTIONS,
        notes=TARGETPOP_NOTES,
        llm_fields=ASSESS_FIELDS,
        text_modules=ASSESS_MODULES,
    ),
    "organ": PromptConfig(
        instructions=ORGAN_INSTRUCTIONS,
        notes=TARGETPOP_NOTES,
        llm_fields=ORGAN_FIELDS,
        text_modules=ORGAN_MODULES,
    ),
    "prognostic": PromptConfig(
        instructions=PROG_INSTRUCTIONS,
        notes=TARGETPOP_NOTES,
        llm_fields=PROG_FIELDS,
        text_modules=PROG_MODULES,
    ),
    "all": PromptConfig(
        instructions=ALL_INSTRUCTIONS,
        notes=TARGETPOP_NOTES,
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
        sys.argv.extend(["--table", "D_Pop"])
    group = pop_arg("--group") or pop_arg("--pass")
    group = (group or "all").strip().lower()
    aliases = {
        "diagnosis": "disease",
        "disease": "disease",
        "performance": "assessment",
        "assessment": "assessment",
        "organ": "organ",
        "safety": "organ",
        "exclusion": "organ",
        "prognostic": "prognostic",
        "predictive": "prognostic",
        "digital": "prognostic",
        "all": "all",
    }
    group = aliases.get(group, group)
    prompt_config = PROMPT_CONFIGS.get(group, PROMPT_CONFIGS["all"])
    fill_main(prompt_config=prompt_config)


if __name__ == "__main__":
    main()
