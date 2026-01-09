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

# study_info, eligibility, design_info, arm_groups, interventions, primary_outcomes, secondary_outcomes
# participant_flow, baseline_results, baseline_measures, results_outcomes, reported_events
# keywords, conditions, location_countries
# endpoint_target, endpoint_matches
TEXT_MODULES = [
    "study_info",
    "eligibility",
]


TARGETPOP_NOTES = {
    "Disease_Stage":(
    "Meaning: Disease definition and stage(s) under investigation, based on (1) pathophysiological classification (e.g., severity, diagnostic findings), (2) clinical progression (e.g., early-stage, advanced, symptomatic/asymptomatic), (3) biomarker/imaging evidence (e.g., tumor size, molecular markers), or (4) trial-specific criteria (e.g., treatment-naive, relapsed, refractory).",
    "Output: Descriptive text summarizing disease stage(s) explicitly mentioned.",
    "Extract: Fill ONLY if text explicitly defines or implies disease stage(s) or disease state criteria in eligibility or study design sections.",
    "Positive cues: 'early-stage breast cancer', 'advanced solid tumor', 'treatment-naive patients', 'relapsed multiple myeloma', 'refractory lymphoma', 'Stage II–III NSCLC'.",
    "Negative cues (leave empty): general disease names without stage indication (e.g., 'breast cancer', 'lung cancer')."
    ), 
    "Prior_Line":(
    "Meaning: Minimum number of prior lines of therapy required for eligibility.",
    "Output: Integer (e.g., '1', '2').",
    "Extract: Fill ONLY if text specifies 'at least X prior lines' or equivalent.",
    "Positive cues: '≥1 prior line of therapy', 'previously treated with at least two regimens'.",
    "Negative cues (leave empty): 'previously treated' (without specifying number)."
    ),
    "Relapsed":(
    "Meaning: Indicates whether the disease population includes relapsed cases (disease has returned after initial response).",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if text explicitly describes relapsed/recurrent disease.",
    "Positive cues: 'relapsed', 'recurrent', 'disease progression after remission'.",
    "Negative cues (leave empty): 'advanced' or 'late-stage' without mention of relapse."
    ),
    "Refractory":(
    "Meaning: Indicates whether the disease population includes refractory cases (disease is resistant to treatment).",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if text explicitly mentions refractory disease.",
    "Positive cues: 'refractory', 'not responsive to prior therapy', 'treatment-resistant'.",
    "Negative cues (leave empty): 'relapsed' (without mention of refractory)."   
    ),
    "Geno_Biomarker":(
    "Meaning: Whether genomic biomarkers are used in identifying participants (e.g., mutations, expression levels).",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if genomic or molecular markers are inclusion criteria.",
    "Positive cues: 'BRCA1/2 mutation-positive', 'EGFR-mutant NSCLC', 'TP53 mutation'.",
    "Negative cues (leave empty): general biomarker mentions not genomic (e.g., 'TSH', 'LDL')."    
    ),
    "Comp_Biomarker":(
    "Meaning: Whether composite biomarkers are used (combined indices such as TMB, CPS, or MSI).",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if composite or integrated biomarker scores are specified.",
    "Positive cues: 'tumor mutational burden (TMB)', 'combined positive score (CPS)', 'microsatellite instability (MSI-high)'.",
    "Negative cues (leave empty): single-molecule biomarkers only."
    ),
    "Other_Biomarker":(
    "Meaning: Whether non-genomic biomarkers are used (e.g., biochemical, serologic, or imaging-based).",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if other biomarker-based inclusion criteria are present.",
    "Positive cues: 'BNP > 100 pg/mL', 'HbA1c > 6.5%', 'LDL-C', 'Amyloid-β42/40 ratio'.",
    "Negative cues (leave empty): purely clinical or demographic eligibility."
    ),
    "ECOG":(
    "Meaning: Minimum ECOG performance status allowed for participation.",
    "Output: Numeric value (0–5 scale).if mentions multiple values,list them all,such as 0 or 1",
    "Extract: Fill if ECOG requirement is specified.",
    "Positive cues: 'ECOG 0–1', 'ECOG ≤2', 'ECOG performance status of 1 or lower'.",
    "Negative cues (leave empty): Karnofsky-only mention."
    ),
    "Karnofsky":(
    "Meaning: Minimum Karnofsky performance score required.",
    "Output: Numeric value (0–100 scale).",
    "Extract: Fill if Karnofsky score threshold is given.",
    "Positive cues: 'Karnofsky ≥70', 'Karnofsky performance status of 80 or higher'.",
    "Negative cues (leave empty): ECOG-only mention."
    ),
    "Kidney_Func": (
    "Meaning: Whether kidney (renal) function criteria are specified.",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if eligibility requires renal function levels or mentions kidney-related disease or dysfunction.",
    "Positive cues: 'creatinine clearance ≥60 mL/min', 'adequate renal function', 'BUN ≤1.5× ULN', 'no renal impairment', 'history of kidney disease excluded'.",
    "Negative cues (leave empty): 'organ function' without renal or kidney reference."
    ),
    "Hepatic_Func": (
    "Meaning: Whether hepatic (liver) function criteria are specified.",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if eligibility includes liver enzyme thresholds or mentions hepatic/liver disease or dysfunction or Hepatitis.",
    "Positive cues: 'AST/ALT ≤2.5× ULN', 'bilirubin ≤1.5× ULN', 'adequate hepatic function', 'no hepatic impairment', 'patients with liver disease excluded','Active Hepatitis B or C'",
    "Negative cues (leave empty): generic 'organ function' without liver reference."
    ),
    "Hemo_Func": (
    "Meaning: Whether hematologic (blood) function criteria are specified.",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if eligibility includes blood cell parameters or mentions hematologic disorders.",
    "Positive cues: 'ANC ≥1500/μL', 'platelets ≥100,000/μL', 'hemoglobin ≥9 g/dL', 'no history of anemia', 'no hematologic malignancy'.",
    "Negative cues (leave empty): 'adequate function' without blood or hematology reference."
    ),
    "Cardio_Func": (
    "Meaning: Whether cardiac (heart) function criteria are specified.",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if eligibility includes cardiac function thresholds or mentions heart disease or cardiovascular conditions.",
    "Positive cues: 'LVEF ≥50%', 'no uncontrolled heart failure', 'adequate cardiac function', 'no history of myocardial infarction', 'patients with arrhythmia excluded'.",
    "Negative cues (leave empty): 'organ function' without heart or cardiovascular reference."
    ),
    "Comorb_Prohib": (
    "Meaning: Whether specific comorbidities disqualify participants.",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if text explicitly prohibits participants with certain comorbid diseases (e.g., cardiac, renal, hepatic, metabolic).",
    "Positive cues: 'patients with uncontrolled hypertension are excluded', 'no history of HIV infection', 'no significant cardiac disease'.",
    "Negative cues (leave empty): vague 'no serious comorbidities' statements."
    ),
    "Baseline_Severity_Score": (
    "Meaning: Specific baseline severity scores required for eligibility (e.g., HAM-D > 20, PASI > 12).",
    "Output: Copy the numeric threshold and the name of the scale (e.g., 'HAM-D > 20', 'PASI ≥ 12').",
    "Extract: Fill ONLY if explicit numeric thresholds or scale-based inclusion criteria are stated.",
    "Positive cues: 'HAM-D ≥20', 'PASI >12', 'CGI-S ≥4', 'baseline severity score above moderate'.",
    "Negative cues (leave empty): 'severe disease' without specific scale or number."
    ),
    "Gender_Criteria": (
    "Meaning: Sex eligibility criteria (Male / Female / Both).",
    "Output: 'Male', 'Female', or 'Both'.",
    "Extract: Fill if eligibility criteria explicitly restrict or allow both sexes.",
    "Positive cues: 'male participants only', 'females only', 'both sexes eligible'.",
    "Negative cues (leave empty): epidemiologic mentions (e.g., 'disease more common in women')."
    ),
    "Washout_Period": (
    "Meaning: Required time interval since last therapy before enrollment.",
    "Output: Numeric value plus time unit (e.g., '14 days', '4 weeks').",
    "Extract: Fill ONLY if text specifies a minimum washout period before screening or randomization.",
    "Positive cues: 'washout period of at least 14 days', 'no prior therapy within 30 days'.",
    "Negative cues (leave empty): 'previously treated' without time restriction."
    ),
    "Pregnancy_Lactation": (
    "Meaning: Exclusion criteria regarding pregnancy or breastfeeding.",
    "Output: 'yes' only; otherwise empty.",
    "Extract: Fill if text explicitly excludes pregnant or lactating women.",
    "Positive cues: 'pregnant or breastfeeding women are excluded', 'negative pregnancy test required', 'must use contraception'.",
    "Negative cues (leave empty): 'female participants' without pregnancy-related restriction."
)
}

PROMPT_CONFIG = PromptConfig(
    instructions=INSTRUCTIONS,
    notes=TARGETPOP_NOTES,
    llm_fields=LLM_FIELDS,
    text_modules=TEXT_MODULES,
)


def main() -> None:
    if "--table" not in sys.argv:
        sys.argv.extend(["--table", "TargetPop"])
    fill_main(prompt_config=PROMPT_CONFIG)


if __name__ == "__main__":
    main()
