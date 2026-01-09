# CTG Tables Extraction Guide (CSR-Vars 2026-01-08.xlsx)

This guide documents how each field is extracted from ClinicalTrials.gov XML.
It is aligned to `data/raw/CSR-Vars 2026-01-08.xlsx` and the unified builder
`tools/ctg_extract/build_ctg_tables.py`.

Legend:
- CTG direct: taken from CTG XML fields
- Derived: deterministic rule from CTG XML
- LLM: use LLM to extract feature from CTG text or SAP/Protocol
- External/TBD: not in CTG text (protocol/CSR/external); to be defined later
- Current build: whether `build_ctg_tables.py` fills the field now

## Design

| Field | Source | Current build |
| --- | --- | --- |
| StudyID | CTG direct: id_info/nct_id | Yes |
| Drug_Name | CTG direct: intervention/intervention_name filtered to normalized drug-like types (drug/biological/combination product, including biological/vaccine variants) | Yes |
| Prot_No | CTG direct: id_info/org_study_id | Yes |
| NCT_No | CTG direct: id_info/nct_id | Yes |
| EUCT_No | Derived: id_info/secondary_id normalized to EudraCT ID (YYYY-NNNNNN-NN) | Yes |
| Other_No | Derived: id_info/secondary_id non-EudraCT | Yes |
| Inv_Prod | Derived: drug-like interventions on experimental arms minus control arms (placebo/sham/vehicle/SOC/BSC excluded); fallback when exactly one non-background drug-like exists | Yes |
| Study_Phase | CTG direct: phase | Yes |
| Start_Date | CTG direct: start_date | Yes |
| Termin_Date | Derived: completion_date when overall_status == terminated (fallback to primary_completion_date if completion_date missing) | Yes |
| Complet_Date | CTG direct: completion_date | Yes |
| Data_Cutoff_Date | LLM (SAP/Protocol) | No |
| Name_PI | CTG direct: overall_official with PI role (role normalized); fallback to responsible_party/investigator_full_name | Yes |
| Sponsor | CTG direct: sponsors/lead_sponsor/agency + collaborator/agency | Yes |
| GCP_Compliance | LLM (CTG text) | No |
| Study_First_Posted_Date | CTG direct: study_first_posted | Yes |
| Route_Admin | LLM (CTG text) | No |
| Treat_Duration | LLM (CTG text) | No |
| Add_On_Treat | LLM (CTG text) | No |
| Arm_Description | CTG direct: arm_group/description (joined) | Yes |
| Adherence_Treat | LLM (CTG text) | No |
| MRCT | Derived: multi-region (>=2 of NA/AP/WEU/EEU/AF) from location countries | Yes |
| No_Center | Derived: unique facility name+city/state+country count | Yes |
| No_Center_NA | Derived: unique facilities mapped to NA region | Yes |
| No_Center_AP | Derived: unique facilities mapped to AP region | Yes |
| No_Center_WEU | Derived: unique facilities mapped to WEU region | Yes |
| No_Center_EEU | Derived: unique facilities mapped to EEU region | Yes |
| No_Center_AF | Derived: unique facilities mapped to AF region | Yes |
| Enroll_Duration_Plan | LLM (CTG text) | No |
| FU_Duration_Plan | LLM (CTG text) | No |
| Central_Lab | LLM (CTG text) | No |
| Run_in | LLM (CTG text) | No |
| DMC | CTG direct: oversight_info/has_dmc normalized to Yes/No when possible | Yes |

## Stat_Reg

| Field | Source | Current build |
| --- | --- | --- |
| StudyID | CTG direct: id_info/nct_id | Yes |
| Drug_Name | CTG direct: intervention/intervention_name filtered to drug/biological/combination product; one row per drug-like intervention | Yes |
| No_Arm | CTG direct: number_of_arms or count of arm_group | Yes |
| Randomization | Derived (interventional only): allocation normalized; non-random -> No, random -> Yes, otherwise blank | Yes |
| Central_Random | LLM (CTG text) | No |
| Rand_Ratio | LLM (CTG text) | No |
| Random_Parallel | Derived (interventional only, if intervention_model present): normalize and contains "parallel" -> Yes else No | Yes |
| Random_Crossover | Derived (interventional only, if intervention_model present): normalize (e.g., cross-over) and contains "crossover" -> Yes else No | Yes |
| Random_Fact | Derived (interventional only, if intervention_model present): contains "factorial" -> Yes else No | Yes |
| Random_Cluster | Not derived from structured fields; left blank in build | No |
| Stratification | Derived (interventional only, if present): no/none -> No, otherwise Yes; NA/unknown/not provided -> blank | Yes |
| No_Stratification | Derived: 0 when Stratification == No; otherwise blank | Yes |
| Single_Arm | Derived (interventional only): No_Arm == 1 | Yes |
| Hist_control | Derived (interventional only): any non-Other arm_group_type contains "histor" -> Yes else No; missing/Other -> blank | Yes |
| Blinding | Derived (interventional only): study_design_info/masking | Yes |
| Level_Blinding | Derived (interventional only): masking none/open=0, single=1, double=2, triple=3, quadruple=4 | Yes |
| Placebo_control | Derived (interventional only): any non-Other arm_group_type contains "placebo" -> Yes else No; missing/Other -> blank | Yes |
| Active_Control | Derived (interventional only): any non-Other arm_group_type contains "active" -> Yes else No; missing/Other -> blank | Yes |
| Control_Drug | Derived: control arm labels -> intervention_name (drug-like only; placebo/sham/vehicle excluded) | Yes |
| Obj_Primary | LLM (CTG text) | No |
| No_Prim_EP | Derived: count of primary_outcome | Yes |
| Primary_EP | CTG direct: primary_outcome/measure (joined) | Yes |
| Success_Criteria_Text | LLM (CTG text) | No |
| Primary_EP_SC | LLM (CTG text) | No |
| Key_Second_EP | CTG direct: secondary_outcome/measure (first only) | Yes |
| Key_Second_EP_SC | LLM (CTG text) | No |
| IRC | LLM (CTG text) | No |
| Sample_Size | CTG direct: enrollment (type=anticipated) | Yes |
| Power | LLM (CTG text) | No |
| Alpha | LLM (CTG text) | No |
| Sided | LLM (CTG text) | No |
| Subgroup | LLM (CTG text) | No |
| Interim | LLM (CTG text) | No |
| Timing_IA | LLM (CTG text) | No |
| Alpha_Spend_Func | LLM (CTG text) | No |
| Adaptive_Design | LLM (CTG text) | No |
| Consistency_Sens_Ana_PE | LLM (CTG text) | No |
| Consistency_Sens_Ana_SE | LLM (CTG text) | No |
| Post_Hoc_Ana | LLM (CTG text) | No |
| Intercurrent_Events | LLM (CTG text) | No |
| Gatekeeping_Strategy | LLM (CTG text) | No |
| Reg_Alignment | LLM (CTG text) | No |
| Fast_Track | LLM (CTG text) | No |
| Breakthrough | LLM (CTG text) | No |
| Priority_Review | LLM (CTG text) | No |
| Accelerated_App | LLM (CTG text) | No |
| Orphan_Drug | LLM (CTG text) | No |
| Pediatric | LLM (CTG text) | No |
| Rare_Disease | LLM (CTG text) | No |
| Reg_Audit | LLM (CTG text) | No |
| Consistency_MRCT | LLM (CTG text) | No |

## TargetPop

| Field | Source | Current build |
| --- | --- | --- |
| StudyID | CTG direct: id_info/nct_id | Yes |
| Drug_Name | CTG direct: intervention/intervention_name filtered to drug/biological/combination product | Yes |
| Disease | CTG direct: condition | Yes |
| Disease_Stage | LLM (CTG text) | No |
| Prevalence | External/TBD (non-CTG) | No |
| Age_Min | CTG direct: eligibility/minimum_age parsed to years (months/weeks/days converted) | Yes |
| Age_Max | CTG direct: eligibility/maximum_age parsed to years (months/weeks/days converted) | Yes |
| Prior_Line | LLM (CTG text) | No |
| Relapsed | LLM (CTG text) | No |
| Refractory | LLM (CTG text) | No |
| Geno_Biomarker | LLM (CTG text) | No |
| Comp_Biomarker | LLM (CTG text) | No |
| Other_Biomarker | LLM (CTG text) | No |
| ECOG | LLM (CTG text) | No |
| Karnofsky | LLM (CTG text) | No |
| Kidney_Func | LLM (CTG text) | No |
| Hepatic_Func | LLM (CTG text) | No |
| Hemo_Func | LLM (CTG text) | No |
| Cardio_Func | LLM (CTG text) | No |
| Comorb_Prohib | LLM (CTG text) | No |
| Concom_Prohib | LLM (CTG text) | No |
| Baseline_Severity_Score | LLM (CTG text) | No |
| Gender_Criteria | CTG direct: eligibility/gender (Male/Female/All -> Both) | Yes |
| Washout_Period | LLM (CTG text) | No |
| Pregnancy_Lactation | LLM (CTG text) | No |

## Drug

| Field | Source | Current build |
| --- | --- | --- |
| StudyID | CTG direct: id_info/nct_id | Yes |
| Drug_Name | CTG direct: intervention/intervention_name filtered to drug/biological/combination product | Yes |
| Molecule_Size | External/TBD (non-CTG) | No |
| SMILES | External/TBD (non-CTG) | No |
| ECFP6 | External/TBD (non-CTG) | No |
| SELFIES | External/TBD (non-CTG) | No |
| Is_Biosimilar | External/TBD (non-CTG) | No |
| F_bioav | External/TBD (non-CTG) | No |
| K_a | External/TBD (non-CTG) | No |
| C_Max | External/TBD (non-CTG) | No |
| T_Max | External/TBD (non-CTG) | No |
| AUC | External/TBD (non-CTG) | No |
| V_d | External/TBD (non-CTG) | No |
| V_ss | External/TBD (non-CTG) | No |
| Cl | External/TBD (non-CTG) | No |
| T_half | External/TBD (non-CTG) | No |
| K_el | External/TBD (non-CTG) | No |
| Cl_R | External/TBD (non-CTG) | No |
| Cl_H | External/TBD (non-CTG) | No |
| Target_Engage | External/TBD (non-CTG) | No |
| Biochem_Change | External/TBD (non-CTG) | No |
| Physio_Change | External/TBD (non-CTG) | No |
| Imaging_Biomarker | External/TBD (non-CTG) | No |
| Molecular_EP | External/TBD (non-CTG) | No |
| EC50 | External/TBD (non-CTG) | No |
| Emax | External/TBD (non-CTG) | No |
| Hill_Coeff | External/TBD (non-CTG) | No |
| AUC_IC50 | External/TBD (non-CTG) | No |
| Excipients | External/TBD (non-CTG) | No |
| ADA_Positive | External/TBD (non-CTG) | No |
| nAb_Positive | External/TBD (non-CTG) | No |

## Others

| Field | Source | Current build |
| --- | --- | --- |
| StudyID | CTG direct: id_info/nct_id | Yes |
| Drug_Name | CTG direct: intervention/intervention_name filtered to drug/biological/combination product | Yes |
| Sponsor_Type | Derived: NIH/U.S. Fed agency_class -> Government; academic keywords in sponsor name -> Academic; otherwise blank | Yes |
| COVID_Impact | External/TBD (non-CTG) | No |
| No_Competing_Trial | LLM (CTG text) | No |
| No_SOC | LLM (CTG text) | No |
| CRO | LLM (CTG text) | No |
| CRO_Oper | LLM (CTG text) | No |
| CRO_Stat | LLM (CTG text) | No |
| CRO_DM | LLM (CTG text) | No |
| RBM | LLM (CTG text) | No |

## Results: Endpoints

| Field | Source | Current build |
| --- | --- | --- |
| StudyID | CTG direct: id_info/nct_id | Yes |
| Drug_Name | Derived: outcome group title == arm_group_label -> intervention_name (drug-like only); otherwise blank | Yes |
| Arm_ID | Derived: StudyID + outcome group_id (fallback to normalized title; no groups -> overall) | Yes |
| Endpoint_Name | CTG direct: clinical_results/outcome_list/outcome/title | Yes |
| Endpoint_Type | Derived: Primary; Key Secondary = trial's first Secondary outcome; others blank | Yes |
| Strategy | LLM (CTG text) | No |
| EP_P_value | CTG direct: analysis_list serialized as JSON array (analysis_id/groups/method/p/effect/ci) | Yes |
| Missing_Imput | LLM (CTG text) | No |
| Covariate_Adjust | LLM (CTG text) | No |
| MCP | LLM (CTG text) | No |
| Subgroup_Ana | LLM (CTG text) | No |
| EP_Value | LLM (CTG text) | No |
| EP_Unit | LLM (CTG text) | No |
| EP_Point | CTG direct: same analysis JSON array as EP_P_value | Yes |
| ARR | LLM (CTG text) | No |
| NNT | LLM (CTG text) | No |
| EP_95CI | CTG direct: same analysis JSON array as EP_P_value | Yes |
| Med_OS | LLM (CTG text) | No |
| OS_YrX | LLM (CTG text) | No |
| Med_PFS | LLM (CTG text) | No |
| ORR | LLM (CTG text) | No |
| pCR | LLM (CTG text) | No |
| Med_DOR | LLM (CTG text) | No |
| RMST | LLM (CTG text) | No |
| PRO | LLM (CTG text) | No |
| QoL | LLM (CTG text) | No |

## Results: Groups

| Field | Source | Current build |
| --- | --- | --- |
| StudyID | CTG direct: id_info/nct_id | Yes |
| Drug_Name | Derived: outcome group title == arm_group_label -> intervention_name (drug-like only); otherwise blank | Yes |
| Arm_ID | Derived: results group_id (participant_flow/baseline/outcome groups) -> StudyID + group_id (fallback to normalized title; no groups -> overall) | Yes |
| Date_End | Derived: completion_date or primary_completion_date | Yes |
| Results_Posted_Or_Updated_Date | Derived: results_first_posted or last_update_posted (Date_Report if column name) | Yes |
| Study_Status | Derived: normalize overall_status to Complete/ongoing/unknown | Yes |
| Outcome | LLM (CTG text) | No |
| Termination | LLM (CTG text) | No |
| No_Amendment | LLM (CTG text) | No |
| No_Substan_Amend | LLM (CTG text) | No |
| No_Sub_Screen | LLM (CTG text) | No |
| No_Sub_Enroll | CTG direct: participant_flow Started/Enrolled/Randomized/Assigned count per group; fallback to enrollment(type=actual) when single group | Yes |
| Screen_Failure | LLM (CTG text) | No |
| Reasons_Screen_Fail | LLM (CTG text) | No |
| Enroll_Rate | LLM (CTG text) | No |
| Time_FPI | LLM (CTG text) | No |
| Prop_Center_Enroll_Target | LLM (CTG text) | No |
| Completer | LLM (CTG text) | No |
| Withdrawer | LLM (CTG text) | No |
| Discont_AE | LLM (CTG text) | No |
| Discont_LE | LLM (CTG text) | No |
| Loss_FU | LLM (CTG text) | No |
| Protocol_Viol | LLM (CTG text) | No |
| Treat_Adherence | LLM (CTG text) | No |
| Visit_Adherence | LLM (CTG text) | No |
| Assess_Adherence | LLM (CTG text) | No |
| Med_Treat | LLM (CTG text) | No |
| Med_Follow_Up | LLM (CTG text) | No |
| Discont | LLM (CTG text) | No |
| Med_Discont | LLM (CTG text) | No |
| Missing_All | LLM (CTG text) | No |
| Missing_PE | LLM (CTG text) | No |
| Missing_Key_SE | LLM (CTG text) | No |
| Med_Age | LLM (CTG text) | No |
| Min_Age | LLM (CTG text) | No |
| Max_Age | LLM (CTG text) | No |
| Prop_NA | LLM (CTG text) | No |
| Prop_AP | LLM (CTG text) | No |
| Prop_EU | LLM (CTG text) | No |
| Prop_AF | LLM (CTG text) | No |
| Prop_White | LLM (CTG text) | No |
| Prop_Asian | LLM (CTG text) | No |
| Prop_Black | LLM (CTG text) | No |
| Prop_Male | LLM (CTG text) | No |
| Heter_Index | LLM (CTG text) | No |
| Prop_Renal | LLM (CTG text) | No |
| Prop_Hepatic | LLM (CTG text) | No |
| Prop_HighRisk | LLM (CTG text) | No |
| Line0 | LLM (CTG text) | No |
| Line1 | LLM (CTG text) | No |
| Line2 | LLM (CTG text) | No |
| Line3 | LLM (CTG text) | No |
| Lapsed | LLM (CTG text) | No |
| Refractory | LLM (CTG text) | No |
| Prop_ECOG0 | LLM (CTG text) | No |
| Prop_ECOG1 | LLM (CTG text) | No |
| Prop_ECOG2 | LLM (CTG text) | No |
| Prop_ECOG3 | LLM (CTG text) | No |
| TEAEs | LLM (CTG text) | No |
| TRAEs | LLM (CTG text) | No |
| SAEs | LLM (CTG text) | No |
| Grade34_TEAEs | LLM (CTG text) | No |
| Grade34_TRAEs | LLM (CTG text) | No |
| AE_Spec | LLM (CTG text) | No |
| Inc_TEAES | LLM (CTG text) | No |
| Inc_Grade34_TEAEs | LLM (CTG text) | No |
| Grade34_Lab | LLM (CTG text) | No |
| Death_rate | LLM (CTG text) | No |
| No_Dosage | LLM (CTG text) | No |
| Dosage_Level | LLM (CTG text) | No |
| Dosing_Frequency | LLM (CTG text) | No |
| Cycle_Length | LLM (CTG text) | No |
| Formulation | LLM (CTG text) | No |

## LLM Extraction Notes

- LLM extraction is intentionally kept separate from the build step.
- The shared interface is `fill_ctg_table_with_llm.py` (requires `--table`).
- Per-table prompt text lives in each `fill_*.py` wrapper for quick edits.
- LLM runners only fill fields listed in each table's `LLM_FIELDS` inside its `fill_*.py`.
- Keep `LLM_FIELDS` aligned with the `LLM` rows in this guide; CTG direct/derived fields are never filled by LLM.
- Design tables use per-table runners:
  - `fill_design_with_llm.py`
  - `fill_stat_reg_with_llm.py`
  - `fill_targetpop_with_llm.py`
  - `fill_drug_with_llm.py`
  - `fill_others_with_llm.py`
- Results tables are split:
  - `fill_endpoints_with_llm.py`
  - `fill_groups_with_llm.py`
