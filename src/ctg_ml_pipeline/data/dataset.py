"""
Dataset class for clinical trial outcome prediction.

This module provides a PyTorch-style Dataset class for loading and preprocessing
clinical trial data for ML modeling.

Usage:
    from ctg_ml_pipeline.data.dataset import TrialDataset
    
    dataset = TrialDataset(
        group_dir="data/ctg_extract_v2/NSCLC_Trialpanorama_pd1",
        target_csv="data/raw/NSCLC_Trialpanorama_pd1_brief_summary.csv",
    )
    
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import polars as pl
import numpy as np


# =============================================================================
# Feature Type Definitions
# =============================================================================

class FeatureType(Enum):
    """Type of feature for preprocessing."""
    NUMERIC = "numeric"         # Continuous numeric values
    CATEGORICAL = "categorical" # Discrete categories (low cardinality)
    TEXT = "text"               # Free-form text (needs embedding)
    EXCLUDE = "exclude"         # Should not be used as feature


# Pre-defined feature types for each table
# Key: column name pattern (exact match or substring)
# Value: FeatureType

FEATURE_TYPE_DEFINITIONS = {
    # =========================
    # D_Design_all columns
    # =========================
    
    # IDs and text - EXCLUDE
    "StudyID": FeatureType.EXCLUDE,
    "Prot_No": FeatureType.EXCLUDE,
    "NCT_No": FeatureType.EXCLUDE,
    "EUCT_No": FeatureType.EXCLUDE,
    "Other_No": FeatureType.EXCLUDE,
    "Name_PI": FeatureType.EXCLUDE,
    "Sponsor": FeatureType.EXCLUDE,
    
    # Long text fields - TEXT (for future embedding)
    "Brief_Title": FeatureType.TEXT,
    "Official_Title": FeatureType.TEXT,
    "Brief_Summary": FeatureType.TEXT,
    "Eligibility_Criteria": FeatureType.TEXT,
    "Inv_Prod": FeatureType.TEXT,
    "Intervention_All": FeatureType.TEXT,
    "Control_Drug": FeatureType.TEXT,
    "Success_Criteria_Text": FeatureType.TEXT,
    "Name_Primary_EP": FeatureType.TEXT,
    "Assess_Primary_EP": FeatureType.TEXT,
    "Name_Key_Second_EP": FeatureType.TEXT,
    "Assess_Key_Second_EP": FeatureType.TEXT,
    "Name_Second_EP": FeatureType.TEXT,
    "Assess_Second_EP": FeatureType.TEXT,
    "Enroll_Duration_Plan": FeatureType.TEXT,
    "FU_Duration_Plan": FeatureType.TEXT,
    "Timing_IA": FeatureType.TEXT,
    "Alpha_Spend_Func": FeatureType.TEXT,
    "Adaptive_Design": FeatureType.TEXT,
    "Intercurrent_Events": FeatureType.TEXT,
    "Gatekeeping_Strategy": FeatureType.TEXT,
    "Reg_Alignment": FeatureType.TEXT,
    
    # Categorical (Yes/No or small set of values)
    "Study_Phase": FeatureType.CATEGORICAL,
    "Study_Type": FeatureType.CATEGORICAL,
    "GCP_Compliance": FeatureType.CATEGORICAL,
    "Central_Lab": FeatureType.CATEGORICAL,
    "Run_in": FeatureType.CATEGORICAL,
    "DMC": FeatureType.CATEGORICAL,
    "Fast_Track": FeatureType.CATEGORICAL,
    "Breakthrough": FeatureType.CATEGORICAL,
    "Priority_Review": FeatureType.CATEGORICAL,
    "Accelerated_App": FeatureType.CATEGORICAL,
    "Orphan_Drug": FeatureType.CATEGORICAL,
    "Pediatric": FeatureType.CATEGORICAL,
    "Rare_Disease": FeatureType.CATEGORICAL,
    "MRCT": FeatureType.CATEGORICAL,
    "Randomization": FeatureType.CATEGORICAL,
    "Central_Random": FeatureType.CATEGORICAL,
    "Rand_Ratio": FeatureType.CATEGORICAL,
    "Random_Parallel": FeatureType.CATEGORICAL,
    "Random_Crossover": FeatureType.CATEGORICAL,
    "Random_Fact": FeatureType.CATEGORICAL,
    "Random_Cluster": FeatureType.CATEGORICAL,
    "Stratification": FeatureType.CATEGORICAL,
    "Single_Arm": FeatureType.CATEGORICAL,
    "Hist_control": FeatureType.CATEGORICAL,
    "Blinding": FeatureType.CATEGORICAL,
    "Placebo_control": FeatureType.CATEGORICAL,
    "Active_Control": FeatureType.CATEGORICAL,
    "Primary_Obj": FeatureType.CATEGORICAL,
    "IRC": FeatureType.CATEGORICAL,
    "Power": FeatureType.CATEGORICAL,
    "Alpha": FeatureType.CATEGORICAL,
    "Subgroup": FeatureType.CATEGORICAL,
    "Interim": FeatureType.CATEGORICAL,
    
    # Numeric
    "No_Center": FeatureType.NUMERIC,
    "No_Center_NA": FeatureType.NUMERIC,
    "No_Center_AP": FeatureType.NUMERIC,
    "No_Center_WEU": FeatureType.NUMERIC,
    "No_Center_EEU": FeatureType.NUMERIC,
    "No_Center_AF": FeatureType.NUMERIC,
    "No_Arm": FeatureType.NUMERIC,
    "No_Stratification": FeatureType.NUMERIC,
    "Level_Blinding": FeatureType.NUMERIC,
    "Sample_Size": FeatureType.NUMERIC,
    "Sample_Size_Actual": FeatureType.NUMERIC,
    "Sided": FeatureType.NUMERIC,
    "No_Primary_EP": FeatureType.NUMERIC,
    "No_Key_Second_EP": FeatureType.NUMERIC,
    "No_Second_EP": FeatureType.NUMERIC,
    
    # =========================
    # D_Pop_all columns
    # =========================
    
    # Text fields
    "Disease": FeatureType.TEXT,
    "Disease_Stage": FeatureType.TEXT,
    "Prevalence": FeatureType.TEXT,
    "Anat_Physio_Measures": FeatureType.TEXT,
    "Clinical_Symptom_Score": FeatureType.TEXT,
    "ECOG": FeatureType.TEXT,
    "Karnofsky": FeatureType.TEXT,
    "Cognitive_Scale": FeatureType.TEXT,
    "Vital_Signs": FeatureType.TEXT,
    "Imaging": FeatureType.TEXT,
    "Baseline_Severity_Score": FeatureType.TEXT,
    "Washout_Period": FeatureType.TEXT,
    "GIST": FeatureType.TEXT,
    "Diag_Dis_Detect": FeatureType.TEXT,
    "Diag_Dis_Subtyp": FeatureType.TEXT,
    "Diag_Molecular": FeatureType.TEXT,
    "Diag_Histologic": FeatureType.TEXT,
    "Diag_Radiographic": FeatureType.TEXT,
    "Diag_Physiologic": FeatureType.TEXT,
    "Prog_High_Risk": FeatureType.TEXT,
    "Prog_Strat": FeatureType.TEXT,
    "Prog_Molecular": FeatureType.TEXT,
    "Prog_Inflamm": FeatureType.TEXT,
    "Prog_Histologic": FeatureType.TEXT,
    "Prog_Circulat": FeatureType.TEXT,
    "Pred_Enrich": FeatureType.TEXT,
    "Pred_Oncogenic": FeatureType.TEXT,
    "Pred_Response": FeatureType.TEXT,
    "Pred_Others": FeatureType.TEXT,
    "Safety_Hema": FeatureType.TEXT,
    "Safety_Hepa": FeatureType.TEXT,
    "Safety_Renal": FeatureType.TEXT,
    "Safety_Cardiac": FeatureType.TEXT,
    "Safety_Immune": FeatureType.TEXT,
    "Digit_Actig": FeatureType.TEXT,
    "Digit_Physio": FeatureType.TEXT,
    "Digit_Vocal": FeatureType.TEXT,
    "Digit_Cognit": FeatureType.TEXT,
    
    # Categorical
    "Gender": FeatureType.CATEGORICAL,
    "Relapsed": FeatureType.CATEGORICAL,
    "Refractory": FeatureType.CATEGORICAL,
    "Kidney_Func": FeatureType.CATEGORICAL,
    "Hepatic_Func": FeatureType.CATEGORICAL,
    "Hemo_Func": FeatureType.CATEGORICAL,
    "Cardio_Func": FeatureType.CATEGORICAL,
    "Comorb_Prohib": FeatureType.CATEGORICAL,
    "Concom_Prohib": FeatureType.CATEGORICAL,
    "Pregnancy_Lactation": FeatureType.CATEGORICAL,
    
    # Numeric
    "Age_Min": FeatureType.NUMERIC,
    "Age_Max": FeatureType.NUMERIC,
    "Prior_Line": FeatureType.NUMERIC,
    
    # =========================
    # D_Drug_all_PKPD columns
    # =========================
    
    # IDs - EXCLUDE
    "DrugBank_ID": FeatureType.EXCLUDE,
    "Generic_Name": FeatureType.EXCLUDE,
    "Brand_Name": FeatureType.EXCLUDE,
    "Generic_Name_DB": FeatureType.EXCLUDE,
    "Brand_Name_DB": FeatureType.EXCLUDE,
    "SMILES": FeatureType.EXCLUDE,
    "SMILES_DB": FeatureType.EXCLUDE,
    "ECFP6": FeatureType.EXCLUDE,
    "SELFIES": FeatureType.EXCLUDE,
    
    # Text fields (descriptions from DrugBank)
    "Molecule_Size": FeatureType.TEXT,
    "V_d_DB": FeatureType.TEXT,
    "Cl_DB": FeatureType.TEXT,
    "T_half_DB": FeatureType.TEXT,
    "Tox_General_DB": FeatureType.TEXT,
    "Target_Engage": FeatureType.TEXT,
    "Target_Engage_DB": FeatureType.TEXT,
    "Biochem_Change": FeatureType.TEXT,
    "Biochem_Change_DB": FeatureType.TEXT,
    "Physio_Change": FeatureType.TEXT,
    "Physio_Change_DB": FeatureType.TEXT,
    "Imaging_Biomarker": FeatureType.TEXT,
    "Molecular_EP": FeatureType.TEXT,
    "K_el": FeatureType.TEXT,
    "Cl_R": FeatureType.TEXT,
    "EC50": FeatureType.TEXT,
    "Emax": FeatureType.TEXT,
    "Hill_Coeff": FeatureType.TEXT,
    "AUC_IC50": FeatureType.TEXT,
    "Excipients": FeatureType.TEXT,
    "dose_unit": FeatureType.CATEGORICAL,
    
    # Categorical
    "Is_Biosimilar": FeatureType.CATEGORICAL,
    
    # Numeric (PK/PD parameters)
    "Molecule_Size_DB": FeatureType.NUMERIC,
    "F_bioav_DB": FeatureType.NUMERIC,
    "dose_value": FeatureType.NUMERIC,
    "AUC": FeatureType.NUMERIC,
    "C_Max": FeatureType.NUMERIC,
    "T_Max": FeatureType.NUMERIC,
    "Cl": FeatureType.NUMERIC,
    "V_d": FeatureType.NUMERIC,
    "F_bioav": FeatureType.NUMERIC,
    "K_a": FeatureType.NUMERIC,
    "V_ss": FeatureType.NUMERIC,
    "T_half": FeatureType.NUMERIC,
    "Cl_H": FeatureType.NUMERIC,
    "LD_50": FeatureType.NUMERIC,
    "Tox_General": FeatureType.NUMERIC,
    "Tox_Hepa": FeatureType.NUMERIC,
    "Tox_Cardiac": FeatureType.NUMERIC,
    "Tox_Nephr": FeatureType.NUMERIC,  # Note: may contain "Invalid Molecule"
    "Tox_Neur": FeatureType.NUMERIC,   # Note: may contain "Invalid Molecule"
}


def get_feature_type(col_name: str) -> FeatureType:
    """
    Get the feature type for a column.
    
    Args:
        col_name: Column name
        
    Returns:
        FeatureType enum value
    """
    # Exact match first
    if col_name in FEATURE_TYPE_DEFINITIONS:
        return FEATURE_TYPE_DEFINITIONS[col_name]
    
    # Default: check if it looks like text (contains certain patterns)
    text_patterns = ["_Text", "Description", "Summary", "Criteria", "Name_", "Assess_"]
    for pattern in text_patterns:
        if pattern in col_name:
            return FeatureType.TEXT
    
    # Default to EXCLUDE for unknown columns
    return FeatureType.EXCLUDE


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TargetConfig:
    """Configuration for target variable preprocessing."""
    
    success_types: list[str] = field(default_factory=lambda: [
        "completed, positive outcome/primary endpoint(s) met"
    ])
    
    fail_types: list[str] = field(default_factory=lambda: [
        "completed, negative outcome/primary endpoint(s) not met",
        "terminated, enrollment issues",
        "terminated, feasibility",
        "terminated, funding issues",
        "terminated, lack of efficacy",
        "terminated, safety issues",
    ])


@dataclass
class FeatureConfig:
    """Configuration for feature selection."""
    
    max_missing_rate: float = 0.5
    keep_features: set[str] = field(default_factory=set)
    text_as_bool: bool = False
    
    feature_tables: list[str] = field(default_factory=lambda: [
        "D_Design_all", "D_Pop_all", "D_Drug_all_PKPD"
    ])
    
    # Which feature types to include
    include_numeric: bool = True
    include_categorical: bool = True
    include_text: bool = False  # Disabled by default (needs embedding)
    
    # Categorical encoding method
    categorical_encoding: Literal["label", "onehot"] = "label"
    
    # Maximum unique values for categorical (above this -> treat as text)
    max_categorical_cardinality: int = 20


@dataclass
class SplitConfig:
    """Configuration for train/test split."""
    
    test_size: float = 0.2
    time_split: bool = True
    random_state: int = 42


@dataclass
class DatasetConfig:
    """Complete dataset configuration."""
    
    target: TargetConfig = field(default_factory=TargetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    
    impute_strategy: Literal["median", "mean", "zero", "none"] = "median"
    scale_features: bool = True


# =============================================================================
# Dataset Class
# =============================================================================

class TrialDataset:
    """
    PyTorch-style Dataset class for clinical trial data.
    
    Handles:
    - Target variable preprocessing (outcome_type -> binary label)
    - Feature loading from multiple tables
    - Feature type classification (numeric/categorical/text)
    - Multi-row aggregation for drug tables
    - Missing value imputation
    - Feature encoding and scaling
    - Time-based or random train/test split
    
    Attributes:
        X: Feature matrix (numpy array)
        y: Target labels (numpy array, 0=fail, 1=success)
        feature_names: List of feature names
        feature_types: Dict mapping feature name to FeatureType
        study_ids: List of StudyID values
        start_dates: List of Start_Date values
        text_features: Dict of text features (for embedding later)
    """
    
    def __init__(
        self,
        group_dir: str | Path,
        target_csv: str | Path,
        config: DatasetConfig | None = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            group_dir: Directory containing trial data
            target_csv: Path to target CSV file with outcome_type column
            config: Dataset configuration
        """
        self.group_dir = Path(group_dir)
        self.target_csv = Path(target_csv)
        self.config = config or DatasetConfig()
        
        self.tables_dir = self.group_dir / "_ml_pipeline" / "merged_tables_notebooklm"
        
        # Data attributes
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.feature_types: dict[str, FeatureType] = {}
        self.feature_stats: dict[str, dict] = {}
        self.study_ids: list[str] = []
        self.start_dates: list[str] = []
        
        # Text features stored separately (for future embedding)
        self.text_features: dict[str, list[str]] = {}
        
        # Preprocessing objects
        self._imputer = None
        self._scaler = None
        self._label_encoders: dict[str, object] = {}
        self._onehot_encoders: dict[str, object] = {}
        
        # Load and preprocess data
        self._load_data()

    def _load_feature_allowlist(
        self,
    ) -> tuple[set[tuple[str, str]] | None, dict[tuple[str, str], str], dict[tuple[str, str], str]]:
        """Load feature allowlist + timepoint/type mapping if available."""
        allowlist_csv = self.group_dir / "_ml_pipeline" / "feature_allowlist.csv"
        if not allowlist_csv.exists():
            allowlist_csv = self.group_dir / "_ml_pipeline" / "feature_allowlist_T0.csv"
        if not allowlist_csv.exists():
            return None, {}, {}

        df = pl.read_csv(allowlist_csv)
        if "table" not in df.columns or "Variable" not in df.columns:
            return None, {}, {}

        allowed: set[tuple[str, str]] = set()
        timepoints: dict[tuple[str, str], str] = {}
        type_map: dict[tuple[str, str], str] = {}

        for row in df.iter_rows(named=True):
            table = str(row.get("table") or "").strip()
            var = str(row.get("Variable") or "").strip()
            if not table or not var:
                continue
            allowed.add((table, var))
            tp = row.get("Availability_Timepoint")
            timepoints[(table, var)] = "" if tp is None else str(tp).strip()
            ft = row.get("FeatureType")
            type_map[(table, var)] = "" if ft is None else str(ft).strip().lower()

        if not allowed:
            return None, {}, {}

        return allowed, timepoints, type_map

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.y) if self.y is not None else 0
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Get a single sample by index."""
        return self.X[idx], self.y[idx]
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def _load_data(self) -> None:
        """Load and preprocess all data."""
        # Step 1: Load and preprocess target
        target_df = self._load_target()
        
        # Step 2: Load and classify features
        df, numeric_cols, categorical_cols, text_cols = self._load_features(target_df)
        
        # Step 3: Get Start_Date for time split
        self._load_start_dates(df)
        
        # Step 4: Store StudyIDs
        self.study_ids = df.get_column("StudyID").to_list()
        
        # Step 5: Store text features separately
        for col in text_cols:
            if col in df.columns:
                self.text_features[col] = df.get_column(col).fill_null("").to_list()
        
        # Step 6: Encode numeric and categorical features
        X, y, feature_names = self._encode_data(df, numeric_cols, categorical_cols)
        
        # Step 7: Impute missing values
        X = self._impute_missing(X)
        
        # Step 8: Scale features
        if self.config.scale_features:
            X = self._scale_features(X)
        
        self.X = X
        self.y = y
        self.feature_names = feature_names
    
    def _load_target(self) -> pl.DataFrame:
        """Load and preprocess target variable."""
        df = pl.read_csv(self.target_csv)
        
        if "nctid" in df.columns:
            df = df.rename({"nctid": "StudyID"})
        
        cfg = self.config.target
        
        df_success = (
            df.filter(pl.col("outcome_type").is_in(cfg.success_types))
            .with_columns(pl.lit(1).alias("label"))
        )
        
        df_fail = (
            df.filter(pl.col("outcome_type").is_in(cfg.fail_types))
            .with_columns(pl.lit(-1).alias("label"))
        )
        
        df_labeled = pl.concat([df_success, df_fail])
        
        return df_labeled.select(["StudyID", "outcome_type", "label"])
    
    def _load_features(
        self, target_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, list[str], list[str], list[str]]:
        """
        Load and classify features from multiple tables.
        
        Returns:
            df: Merged DataFrame
            numeric_cols: List of numeric feature columns
            categorical_cols: List of categorical feature columns
            text_cols: List of text feature columns
        """
        cfg = self.config.features
        
        df = target_df.select(["StudyID", "label"])
        numeric_cols = []
        categorical_cols = []
        text_cols = []

        allowlist, timepoint_map, type_map = self._load_feature_allowlist()
        if allowlist:
            allowed_tables = []
            for table, _ in allowlist:
                table_name = table if table.endswith("_all") else f"{table}_all"
                allowed_tables.append(table_name)
            cfg.feature_tables = list(dict.fromkeys(allowed_tables))
        
        for table_name in cfg.feature_tables:
            csv_path = self.tables_dir / f"{table_name}.csv"
            if not csv_path.exists():
                continue
            
            table_df = pl.read_csv(csv_path)
            table_key = table_name.replace("_all", "")

            # Identify columns with multiple values per StudyID (one-to-many)
            multi_value_cols: set[str] = set()
            if table_df.height > 0 and table_df.get_column("StudyID").n_unique() < table_df.height:
                candidate_cols = [c for c in table_df.columns if c != "StudyID"]
                if candidate_cols:
                    agg = table_df.group_by("StudyID").agg(
                        [pl.col(c).n_unique().alias(c) for c in candidate_cols]
                    )
                    max_unique = agg.select(
                        [pl.col(c).max().alias(c) for c in candidate_cols]
                    ).row(0)
                    for col_name, max_val in zip(candidate_cols, max_unique):
                        if max_val is not None and max_val > 1:
                            multi_value_cols.add(col_name)
            
            # Classify and select features
            selected_cols = ["StudyID"]
            
            for col in table_df.columns:
                if col == "StudyID":
                    continue
                if col in multi_value_cols:
                    continue
                
                # Force-include overrides for vetted features
                if allowlist is not None and (table_key, col) not in allowlist:
                    continue
                force_keep = (allowlist is not None and (table_key, col) in allowlist) or (col in cfg.keep_features)
                if allowlist is not None:
                    ft_raw = type_map.get((table_key, col), "")
                    if ft_raw in ("continuous", "numeric"):
                        feat_type = FeatureType.NUMERIC
                    elif ft_raw in ("categorical", "category"):
                        feat_type = FeatureType.CATEGORICAL
                    elif ft_raw in ("string", "text"):
                        feat_type = FeatureType.TEXT
                    else:
                        feat_type = get_feature_type(col)
                elif force_keep:
                    dtype = table_df.get_column(col).dtype
                    if dtype.is_numeric():
                        feat_type = FeatureType.NUMERIC
                    else:
                        feat_type = FeatureType.CATEGORICAL
                else:
                    # Get predefined feature type
                    feat_type = get_feature_type(col)
                
                # Skip excluded columns
                if feat_type == FeatureType.EXCLUDE and not force_keep:
                    continue
                
                # Check missing rate
                col_series = table_df.get_column(col)
                if col_series.dtype == pl.Utf8:
                    missing_rate = table_df.select(
                        (pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")).mean()
                    ).item()
                else:
                    missing_rate = col_series.null_count() / len(table_df)
                if missing_rate > cfg.max_missing_rate:
                    continue
                
                # Handle based on feature type
                dtype = table_df.get_column(col).dtype
                
                if feat_type == FeatureType.NUMERIC:
                    # Verify it's actually numeric or can be converted
                    if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                        if cfg.include_numeric:
                            selected_cols.append(col)
                            numeric_cols.append(col)
                            self.feature_types[col] = FeatureType.NUMERIC
                    elif dtype == pl.Utf8:
                        # Try to detect if it's a string-encoded numeric
                        # (e.g., "Invalid Molecule" mixed with numbers)
                        # Skip for now, treat as text
                        if cfg.include_text:
                            selected_cols.append(col)
                            text_cols.append(col)
                            self.feature_types[col] = FeatureType.TEXT
                        continue
                    
                elif feat_type == FeatureType.CATEGORICAL:
                    if dtype == pl.Utf8:
                        if force_keep:
                            if cfg.include_categorical:
                                selected_cols.append(col)
                                categorical_cols.append(col)
                                self.feature_types[col] = FeatureType.CATEGORICAL
                        else:
                            # Check cardinality
                            n_unique = table_df.get_column(col).n_unique()
                            if n_unique <= cfg.max_categorical_cardinality:
                                if cfg.include_categorical:
                                    selected_cols.append(col)
                                    categorical_cols.append(col)
                                    self.feature_types[col] = FeatureType.CATEGORICAL
                            else:
                                # Too many categories -> treat as text
                                if cfg.include_text:
                                    selected_cols.append(col)
                                    text_cols.append(col)
                                    self.feature_types[col] = FeatureType.TEXT
                    elif dtype in [pl.Int64, pl.Int32]:
                        # Integer categorical (e.g., Level_Blinding)
                        if cfg.include_categorical:
                            selected_cols.append(col)
                            categorical_cols.append(col)
                            self.feature_types[col] = FeatureType.CATEGORICAL
                    
                elif feat_type == FeatureType.TEXT:
                    if cfg.text_as_bool:
                        if cfg.include_numeric:
                            present_expr = pl.col(col).is_not_null()
                            if col_series.dtype == pl.Utf8:
                                present_expr = present_expr & (pl.col(col).str.strip_chars() != "")
                            table_df = table_df.with_columns(present_expr.cast(pl.Int8).alias(col))
                            selected_cols.append(col)
                            numeric_cols.append(col)
                            self.feature_types[col] = FeatureType.NUMERIC
                            self.feature_stats[col] = {
                                "table": table_name,
                                "missing_rate": missing_rate,
                                "dtype": "int8",
                                "type": "text_boolean",
                                "display_type": "Text(boolean)",
                                "timepoint": timepoint_map.get((table_key, col), ""),
                            }
                        continue
                    if cfg.include_text:
                        selected_cols.append(col)
                        text_cols.append(col)
                        self.feature_types[col] = FeatureType.TEXT
                
                # Store stats
                if col in selected_cols:
                    self.feature_stats[col] = {
                        "table": table_name,
                        "missing_rate": missing_rate,
                        "dtype": str(dtype),
                        "type": feat_type.value,
                        "timepoint": timepoint_map.get((table_key, col), ""),
                    }
            
            # Join table to main DataFrame
            if len(selected_cols) > 1:
                table_subset = table_df.select(selected_cols)
                
                # Aggregate drug tables
                if "Drug" in table_name:
                    table_subset = self._aggregate_drug_table(table_subset)
                
                suffix = f"_{table_name.split('_')[1]}" if "_" in table_name else ""
                df = df.join(table_subset, on="StudyID", how="left", suffix=suffix)
        
        return df, numeric_cols, categorical_cols, text_cols
    
    def _aggregate_drug_table(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate drug table to StudyID level."""
        agg_exprs = []
        for col in df.columns:
            if col == "StudyID":
                continue
            dtype = df.get_column(col).dtype
            if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                agg_exprs.append(pl.col(col).mean().alias(col))
            else:
                agg_exprs.append(pl.col(col).drop_nulls().first().alias(col))
        
        if agg_exprs:
            return df.group_by("StudyID").agg(agg_exprs)
        return df.unique(subset=["StudyID"])
    
    def _load_start_dates(self, df: pl.DataFrame) -> None:
        """Load Start_Date for time-based split."""
        r_study_path = self.tables_dir / "R_Study_all.csv"
        if r_study_path.exists():
            r_study_df = pl.read_csv(r_study_path)
            if "Start_Date" in r_study_df.columns:
                date_df = r_study_df.select(["StudyID", "Start_Date"])
                df_with_date = df.join(date_df, on="StudyID", how="left")
                self.start_dates = df_with_date.get_column("Start_Date").to_list()
    
    # =========================================================================
    # Feature Encoding
    # =========================================================================
    
    def _encode_data(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Encode numeric and categorical features.
        
        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names after encoding
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        X_parts = []
        feature_names = []
        cfg = self.config.features
        
        # Process numeric columns
        for col in numeric_cols:
            if col not in df.columns:
                continue
            arr = df.get_column(col).to_numpy().astype(float)
            X_parts.append(arr.reshape(-1, 1))
            feature_names.append(col)
        
        # Process categorical columns
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            col_data = df.get_column(col)
            dtype = col_data.dtype
            
            if dtype == pl.Utf8:
                arr = col_data.fill_null("_missing_").to_numpy()
            else:
                arr = col_data.fill_null(-999).to_numpy().astype(str)
            
            if cfg.categorical_encoding == "onehot":
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = ohe.fit_transform(arr.reshape(-1, 1))
                X_parts.append(encoded)
                self._onehot_encoders[col] = ohe
                for cat in ohe.categories_[0]:
                    feature_names.append(f"{col}_{cat}")
            else:
                le = LabelEncoder()
                encoded = le.fit_transform(arr)
                X_parts.append(encoded.reshape(-1, 1))
                self._label_encoders[col] = le
                feature_names.append(col)
        
        if X_parts:
            X = np.hstack(X_parts)
        else:
            X = np.empty((len(df), 0))
        
        # Convert label from {-1, 1} to {0, 1}
        y = df.get_column("label").to_numpy()
        y = ((y + 1) / 2).astype(int)
        
        return X, y, feature_names
    
    def _impute_missing(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values."""
        if self.config.impute_strategy == "none":
            # Keep NaN values for models that handle them natively
            return X
        
        from sklearn.impute import SimpleImputer
        
        self._imputer = SimpleImputer(strategy=self.config.impute_strategy)
        return self._imputer.fit_transform(X)
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features."""
        from sklearn.preprocessing import StandardScaler
        
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(X)
    
    # =========================================================================
    # Train/Test Split
    # =========================================================================
    
    def get_train_test_split(
        self,
        return_indices: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/test split.
        
        Args:
            return_indices: If True, also return train and test indices
            
        Returns:
            If return_indices=False: X_train, X_test, y_train, y_test
            If return_indices=True: X_train, X_test, y_train, y_test, train_indices, test_indices
        """
        cfg = self.config.split
        
        if cfg.time_split and self.start_dates:
            return self._time_split(return_indices)
        else:
            return self._random_split(return_indices)
    
    def _parse_date(self, date_str: str | None) -> int:
        """Parse date string to ordinal."""
        if date_str is None:
            return 0
        try:
            dt = datetime.strptime(date_str, "%B %d, %Y")
            return dt.toordinal()
        except:
            try:
                dt = datetime.strptime(date_str, "%B %Y")
                return dt.toordinal()
            except:
                return 0
    
    def _time_split(
        self,
        return_indices: bool = False,
    ) -> tuple:
        """Split by time."""
        cfg = self.config.split
        
        date_ordinals = np.array([self._parse_date(d) for d in self.start_dates])
        sorted_indices = np.argsort(date_ordinals)
        
        n_test = int(len(sorted_indices) * cfg.test_size)
        n_train = len(sorted_indices) - n_test
        
        train_indices = sorted_indices[:n_train]
        test_indices = sorted_indices[n_train:]
        
        result = (
            self.X[train_indices], self.X[test_indices],
            self.y[train_indices], self.y[test_indices]
        )
        
        if return_indices:
            return result + (train_indices, test_indices)
        return result
    
    def _random_split(
        self,
        return_indices: bool = False,
    ) -> tuple:
        """Random stratified split."""
        from sklearn.model_selection import train_test_split
        
        cfg = self.config.split
        
        indices = np.arange(len(self.y))
        
        train_idx, test_idx = train_test_split(
            indices,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=self.y,
        )
        
        result = (
            self.X[train_idx], self.X[test_idx],
            self.y[train_idx], self.y[test_idx]
        )
        
        if return_indices:
            return result + (train_idx, test_idx)
        return result
    
    # =========================================================================
    # Text Feature Methods (for future embedding)
    # =========================================================================
    
    def get_text_features(self) -> dict[str, list[str]]:
        """
        Get text features for embedding.
        
        Returns:
            Dict mapping feature name to list of text values
        """
        return self.text_features

    def select_features(self, selected: list[str]) -> None:
        """Restrict dataset to selected feature names."""
        if not selected:
            return
        selected_set = set(selected)
        indices = [i for i, name in enumerate(self.feature_names) if name in selected_set]
        if not indices:
            return
        self.X = self.X[:, indices]
        self.feature_names = [self.feature_names[i] for i in indices]
        self.feature_types = {k: v for k, v in self.feature_types.items() if k in selected_set}
        self.feature_stats = {k: v for k, v in self.feature_stats.items() if k in selected_set}
    
    def set_text_embeddings(
        self,
        embeddings: dict[str, np.ndarray],
        prefix: str = "emb_"
    ) -> None:
        """
        Add text embeddings to feature matrix.
        
        Args:
            embeddings: Dict mapping feature name to embedding matrix (n_samples x dim)
            prefix: Prefix for embedding feature names
        """
        for feat_name, emb_matrix in embeddings.items():
            if emb_matrix.shape[0] != len(self.y):
                raise ValueError(
                    f"Embedding for {feat_name} has wrong shape: "
                    f"{emb_matrix.shape[0]} vs {len(self.y)}"
                )
            
            self.X = np.hstack([self.X, emb_matrix])
            
            for i in range(emb_matrix.shape[1]):
                new_name = f"{prefix}{feat_name}_{i}"
                self.feature_names.append(new_name)
                self.feature_types[new_name] = FeatureType.NUMERIC
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_feature_info(self) -> pl.DataFrame:
        """Get feature information as DataFrame."""
        data = [
            {
                "feature": name,
                "table": self.feature_stats.get(name, {}).get("table", "unknown"),
                "type": self.feature_stats.get(name, {}).get(
                    "display_type",
                    self.feature_types.get(name, FeatureType.EXCLUDE).value,
                ),
                "missing_rate": self.feature_stats.get(name, {}).get("missing_rate", 0),
                "dtype": self.feature_stats.get(name, {}).get("dtype", "unknown"),
            }
            for name in self.feature_names
        ]
        return pl.DataFrame(data)
    
    def get_split_info(self) -> dict:
        """Get split information."""
        X_train, X_test, y_train, y_test = self.get_train_test_split()
        
        info = {
            "total_samples": len(self),
            "n_features": len(self.feature_names),
            "n_numeric": sum(1 for t in self.feature_types.values() if t == FeatureType.NUMERIC),
            "n_categorical": sum(1 for t in self.feature_types.values() if t == FeatureType.CATEGORICAL),
            "n_text": len(self.text_features),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "train_pos_rate": y_train.mean(),
            "test_pos_rate": y_test.mean(),
            "split_method": "time" if self.config.split.time_split else "random",
        }
        
        if self.config.split.time_split and self.start_dates:
            sorted_indices = np.argsort([self._parse_date(d) for d in self.start_dates])
            n_train = len(sorted_indices) - int(len(sorted_indices) * self.config.split.test_size)
            
            train_dates = [self.start_dates[int(i)] for i in sorted_indices[:n_train]]
            test_dates = [self.start_dates[int(i)] for i in sorted_indices[n_train:]]
            
            info["train_date_range"] = f"{train_dates[0]} ~ {train_dates[-1]}"
            info["test_date_range"] = f"{test_dates[0]} ~ {test_dates[-1]}"
        
        return info
    
    def summary(self) -> None:
        """Print dataset summary."""
        print("=" * 70)
        print("TrialDataset Summary")
        print("=" * 70)
        
        split_info = self.get_split_info()
        
        print(f"\nSamples: {split_info['total_samples']}")
        print(f"Train: {split_info['n_train']} (pos rate: {split_info['train_pos_rate']:.1%})")
        print(f"Test: {split_info['n_test']} (pos rate: {split_info['test_pos_rate']:.1%})")
        print(f"Split method: {split_info['split_method']}")
        
        if "train_date_range" in split_info:
            print(f"\nTrain date range: {split_info['train_date_range']}")
            print(f"Test date range: {split_info['test_date_range']}")
        
        print("\n" + "-" * 70)
        print("Feature Types Summary")
        print("-" * 70)
        print(f"  Numeric features:     {split_info['n_numeric']}")
        print(f"  Categorical features: {split_info['n_categorical']}")
        print(f"  Text features:        {split_info['n_text']} (not in X, use get_text_features())")
        print(f"  Total in X:           {split_info['n_features']}")
        
        print("\n" + "-" * 70)
        print(f"{'Feature':<35} {'Type':<12} {'Table':<18} {'Missing':>8}")
        print("-" * 70)
        
        feature_info = self.get_feature_info()
        for row in feature_info.sort("type").iter_rows(named=True):
            feat_display = row['feature'][:33] + ".." if len(row['feature']) > 35 else row['feature']
            table_display = row['table'].replace("_all", "")[:16]
            print(f"{feat_display:<35} {row['type']:<12} {table_display:<18} {row['missing_rate']:>7.1%}")
        
        if self.text_features:
            print("\n" + "-" * 70)
            print("Text Features (for embedding)")
            print("-" * 70)
            for name in self.text_features.keys():
                print(f"  {name}")


# =============================================================================
# Convenience Function
# =============================================================================

def load_trial_dataset(
    group_dir: str | Path,
    target_csv: str | Path,
    max_missing_rate: float = 0.5,
    time_split: bool = True,
    test_size: float = 0.2,
    include_text: bool = False,
    categorical_encoding: Literal["label", "onehot"] = "label",
    impute_strategy: Literal["median", "mean", "zero", "none"] = "median",
    scale_features: bool = True,
    text_as_bool: bool = False,
    keep_features_csv: str | Path | None = None,
) -> TrialDataset:
    """
    Convenience function to load a trial dataset.
    
    Args:
        group_dir: Directory containing trial data
        target_csv: Path to target CSV file
        max_missing_rate: Maximum missing rate for features
        time_split: Whether to use time-based split
        test_size: Test set size ratio
        include_text: Whether to include text features
        categorical_encoding: "label" or "onehot"
        impute_strategy: "median", "mean", "zero", or "none" (keep NaN for native handling)
        scale_features: Whether to scale features
        keep_features_csv: CSV with columns [table, Variable] to force-include
        
    Returns:
        TrialDataset instance
    """
    config = DatasetConfig()
    config.features.max_missing_rate = max_missing_rate
    config.features.include_text = include_text
    config.features.text_as_bool = text_as_bool
    config.features.categorical_encoding = categorical_encoding
    config.split.time_split = time_split
    config.split.test_size = test_size
    config.impute_strategy = impute_strategy
    config.scale_features = scale_features

    if keep_features_csv:
        keep_df = pl.read_csv(keep_features_csv)
        if "Variable" in keep_df.columns:
            keep_vars = [str(v).strip() for v in keep_df.get_column("Variable").to_list() if v]
            config.features.keep_features = set(keep_vars)
        if "table" in keep_df.columns:
            extra_tables = []
            for t in keep_df.get_column("table").to_list():
                if not t:
                    continue
                name = str(t).strip()
                table_name = name if name.endswith("_all") else f"{name}_all"
                extra_tables.append(table_name)
            if extra_tables:
                merged = list(dict.fromkeys(list(config.features.feature_tables) + extra_tables))
                config.features.feature_tables = merged
    
    return TrialDataset(group_dir, target_csv, config)
