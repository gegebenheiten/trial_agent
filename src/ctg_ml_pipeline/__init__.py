"""CTG NotebookLM downstream ML pipeline (framework)."""

from ctg_ml_pipeline.config import PipelineConfig
from ctg_ml_pipeline.data.status import scan_notebooklm_group
from ctg_ml_pipeline.data.merge import (
    merge_group_tables,
    merge_group_tables_by_table,
    backfill_notebooklm_columns,
)
from ctg_ml_pipeline.analysis.visualize import summarize_missing, summarize_ranges
from ctg_ml_pipeline.preprocess.impute import impute_simple
from ctg_ml_pipeline.data.targets import build_target_labels

# Variable statistics
from ctg_ml_pipeline.analysis.var_stats import (
    VarStat,
    analyze_var,
    analyze_vars,
    summarize_var_stats,
    summarize_var_stats_multi_tables,
    filter_problematic_vars,
    print_var_stats_summary,
)

# Dataset
from ctg_ml_pipeline.data.dataset import (
    TrialDataset,
    DatasetConfig,
    TargetConfig,
    FeatureConfig,
    SplitConfig,
    FeatureType,
    load_trial_dataset,
)

# Modeling
from ctg_ml_pipeline.modeling.modeling import (
    ModelResult,
    ComparisonResult,
    train_and_evaluate,
    compare_models,
    run_experiment,
)
from ctg_ml_pipeline.modeling.tuning import TuneResult, tune_model, tune_models

__all__ = [
    # Config
    "PipelineConfig",
    # Data processing
    "scan_notebooklm_group",
    "merge_group_tables",
    "merge_group_tables_by_table",
    "backfill_notebooklm_columns",
    "summarize_missing",
    "summarize_ranges",
    "impute_simple",
    "build_target_labels",
    # Variable statistics
    "VarStat",
    "analyze_var",
    "analyze_vars",
    "summarize_var_stats",
    "summarize_var_stats_multi_tables",
    "filter_problematic_vars",
    "print_var_stats_summary",
    # Dataset
    "TrialDataset",
    "DatasetConfig",
    "TargetConfig",
    "FeatureConfig",
    "SplitConfig",
    "FeatureType",
    "load_trial_dataset",
    # Modeling
    "ModelResult",
    "ComparisonResult",
    "train_and_evaluate",
    "compare_models",
    "run_experiment",
    "TuneResult",
    "tune_model",
    "tune_models",
]
