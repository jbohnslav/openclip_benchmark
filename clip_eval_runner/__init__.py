"""OpenCLIP Benchmark module for distributed model evaluation."""

from .datasets import (
    get_benchmark_datasets,
    get_dataset_config,
    load_dataset_mapping,
)
from .models import (
    estimate_job_runtime,
    generate_pairs_dataframe,
    get_model_gpu_requirement,
    get_model_metadata,
    get_openclip_models,
)
from .tracking import (
    init_tracking,
    parse_results,
    sync_r2_results,
    update_tracking,
)
from .utils import (
    estimate_job_cost,
    parse_result_filename,
    safe_load_json,
    setup_logging,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "get_openclip_models",
    "generate_pairs_dataframe",
    "get_model_metadata",
    "get_model_gpu_requirement",
    "estimate_job_runtime",
    # Tracking
    "init_tracking",
    "sync_r2_results",
    "update_tracking",
    "parse_results",
    # Utils
    "safe_load_json",
    "parse_result_filename",
    "setup_logging",
    "estimate_job_cost",
    # Datasets
    "load_dataset_mapping",
    "get_dataset_config",
    "get_benchmark_datasets",
]
