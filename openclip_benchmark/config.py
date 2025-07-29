"""Configuration constants for OpenCLIP benchmarking.

This module contains all configuration constants and settings used throughout
the OpenCLIP benchmarking project, including job limits, R2 storage settings,
file paths, and safety limits.
"""

from pathlib import Path
from typing import Final

# Job limits and safety
DEFAULT_BATCH_SIZE: Final[int] = 10
"""Default number of jobs to process in a single batch."""

MAX_CONCURRENT_JOBS: Final[int] = 90
"""Maximum number of concurrent jobs allowed to run simultaneously."""

MAX_TEST_JOBS: Final[int] = 10
"""Maximum number of jobs to run in test mode."""

# R2 configuration
R2_BUCKET_NAME: Final[str] = "openclip-results"
"""Name of the Cloudflare R2 bucket for storing benchmark results."""

R2_MOUNT_PATH: Final[str] = "/openclip_results"
"""Path where R2 bucket is mounted in the cloud environment."""

# File paths
DATASET_MAPPING_FILE: Final[Path] = (
    Path(__file__).parent.parent / "dataset_mapping.json"
)
"""Path to the dataset mapping configuration file."""

RESULTS_CSV: Final[str] = "results.csv"
"""Filename for the consolidated results CSV file."""

RESULTS_DIR: Final[str] = "results"
"""Directory name for storing downloaded benchmark results."""

# Job configuration
DEFAULT_GPU_TYPE: Final[str] = "A100:1"
"""Default GPU type and count for benchmark jobs."""

DEFAULT_DISK_SIZE: Final[int] = 50
"""Default disk size in GB for benchmark jobs."""

USE_SPOT_INSTANCES: Final[bool] = True
"""Whether to use spot instances for cost optimization."""

# Output file naming
OUTPUT_FILENAME_TEMPLATE: Final[str] = (
    "{dataset}_{pretrained}_{model}_en_{task_type}.json"
)
"""Template for generating output filenames for benchmark results.

The template uses the following placeholders:
- {dataset}: The benchmark dataset name
- {pretrained}: The pretrained model identifier
- {model}: The model architecture name
- {task_type}: The type of task (e.g., zeroshot_classification, retrieval)
"""

# Safety limits
MAX_HOURLY_BUDGET: Final[float] = 100.0
"""Maximum hourly budget in USD for cloud computing costs."""

JOB_TIMEOUT_MINUTES: Final[int] = 120
"""Timeout in minutes for individual benchmark jobs."""

# Logging
LOG_FILE: Final[str] = "benchmark_tracker.log"
"""Filename for the main application log file."""
