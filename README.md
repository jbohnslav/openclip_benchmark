# CLIP Eval Runner

A distributed evaluation orchestration system for running OpenCLIP models on zero-shot classification and retrieval tasks using SkyPilot managed jobs. Includes the `clipeval` command-line interface for streamlined workflow management.

## Overview

This project benchmarks all available OpenCLIP models across multiple datasets using [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) and [SkyPilot](https://github.com/skypilot-org/skypilot) for distributed cloud computing. It supports both interactive manual benchmarking and automated managed job submission.

## Quick Start

### Install Dependencies

**Method 1: Using uv (recommended for this project)**
```bash
uv sync
```

**Method 2: Using virtual environment**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Initialize Benchmark Tracking

**With uv:**
```bash
# Generate all model-benchmark pairs (9,250+ combinations)
uv run clipeval track init
```

**With virtual environment:**
```bash
# After activating venv (source venv/bin/activate)
clipeval track init
```

### Option 1: Grouped Managed Jobs (Recommended)

**With uv:**
```bash
# Preview jobs without launching (dry-run)
uv run clipeval launch-managed --dry-run

# Launch a specific number of job groups
uv run clipeval launch-managed --max-jobs 10

# Monitor running jobs
sky jobs queue

# Sync results and update tracking
uv run clipeval track update --sync

# Generate analysis CSVs
uv run clipeval track parse
```

**With virtual environment:**
```bash
# After activating venv (source venv/bin/activate)
clipeval launch-managed --dry-run
clipeval launch-managed --max-jobs 10
sky jobs queue
clipeval track update --sync
clipeval track parse
```

### Option 2: Manual Interactive Benchmarking

```bash
# Launch interactive cluster
sky launch -c interactive configs/interactive.yaml

# SSH into cluster and run individual benchmarks
sky ssh interactive

# On remote machine - Classification example:
uv run clip_benchmark eval --pretrained_model ViT-B-32,openai \
    --dataset cifar10 \
    --task zeroshot_classification \
    --output "/clip_eval_runner_results/{dataset}_{pretrained}_{model}_{language}_{task}.json"

# Retrieval example (webdataset format):
uv run clip_benchmark eval --pretrained_model ViT-B-32,openai \
    --dataset wds/mscoco_captions \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_mscoco_captions/tree/main" \
    --task zeroshot_retrieval \
    --output "/clip_eval_runner_results/{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

## CLIP Eval Runner CLI Commands

The new `clipeval` command provides a unified interface for all operations:

### Tracking Commands
```bash
clipeval track init          # Initialize model-benchmark pairs
clipeval track init --reset  # Reset and regenerate all pairs
clipeval track sync          # Download results from R2 bucket
clipeval track update        # Update tracking from local results
clipeval track update --sync # Sync from R2 first, then update
clipeval track parse         # Generate analysis CSV files
clipeval track parse --format merged  # Single combined CSV
```

### Launch Commands
```bash
# Managed jobs (recommended)
clipeval launch-managed --dry-run          # Preview jobs
clipeval launch-managed --max-jobs 10      # Launch 10 job groups
clipeval launch-managed --max-benchmarks 5 # Limit benchmarks per job (testing)

# Cluster execution (for shared resources)
clipeval launch-cluster --cluster my-cluster --dry-run
clipeval launch-cluster --cluster my-cluster --max-jobs 5
clipeval launch-cluster --cluster my-cluster --gpu-fraction 0.25
```

### Legacy Scripts (Backward Compatibility)

The original scripts are still available:
```bash
# Legacy tracking
uv run benchmark_tracker.py init
uv run benchmark_tracker.py update --sync
uv run benchmark_tracker.py parse

# Legacy job launching
uv run python scripts/grouped_job_launcher.py --dry-run
uv run python scripts/submit_to_cluster.py --cluster my-cluster
```

## Architecture

### Module Structure

```
clip_eval_runner/
├── __init__.py          # Package exports
├── cli.py              # CLIP Eval Runner CLI interface
├── config.py           # Configuration constants
├── datasets.py         # Dataset mapping and configuration
├── models.py           # OpenCLIP model utilities and metadata
├── tracking.py         # Benchmark tracking and R2 sync
└── utils.py            # Shared utilities

scripts/
├── grouped_job_launcher.py  # Grouped job submission (legacy)
├── submit_to_cluster.py     # Submit jobs to existing cluster (legacy)
└── run_benchmark_group.py   # Runs all benchmarks for a model

configs/
├── grouped_job_template.yaml  # Template for grouped jobs
├── cluster_8xh100.yaml        # Cluster configuration
├── interactive.yaml           # Interactive cluster configuration
└── experiment_spot.yaml       # Example single experiment config
```

### Key Features

**Grouped Job System:**

- Efficient batch processing by grouping benchmarks per model
- Loads model once and runs all benchmarks sequentially
- GPU requirement detection based on model architecture
- Automatic resource cleanup after completion
- Timing information for each benchmark and total runtime

**Model Metadata:**

- GPU memory requirements for 23+ model architectures
- Intelligent fallback for unknown models
- Runtime estimation based on model size and dataset
- Cost estimation with spot instance support

**Dataset Integration:**

- Comprehensive mapping for 44+ webdatasets from Hugging Face
- Automatic URL resolution for retrieval benchmarks
- Support for both classification (vtab+) and retrieval tasks

## Commands

### Benchmark Tracker

```bash
uv run benchmark_tracker.py init          # Initialize tracking
uv run benchmark_tracker.py sync          # Sync from R2 bucket
uv run benchmark_tracker.py update --sync # Sync and update tracking
uv run benchmark_tracker.py parse         # Generate analysis CSVs
```

### Grouped Job Launcher

```bash
uv run python scripts/grouped_job_launcher.py --dry-run              # Preview all pending groups
uv run python scripts/grouped_job_launcher.py --max-jobs 10          # Launch 10 job groups
uv run python scripts/grouped_job_launcher.py                        # Launch all pending groups
```

### Key Files

- `results.csv` - Master tracking file with all model-benchmark pairs
- `dataset_mapping.json` - Webdataset URL mappings
- `classification_metrics.csv` - Parsed classification results
- `retrieval_metrics.csv` - Parsed retrieval results

## Requirements

- Python 3.12+
- SkyPilot CLI (for cloud orchestration)
- AWS CLI (for R2 storage sync)
- Environment variables:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `R2_ENDPOINT_URL`

## RunPod Storage Note

RunPod doesn't support storage mounting, so we write results locally and sync to R2 after completion. This workaround enables access to cheap RunPod spot instances ($0.33/hr) instead of more expensive alternatives ($1.29/hr+).

## SkyPilot Environment Variables

When passing environment variables to cloud jobs, SkyPilot requires using `task.update_envs()` instead of YAML placeholders. The correct pattern:

```python
# In grouped_job_launcher.py
task.update_envs({
    "R2_ENDPOINT_URL": os.environ.get("R2_ENDPOINT_URL"),
    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
})
```

The YAML template should use empty strings for environment variables:
```yaml
envs:
  R2_ENDPOINT_URL: ""
  AWS_ACCESS_KEY_ID: ""
  AWS_SECRET_ACCESS_KEY: ""
```

## Safety Features

- Dry-run mode for testing
- Confirmation prompts for large batches
- Automatic spot instance usage for cost savings
- Resource cleanup after job completion
- Efficient grouped processing to minimize GPU time
- Incremental R2 syncing after each benchmark for fault tolerance
