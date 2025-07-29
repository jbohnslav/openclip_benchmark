# OpenCLIP Benchmark

A distributed benchmarking system for evaluating OpenCLIP models on zero-shot classification and retrieval tasks using SkyPilot managed jobs.

## Overview

This project benchmarks all available OpenCLIP models across multiple datasets using [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) and [SkyPilot](https://github.com/skypilot-org/skypilot) for distributed cloud computing. It supports both interactive manual benchmarking and automated managed job submission.

## Quick Start

### Install Dependencies

```bash
uv sync
```

### Initialize Benchmark Tracking

```bash
# Generate all model-benchmark pairs (9,250+ combinations)
uv run benchmark_tracker.py init
```

### Option 1: Automated Managed Jobs (Recommended)

```bash
# Preview jobs without launching (dry-run)
uv run python scripts/job_launcher.py --batch-size 10 --dry-run

# Launch actual jobs after review
uv run python scripts/job_launcher.py --batch-size 10

# Monitor running jobs
sky jobs queue

# Sync results and update tracking
uv run benchmark_tracker.py update --sync

# Generate analysis CSVs
uv run benchmark_tracker.py parse
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
    --output "/openclip_results/{dataset}_{pretrained}_{model}_{language}_{task}.json"

# Retrieval example (webdataset format):
uv run clip_benchmark eval --pretrained_model ViT-B-32,openai \
    --dataset wds/mscoco_captions \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_mscoco_captions/tree/main" \
    --task zeroshot_retrieval \
    --output "/openclip_results/{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

## Architecture

### Module Structure

```
openclip_benchmark/
├── __init__.py          # Package exports
├── config.py           # Configuration constants
├── datasets.py         # Dataset mapping and configuration
├── models.py           # OpenCLIP model utilities and metadata
├── tracking.py         # Benchmark tracking and R2 sync
└── utils.py            # Shared utilities

scripts/
└── job_launcher.py     # Managed job submission system

configs/
├── managed_job_template.yaml  # SkyPilot managed job template
└── interactive.yaml           # Interactive cluster configuration
```

### Key Features

**Managed Job System:**

- Simple batch job submission with cost estimation
- GPU requirement detection based on model architecture
- Budget limits and confirmation prompts
- Automatic resource cleanup after completion

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

### Job Launcher

```bash
uv run python scripts/job_launcher.py --batch-size 10 --dry-run      # Preview jobs
uv run python scripts/job_launcher.py --batch-size 10                # Launch jobs
uv run python scripts/job_launcher.py --batch-size 20 --budget-limit 50.0  # With budget limit
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

## Safety Features

- Maximum 10 jobs by default (expandable to 90)
- Budget estimation with configurable limits
- Dry-run mode for testing
- Confirmation prompts for large batches
- Automatic spot instance usage for cost savings
- Resource cleanup after job completion
