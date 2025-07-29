# OpenCLIP Benchmark

A minimal benchmark tracking system for evaluating OpenCLIP models on zero-shot classification and retrieval tasks.

## Overview

This project benchmarks all available OpenCLIP models across multiple datasets using [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) and [SkyPilot](https://github.com/skypilot-org/skypilot) for distributed cloud computing.

## Quick Start

```bash
# Install dependencies
uv sync

# Initialize benchmark tracking (9,000+ model-dataset pairs)
uv run benchmark_tracker.py init

# Launch interactive cluster for manual benchmarking
sky launch -c interactive configs/interactive.yaml

# SSH into the cluster and run benchmarks manually
sky ssh interactive

# On the remote machine:
# Classification example:
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

# Sync results from R2 and update tracking
uv run benchmark_tracker.py update --sync

# Generate analysis-ready CSVs
uv run benchmark_tracker.py parse
```

## Benchmark Tracker Commands

- `init` - Generate all model-benchmark pairs and initialize tracking
- `sync` - Download results from R2 bucket  
- `update` - Update tracking from local results
- `update --sync` - Sync from R2 and update in one command
- `parse` - Generate analysis CSVs from completed benchmarks

## Files

- `benchmark_tracker.py` - Main tracking system
- `configs/interactive.yaml` - SkyPilot configuration for interactive GPU cluster
- `results.csv` - Tracking file with benchmark completion status
- `classification_metrics.csv` - Parsed classification benchmark results

## Requirements

- Python 3.12+
- AWS CLI (for R2 sync)
- Environment variables:
  - `AWS_ACCESS_KEY_ID` 
  - `AWS_SECRET_ACCESS_KEY`
  - `R2_ENDPOINT_URL`