# Shared Cache Execution

This document describes how to run OpenCLIP benchmarks on a multi-GPU cluster to benefit from shared HuggingFace dataset caching.

## Overview

Instead of using SkyPilot managed jobs (which distribute jobs across different machines), we can launch a single large cluster and submit many jobs to it using `sky.exec()`. This allows all jobs to share the same HuggingFace dataset cache, dramatically reducing redundant downloads.

The script reuses our existing `run_benchmark_group.py` to process all benchmarks for each (model, pretrained) pair, maintaining compatibility with R2 syncing and all existing functionality.

## Usage

### 1. Launch the cluster

First, manually launch an 8xH100 cluster:

```bash
sky launch -c openclip-cluster configs/cluster_8xh100.yaml
```

This creates a cluster with:
- 8x H100 GPUs
- 500GB disk for dataset caching
- All dependencies installed

### 2. Submit jobs

Then submit benchmarks to the cluster using fractional GPUs:

```bash
# Submit first 10 model groups (each group runs all benchmarks for that model)
python scripts/submit_to_cluster.py --cluster openclip-cluster --max-jobs 10

# Submit with different GPU fraction (0.25 = 4 jobs per GPU = 32 concurrent)
python scripts/submit_to_cluster.py --cluster openclip-cluster --gpu-fraction 0.25

# Dry run to preview
python scripts/submit_to_cluster.py --cluster openclip-cluster --dry-run
```

### 3. Monitor progress

```bash
# Check job queue on cluster
sky queue openclip-cluster

# Watch job progress
watch -n 5 'sky queue openclip-cluster'

# View logs for specific job
sky logs openclip-cluster --job-id <job-id>
```

### 4. Teardown

When done, teardown the cluster:

```bash
sky down openclip-cluster
```

## How it works

1. **Shared cluster**: All jobs run on the same 8xH100 node
2. **Fractional GPUs**: Each job uses 0.125 GPU (1/8), allowing 64 concurrent jobs
3. **Shared cache**: HuggingFace datasets cached at `~/.cache/huggingface`, webdatasets at `~/.cache/webdataset/{dataset}/`
4. **Automatic scheduling**: SkyPilot queues jobs when resources are full
5. **sky.exec()**: Skips provisioning/setup, just syncs workdir and enqueues the job

## Technical Details

The script uses `sky.exec()` instead of `sky.launch()`:
- `sky.exec()`: Submits job to existing cluster, skips setup
- `sky.launch()`: Would provision new resources or re-run setup

This is crucial for our use case because:
- Jobs start faster (no setup overhead)
- All jobs share the same environment and cache
- We can submit hundreds of jobs quickly

### Cache Isolation

Each dataset gets its own webdataset cache directory to prevent "noisy neighbor" issues:
- Different datasets don't overwrite each other's cache files
- Multiple jobs can download different datasets concurrently without conflicts
- Cache files (0.tar, 1.tar, etc.) are isolated per dataset

## Benefits

- **Efficient caching**: Download each dataset only once
- **High utilization**: Run 64 jobs concurrently on 8 GPUs
- **Simple**: No complex scheduling logic needed
- **Flexible**: Easily adjust GPU fraction per job

## Comparison with managed jobs

| Aspect | Managed Jobs | Shared Cluster |
|--------|--------------|----------------|
| Setup | Automatic | Manual cluster launch |
| Caching | Per-machine | Shared across all jobs |
| Scheduling | Across clouds | Within single cluster |
| Cost | Pay per job | Pay for cluster duration |
| Best for | Distributed workloads | Cache-heavy workloads |