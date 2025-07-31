#!/usr/bin/env python3
"""
Submit OpenCLIP benchmark jobs to an existing cluster using sky.exec().

This script submits multiple jobs to the same cluster to benefit from
shared HuggingFace dataset caching. It reuses the existing run_benchmark_group.py
script to process all benchmarks for each model.

Usage:
    # First, manually launch the cluster:
    sky launch -c openclip-cluster configs/cluster_8xh100.yaml

    # Then submit jobs:
    python scripts/submit_to_cluster.py --cluster openclip-cluster --max-jobs 10
    python scripts/submit_to_cluster.py --cluster openclip-cluster --gpu-fraction 0.25
"""

import argparse
import os
import random
import sys
from pathlib import Path

import boto3
import polars as pl
import sky

from clip_eval_runner import get_model_gpu_requirement
from clip_eval_runner.config import RESULTS_CSV


def load_pending_groups(max_groups: int = None) -> list[dict]:
    """
    Load pending benchmarks grouped by (model, pretrained) pairs.
    Reuses logic from grouped_job_launcher.py
    """
    df = pl.read_csv(RESULTS_CSV)

    # Filter pending benchmarks
    pending_df = df.filter(pl.col("status") == "pending")

    # Group by model and pretrained
    groups = (
        pending_df.group_by(["model", "pretrained"])
        .agg(
            [
                pl.col("benchmark").alias("benchmarks"),
                pl.col("task_type").alias("task_types"),
            ]
        )
        .sort("model", "pretrained")
    )

    # Convert to list for shuffling
    groups_list = groups.to_dicts()

    # Randomize order to test different models
    random.shuffle(groups_list)

    if max_groups:
        groups_list = groups_list[:max_groups]

    # Convert to list of dicts with additional metadata
    group_list = []
    for row in groups_list:
        model_req = get_model_gpu_requirement(row["model"])

        group_list.append(
            {
                "model": row["model"],
                "pretrained": row["pretrained"],
                "benchmarks": row["benchmarks"],
                "task_types": row["task_types"],
                "num_benchmarks": len(row["benchmarks"]),
                "gpu_requirement": model_req["recommended_gpu"],
                "estimated_runtime_minutes": len(row["benchmarks"])
                * 2,  # ~2 min per benchmark
            }
        )

    return group_list


def check_r2_credentials():
    """Check and load R2 credentials."""
    if not os.environ.get("R2_ENDPOINT_URL"):
        raise ValueError(
            "R2_ENDPOINT_URL is not set. Run `export R2_ENDPOINT_URL='https://...'` to set it."
        )

    # Try to load from AWS profile 'r2' if not already in environment
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
        try:
            session = boto3.Session(profile_name="r2")
            credentials = session.get_credentials()
            if credentials:
                os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
                os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
                print("Loaded AWS credentials from profile 'r2'")
            else:
                raise ValueError(
                    "No AWS credentials found. Run `aws configure --profile r2` to set them."
                )
        except Exception:
            raise ValueError(
                "No AWS credentials found. Run `aws configure --profile r2` to set them."
            ) from None


def submit_group_to_cluster(
    cluster_name: str,
    group: dict,
    gpu_fraction: float = 0.125,
) -> str:
    """Submit a model group job to the cluster using sky.exec()."""

    model = group["model"]
    pretrained = group["pretrained"]

    # Create task to run the benchmark group
    cmd = (
        f"uv run python scripts/run_benchmark_group.py "
        f"--model '{model}' "
        f"--pretrained '{pretrained}' "
        f"--output-dir ~/openclip_results"
    )

    task = sky.Task(
        run=cmd,
        workdir=".",
        envs={
            "R2_ENDPOINT_URL": os.environ.get("R2_ENDPOINT_URL"),
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        },
    )

    # Set fractional GPU requirement
    task.set_resources(sky.Resources(accelerators=f"H100:{gpu_fraction}"))

    # Submit to cluster using exec (skips setup/provisioning)
    request_id = sky.exec(task, cluster_name=cluster_name)

    return request_id


def main():
    parser = argparse.ArgumentParser(
        description="Submit OpenCLIP benchmark jobs to existing cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cluster",
        required=True,
        help="Name of the existing cluster",
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Maximum number of job groups to submit",
    )

    parser.add_argument(
        "--gpu-fraction",
        type=float,
        default=0.125,
        help="Fraction of GPU per job (default: 0.125 = 1/8 GPU)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without submitting",
    )

    args = parser.parse_args()

    # Check prerequisites
    if not Path(RESULTS_CSV).exists():
        print(f"ERROR: Results CSV not found: {RESULTS_CSV}")
        sys.exit(1)

    # Check R2 credentials
    try:
        check_r2_credentials()
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load pending groups
    groups = load_pending_groups(args.max_jobs)

    if not groups:
        print("No pending benchmarks found.")
        return

    # Show summary
    total_benchmarks = sum(g["num_benchmarks"] for g in groups)
    print(
        f"\nFound {len(groups)} model groups with {total_benchmarks} total benchmarks"
    )
    print(f"Cluster: {args.cluster}")
    print(f"GPU fraction per job: {args.gpu_fraction}")
    print(f"Max concurrent jobs: {int(8 / args.gpu_fraction)}")

    # Show examples
    print("\nFirst few groups:")
    for i, g in enumerate(groups[:5]):
        runtime = g["estimated_runtime_minutes"]
        print(
            f"  {i + 1}. {g['model']}/{g['pretrained']}: "
            f"{g['num_benchmarks']} benchmarks, ~{runtime}min"
        )
    if len(groups) > 5:
        print(f"  ... and {len(groups) - 5} more groups")

    # Confirmation
    if not args.dry_run and len(groups) > 5:
        response = input(f"\nSubmit {len(groups)} job groups? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("Cancelled.")
            return

    # Submit jobs
    print(f"\n{'[DRY RUN] Would submit' if args.dry_run else 'Submitting'} jobs...")

    request_ids = []
    for i, group in enumerate(groups):
        print(
            f"\n[{i + 1}/{len(groups)}] {group['model']}/{group['pretrained']} "
            f"({group['num_benchmarks']} benchmarks)"
        )

        if args.dry_run:
            print("  [DRY RUN] Would submit job")
        else:
            request_id = submit_group_to_cluster(
                cluster_name=args.cluster,
                group=group,
                gpu_fraction=args.gpu_fraction,
            )
            print(f"  ✓ Submitted (request_id: {request_id})")
            request_ids.append(request_id)

    # Summary
    print(
        f"\n{'[DRY RUN] Would submit' if args.dry_run else '✅ Submitted'} {len(groups)} jobs"
    )
    print("\nMonitor cluster with:")
    print(f"  sky queue {args.cluster}")
    print(f"  watch -n 5 'sky queue {args.cluster}'")
    print("\nView logs:")
    print(f"  sky logs {args.cluster} --job-id <job-id>")


if __name__ == "__main__":
    main()
