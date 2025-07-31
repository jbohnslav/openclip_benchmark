#!/usr/bin/env python3
"""
Grouped Job Launcher for OpenCLIP Benchmark

Submits jobs grouped by (model, pretrained) pairs to maximize GPU utilization.
Each job processes all benchmarks for a single model configuration.

Usage:
    python scripts/grouped_job_launcher.py --dry-run
    python scripts/grouped_job_launcher.py --max-jobs 10
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

    Returns:
        List of groups, each containing model, pretrained, and list of benchmarks
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


def create_grouped_job_task(group: dict, max_benchmarks: int = None) -> sky.Task:
    """Create SkyPilot task for a model group."""
    # Create task name
    task_name = f"openclip-group-{group['model']}-{group['pretrained']}".replace(
        "/", "-"
    )

    # Load base task from template
    task = sky.Task.from_yaml("configs/grouped_job_template.yaml")
    task.name = task_name

    model = group["model"]
    pretrained = group["pretrained"]
    max_benchmarks_flag = f"--max-benchmarks {max_benchmarks}" if max_benchmarks else ""

    # Create run command with proper formatting
    run_cmd = (
        "uv run python scripts/run_benchmark_group.py "
        f'--model "{model}" '
        f'--pretrained "{pretrained}" '
        '--output-dir "$HOME/openclip_results"'
    )
    if max_benchmarks_flag:
        run_cmd += f" {max_benchmarks_flag}"

    task.run = run_cmd

    # Update environment variables with model info and R2 credentials
    task.update_envs(
        {
            "MODEL": model,
            "PRETRAINED": pretrained,
            "R2_ENDPOINT_URL": os.environ.get("R2_ENDPOINT_URL"),
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        }
    )

    return task


def submit_group_jobs(
    groups: list[dict],
    dry_run: bool = False,
    max_benchmarks: int = None,
) -> list[str]:
    """Submit grouped jobs."""
    launched_jobs = []

    for i, group in enumerate(groups):
        actual_benchmarks = (
            min(group["num_benchmarks"], max_benchmarks)
            if max_benchmarks
            else group["num_benchmarks"]
        )
        print(
            f"\n[{i + 1}/{len(groups)}] {group['model']}/{group['pretrained']} ({actual_benchmarks} benchmarks)"
        )

        task = create_grouped_job_task(group, max_benchmarks)

        if dry_run:
            print(f"  [DRY RUN] Would launch: {task.name}")
            estimated_runtime = actual_benchmarks * 2  # ~2 min per benchmark
            print(
                f"  GPU: {group['gpu_requirement']}, Runtime: ~{estimated_runtime}min"
            )
            launched_jobs.append(task.name)
        else:
            print(f"  Launching managed job: {task.name}")
            sky.jobs.launch(task, name=task.name)
            launched_jobs.append(task.name)

    return launched_jobs


def main():
    parser = argparse.ArgumentParser(
        description="Launch grouped OpenCLIP benchmark jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Maximum number of job groups to launch",
    )

    parser.add_argument(
        "--max-benchmarks",
        type=int,
        help="Maximum number of benchmarks per group (for testing)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without launching",
    )

    args = parser.parse_args()

    # Check R2 credentials are set locally (SkyPilot will pass them through)
    if not os.environ.get("R2_ENDPOINT_URL"):
        raise ValueError(
            "R2_ENDPOINT_URL is not set. Run `export R2_ENDPOINT_URL='https://...'` to set it."
        )

    # Try to load from AWS profile 'r2' if not already in environment
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
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

    # Check files
    template_path = Path("configs/grouped_job_template.yaml")
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)

    if not Path(RESULTS_CSV).exists():
        print(f"ERROR: Results CSV not found: {RESULTS_CSV}")
        sys.exit(1)

    # Template is now loaded directly in create_grouped_job_task
    # Just verify it exists above

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

    # Show examples
    print("\nFirst few groups:")
    for i, g in enumerate(groups[:5]):
        runtime = g["estimated_runtime_minutes"]
        print(
            f"  {i + 1}. {g['model']}/{g['pretrained']}: {g['num_benchmarks']} benchmarks, ~{runtime}min on {g['gpu_requirement']}"
        )
    if len(groups) > 5:
        print(f"  ... and {len(groups) - 5} more groups")

    # Confirmation
    if not args.dry_run and len(groups) > 5:
        response = input(f"\nLaunch {len(groups)} job groups? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("Cancelled.")
            return

    # Submit
    launched = submit_group_jobs(groups, args.dry_run, args.max_benchmarks)

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch {len(launched)} job groups")
    else:
        print(f"\nLaunched {len(launched)} job groups")
        print("\nMonitor with:")
        print("  sky jobs queue")
        print("  sky jobs logs <job-name>")


if __name__ == "__main__":
    main()
