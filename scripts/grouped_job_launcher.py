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
import sys
from pathlib import Path

import polars as pl
import sky
import yaml

from openclip_benchmark import get_model_gpu_requirement
from openclip_benchmark.cloud_gpus import generate_gpu_configs
from openclip_benchmark.config import (
    DEFAULT_DISK_SIZE,
    RESULTS_CSV,
)


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

    if max_groups:
        groups = groups.head(max_groups)

    # Convert to list of dicts with additional metadata
    group_list = []
    for row in groups.iter_rows(named=True):
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


def create_grouped_job_task(group: dict, template: dict) -> sky.Task:
    """Create SkyPilot task for a model group."""
    # Create task name
    task_name = f"openclip-group-{group['model']}-{group['pretrained']}".replace(
        "/", "-"
    )

    # Check if template resources need GPU configs to be generated
    resources = template["resources"].copy()

    # If resources contains placeholder for GPU configs, generate them
    if "any_of" in resources and resources["any_of"] == "{gpu_configs}":
        # Generate GPU configurations programmatically
        gpu_configs = generate_gpu_configs()
        resources["any_of"] = gpu_configs

    # Handle disk_size substitution
    if "disk_size" in resources:
        if resources["disk_size"] == "{disk_size}":
            resources["disk_size"] = DEFAULT_DISK_SIZE

    # Create task config with template substitutions
    task_config = {
        "name": task_name,
        "resources": resources,
        "setup": template["setup"],
        "run": template["run"].format(
            model=group["model"],
            pretrained=group["pretrained"],
        ),
        "workdir": template.get("workdir", "."),
        "file_mounts": template.get("file_mounts", {}),
    }

    # Add envs if present in template
    if "envs" in template:
        task_config["envs"] = {}
        for k, v in template["envs"].items():
            if isinstance(v, str):
                task_config["envs"][k] = v.format(
                    model=group["model"], pretrained=group["pretrained"]
                )
            else:
                task_config["envs"][k] = v

    # Create task from the config (this includes the resources from the template)
    task = sky.Task.from_yaml_config(task_config)

    return task


def submit_group_jobs(
    groups: list[dict], template: dict, dry_run: bool = False
) -> list[str]:
    """Submit grouped jobs."""
    launched_jobs = []

    for i, group in enumerate(groups):
        print(
            f"\n[{i + 1}/{len(groups)}] {group['model']}/{group['pretrained']} ({group['num_benchmarks']} benchmarks)"
        )

        task = create_grouped_job_task(group, template)

        if dry_run:
            print(f"  [DRY RUN] Would launch: {task.name}")
            print(
                f"  GPU: {group['gpu_requirement']}, Runtime: ~{group['estimated_runtime_minutes']}min"
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
        "--dry-run",
        action="store_true",
        help="Preview jobs without launching",
    )

    args = parser.parse_args()

    # Check files
    template_path = Path("configs/grouped_job_template.yaml")
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)

    if not Path(RESULTS_CSV).exists():
        print(f"ERROR: Results CSV not found: {RESULTS_CSV}")
        sys.exit(1)

    # Load template
    with open(template_path) as f:
        template = yaml.safe_load(f)

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
    launched = submit_group_jobs(groups, template, args.dry_run)

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch {len(launched)} job groups")
    else:
        print(f"\nLaunched {len(launched)} job groups")
        print("\nMonitor with:")
        print("  sky jobs queue")
        print("  sky jobs logs <job-name>")


if __name__ == "__main__":
    main()
