#!/usr/bin/env python3
"""
Job Launcher for OpenCLIP Benchmark

Submits batches of benchmark jobs to SkyPilot with cost estimation and safety controls.

Usage:
    python scripts/job_launcher.py --batch-size 5 --dry-run
    python scripts/job_launcher.py --batch-size 20 --budget-limit 50.0
"""

import argparse
import sys
from pathlib import Path

import polars as pl
import sky
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openclip_benchmark import (
    estimate_job_runtime,
    get_model_gpu_requirement,
)
from openclip_benchmark.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DISK_SIZE,
    DEFAULT_GPU_TYPE,
    MAX_HOURLY_BUDGET,
    RESULTS_CSV,
    USE_SPOT_INSTANCES,
)
from openclip_benchmark.utils import estimate_job_cost


def load_pending_benchmarks(limit: int = DEFAULT_BATCH_SIZE) -> list[dict]:
    """
    Load next batch of pending jobs from results.csv.

    Args:
        limit: Maximum number of benchmarks to load

    Returns:
        List of dictionaries with benchmark data
    """
    df = pl.read_csv(RESULTS_CSV)
    pending_df = df.filter(pl.col("status") == "pending").head(limit)
    benchmarks = pending_df.to_dicts()

    # Add GPU requirements and runtime estimates
    for benchmark in benchmarks:
        model_req = get_model_gpu_requirement(benchmark["model"])
        benchmark["gpu_requirement"] = model_req["recommended_gpu"]
        benchmark["estimated_runtime"] = estimate_job_runtime(
            benchmark["model"],
            benchmark["task_type"],
            benchmark["benchmark"],
        )

    return benchmarks


def create_job_task(benchmark: dict, template: dict) -> sky.Task:
    """
    Create SkyPilot task for a benchmark.

    Args:
        benchmark: Benchmark configuration
        template: YAML template loaded from file

    Returns:
        Configured SkyPilot Task
    """
    # Map GPU requirement
    gpu_mapping = {
        "T4": "T4:1",
        "V100": "V100:1",
        "A100": "A100:1",
        "A100-80GB": "A100-80GB:1",
    }
    model_req = get_model_gpu_requirement(benchmark["model"])
    gpu_type = gpu_mapping.get(model_req["recommended_gpu"], DEFAULT_GPU_TYPE)

    # Create task name
    task_name = f"openclip-{benchmark['model']}-{benchmark['benchmark']}".replace(
        "/", "-"
    )

    # Create task with formatted run command
    run_command = template["run"].format(
        model=benchmark["model"],
        pretrained=benchmark["pretrained"],
        dataset=benchmark["benchmark"],
        dataset_root="auto",  # Let clip_benchmark handle dataset download
        task_type=benchmark["task_type"],
        language="en",  # Default language
        task=benchmark["task_type"],  # For output filename
    )

    task = sky.Task(
        name=task_name,
        setup=template["setup"],
        run=run_command,
        workdir=".",
    )

    # Set up R2 storage mount
    storage = sky.Storage(
        name="openclip-results",
        source=None,
        stores=[sky.StoreType.R2],
    )
    task.set_file_mounts({"/openclip_results": storage})

    # Set resources
    task.set_resources(
        sky.Resources(
            accelerators=gpu_type,
            use_spot=USE_SPOT_INSTANCES,
            disk_size=DEFAULT_DISK_SIZE,
        )
    )

    return task


def submit_job_batch(
    benchmarks: list[dict], template: dict, dry_run: bool = False
) -> list[str]:
    """
    Submit batch of jobs.

    Args:
        benchmarks: List of benchmark configurations
        template: YAML template
        dry_run: If True, only simulate submission

    Returns:
        List of launched job names
    """
    if not benchmarks:
        return []

    launched_jobs = []

    for i, benchmark in enumerate(benchmarks):
        print(
            f"\n[{i + 1}/{len(benchmarks)}] {benchmark['model']}/{benchmark['benchmark']}"
        )

        # Create task
        task = create_job_task(benchmark, template)

        if dry_run:
            print(f"  [DRY RUN] Would launch: {task.name}")
            launched_jobs.append(task.name)
        else:
            # Launch the job as a managed job
            print(f"  Launching managed job: {task.name}")
            sky.jobs.launch(
                task,
                name=task.name,
                retry_until_up=True,
                detach_run=True,
            )
            launched_jobs.append(task.name)

    return launched_jobs


def estimate_batch_cost(benchmarks: list[dict]) -> float:
    """
    Estimate total cost for a batch of benchmarks.

    Args:
        benchmarks: List of benchmark configurations

    Returns:
        Total estimated cost in USD
    """
    total_cost = 0.0

    for benchmark in benchmarks:
        runtime_hours = benchmark.get("estimated_runtime", 30) / 60.0

        # Map GPU type
        gpu_mapping = {
            "T4": "T4:1",
            "V100": "V100:1",
            "A100": "A100:1",
            "A100-80GB": "A100-80GB:1",
        }
        gpu_type = gpu_mapping.get(benchmark.get("gpu_requirement", "V100"), "V100:1")

        job_cost = estimate_job_cost(gpu_type, runtime_hours, USE_SPOT_INSTANCES)
        total_cost += job_cost

    return total_cost


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Launch OpenCLIP benchmark jobs using SkyPilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview 5 jobs without launching
  python scripts/job_launcher.py --batch-size 5 --dry-run

  # Launch 20 jobs with budget limit
  python scripts/job_launcher.py --batch-size 20 --budget-limit 50.0
        """,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of jobs to launch (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without launching",
    )

    parser.add_argument(
        "--budget-limit",
        type=float,
        default=MAX_HOURLY_BUDGET,
        help=f"Maximum budget in USD (default: ${MAX_HOURLY_BUDGET})",
    )

    args = parser.parse_args()

    # Check required files
    template_path = Path("configs/managed_job_template.yaml")
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)

    if not Path(RESULTS_CSV).exists():
        print(f"ERROR: Results CSV not found: {RESULTS_CSV}")
        print("Run 'uv run benchmark_tracker.py init' first")
        sys.exit(1)

    # Load template
    with open(template_path) as f:
        template = yaml.safe_load(f)

    # Load pending benchmarks
    benchmarks = load_pending_benchmarks(args.batch_size)

    if not benchmarks:
        print("No pending benchmarks found.")
        return

    # Estimate cost
    estimated_cost = estimate_batch_cost(benchmarks)

    # Check budget
    if estimated_cost > args.budget_limit:
        print(
            f"\nERROR: Estimated cost ${estimated_cost:.2f} exceeds budget ${args.budget_limit:.2f}"
        )
        sys.exit(1)

    # Show summary
    print(f"\nFound {len(benchmarks)} pending benchmarks")
    print(f"Estimated cost: ${estimated_cost:.2f}")

    # Show first few
    print("\nFirst few jobs:")
    for i, b in enumerate(benchmarks[:5]):
        runtime = b.get("estimated_runtime", 30)
        print(
            f"  {i + 1}. {b['model']}/{b['benchmark']} ({b['task_type']}) - ~{runtime}min on {b.get('gpu_requirement', 'V100')}"
        )
    if len(benchmarks) > 5:
        print(f"  ... and {len(benchmarks) - 5} more")

    # Confirmation for large batches
    if not args.dry_run and len(benchmarks) > 10:
        response = input(f"\nLaunch {len(benchmarks)} jobs? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("Cancelled.")
            return

    # Submit jobs
    launched = submit_job_batch(benchmarks, template, args.dry_run)

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch {len(launched)} jobs")
    else:
        print(f"\nLaunched {len(launched)} managed jobs")
        print("\nMonitor with:")
        print("  sky jobs queue  # List all jobs")
        print("  sky jobs logs <job-name>  # View job logs")
        print("  uv run benchmark_tracker.py update --sync  # Sync results")


if __name__ == "__main__":
    main()
