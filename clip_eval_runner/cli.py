#!/usr/bin/env python3
"""
CLI for OpenCLIP Benchmark Runner

Provides a unified command-line interface for tracking benchmarks,
launching managed jobs, and submitting to existing clusters.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import boto3
import polars as pl
import sky

from . import get_model_gpu_requirement
from .config import RESULTS_CSV
from .tracking import init_tracking, parse_results, sync_r2_results, update_tracking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
                "estimated_runtime_minutes": len(row["benchmarks"]) * 2,  # ~2 min per benchmark
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


# Command handlers
def handle_track_init(args):
    """Handle track init command."""
    init_tracking(force_reset=args.reset)


def handle_track_sync(args):
    """Handle track sync command."""
    sync_r2_results()


def handle_track_update(args):
    """Handle track update command."""
    update_tracking(sync_first=args.sync)


def handle_track_parse(args):
    """Handle track parse command."""
    parse_results(output_format=args.format)


def handle_launch_managed(args):
    """Handle launch-managed command."""
    # Check R2 credentials
    check_r2_credentials()

    # Check files
    template_path = Path("configs/grouped_job_template.yaml")
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)

    if not Path(RESULTS_CSV).exists():
        print(f"ERROR: Results CSV not found: {RESULTS_CSV}")
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


def handle_launch_cluster(args):
    """Handle launch-cluster command."""
    # Check prerequisites
    if not Path(RESULTS_CSV).exists():
        print(f"ERROR: Results CSV not found: {RESULTS_CSV}")
        sys.exit(1)

    # Check R2 credentials
    check_r2_credentials()

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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenCLIP Benchmark Runner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tracking commands
  clipeval track init                    # Initialize tracking from model pairs
  clipeval track init --reset           # Reset and recreate tracking file
  clipeval track sync                   # Sync results from R2 bucket
  clipeval track update                 # Update tracking from results directory
  clipeval track update --sync          # Sync from R2 then update tracking
  clipeval track parse                  # Generate separate analysis CSVs
  clipeval track parse --format merged  # Generate single combined CSV

  # Launch managed jobs
  clipeval launch-managed --dry-run     # Preview managed jobs
  clipeval launch-managed --max-jobs 5  # Launch up to 5 job groups
  
  # Launch on existing cluster
  clipeval launch-cluster --cluster openclip-cluster --dry-run
  clipeval launch-cluster --cluster openclip-cluster --max-jobs 10
        """,
    )

    # Global options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Track subcommand
    track_parser = subparsers.add_parser("track", help="Benchmark tracking commands")
    track_subparsers = track_parser.add_subparsers(dest="track_command", help="Track commands")

    # Track init
    track_init_parser = track_subparsers.add_parser("init", help="Initialize or update results.csv")
    track_init_parser.add_argument(
        "--reset", action="store_true", help="Force regenerate from scratch"
    )

    # Track sync
    track_subparsers.add_parser("sync", help="Sync results from R2 bucket")

    # Track update
    track_update_parser = track_subparsers.add_parser(
        "update", help="Update tracking from results directory"
    )
    track_update_parser.add_argument(
        "--sync", action="store_true", help="Sync from R2 before updating tracking"
    )

    # Track parse
    track_parse_parser = track_subparsers.add_parser(
        "parse", help="Generate analysis-ready CSV files"
    )
    track_parse_parser.add_argument(
        "--format",
        choices=["separate", "merged"],
        default="separate",
        help="Output format (default: separate)",
    )

    # Launch-managed subcommand
    launch_managed_parser = subparsers.add_parser(
        "launch-managed", help="Launch managed jobs on cloud"
    )
    launch_managed_parser.add_argument(
        "--max-jobs",
        type=int,
        help="Maximum number of job groups to launch",
    )
    launch_managed_parser.add_argument(
        "--max-benchmarks",
        type=int,
        help="Maximum number of benchmarks per group (for testing)",
    )
    launch_managed_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without launching",
    )

    # Launch-cluster subcommand
    launch_cluster_parser = subparsers.add_parser(
        "launch-cluster", help="Submit jobs to existing cluster"
    )
    launch_cluster_parser.add_argument(
        "--cluster",
        required=True,
        help="Name of the existing cluster",
    )
    launch_cluster_parser.add_argument(
        "--max-jobs",
        type=int,
        help="Maximum number of job groups to submit",
    )
    launch_cluster_parser.add_argument(
        "--gpu-fraction",
        type=float,
        default=0.125,
        help="Fraction of GPU per job (default: 0.125 = 1/8 GPU)",
    )
    launch_cluster_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without submitting",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "track":
            if not args.track_command:
                track_parser.print_help()
                sys.exit(1)
            
            if args.track_command == "init":
                handle_track_init(args)
            elif args.track_command == "sync":
                handle_track_sync(args)
            elif args.track_command == "update":
                handle_track_update(args)
            elif args.track_command == "parse":
                handle_track_parse(args)
                
        elif args.command == "launch-managed":
            handle_launch_managed(args)
            
        elif args.command == "launch-cluster":
            handle_launch_cluster(args)

        logger.info("Command completed successfully")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()