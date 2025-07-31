#!/usr/bin/env python3
"""
Run all benchmarks for a specific (model, pretrained) pair.

This script is executed on GPU nodes to process a group of benchmarks
efficiently by loading the model once and running all benchmarks sequentially.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from clip_eval_runner.config import RESULTS_CSV
from clip_eval_runner.datasets import get_dataset_config


def check_r2_credentials():
    """
    Check if R2 credentials are configured.

    Raises:
        SystemExit if credentials are missing
    """
    missing = []

    if not os.environ.get("R2_ENDPOINT_URL"):
        missing.append("R2_ENDPOINT_URL")
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        missing.append("AWS_ACCESS_KEY_ID")
    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        missing.append("AWS_SECRET_ACCESS_KEY")

    if missing:
        print("ERROR: Missing R2 credentials:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set these environment variables before running.")
        sys.exit(1)


def sync_to_r2(output_dir: str) -> bool:
    """
    Sync results to R2 bucket.

    Returns:
        True if sync succeeded, False otherwise
    """
    try:
        r2_endpoint = os.environ.get("R2_ENDPOINT_URL", "")
        if not r2_endpoint:
            print("  Warning: R2_ENDPOINT_URL not set")
            return False

        cmd = [
            "aws",
            "s3",
            "sync",
            output_dir,
            "s3://openclip-results/",
            "--endpoint-url",
            r2_endpoint,
            "--region",
            "auto",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"  Warning: R2 sync failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Warning: R2 sync exception: {str(e)}")
        return False


def run_single_benchmark(
    model: str,
    pretrained: str,
    benchmark: str,
    task_type: str,
    output_dir: str,
    sync_after_each: bool = True,
) -> dict:
    """
    Run a single benchmark and save results.

    Returns:
        Dict with status and result_file path
    """
    dataset_info = get_dataset_config(benchmark, task_type)

    # Construct output filename
    language = "en"  # Default language
    output_file = (
        Path(output_dir)
        / f"{benchmark}_{pretrained}_{model}_{language}_{task_type}.json"
    )

    try:
        # Run the benchmark using clip_benchmark CLI
        # Use webdataset name for retrieval tasks
        dataset_name = dataset_info.get("webdataset_name", benchmark)
        dataset_root = dataset_info.get("dataset_root", "auto")

        # Create unique cache directory per dataset to avoid noisy neighbor issues
        # Each dataset gets its own subdirectory: ~/.cache/webdataset/dataset_name/
        wds_cache_dir = os.path.expanduser(f"~/.cache/webdataset/{benchmark}")
        os.makedirs(wds_cache_dir, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "clip_benchmark",
            "eval",
            "--pretrained_model",
            f"{model},{pretrained}",
            "--dataset",
            dataset_name,
            "--dataset_root",
            dataset_root,
            "--task",
            task_type,
            "--batch_size",
            "16",  # Default batch size
            "--wds_cache_dir",
            wds_cache_dir,  # Unique cache dir per dataset
            "--output",
            str(output_file),
        ]

        print(f"Running: {' '.join(cmd)}")
        start_time = time.time()

        result = subprocess.run(cmd, capture_output=True, text=True)

        runtime = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ Completed in {runtime:.1f}s: {output_file.name}")

            # Sync to R2 after successful benchmark
            if sync_after_each:
                print("  Syncing to R2...")
                sync_success = sync_to_r2(output_dir)
                if sync_success:
                    print("  ✓ Synced to R2")
                else:
                    print("  ⚠ R2 sync failed, but result saved locally")

            return {
                "status": "completed",
                "result_file": str(output_file),
                "runtime_seconds": runtime,
            }
        else:
            print(f"✗ Failed: {result.stderr}")
            return {
                "status": "failed",
                "error": result.stderr,
                "runtime_seconds": runtime,
            }

    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmarks for a model/pretrained pair"
    )

    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--pretrained", required=True, help="Pretrained weights")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--max-benchmarks",
        type=int,
        help="Maximum number of benchmarks to run (for testing)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Disable syncing to R2 after each benchmark",
    )

    args = parser.parse_args()

    # Check R2 credentials if syncing is enabled
    if not args.no_sync:
        check_r2_credentials()

    # Start overall timer
    overall_start = time.time()

    # Load benchmarks for this model/pretrained pair
    df = pl.read_csv(RESULTS_CSV)

    group_df = df.filter(
        (pl.col("model") == args.model)
        & (pl.col("pretrained") == args.pretrained)
        & (pl.col("status") == "pending")
    )

    benchmarks = group_df.to_dicts()

    if not benchmarks:
        print(f"No pending benchmarks found for {args.model}/{args.pretrained}")
        return

    # Limit benchmarks if requested
    if args.max_benchmarks:
        benchmarks = benchmarks[: args.max_benchmarks]
        print(f"\nLimiting to first {args.max_benchmarks} benchmarks (for testing)")

    print(
        f"\nProcessing {len(benchmarks)} benchmarks for {args.model}/{args.pretrained}"
    )
    print("=" * 60)

    # Track results for exit code only
    failed = 0
    total_benchmark_time = 0

    # Process each benchmark
    for i, benchmark in enumerate(benchmarks):
        print(
            f"\n[{i + 1}/{len(benchmarks)}] {benchmark['benchmark']} ({benchmark['task_type']})"
        )

        result = run_single_benchmark(
            model=args.model,
            pretrained=args.pretrained,
            benchmark=benchmark["benchmark"],
            task_type=benchmark["task_type"],
            output_dir=args.output_dir,
            sync_after_each=not args.no_sync,
        )

        if result["status"] != "completed":
            failed += 1

        if "runtime_seconds" in result:
            total_benchmark_time += result["runtime_seconds"]

    # Calculate total time
    overall_time = time.time() - overall_start

    # Print timing summary
    print("\n" + "=" * 60)
    print(f"Completed {len(benchmarks)} benchmarks for {args.model}/{args.pretrained}")
    print(
        f"Total benchmark runtime: {total_benchmark_time:.1f} seconds ({total_benchmark_time / 60:.1f} minutes)"
    )
    print(
        f"Total overall runtime: {overall_time:.1f} seconds ({overall_time / 60:.1f} minutes)"
    )
    print(f"Overhead time: {overall_time - total_benchmark_time:.1f} seconds")

    if failed > 0:
        print(f"\n⚠️  {failed} benchmarks failed")
    else:
        print("\n✅ All benchmarks completed successfully")

    # Final sync to ensure everything is uploaded
    if not args.no_sync:
        print("\nPerforming final R2 sync...")
        if sync_to_r2(args.output_dir):
            print("✓ Final sync completed")
        else:
            print("⚠ Final sync failed")

    # Exit with error if any failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
