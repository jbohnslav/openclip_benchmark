#!/usr/bin/env python3
"""
Single-file benchmark tracking system for OpenCLIP benchmarks.
Tracks completion status and parses results.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark_tracker.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_result_filename(filename: str) -> dict[str, str] | None:
    """
    Parse result filename to extract benchmark components.

    Expected format: {benchmark}_{pretrained}_{model}_{lang}_zeroshot_{task_type}.json
    Example: cifar10_openai_ViT-B-32_en_zeroshot_classification.json
    """
    # Remove .json extension
    name = filename.replace(".json", "")

    # Split by underscores and try to identify components
    parts = name.split("_")

    if len(parts) < 5:
        logger.warning(f"Unexpected filename format: {filename}")
        return None

    # Extract task type (classification or retrieval)
    task_type = None
    if "classification" in parts:
        task_type = "classification"
    elif "retrieval" in parts:
        task_type = "retrieval"
    else:
        logger.warning(f"Could not determine task type from filename: {filename}")
        return None

    # Find the benchmark (first part)
    benchmark = parts[0]

    # Find pretrained and model - they're typically together
    # Look for common pretrained patterns
    pretrained = None
    model = None

    for i, part in enumerate(parts[1:], 1):
        if part in ["openai", "laion2b_s34b_b79k", "laion400m_e31", "laion400m_e32"]:
            pretrained = part
            # Model is typically the next part(s) until we hit language or zeroshot
            model_parts = []
            for j in range(i + 1, len(parts)):
                if parts[j] in ["en", "zeroshot", "classification", "retrieval"]:
                    break
                model_parts.append(parts[j])
            model = "_".join(model_parts) if model_parts else "unknown"
            break

    if not pretrained or not model:
        # Fallback: assume second part is pretrained, third is model
        if len(parts) >= 3:
            pretrained = parts[1]
            model = parts[2]
        else:
            logger.warning(
                f"Could not parse pretrained/model from filename: {filename}"
            )
            return None

    return {
        "benchmark": benchmark,
        "pretrained": pretrained,
        "model": model,
        "task_type": task_type,
    }


def safe_load_json(filepath: str) -> tuple[dict | None, str | None]:
    """
    Safely load JSON file with error handling.

    Returns:
        tuple: (data, error_message)
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    except FileNotFoundError:
        return None, f"File not found: {filepath}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def get_openclip_models():
    """Get all available pretrained OpenCLIP models"""
    import open_clip

    pretrained = open_clip.list_pretrained()
    return [(model, pretrained_name) for model, pretrained_name in pretrained]


def get_benchmark_datasets():
    """Get zero-shot classification and retrieval benchmarks from CLIP Benchmark"""
    from clip_benchmark.cli import dataset_collection

    # Get all classification benchmarks from vtab+ collection
    classification_benchmarks = dataset_collection["vtab+"]

    # Get retrieval benchmarks
    retrieval_benchmarks = dataset_collection["retrieval"]

    # Add additional classification datasets not in vtab+
    additional_classification = [
        "cifar10",
        "cifar100",
        "caltech101",
        "dtd",
        "eurosat",
        "flowers",
        "food101",
        "pets",
        "pcam",
        "resisc45",
    ]

    # Add any additional datasets that aren't already in vtab+
    for dataset in additional_classification:
        if dataset not in classification_benchmarks:
            classification_benchmarks.append(dataset)

    # Remove duplicates and sort
    classification_benchmarks = sorted(set(classification_benchmarks))
    retrieval_benchmarks = sorted(set(retrieval_benchmarks))

    return classification_benchmarks, retrieval_benchmarks


def generate_pairs_dataframe():
    """Generate DataFrame with all model-benchmark combinations"""
    models = get_openclip_models()
    classification_benchmarks, retrieval_benchmarks = get_benchmark_datasets()

    all_benchmarks = [
        ("classification", bench) for bench in classification_benchmarks
    ] + [("retrieval", bench) for bench in retrieval_benchmarks]

    rows = []
    for model, pretrained in models:
        for task_type, benchmark in all_benchmarks:
            rows.append(
                {
                    "model": model,
                    "pretrained": pretrained,
                    "task_type": task_type,
                    "benchmark": benchmark,
                }
            )

    return pl.DataFrame(rows)


def init_tracking(force_reset: bool = False) -> None:
    """
    Initialize results.csv by generating all model-benchmark pairs

    Args:
        force_reset: If True, recreate results.csv from scratch
    """
    logger.info("Initializing benchmark tracking...")

    # Generate model-benchmark pairs
    logger.info("Generating model-benchmark pairs...")
    pairs_df = generate_pairs_dataframe()
    logger.info(f"Generated {len(pairs_df)} model-benchmark pairs")

    # Check if results.csv exists
    results_file = "results.csv"

    if os.path.exists(results_file) and not force_reset:
        logger.info("Loading existing results.csv...")
        existing_df = pl.read_csv(results_file)
        logger.info(f"Loaded {len(existing_df)} existing entries")

        # Join with existing results, keeping existing entries where they match
        results_df = pairs_df.join(
            existing_df,
            on=["model", "pretrained", "task_type", "benchmark"],
            how="left",
        ).with_columns(
            [
                # Fill missing values for new entries
                pl.col("result_file").fill_null(""),
                pl.col("timestamp").fill_null(""),
                pl.col("status").fill_null("pending"),
            ]
        )

        new_count = len(results_df.filter(pl.col("status") == "pending"))
        if new_count > 0:
            logger.info(f"Added {new_count} new entries")
    else:
        # Create new results structure
        results_df = pairs_df.with_columns(
            [
                pl.lit("").alias("result_file"),
                pl.lit("").alias("timestamp"),
                pl.lit("pending").alias("status"),
            ]
        )
        new_count = len(results_df)

    # Write results.csv
    results_df.write_csv(results_file)

    # Count status
    status_counts = results_df.group_by("status").len()
    completed = status_counts.filter(pl.col("status") == "completed")
    pending = status_counts.filter(pl.col("status") == "pending")

    completed_count = completed["len"][0] if len(completed) > 0 else 0
    pending_count = pending["len"][0] if len(pending) > 0 else 0

    logger.info(f"Initialized results.csv with {len(results_df)} total benchmarks")
    logger.info(f"Status: {completed_count} completed, {pending_count} pending")


def sync_r2_results() -> None:
    """Sync R2 bucket contents to local results directory using AWS CLI"""
    logger.info("Syncing R2 bucket contents to local results directory...")

    # Ensure results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # The bucket name from hello-sky.yaml is 'openclip-results'
    bucket_name = "openclip-results"

    try:
        # Check if aws cli is installed
        subprocess.run(["aws", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("AWS CLI is not installed. Please install AWS CLI first.")
        logger.error("Visit: https://aws.amazon.com/cli/")
        sys.exit(1)

    # Sync command for R2 using AWS CLI
    # R2 uses S3-compatible API
    cmd = [
        "aws",
        "s3",
        "sync",
        f"s3://{bucket_name}",
        results_dir,
        "--endpoint-url",
        os.environ.get(
            "R2_ENDPOINT_URL", "https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com"
        ),
    ]

    logger.info(
        f"Syncing R2 bucket '{bucket_name}' to local '{results_dir}' directory..."
    )

    try:
        subprocess.run(cmd, check=True)
        logger.info("R2 sync completed successfully!")

        # Count downloaded files
        json_files = list(Path(results_dir).glob("*.json"))
        logger.info(f"Total JSON files in results directory: {len(json_files)}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error syncing R2 bucket: {e}")
        logger.error("Make sure you have:")
        logger.error(
            "1. Set AWS credentials (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)"
        )
        logger.error("2. Set R2_ENDPOINT_URL environment variable")
        logger.error(
            "   Example: export R2_ENDPOINT_URL=https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com"
        )
        sys.exit(1)


def update_tracking(sync_first: bool = False) -> None:
    """
    Scan results directory and update tracking with completed benchmarks.

    Args:
        sync_first: If True, sync from R2 before updating tracking
    """
    if sync_first:
        sync_r2_results()

    logger.info("Updating tracking from results directory...")

    # Read current results.csv
    if not os.path.exists("results.csv"):
        logger.error("results.csv not found. Run 'init' command first.")
        exit(2)

    results_df = pl.read_csv("results.csv")
    logger.info(f"Loaded {len(results_df)} tracking entries")

    # Scan results directory
    results_dir = Path("results")
    if not results_dir.exists():
        logger.warning("Results directory does not exist")
        return

    json_files = list(results_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in results directory")

    # Parse all filenames and create updates dataframe
    file_updates = []
    unmatched_files = []

    for json_file in json_files:
        parsed = parse_result_filename(json_file.name)
        if parsed:
            file_updates.append(
                {
                    "model": parsed["model"],
                    "pretrained": parsed["pretrained"],
                    "task_type": parsed["task_type"],
                    "benchmark": parsed["benchmark"],
                    "result_file": str(json_file),
                    "timestamp": datetime.fromtimestamp(
                        json_file.stat().st_mtime
                    ).isoformat(),
                    "status": "completed",
                }
            )
        else:
            unmatched_files.append(json_file.name)

    if file_updates:
        # Create updates dataframe
        updates_df = pl.DataFrame(file_updates)

        # Update results with completed benchmarks
        results_df = (
            results_df.join(
                updates_df,
                on=["model", "pretrained", "task_type", "benchmark"],
                how="left",
            )
            .with_columns(
                [
                    # Use right-side values where available, otherwise keep left values
                    pl.coalesce(
                        [pl.col("result_file_right"), pl.col("result_file")]
                    ).alias("result_file"),
                    pl.coalesce([pl.col("timestamp_right"), pl.col("timestamp")]).alias(
                        "timestamp"
                    ),
                    pl.coalesce([pl.col("status_right"), pl.col("status")]).alias(
                        "status"
                    ),
                ]
            )
            .select(
                [
                    "model",
                    "pretrained",
                    "task_type",
                    "benchmark",
                    "result_file",
                    "timestamp",
                    "status",
                ]
            )
        )

        updated_count = len(updates_df)
    else:
        updated_count = 0

    # Write updated results.csv
    results_df.write_csv("results.csv")

    logger.info(f"Updated {updated_count} entries")
    if unmatched_files:
        logger.warning(f"Could not match {len(unmatched_files)} files:")
        for file in unmatched_files[:5]:
            logger.warning(f"  - {file}")
        if len(unmatched_files) > 5:
            logger.warning(f"  ... and {len(unmatched_files) - 5} more")


def parse_results(output_format: str = "separate") -> None:
    """
    Parse completed results and generate analysis-ready CSV files.

    Args:
        output_format: 'separate' for task-specific CSVs, 'merged' for single CSV
    """
    logger.info("Parsing benchmark results...")

    # Read results.csv
    if not os.path.exists("results.csv"):
        logger.error("results.csv not found. Run 'init' command first.")
        exit(2)

    results_df = pl.read_csv("results.csv").filter(pl.col("status") == "completed")
    logger.info(f"Found {len(results_df)} completed results to parse")

    classification_data = []
    retrieval_data = []
    parse_errors = []

    # Process each completed result
    for row in results_df.iter_rows(named=True):
        if not row["result_file"]:
            continue

        # Load JSON
        data, error = safe_load_json(row["result_file"])
        if error:
            parse_errors.append(f"{row['result_file']}: {error}")
            continue

        base_info = {
            "model": row["model"],
            "pretrained": row["pretrained"],
            "benchmark": row["benchmark"],
            "benchmark_run_at": row["timestamp"],  # When the benchmark was completed
        }

        if row["task_type"] == "classification":
            # Extract classification metrics
            metrics = data.get("metrics", {})
            classification_data.append(
                {
                    **base_info,
                    "accuracy": metrics.get("acc1", None),
                    "top5_accuracy": metrics.get("acc5", None),
                    "mean_per_class_recall": metrics.get("mean_per_class_recall", None),
                }
            )

        elif row["task_type"] == "retrieval":
            # Extract retrieval metrics
            metrics = data.get("metrics", {})
            retrieval_data.append(
                {
                    **base_info,
                    "image_retrieval_recall_at_5": metrics.get(
                        "image_retrieval_recall_at_5", None
                    ),
                    "text_retrieval_recall_at_5": metrics.get(
                        "text_retrieval_recall_at_5", None
                    ),
                    "mean_average_precision": metrics.get(
                        "mean_average_precision", None
                    ),
                }
            )

    # Output results using Polars
    if output_format == "separate":
        # Write classification results
        if classification_data:
            classification_df = pl.DataFrame(classification_data)
            classification_df.write_csv("classification_metrics.csv")
            logger.info(
                f"Wrote {len(classification_data)} classification results to classification_metrics.csv"
            )

        # Write retrieval results
        if retrieval_data:
            retrieval_df = pl.DataFrame(retrieval_data)
            retrieval_df.write_csv("retrieval_metrics.csv")
            logger.info(
                f"Wrote {len(retrieval_data)} retrieval results to retrieval_metrics.csv"
            )

    elif output_format == "merged":
        # Create separate DataFrames and then combine
        all_data = []

        if classification_data:
            classification_df = pl.DataFrame(classification_data).with_columns(
                [
                    pl.lit("classification").alias("task_type"),
                    pl.lit(None).alias("image_retrieval_recall_at_5"),
                    pl.lit(None).alias("text_retrieval_recall_at_5"),
                    pl.lit(None).alias("mean_average_precision"),
                ]
            )
            all_data.append(classification_df)

        if retrieval_data:
            retrieval_df = pl.DataFrame(retrieval_data).with_columns(
                [
                    pl.lit("retrieval").alias("task_type"),
                    pl.lit(None).alias("accuracy"),
                    pl.lit(None).alias("top5_accuracy"),
                    pl.lit(None).alias("mean_per_class_recall"),
                ]
            )
            all_data.append(retrieval_df)

        if all_data:
            combined_df = pl.concat(all_data, how="diagonal")
            # Reorder columns
            column_order = [
                "model",
                "pretrained",
                "benchmark",
                "task_type",
                "accuracy",
                "top5_accuracy",
                "mean_per_class_recall",
                "image_retrieval_recall_at_5",
                "text_retrieval_recall_at_5",
                "mean_average_precision",
                "benchmark_run_at",
            ]
            combined_df = combined_df.select(column_order)
            combined_df.write_csv("all_metrics.csv")
            logger.info(f"Wrote {len(combined_df)} combined results to all_metrics.csv")

    if parse_errors:
        logger.warning(f"Failed to parse {len(parse_errors)} result files:")
        for error in parse_errors[:5]:
            logger.warning(f"  - {error}")
        if len(parse_errors) > 5:
            logger.warning(f"  ... and {len(parse_errors) - 5} more")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Benchmark tracking system for OpenCLIP benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                    # Initialize tracking from model pairs
  %(prog)s init --reset           # Reset and recreate tracking file
  %(prog)s sync                   # Sync results from R2 bucket
  %(prog)s update                 # Update tracking from results directory
  %(prog)s update --sync          # Sync from R2 then update tracking
  %(prog)s parse                  # Generate separate analysis CSVs
  %(prog)s parse --format merged  # Generate single combined CSV
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize or update results.csv")
    init_parser.add_argument(
        "--reset", action="store_true", help="Force regenerate from scratch"
    )

    # Update command
    update_parser = subparsers.add_parser(
        "update", help="Update tracking from results directory"
    )
    update_parser.add_argument(
        "--sync", action="store_true", help="Sync from R2 before updating tracking"
    )

    # Sync command
    subparsers.add_parser("sync", help="Sync results from R2 bucket")

    # Parse command
    parse_parser = subparsers.add_parser(
        "parse", help="Generate analysis-ready CSV files"
    )
    parse_parser.add_argument(
        "--format",
        choices=["separate", "merged"],
        default="separate",
        help="Output format (default: separate)",
    )

    # Global options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        exit(1)

    try:
        if args.command == "init":
            init_tracking(force_reset=args.reset)
        elif args.command == "update":
            update_tracking(sync_first=args.sync)
        elif args.command == "sync":
            sync_r2_results()
        elif args.command == "parse":
            parse_results(output_format=args.format)

        logger.info("Command completed successfully")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
