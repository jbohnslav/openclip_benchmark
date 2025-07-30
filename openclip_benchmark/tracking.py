"""
Tracking functions for OpenCLIP benchmarking.

This module provides functions for initializing benchmark tracking, syncing results
from R2 storage, updating tracking status, and parsing benchmark results into
analysis-ready formats.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

from .config import (
    R2_BUCKET_NAME,
    RESULTS_CSV,
    RESULTS_DIR,
)
from .models import generate_pairs_dataframe
from .utils import parse_result_filename, safe_load_json

# Configure logging
logger = logging.getLogger(__name__)


def init_tracking(force_reset: bool = False) -> None:
    """
    Initialize results.csv by generating all model-benchmark pairs.

    Args:
        force_reset: If True, recreate results.csv from scratch
    """
    logger.info("Initializing benchmark tracking...")

    # Generate model-benchmark pairs
    logger.info("Generating model-benchmark pairs...")
    pairs_df = generate_pairs_dataframe()
    logger.info(f"Generated {len(pairs_df)} model-benchmark pairs")

    # Check if results.csv exists
    if os.path.exists(RESULTS_CSV) and not force_reset:
        logger.info("Loading existing results.csv...")
        existing_df = pl.read_csv(RESULTS_CSV)
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
    results_df.write_csv(RESULTS_CSV)

    # Count status
    status_counts = results_df.group_by("status").len()
    completed = status_counts.filter(pl.col("status") == "completed")
    pending = status_counts.filter(pl.col("status") == "pending")

    completed_count = completed["len"][0] if len(completed) > 0 else 0
    pending_count = pending["len"][0] if len(pending) > 0 else 0

    logger.info(f"Initialized {RESULTS_CSV} with {len(results_df)} total benchmarks")
    logger.info(f"Status: {completed_count} completed, {pending_count} pending")


def sync_r2_results() -> None:
    """Sync R2 bucket contents to local results directory using AWS CLI."""
    logger.info("Syncing R2 bucket contents to local results directory...")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
        f"s3://{R2_BUCKET_NAME}",
        RESULTS_DIR,
        "--endpoint-url",
        os.environ.get(
            "R2_ENDPOINT_URL", "https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com"
        ),
        "--profile", 
        "r2",
    ]

    logger.info(
        f"Syncing R2 bucket '{R2_BUCKET_NAME}' to local '{RESULTS_DIR}' directory..."
    )

    try:
        subprocess.run(cmd, check=True)
        logger.info("R2 sync completed successfully!")

        # Count downloaded files
        json_files = list(Path(RESULTS_DIR).glob("*.json"))
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
    if not os.path.exists(RESULTS_CSV):
        logger.error(f"{RESULTS_CSV} not found. Run 'init' command first.")
        sys.exit(2)

    results_df = pl.read_csv(RESULTS_CSV)
    logger.info(f"Loaded {len(results_df)} tracking entries")

    # Scan results directory
    results_dir = Path(RESULTS_DIR)
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
    results_df.write_csv(RESULTS_CSV)

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
    if not os.path.exists(RESULTS_CSV):
        logger.error(f"{RESULTS_CSV} not found. Run 'init' command first.")
        sys.exit(2)

    results_df = pl.read_csv(RESULTS_CSV).filter(pl.col("status") == "completed")
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

        if row["task_type"] == "zeroshot_classification":
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

        elif row["task_type"] == "zeroshot_retrieval":
            # Extract retrieval metrics
            metrics = data.get("metrics", {})
            retrieval_data.append(
                {
                    **base_info,
                    "image_retrieval_recall_at_5": metrics.get(
                        "image_retrieval_recall@5", None
                    ),
                    "text_retrieval_recall_at_5": metrics.get(
                        "text_retrieval_recall@5", None
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
                    pl.lit("zeroshot_classification").alias("task_type"),
                    pl.lit(None).alias("image_retrieval_recall_at_5"),
                    pl.lit(None).alias("text_retrieval_recall_at_5"),
                    pl.lit(None).alias("mean_average_precision"),
                ]
            )
            all_data.append(classification_df)

        if retrieval_data:
            retrieval_df = pl.DataFrame(retrieval_data).with_columns(
                [
                    pl.lit("zeroshot_retrieval").alias("task_type"),
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
