#!/usr/bin/env python3
"""
Single-file benchmark tracking system for OpenCLIP benchmarks.
Tracks completion status and parses results.
"""

import argparse
import logging

from openclip_benchmark.tracking import (
    init_tracking,
    parse_results,
    sync_r2_results,
    update_tracking,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark_tracker.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
