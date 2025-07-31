"""
Dataset-related functions for OpenCLIP benchmarking.

This module provides functions to load dataset configurations and mappings
for both classification and retrieval benchmarks.
"""

import json
from pathlib import Path

import clip_benchmark.cli


def load_dataset_mapping() -> dict:
    """
    Load dataset mapping from JSON file.

    Returns:
        dict: Dataset mapping containing classification and retrieval datasets

    Raises:
        FileNotFoundError: If dataset_mapping.json is not found
        json.JSONDecodeError: If the JSON file is malformed
    """
    mapping_file = Path(__file__).parent.parent / "dataset_mapping.json"
    with open(mapping_file) as f:
        return json.load(f)


def get_dataset_config(dataset_name: str, task_type: str) -> dict:
    """
    Get dataset configuration from mapping.

    Args:
        dataset_name: Name of the dataset
        task_type: Type of task ("zeroshot_retrieval" or other)

    Returns:
        dict: Dataset configuration with keys like webdataset_name, dataset_root, task
    """
    dataset_mapping = load_dataset_mapping()

    if task_type == "zeroshot_retrieval":
        return dataset_mapping["retrieval_datasets"].get(
            dataset_name,
            {
                "webdataset_name": dataset_name,
                "dataset_root": "root",
                "task": task_type,
            },
        )
    else:
        return dataset_mapping["classification_datasets"].get(
            dataset_name,
            {
                "webdataset_name": dataset_name,
                "dataset_root": "root",
                "task": task_type,
            },
        )


def get_benchmark_datasets() -> tuple[list[str], list[str]]:
    """
    Get zero-shot classification and retrieval benchmarks from CLIP Benchmark.

    Returns:
        tuple: (classification_benchmarks, retrieval_benchmarks)
            - classification_benchmarks: List of classification dataset names
            - retrieval_benchmarks: List of retrieval dataset names
    """
    # Load dataset mapping
    dataset_mapping = load_dataset_mapping()

    # Get all classification benchmarks from vtab+ collection
    classification_benchmarks = clip_benchmark.cli.dataset_collection["vtab+"]

    # Get retrieval benchmarks from mapping (use the original names, not webdataset names)
    retrieval_benchmarks = list(dataset_mapping["retrieval_datasets"].keys())

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

    return classification_benchmarks, retrieval_benchmarks
