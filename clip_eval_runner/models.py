"""Model-related utilities for OpenCLIP benchmarking.

This module provides functions for retrieving OpenCLIP models, generating
model-benchmark pairs, and estimating GPU requirements for different model sizes.
"""

from typing import Any

import polars as pl


def get_openclip_models() -> list[tuple[str, str]]:
    """Get all available pretrained OpenCLIP models.

    Returns:
        List of tuples containing (model_name, pretrained_name) pairs.

    Example:
        >>> models = get_openclip_models()
        >>> print(models[:3])
        [('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ...]
    """
    import open_clip

    pretrained = open_clip.list_pretrained()
    return [(model, pretrained_name) for model, pretrained_name in pretrained]


def generate_pairs_dataframe() -> pl.DataFrame:
    """Generate DataFrame with all model-benchmark combinations.

    This function creates a comprehensive DataFrame containing all possible
    combinations of OpenCLIP models and benchmark datasets for both
    zero-shot classification and retrieval tasks.

    Returns:
        Polars DataFrame with columns: model, pretrained, task_type, benchmark

    Example:
        >>> df = generate_pairs_dataframe()
        >>> print(df.shape)
        (15420, 4)  # Approximate number of combinations
        >>> print(df.columns)
        ['model', 'pretrained', 'task_type', 'benchmark']
    """
    from .datasets import get_benchmark_datasets

    models = get_openclip_models()
    classification_benchmarks, retrieval_benchmarks = get_benchmark_datasets()

    all_benchmarks = [
        ("zeroshot_classification", bench) for bench in classification_benchmarks
    ] + [("zeroshot_retrieval", bench) for bench in retrieval_benchmarks]

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


def get_model_metadata() -> dict[str, dict[str, Any]]:
    """Get metadata about OpenCLIP models including estimated GPU requirements.

    This function returns a mapping of model architectures to their estimated
    GPU memory requirements, parameter counts, and recommended instance types.

    Returns:
        Dictionary mapping model names to metadata dictionaries containing:
        - gpu_memory_gb: Estimated GPU memory requirement in GB
        - parameters_millions: Approximate parameter count in millions
        - recommended_gpu: Recommended GPU type for optimal performance
        - batch_size_hint: Suggested batch size for inference

    Example:
        >>> metadata = get_model_metadata()
        >>> print(metadata['ViT-B-32'])
        {
            'gpu_memory_gb': 4,
            'parameters_millions': 88,
            'recommended_gpu': 'T4',
            'batch_size_hint': 512
        }
    """
    return {
        # Vision Transformer Base models
        "ViT-B-32": {
            "gpu_memory_gb": 4,
            "parameters_millions": 88,
            "recommended_gpu": "T4",
            "batch_size_hint": 512,
            "architecture_family": "ViT-Base",
        },
        "ViT-B-16": {
            "gpu_memory_gb": 6,
            "parameters_millions": 86,
            "recommended_gpu": "T4",
            "batch_size_hint": 256,
            "architecture_family": "ViT-Base",
        },
        # Vision Transformer Large models
        "ViT-L-14": {
            "gpu_memory_gb": 12,
            "parameters_millions": 304,
            "recommended_gpu": "V100",
            "batch_size_hint": 128,
            "architecture_family": "ViT-Large",
        },
        "ViT-L-14-336": {
            "gpu_memory_gb": 16,
            "parameters_millions": 304,
            "recommended_gpu": "A100",
            "batch_size_hint": 64,
            "architecture_family": "ViT-Large",
        },
        # Vision Transformer Huge models
        "ViT-H-14": {
            "gpu_memory_gb": 24,
            "parameters_millions": 632,
            "recommended_gpu": "A100",
            "batch_size_hint": 32,
            "architecture_family": "ViT-Huge",
        },
        # Vision Transformer Giant models
        "ViT-g-14": {
            "gpu_memory_gb": 32,
            "parameters_millions": 1012,
            "recommended_gpu": "A100-80GB",
            "batch_size_hint": 16,
            "architecture_family": "ViT-Giant",
        },
        "ViT-G-14": {
            "gpu_memory_gb": 32,
            "parameters_millions": 1012,
            "recommended_gpu": "A100-80GB",
            "batch_size_hint": 16,
            "architecture_family": "ViT-Giant",
        },
        # ResNet models
        "RN50": {
            "gpu_memory_gb": 3,
            "parameters_millions": 38,
            "recommended_gpu": "T4",
            "batch_size_hint": 512,
            "architecture_family": "ResNet",
        },
        "RN101": {
            "gpu_memory_gb": 4,
            "parameters_millions": 57,
            "recommended_gpu": "T4",
            "batch_size_hint": 384,
            "architecture_family": "ResNet",
        },
        "RN50x4": {
            "gpu_memory_gb": 8,
            "parameters_millions": 87,
            "recommended_gpu": "V100",
            "batch_size_hint": 256,
            "architecture_family": "ResNet",
        },
        "RN50x16": {
            "gpu_memory_gb": 16,
            "parameters_millions": 168,
            "recommended_gpu": "A100",
            "batch_size_hint": 128,
            "architecture_family": "ResNet",
        },
        "RN50x64": {
            "gpu_memory_gb": 32,
            "parameters_millions": 623,
            "recommended_gpu": "A100-80GB",
            "batch_size_hint": 64,
            "architecture_family": "ResNet",
        },
        # ConvNeXT models
        "convnext_base": {
            "gpu_memory_gb": 8,
            "parameters_millions": 89,
            "recommended_gpu": "V100",
            "batch_size_hint": 256,
            "architecture_family": "ConvNeXT",
        },
        "convnext_base_w": {
            "gpu_memory_gb": 8,
            "parameters_millions": 89,
            "recommended_gpu": "V100",
            "batch_size_hint": 256,
            "architecture_family": "ConvNeXT",
        },
        "convnext_base_w_320": {
            "gpu_memory_gb": 12,
            "parameters_millions": 89,
            "recommended_gpu": "V100",
            "batch_size_hint": 128,
            "architecture_family": "ConvNeXT",
        },
        "convnext_large_d": {
            "gpu_memory_gb": 16,
            "parameters_millions": 198,
            "recommended_gpu": "A100",
            "batch_size_hint": 128,
            "architecture_family": "ConvNeXT",
        },
        "convnext_large_d_320": {
            "gpu_memory_gb": 20,
            "parameters_millions": 198,
            "recommended_gpu": "A100",
            "batch_size_hint": 64,
            "architecture_family": "ConvNeXT",
        },
        "convnext_xxlarge": {
            "gpu_memory_gb": 40,
            "parameters_millions": 846,
            "recommended_gpu": "A100-80GB",
            "batch_size_hint": 32,
            "architecture_family": "ConvNeXT",
        },
        # EVA models
        "EVA01-g-14": {
            "gpu_memory_gb": 32,
            "parameters_millions": 1012,
            "recommended_gpu": "A100-80GB",
            "batch_size_hint": 16,
            "architecture_family": "EVA",
        },
        "EVA01-g-14-plus": {
            "gpu_memory_gb": 40,
            "parameters_millions": 1012,
            "recommended_gpu": "A100-80GB",
            "batch_size_hint": 16,
            "architecture_family": "EVA",
        },
        "EVA02-B-16": {
            "gpu_memory_gb": 8,
            "parameters_millions": 86,
            "recommended_gpu": "V100",
            "batch_size_hint": 256,
            "architecture_family": "EVA",
        },
        "EVA02-L-14": {
            "gpu_memory_gb": 16,
            "parameters_millions": 304,
            "recommended_gpu": "A100",
            "batch_size_hint": 64,
            "architecture_family": "EVA",
        },
        "EVA02-L-14-336": {
            "gpu_memory_gb": 20,
            "parameters_millions": 304,
            "recommended_gpu": "A100",
            "batch_size_hint": 32,
            "architecture_family": "EVA",
        },
    }


def get_model_gpu_requirement(model_name: str) -> dict[str, Any]:
    """Get GPU requirements for a specific model.

    Args:
        model_name: Name of the OpenCLIP model

    Returns:
        Dictionary with GPU requirements, or default values if model not found

    Example:
        >>> req = get_model_gpu_requirement('ViT-L-14')
        >>> print(req['recommended_gpu'])
        'V100'
        >>> print(req['gpu_memory_gb'])
        12
    """
    metadata = get_model_metadata()

    if model_name in metadata:
        return metadata[model_name]

    # Fallback logic for unknown models based on naming patterns
    if "ViT-g-" in model_name or "ViT-G-" in model_name or "EVA01-g-" in model_name:
        return {
            "gpu_memory_gb": 32,
            "parameters_millions": 1000,
            "recommended_gpu": "A100-80GB",
            "batch_size_hint": 16,
            "architecture_family": "Unknown-Giant",
        }
    elif "ViT-H-" in model_name or "xxlarge" in model_name:
        return {
            "gpu_memory_gb": 24,
            "parameters_millions": 600,
            "recommended_gpu": "A100",
            "batch_size_hint": 32,
            "architecture_family": "Unknown-Huge",
        }
    elif "ViT-L-" in model_name or "large" in model_name or "EVA02-L-" in model_name:
        return {
            "gpu_memory_gb": 12,
            "parameters_millions": 300,
            "recommended_gpu": "V100",
            "batch_size_hint": 128,
            "architecture_family": "Unknown-Large",
        }
    elif "ViT-B-" in model_name or "base" in model_name or "EVA02-B-" in model_name:
        return {
            "gpu_memory_gb": 6,
            "parameters_millions": 90,
            "recommended_gpu": "T4",
            "batch_size_hint": 256,
            "architecture_family": "Unknown-Base",
        }
    else:
        # Default for unknown architectures
        return {
            "gpu_memory_gb": 8,
            "parameters_millions": 100,
            "recommended_gpu": "V100",
            "batch_size_hint": 128,
            "architecture_family": "Unknown",
        }


def estimate_job_runtime(model_name: str, task_type: str, benchmark: str) -> int:
    """Estimate runtime in minutes for a model-benchmark combination.

    Args:
        model_name: Name of the OpenCLIP model
        task_type: Type of task ('zeroshot_classification' or 'zeroshot_retrieval')
        benchmark: Name of the benchmark dataset

    Returns:
        Estimated runtime in minutes

    Example:
        >>> runtime = estimate_job_runtime('ViT-L-14', 'zeroshot_classification', 'imagenet1k')
        >>> print(f"Estimated runtime: {runtime} minutes")
        Estimated runtime: 45 minutes
    """
    model_req = get_model_gpu_requirement(model_name)

    # Base runtime factors
    base_runtime = 20  # Base runtime in minutes

    # Model size factor (larger models take longer)
    if model_req["architecture_family"].endswith("Giant"):
        model_factor = 3.0
    elif model_req["architecture_family"].endswith("Huge"):
        model_factor = 2.5
    elif model_req["architecture_family"].endswith("Large"):
        model_factor = 2.0
    elif model_req["architecture_family"].endswith("Base"):
        model_factor = 1.0
    else:
        model_factor = 1.5

    # Task type factor
    task_factor = 1.5 if task_type == "zeroshot_retrieval" else 1.0

    # Dataset size factor (rough estimates)
    dataset_factors = {
        "imagenet1k": 2.0,
        "mscoco_captions": 1.8,
        "flickr30k": 1.2,
        "cifar10": 0.3,
        "cifar100": 0.4,
        "caltech101": 0.5,
        "dtd": 0.4,
        "eurosat": 0.3,
        "flowers": 0.3,
        "food101": 1.0,
        "pets": 0.4,
        "pcam": 0.8,
        "resisc45": 0.6,
    }

    dataset_factor = dataset_factors.get(benchmark, 1.0)

    estimated_minutes = int(base_runtime * model_factor * task_factor * dataset_factor)

    # Ensure minimum runtime and reasonable maximum
    return max(10, min(estimated_minutes, 180))
