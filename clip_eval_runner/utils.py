"""Shared utilities for OpenCLIP benchmarking."""

import json
import logging


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


def parse_result_filename(filename: str) -> dict[str, str] | None:
    """
    Parse result filename to extract benchmark components.

    Expected format: {benchmark}_{pretrained}_{model}_{lang}_zeroshot_{task_type}.json
    Example: cifar10_openai_ViT-B-32_en_zeroshot_classification.json

    Args:
        filename: The result filename to parse

    Returns:
        Dictionary with parsed components or None if parsing fails
    """
    # Remove .json extension
    name = filename.replace(".json", "")

    # Split by underscores and try to identify components
    parts = name.split("_")

    if len(parts) < 5:
        return None

    # Extract task type (zeroshot_classification or zeroshot_retrieval)
    task_type = None
    if "classification" in parts:
        task_type = "zeroshot_classification"
    elif "retrieval" in parts:
        task_type = "zeroshot_retrieval"
    else:
        return None

    # Find the benchmark - handle multi-part benchmark names
    benchmark = parts[0]

    # Check for multi-part benchmark names (like mscoco_captions)
    if len(parts) > 1 and parts[1] in ["captions"]:
        benchmark = f"{parts[0]}_{parts[1]}"
        # Remove the consumed part
        parts = [benchmark] + parts[2:]

    # Keep the original benchmark name for retrieval tasks
    # The mapping system will handle webdataset format internally

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
            return None

    return {
        "benchmark": benchmark,
        "pretrained": pretrained,
        "model": model,
        "task_type": task_type,
    }


def setup_logging(
    log_file: str = "benchmark.log", verbose: bool = False
) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def estimate_job_cost(
    gpu_type: str, duration_hours: float, use_spot: bool = True
) -> float:
    """Estimate cost for a job based on GPU type and duration."""
    # Basic cost estimation - can be expanded
    gpu_costs = {
        "A100:1": 3.0,  # USD per hour on-demand
        "V100:1": 2.0,
        "T4:1": 0.5,
    }

    base_cost = gpu_costs.get(gpu_type, 2.0) * duration_hours
    if use_spot:
        base_cost *= 0.3  # Spot instances are ~70% cheaper

    return base_cost
