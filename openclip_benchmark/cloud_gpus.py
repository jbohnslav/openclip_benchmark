"""
Cloud GPU availability mapping for SkyPilot.

This module defines which GPUs are available on each cloud provider
based on actual availability data.
"""

# GPU availability per cloud provider
CLOUD_GPU_AVAILABILITY = {
    "lambda": {
        # Lambda doesn't support spot instances
        "gpus": ["A100:1", "H100:1"],
        "supports_spot": False,
    },
    "runpod": {
        # RunPod has many GPU options with spot support
        "gpus": [
            "A100-80GB:1",
            "A100-80GB-SXM:1",
            "H100:1",
            "H100-SXM:1",
            "H200-SXM:1",
            "L40S:1",
            "L40:1",
            "L4:1",
            # High VRAM consumer GPUs
            "RTX4090:1",  # 24GB VRAM
            "RTX6000-Ada:1",  # 48GB VRAM
            "RTXA6000:1",  # 48GB VRAM
        ],
        "supports_spot": True,
    },
    "nebius": {
        # Nebius doesn't support spot instances
        "gpus": ["H100:1", "H200:1", "L40S:1"],
        "supports_spot": False,
    },
}

# Minimum VRAM requirements for filtering
MIN_VRAM_GB = 24  # Minimum VRAM for OpenCLIP models

# GPU VRAM mapping (in GB)
GPU_VRAM = {
    "A100": 40,
    "A100-80GB": 80,
    "A100-80GB-SXM": 80,
    "H100": 80,
    "H100-SXM": 80,
    "H100-NVL": 94,
    "H200": 141,
    "H200-SXM": 141,
    "L40S": 48,
    "L40": 48,
    "L4": 24,
    "RTX4090": 24,
    "RTX6000-Ada": 48,
    "RTXA6000": 48,
    "RTXA5000": 24,
    "RTXA4500": 20,  # Below threshold
    "RTXA4000": 16,  # Below threshold
    "RTX3090": 24,
}


def get_gpu_vram(gpu_spec: str) -> int:
    """Extract VRAM from GPU specification."""
    gpu_name = gpu_spec.split(":")[0]
    return GPU_VRAM.get(gpu_name, 0)


def generate_gpu_configs(min_vram_gb: int = MIN_VRAM_GB) -> list[dict]:
    """
    Generate all GPU configurations for the any_of structure.

    Args:
        min_vram_gb: Minimum VRAM requirement in GB

    Returns:
        List of GPU configurations for any_of structure
    """
    configs = []

    for cloud, info in CLOUD_GPU_AVAILABILITY.items():
        gpus = info["gpus"]
        supports_spot = info["supports_spot"]

        # Filter GPUs by VRAM requirement
        valid_gpus = [gpu for gpu in gpus if get_gpu_vram(gpu) >= min_vram_gb]

        for gpu in valid_gpus:
            if supports_spot:
                # Add both spot and on-demand options
                configs.append({"accelerators": gpu, "infra": cloud, "use_spot": True})
                configs.append({"accelerators": gpu, "infra": cloud, "use_spot": False})
            else:
                # Only on-demand option
                configs.append({"accelerators": gpu, "infra": cloud})

    return configs


def generate_yaml_any_of_structure(min_vram_gb: int = MIN_VRAM_GB) -> str:
    """
    Generate YAML-formatted any_of structure for GPU configurations.

    Args:
        min_vram_gb: Minimum VRAM requirement in GB

    Returns:
        YAML-formatted string for any_of structure
    """
    configs = generate_gpu_configs(min_vram_gb)

    yaml_lines = ["  any_of:"]

    # Group by cloud for better readability
    for cloud in ["lambda", "runpod", "nebius"]:
        cloud_configs = [c for c in configs if c["infra"] == cloud]
        if not cloud_configs:
            continue

        cloud_info = CLOUD_GPU_AVAILABILITY[cloud]
        if cloud_info["supports_spot"]:
            yaml_lines.append(f"    # {cloud.capitalize()}")
        else:
            yaml_lines.append(f"    # {cloud.capitalize()} (no spot support)")

        for config in cloud_configs:
            gpu = config["accelerators"]
            vram = get_gpu_vram(gpu)

            if "use_spot" in config:
                spot_str = (
                    ", use_spot: true" if config["use_spot"] else ", use_spot: false"
                )
                yaml_lines.append(
                    f"    - {{accelerators: {gpu:<15} infra: {cloud}{spot_str}}}  # {vram}GB"
                )
            else:
                yaml_lines.append(
                    f"    - {{accelerators: {gpu:<15} infra: {cloud}}}  # {vram}GB"
                )

        yaml_lines.append("")  # Empty line between clouds

    return "\n".join(yaml_lines).rstrip()


if __name__ == "__main__":
    # Generate and print the YAML structure
    print(generate_yaml_any_of_structure())
