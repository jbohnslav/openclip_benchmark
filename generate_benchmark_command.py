#!/usr/bin/env python3
"""
Generate CLIP benchmark commands using the dataset mapping.
"""

import argparse

from benchmark_tracker import get_dataset_config


def generate_command(model: str, pretrained: str, dataset: str, task_type: str) -> str:
    """Generate a clip_benchmark command using the dataset mapping"""

    config = get_dataset_config(dataset, task_type)

    cmd_parts = [
        "uv run clip_benchmark eval",
        f"--pretrained_model {model},{pretrained}",
        f"--dataset {config['webdataset_name']}",
    ]

    # Add dataset_root if not default
    if config["dataset_root"] != "root":
        cmd_parts.append(f'--dataset_root "{config["dataset_root"]}"')

    cmd_parts.extend(
        [
            f"--task {config['task']}",
            '--output "/openclip_results/{dataset}_{pretrained}_{model}_{language}_{task}.json"',
        ]
    )

    return " \\\n    ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP benchmark commands")
    parser.add_argument("--model", default="ViT-B-32", help="Model architecture")
    parser.add_argument("--pretrained", default="openai", help="Pretrained weights")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--task",
        choices=["zeroshot_classification", "zeroshot_retrieval"],
        required=True,
        help="Task type",
    )

    args = parser.parse_args()

    command = generate_command(args.model, args.pretrained, args.dataset, args.task)
    print(command)


if __name__ == "__main__":
    main()
