#!/usr/bin/env python3
"""Test script to debug SkyPilot environment variable handling"""

import os

import sky

# Set test environment variables
os.environ["TEST_VAR1"] = "value1"
os.environ["TEST_VAR2"] = "value2"

# Create task from YAML
task = sky.Task.from_yaml("configs/grouped_job_template.yaml")

print("Task loaded from YAML")
print(f"Task envs attribute exists: {hasattr(task, 'envs')}")
print(f"Task envs value: {task.envs if hasattr(task, 'envs') else 'No envs attribute'}")

# Try updating envs
task.update_envs({"CUSTOM_VAR": "custom_value"})

print("\nAfter update_envs:")
print(f"Task envs: {task.envs if hasattr(task, 'envs') else 'No envs attribute'}")

# Check the YAML structure
print("\nChecking YAML template for env vars...")
with open("configs/grouped_job_template.yaml") as f:
    import yaml

    config = yaml.safe_load(f)
    if "envs" in config:
        print("Template envs section:")
        for k, v in config["envs"].items():
            print(f"  {k}: {v}")
