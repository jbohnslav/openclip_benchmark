#!/usr/bin/env python3
import polars as pl

# Load the CSV
df = pl.read_csv("results.csv")

print(f"Total benchmarks: {len(df)}")
print(f"\nColumns: {df.columns}")

# Count unique values
print(f"\nUnique models: {df['model'].n_unique()}")
print(f"Unique pretrained: {df['pretrained'].n_unique()}")
print(
    f"Unique (model, pretrained) pairs: {df.group_by(['model', 'pretrained']).len().height}"
)
print(f"Unique task types: {df['task_type'].n_unique()}")
print(f"Unique benchmarks: {df['benchmark'].n_unique()}")

# Group by model-pretrained and count benchmarks
model_pretrained_counts = (
    df.group_by(["model", "pretrained"]).len().sort("len", descending=True)
)

counts_series = model_pretrained_counts["len"]
print("\nBenchmarks per (model, pretrained) pair:")
print(f"Min: {counts_series.min()}")
print(f"Max: {counts_series.max()}")
print(f"Mean: {counts_series.mean():.1f}")
print(f"Median: {counts_series.median():.1f}")

# Show distribution
print("\nTop 10 (model, pretrained) pairs by benchmark count:")
for row in model_pretrained_counts.head(10).iter_rows():
    print(f"  {row[0]}, {row[1]}: {row[2]} benchmarks")

print("\nBottom 10 (model, pretrained) pairs by benchmark count:")
for row in model_pretrained_counts.tail(10).iter_rows():
    print(f"  {row[0]}, {row[1]}: {row[2]} benchmarks")

# Analyze task type distribution
print("\nTask type distribution:")
task_counts = df["task_type"].value_counts()
total = len(df)
for row in task_counts.iter_rows():
    task, count = row
    print(f"  {task}: {count} ({count / total * 100:.1f}%)")

# Calculate optimal batch sizes
num_groups = model_pretrained_counts.height
mean_benchmarks = counts_series.mean()
print("\nGrouping recommendations:")
print(f"- Total (model, pretrained) groups: {num_groups}")
print("- If we process one group per node:")
print(f"  - Average benchmarks per node: {mean_benchmarks:.1f}")
print(f"  - At ~2 min per benchmark: ~{mean_benchmarks * 2:.0f} minutes per node")

# Show some example groups
print("\nExample groups with ~50 benchmarks each:")
target = 50
close_to_target = model_pretrained_counts.filter(
    (pl.col("len") - target).abs() < 5
).head(5)
for row in close_to_target.iter_rows():
    print(f"  {row[0]}, {row[1]}: {row[2]} benchmarks")
