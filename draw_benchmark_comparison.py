# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
"""
Benchmark visualization script for comparing old vs new serialization performance.
Generates comparison charts from bench-0.15.json (before) and bench-0.15-new.json (after).
"""
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'


def load_benchmark_data(json_path: str, label: str) -> pd.DataFrame:
    """Load benchmark JSON and convert to DataFrame."""
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    rows = []
    for item in raw_data:
        config = item['config']
        for perf in item['performance_data']:
            rows.append({
                "Version": label,
                "Config": config,
                "Data Volume": perf['data_volume'],
                "Payload (GB)": perf['payload_gb'],
                "Operation": perf['operation'],
                "Mean Gbps": perf['stats_gbps']['mean'],
                "Max Gbps": perf['stats_gbps']['max'],
                "Min Gbps": perf['stats_gbps']['min'],
                "P99 Gbps": perf['stats_gbps']['p99'],
            })

    return pd.DataFrame(rows)


def plot_throughput_comparison(df: pd.DataFrame):
    """
    Chart 1: Bar chart comparing PUT/GET throughput between versions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    operations = ["PUT", "GET"]
    colors = {"Before (v0.15)": "#3498db", "After (Optimized)": "#e74c3c"}
    
    for idx, op in enumerate(operations):
        subset = df[df["Operation"] == op].copy()
        
        # Create grouped bar chart
        pivot = subset.pivot(index="Config", columns="Version", values="Mean Gbps")
        pivot = pivot.reindex(["debug", "tiny", "small", "medium", "large", "xlarge", "huge"])
        
        x = np.arange(len(pivot.index))
        width = 0.35
        
        ax = axes[idx]
        bars1 = ax.bar(x - width/2, pivot["Before (v0.15)"], width, 
                       label="Before (v0.15)", color=colors["Before (v0.15)"], alpha=0.8)
        bars2 = ax.bar(x + width/2, pivot["After (Optimized)"], width,
                       label="After (Optimized)", color=colors["After (Optimized)"], alpha=0.8)
        
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Throughput (Gbps)")
        ax.set_title(f"{op} Operation Throughput")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
        
        # Add speedup annotations
        for i, (before, after) in enumerate(zip(pivot["Before (v0.15)"], pivot["After (Optimized)"])):
            if pd.notna(before) and pd.notna(after) and before > 0:
                speedup = after / before
                color = "green" if speedup > 1 else "red"
                ax.annotate(f'{speedup:.1f}x', 
                           xy=(i + width/2, after),
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           color=color)
    
    plt.suptitle("Serialization Optimization: Before vs After", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("benchmark_comparison.png", bbox_inches='tight', dpi=300)
    print("Generated: benchmark_comparison.png")


def plot_speedup_chart(df: pd.DataFrame):
    """
    Chart 2: Speedup factor visualization.
    """
    # Pivot data to calculate speedup
    pivot = df.pivot_table(index=["Config", "Operation"], columns="Version", values="Mean Gbps").reset_index()
    pivot["Speedup"] = pivot["After (Optimized)"] / pivot["Before (v0.15)"]
    
    # Order configs properly
    config_order = ["debug", "tiny", "small", "medium", "large", "xlarge", "huge"]
    pivot["Config"] = pd.Categorical(pivot["Config"], categories=config_order, ordered=True)
    pivot = pivot.sort_values("Config")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create grouped bar chart
    pivot_wide = pivot.pivot(index="Config", columns="Operation", values="Speedup")
    
    x = np.arange(len(pivot_wide.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pivot_wide["PUT"], width, label="PUT", color="#2ecc71", alpha=0.8)
    bars2 = ax.bar(x + width/2, pivot_wide["GET"], width, label="GET", color="#9b59b6", alpha=0.8)
    
    # Add reference line at 1x
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Baseline (1x)')
    
    ax.set_xlabel("Configuration (Data Size)")
    ax.set_ylabel("Speedup Factor (x)")
    ax.set_title("Performance Speedup: New vs Old Serialization", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}" for c in pivot_wide.index], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', ls="--", linewidth=0.5, alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("benchmark_speedup.png", bbox_inches='tight', dpi=300)
    print("Generated: benchmark_speedup.png")


def generate_comparison_table(df: pd.DataFrame) -> str:
    """Generate markdown table for PR message."""
    pivot = df.pivot_table(index=["Config", "Operation"], columns="Version", values="Mean Gbps").reset_index()
    pivot["Speedup"] = pivot["After (Optimized)"] / pivot["Before (v0.15)"]
    
    config_order = ["debug", "tiny", "small", "medium", "large", "xlarge", "huge"]
    pivot["Config"] = pd.Categorical(pivot["Config"], categories=config_order, ordered=True)
    pivot = pivot.sort_values(["Config", "Operation"])
    
    # Data volume mapping
    volume_map = {
        "debug": "0.05 MB",
        "tiny": "0.62 MB", 
        "small": "50 MB",
        "medium": "500 MB",
        "large": "2.9 GB",
        "xlarge": "5.9 GB",
        "huge": "9.8 GB"
    }
    
    print("\n### PUT Operation\n")
    print("| Data Size | Before (Gbps) | After (Gbps) | Speedup |")
    print("|-----------|---------------|--------------|---------|")
    for _, row in pivot[pivot["Operation"] == "PUT"].iterrows():
        print(f"| {volume_map[row['Config']]} | {row['Before (v0.15)']:.2f} | {row['After (Optimized)']:.2f} | **{row['Speedup']:.2f}x** |")
    
    print("\n### GET Operation\n")
    print("| Data Size | Before (Gbps) | After (Gbps) | Speedup |")
    print("|-----------|---------------|--------------|---------|")
    for _, row in pivot[pivot["Operation"] == "GET"].iterrows():
        print(f"| {volume_map[row['Config']]} | {row['Before (v0.15)']:.2f} | {row['After (Optimized)']:.2f} | **{row['Speedup']:.2f}x** |")


if __name__ == "__main__":
    try:
        # Load both benchmark files
        df_before = load_benchmark_data("bench-0.15.json", "Before (v0.15)")
        df_after = load_benchmark_data("bench-0.15-new.json", "After (Optimized)")
        
        # Combine data
        df = pd.concat([df_before, df_after], ignore_index=True)
        
        # Generate visualizations
        plot_throughput_comparison(df)
        plot_speedup_chart(df)
        
        # Print table for PR message
        generate_comparison_table(df)
        
        print("\n✅ Success. Check the .png files in the current directory.")

    except FileNotFoundError as e:
        print(f"Error: Benchmark file not found - {e}")
    except Exception as e:
        print(f"Plotting Error: {e}")
        import traceback
        traceback.print_exc()
