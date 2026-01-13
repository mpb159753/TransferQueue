# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'


def load_and_process_data(json_path):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    rows = []
    for item in raw_data:
        cfg = item['config']
        total_elements = cfg['list_size'] * cfg['tensor_size']

        rows.append({
            "List Size": cfg['list_size'],
            "Tensor Size": cfg['tensor_size'],
            "Is Pad": "Yes" if cfg['is_pad'] else "No",
            "Input Type": "2D Tensor" if cfg['type'] == "tensor" else "List[Tensor]",
            "Put Latency (ms)": item['put_lat'][0],
            "Get Latency (ms)": item['get_lat'][0],
            "Put CPU (ms)": item['put_cpu'][0],
            "Get CPU (ms)": item['get_cpu'][0],
            "Memory (MB)": item['mem'][0],
            "Total Elements": total_elements,
            "Config Label": f"L={cfg['list_size']}, T={cfg['tensor_size']}"
        })

    return pd.DataFrame(rows)


def plot_latency_log_scale(df):
    """
    Chart 1: Latency Analysis (Log Scale)
    """
    df_melt = df.melt(
        id_vars=["List Size", "Tensor Size", "Input Type", "Is Pad"],
        value_vars=["Put Latency (ms)", "Get Latency (ms)"],
        var_name="Operation", value_name="Latency (ms)"
    )

    # 简化 Operation 标签
    df_melt["Operation"] = df_melt["Operation"].replace({
        "Put Latency (ms)": "Put",
        "Get Latency (ms)": "Get"
    })

    g = sns.relplot(
        data=df_melt,
        x="Tensor Size", y="Latency (ms)",
        hue="Input Type", style="Operation",
        col="List Size", row="Is Pad",
        kind="line", markers=True, dashes=False,
        height=4, aspect=1.2,
        facet_kws={'sharey': False, 'sharex': True}
    )

    g.set(xscale="log", yscale="log")

    for ax in g.axes.flat:
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    g.fig.suptitle("Latency vs Tensor Size (Log Scale)", y=1.02, fontsize=16, fontweight='bold')
    plt.savefig("bench_latency_log.png", bbox_inches='tight', dpi=300)
    print("Generated: bench_latency_log.png")


def plot_throughput_heatmap(df):
    """
    Chart 2: Throughput Heatmap (Efficiency)
    """
    df = df.copy()
    df["Total Latency"] = df["Put Latency (ms)"] + df["Get Latency (ms)"]
    # Metric: Thousands of elements per millisecond
    df["Throughput (k_Elems/ms)"] = (df["Total Elements"] / df["Total Latency"]) / 1000

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    input_types = ["List[Tensor]", "2D Tensor"]

    for i, i_type in enumerate(input_types):
        # Filter for Is Pad = Yes (typical scenario)
        subset = df[(df["Input Type"] == i_type) & (df["Is Pad"] == "Yes")]

        if subset.empty:
            continue

        pivot_table = subset.pivot(index="List Size", columns="Tensor Size", values="Throughput (k_Elems/ms)")

        # FIX: "Viridis" -> "viridis" (lowercase)
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", ax=axes[i], linewidths=.5)
        axes[i].set_title(f"Throughput (k_Elems/ms) - {i_type}")
        axes[i].invert_yaxis()

    plt.suptitle("Throughput Efficiency (List vs Tensor) - Padding Mode", y=1.02, fontsize=16, fontweight='bold')
    plt.savefig("bench_throughput_heatmap.png", bbox_inches='tight', dpi=300)
    print("Generated: bench_throughput_heatmap.png")


def plot_resources(df):
    """
    Chart 3: CPU & Memory Usage
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))

    # Memory
    sns.lineplot(
        data=df, x="Total Elements", y="Memory (MB)",
        hue="Input Type", style="Is Pad",
        markers=True, ax=axes[0]
    )
    axes[0].set_title("Memory Usage vs Total Elements")
    axes[0].set_xscale("log")
    axes[0].grid(True, which="minor", ls="--")

    # CPU
    df["Total CPU (ms)"] = df["Put CPU (ms)"] + df["Get CPU (ms)"]
    sns.lineplot(
        data=df, x="Total Elements", y="Total CPU (ms)",
        hue="Input Type", style="Is Pad",
        markers=True, ax=axes[1]
    )
    axes[1].set_title("Total CPU Time (Serialize + Deserialize)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="minor", ls="--")

    plt.tight_layout()
    plt.savefig("bench_resources.png", bbox_inches='tight', dpi=300)
    print("Generated: bench_resources.png")


if __name__ == "__main__":
    try:
        # Load data
        df = load_and_process_data("benchmark_results.json")

        plot_latency_log_scale(df)
        plot_throughput_heatmap(df)
        plot_resources(df)

        print("\nSuccess. Check the .png files in the current directory.")

    except FileNotFoundError:
        print("Error: benchmark_results.json not found.")
    except Exception as e:
        print(f"Plotting Error: {e}")