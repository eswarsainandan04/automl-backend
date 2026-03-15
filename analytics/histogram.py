import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_filename(title, prefix):
    safe = "".join(c if c.isalnum() or c == " " else "_" for c in title)[:50]
    return f"{prefix}_{safe.strip().replace(' ', '_')}.png"


def generate_histogram(chart_data, output_dir):
    """
    chart_data keys used:
        title, x_column, values (flat list of numbers)
    """
    title   = chart_data.get("title",    "Histogram")
    values  = chart_data.get("values",   [])
    x_label = chart_data.get("x_column", "Value")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    filename = _safe_filename(title, "hist")
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    return filename
