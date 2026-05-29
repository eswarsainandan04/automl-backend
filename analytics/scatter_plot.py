import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_filename(title, prefix):
    safe = "".join(c if c.isalnum() or c == " " else "_" for c in title)[:50]
    return f"{prefix}_{safe.strip().replace(' ', '_')}.png"


def generate_scatter_plot(chart_data, output_dir):
    """
    chart_data keys used:
        title, x_column, y_column, x_values (list), y_values (list)
    """
    title    = chart_data.get("title",    "Scatter Plot")
    x_values = chart_data.get("x_values", [])
    y_values = chart_data.get("y_values", [])
    x_label  = chart_data.get("x_column", "X")
    y_label  = chart_data.get("y_column", "Y")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        x_values, y_values,
        alpha=0.5, color="#4C72B0",
        edgecolors="white", linewidth=0.5, s=30,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()

    filename = _safe_filename(title, "scatter")
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    return filename
