import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_filename(title, prefix):
    safe = "".join(c if c.isalnum() or c == " " else "_" for c in title)[:50]
    return f"{prefix}_{safe.strip().replace(' ', '_')}.png"


def generate_line_chart(chart_data, output_dir):
    """
    chart_data keys used:
        title, x_column, y_column, x_values (list), y_values (list)
    """
    title    = chart_data.get("title",    "Line Chart")
    x_values = chart_data.get("x_values", [])
    y_values = chart_data.get("y_values", [])
    x_label  = chart_data.get("x_column", "X")
    y_label  = chart_data.get("y_column", "count")
    y_label  = "Count" if y_label == "count" else y_label.capitalize()

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = list(range(len(x_values)))

    ax.plot(
        positions, y_values,
        marker="o", color="#4C72B0",
        linewidth=2, markersize=5,
        markerfacecolor="white", markeredgewidth=1.5,
    )
    ax.fill_between(positions, y_values, alpha=0.15, color="#4C72B0")

    ax.set_xticks(positions)
    ax.set_xticklabels(x_values, rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()

    filename = _safe_filename(title, "line")
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    return filename
