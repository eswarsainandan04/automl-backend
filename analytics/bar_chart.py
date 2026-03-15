import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend - safe in server / CLI mode
import matplotlib.pyplot as plt


def _safe_filename(title, prefix):
    safe = "".join(c if c.isalnum() or c == " " else "_" for c in title)[:50]
    return f"{prefix}_{safe.strip().replace(' ', '_')}.png"


def generate_bar_chart(chart_data, output_dir):
    """
    chart_data keys used:
        title, x_column, y_column, x_values (list), y_values (list)
    """
    title    = chart_data.get("title",    "Bar Chart")
    x_values = chart_data.get("x_values", [])
    y_values = chart_data.get("y_values", [])
    x_label  = chart_data.get("x_column", "Category")
    y_label  = chart_data.get("y_column", "count")
    y_label  = "Count" if y_label == "count" else y_label.capitalize()

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = range(len(x_values))
    bars = ax.bar(positions, y_values, color="#4C72B0", edgecolor="white", linewidth=0.8)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(x_values, rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    max_val = max(y_values) if y_values else 1
    for bar, val in zip(bars, y_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.01,
            str(round(val, 1)),
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    filename = _safe_filename(title, "bar")
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    return filename
