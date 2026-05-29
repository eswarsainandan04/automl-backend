import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_filename(title, prefix):
    safe = "".join(c if c.isalnum() or c == " " else "_" for c in title)[:50]
    return f"{prefix}_{safe.strip().replace(' ', '_')}.png"


def generate_box_plot(chart_data, output_dir):
    """
    chart_data keys used:
        title, x_column, y_column,
        groups: {label: [values, ...], ...}
    """
    title   = chart_data.get("title",    "Box Plot")
    groups  = chart_data.get("groups",   {})
    x_label = chart_data.get("x_column", "Category")
    y_label = chart_data.get("y_column", "Value")

    if not groups:
        return None

    labels = list(groups.keys())
    data   = [groups[k] for k in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        medianprops=dict(color="red", linewidth=2),
    )
    colors = list(plt.cm.Set3.colors[: len(data)])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    filename = _safe_filename(title, "box")
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    return filename
