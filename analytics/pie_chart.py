import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_filename(title, prefix):
    safe = "".join(c if c.isalnum() or c == " " else "_" for c in title)[:50]
    return f"{prefix}_{safe.strip().replace(' ', '_')}.png"


def generate_pie_chart(chart_data, output_dir):
    """
    chart_data keys used:
        title, x_values (list of labels), y_values (list of counts)
    """
    title    = chart_data.get("title",    "Pie Chart")
    x_values = chart_data.get("x_values", [])
    y_values = chart_data.get("y_values", [])

    if len(x_values) > 12:
        x_values = x_values[:12]
        y_values = y_values[:12]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = list(plt.cm.Set3.colors[: len(x_values)])

    wedges, texts, autotexts = ax.pie(
        y_values,
        labels=x_values,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.82,
    )
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight("bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    filename = _safe_filename(title, "pie")
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    return filename
