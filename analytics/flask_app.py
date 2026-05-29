"""
Analytics Flask App
===================
Usage:
    python flask_app.py

The CLI will prompt for a User ID, run the full LLM → chart pipeline,
then start a Flask server so you can browse the generated charts.

Endpoints:
    GET /             → pipeline summary (JSON)
    GET /charts       → list of generated chart files (JSON)
    GET /charts/<file> → serve a chart PNG directly
"""

import os
import sys
from flask import Flask, jsonify, send_from_directory

# Import chart generators
from decision_maker  import run_decision_maker
from bar_chart       import generate_bar_chart
from pie_chart       import generate_pie_chart
from histogram       import generate_histogram
from line_chart      import generate_line_chart
from scatter_plot    import generate_scatter_plot
from box_plot        import generate_box_plot

app = Flask(__name__)

# Map LLM chart-type strings to generator functions
CHART_GENERATORS = {
    "bar":       generate_bar_chart,
    "pie":       generate_pie_chart,
    "line":      generate_line_chart,
    "histogram": generate_histogram,
    "scatter":   generate_scatter_plot,
    "box":       generate_box_plot,
}

# In-memory state populated at startup
_state = {
    "user_id":          None,
    "output_dir":       None,
    "charts_generated": [],
}


# ─────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────

def run_analytics_pipeline(user_id):
    """
    1. Call the LLM decision maker to get a filled chart plan.
    2. For each chart in the plan, call the appropriate generator.
    3. Collect results.
    """
    charts, output_dir = run_decision_maker(user_id)

    print(f"\nGenerating {len(charts)} chart image(s)…")
    generated = []

    for idx, chart in enumerate(charts, 1):
        chart_type = chart.get("type", "")
        generator  = CHART_GENERATORS.get(chart_type)

        if generator is None:
            print(f"  [{idx}] SKIP  — unknown chart type: '{chart_type}'")
            continue

        try:
            filename = generator(chart, output_dir)
            if filename:
                generated.append({
                    "type":        chart_type,
                    "title":       chart.get("title", ""),
                    "description": chart.get("description", ""),
                    "file":        filename,
                })
                print(f"  [{idx}] SAVED — {filename}")
        except Exception as exc:
            print(f"  [{idx}] ERROR — {chart_type} '{chart.get('title')}': {exc}")

    print(f"\n✓ {len(generated)} chart(s) saved to: {output_dir}")
    return generated, output_dir


# ─────────────────────────────────────────────
# Flask endpoints
# ─────────────────────────────────────────────

@app.route("/")
def index():
    if _state["user_id"] is None:
        return jsonify({"status": "no analytics run yet"})
    return jsonify({
        "status":           "ready",
        "user_id":          _state["user_id"],
        "output_dir":       _state["output_dir"],
        "charts_generated": len(_state["charts_generated"]),
        "charts":           _state["charts_generated"],
    })


@app.route("/charts")
def list_charts():
    return jsonify({
        "output_dir": _state["output_dir"],
        "charts":     _state["charts_generated"],
    })


@app.route("/charts/<path:filename>")
def serve_chart(filename):
    output_dir = _state.get("output_dir")
    if not output_dir or not os.path.isdir(output_dir):
        return jsonify({"error": "Output directory not found"}), 404
    return send_from_directory(output_dir, filename)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Analytics Dashboard Generator  (LLM-powered)")
    print("=" * 55)

    user_id = input("\nEnter User ID: ").strip()
    if not user_id:
        print("Error: User ID cannot be empty.")
        sys.exit(1)

    try:
        generated, output_dir = run_analytics_pipeline(user_id)
    except FileNotFoundError as exc:
        print(f"\nFile Error: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\nConfig Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected Error: {exc}")
        sys.exit(1)

    _state["user_id"]          = user_id
    _state["output_dir"]       = output_dir
    _state["charts_generated"] = generated

    print("\n" + "=" * 55)
    print(f"  Flask server starting on http://localhost:5001")
    print(f"  Browse charts : http://localhost:5001/charts")
    print("=" * 55 + "\n")

    app.run(host="0.0.0.0", port=5001, debug=False)
