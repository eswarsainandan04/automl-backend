import os
import json
import glob
import pandas as pd
from google import genai
from dotenv import load_dotenv

# Prefer global backend .env (root project settings), then analytics/.env for overrides.
backend_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
analytics_env = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(backend_env)
load_dotenv(analytics_env)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model_candidates(primary_model: str) -> list:
    """
    Build an ordered list of Gemini models to try.

    Order:
      1) Requested model from env/code.
      2) Optional comma-separated GEMINI_FALLBACK_MODELS env var.
      3) A broadly available default fallback.
    """
    configured = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
    candidates = [primary_model] + configured + ["gemini-2.0-flash"]

    # Preserve order while removing duplicates.
    deduped = []
    seen = set()
    for model in candidates:
        if model in seen:
            continue
        seen.add(model)
        deduped.append(model)
    return deduped


def _is_model_access_error(exc: Exception) -> bool:
    """Return True for model-level access errors where fallback is sensible."""
    msg = str(exc).lower()
    return (
        "403" in msg
        or "forbidden" in msg
        or "permission" in msg
        or "404" in msg
        or "not found" in msg
    )


def _generate_with_model_fallback(client, model_name: str, prompt: str) -> str:
    """
    Try the configured Gemini model and, on model-access failures,
    retry with fallback models.
    """
    attempts = []
    candidates = _model_candidates(model_name)

    for idx, candidate in enumerate(candidates):
        try:
            response = client.models.generate_content(model=candidate, contents=prompt)
            if idx > 0:
                print(
                    f"      INFO: Primary Gemini model '{model_name}' unavailable; "
                    f"using fallback model '{candidate}'."
                )
            return response.text
        except Exception as exc:
            attempts.append((candidate, str(exc)))
            is_last = idx == len(candidates) - 1
            if is_last or not _is_model_access_error(exc):
                attempted_models = ", ".join(m for m, _ in attempts)
                raise RuntimeError(
                    "Gemini generate_content failed after model retries. "
                    f"attempted_models=[{attempted_models}] last_error={attempts[-1][1]}"
                ) from exc

    # Unreachable, but keeps static analyzers happy.
    raise RuntimeError("Gemini generate_content failed with no model candidates.")


# ─────────────────────────────────────────────
# 1.  File discovery
# ─────────────────────────────────────────────

def find_user_files(user_id):
    """Return (profiling_json_path, cleaned_csv_path) for a given user."""
    meta_dir   = os.path.join(BASE_DIR, "storage", "meta_data", user_id)
    output_dir = os.path.join(BASE_DIR, "storage", "output",    user_id)

    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(f"No metadata folder found for user '{user_id}'.")

    profiling_files = glob.glob(os.path.join(meta_dir, "*_profiling.json"))
    if not profiling_files:
        raise FileNotFoundError(f"No *_profiling.json found in: {meta_dir}")

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"No output folder found for user '{user_id}'.")

    cleaned_files = glob.glob(os.path.join(output_dir, "*_cleaned.csv"))
    if not cleaned_files:
        raise FileNotFoundError(f"No *_cleaned.csv found in: {output_dir}")

    # Use first match (alphabetical); a single dataset is the common case
    return sorted(profiling_files)[0], sorted(cleaned_files)[0]


# ─────────────────────────────────────────────
# 2.  Column-data extraction (Python logic)
# ─────────────────────────────────────────────

def extract_column_data(df, profiling):
    """
    Build a compact column summary for the LLM prompt.
    Categorical columns (unique ≤ 30) get a top-10 value distribution.
    Numeric / date columns get min / max / mean stats.
    Identifier columns are skipped.
    """
    column_data = []

    for col in profiling.get("column_wise_summary", []):
        name            = col["column_name"]
        structural_type = col.get("structural_type", "")
        unique_count    = col.get("unique_count", 0)

        # Skip identifiers – they produce meaningless charts
        if structural_type == "identifier":
            continue

        # Skip columns not present in the DataFrame
        if name not in df.columns:
            continue

        entry = {
            "column":          name,
            "structural_type": structural_type,
            "semantic_type":   col.get("semantic_type", ""),
            "unique_count":    unique_count,
        }

        # Categorical columns: expose every distinct label + how often each appears
        if structural_type == "categorical":
            unique_vals = [str(v) for v in df[name].dropna().unique().tolist()]
            entry["unique_values"] = sorted(unique_vals)          # full sorted label list
            vc = df[name].value_counts().head(20)
            entry["value_counts"] = {str(k): int(v) for k, v in vc.items()}  # top-20 frequency

        # Numeric columns: rich stats so the LLM understands the distribution
        elif structural_type == "numeric":
            try:
                desc = df[name].dropna().describe(percentiles=[0.25, 0.5, 0.75])
                entry["numeric_stats"] = {
                    "min":    round(float(desc["min"]),   4),
                    "q25":    round(float(desc["25%"]),   4),
                    "median": round(float(desc["50%"]),   4),
                    "mean":   round(float(desc["mean"]),  4),
                    "q75":    round(float(desc["75%"]),   4),
                    "max":    round(float(desc["max"]),   4),
                }
                entry["sample_values"] = [
                    round(float(v), 4)
                    for v in df[name].dropna().sample(
                        min(5, df[name].dropna().shape[0]), random_state=42
                    ).tolist()
                ]
            except Exception:
                pass

        # Date / temporal columns: sample a few values so the LLM sees the format
        elif structural_type == "date":
            entry["sample_values"] = [
                str(v) for v in df[name].dropna().head(5).tolist()
            ]
            try:
                entry["date_range"] = {
                    "earliest": str(df[name].min()),
                    "latest":   str(df[name].max()),
                }
            except Exception:
                pass

        column_data.append(entry)

    return column_data


# ─────────────────────────────────────────────
# 3.  LLM prompt & response
# ─────────────────────────────────────────────

def ask_llm_for_chart_plan(column_data, profiling, client, model_name):
    """Send column metadata to Gemini and get a chart plan as a JSON string."""

    prompt = f"""You are a data visualization expert. You have been given the FULL column-level
data for a dataset. Read it carefully before deciding which charts to produce.

Dataset : {profiling.get("file_name", "Unknown")}
Rows    : {profiling.get("number_of_rows", 0):,}
Columns : {len(column_data)}

── HOW TO READ THE COLUMN DATA ──────────────────────────────────────────────
Each column entry contains:
  • structural_type  — "categorical" | "numeric" | "date"
  • For CATEGORICAL columns:
      - unique_values  : sorted list of every distinct label in that column
                         e.g. {{"transportation": ["bus", "car", "train"]}}
      - value_counts   : {{label: frequency}} for the top-20 most common values
      - unique_count   : total number of distinct labels
  • For NUMERIC columns:
      - numeric_stats  : min, q25, median, mean, q75, max
      - sample_values  : 5 random actual values from the column
  • For DATE columns:
      - sample_values  : first 5 raw date strings
      - date_range     : {{"earliest": ..., "latest": ...}}
─────────────────────────────────────────────────────────────────────────────

Column Details (read every field before choosing charts):
{json.dumps(column_data, indent=2)}

Using the actual values and statistics above, generate 5–8 meaningful,
non-redundant charts that reveal real insights about this specific dataset.

Chart-type rules:
  bar       — categorical x_column (unique ≤ 20); y_column = "count" OR a numeric column name
  pie       — categorical x_column with unique ≤ 12; y_column = "count"
  line      — date/year/temporal x_column; y_column = "count" or a numeric column
  histogram — single numeric x_column; omit y_column entirely
  scatter   — two DIFFERENT numeric columns: x_column and y_column
  box       — categorical x_column + numeric y_column

Return ONLY a valid JSON object — NO markdown fences, NO extra text:
{{
  "charts": [
    {{
      "type": "bar",
      "title": "Descriptive human-readable chart title",
      "x_column": "exact_column_name",
      "y_column": "count",
      "description": "One sentence explaining the insight this chart reveals"
    }}
  ]
}}

Allowed types: bar, pie, line, histogram, scatter, box"""

    return _generate_with_model_fallback(client, model_name, prompt)


def parse_llm_response(text):
    """Parse the LLM JSON, tolerating markdown code fences."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


# ─────────────────────────────────────────────
# 4.  Fill chart data from DataFrame (Python logic)
# ─────────────────────────────────────────────

def fill_chart_data(chart_spec, df):
    """
    Given an LLM chart spec, compute the actual x_values / y_values / values / groups
    from the DataFrame and return the enriched spec dict.
    Returns None if the required columns are missing or the data cannot be computed.
    """
    chart_type = chart_spec.get("type", "")
    x_col      = chart_spec.get("x_column", "")
    y_col      = chart_spec.get("y_column", "count")

    data = dict(chart_spec)   # shallow copy – we add computed fields

    try:
        if chart_type in ("bar", "pie"):
            if x_col not in df.columns:
                return None
            if y_col == "count" or y_col not in df.columns:
                counts = df[x_col].value_counts().head(15)

                # Lookup-table guard: if every category appears exactly once,
                # value_counts() gives all-equal slices which is meaningless.
                # Instead, find the first numeric column and use its actual values.
                if (counts == 1).all():
                    numeric_cols = [
                        c for c in df.select_dtypes(include="number").columns
                        if c != x_col
                    ]
                    if numeric_cols:
                        y_col = numeric_cols[0]
                        data["y_column"] = y_col
                        # Fall through to the grouped-mean branch below
                        grouped = df.groupby(x_col)[y_col].sum()
                        data["x_values"] = [str(k) for k in grouped.index.tolist()]
                        data["y_values"] = [round(float(v), 4) for v in grouped.values.tolist()]
                    else:
                        data["x_values"] = [str(k) for k in counts.index.tolist()]
                        data["y_values"] = [int(v)  for v in counts.values.tolist()]
                else:
                    data["x_values"] = [str(k) for k in counts.index.tolist()]
                    data["y_values"] = [int(v)  for v in counts.values.tolist()]
            else:
                grouped = df.groupby(x_col)[y_col].mean()
                data["x_values"] = [str(k)         for k in grouped.index.tolist()]
                data["y_values"] = [round(float(v), 4) for v in grouped.values.tolist()]

        elif chart_type == "line":
            if x_col not in df.columns:
                return None
            if y_col == "count" or y_col not in df.columns:
                counts = df[x_col].value_counts().sort_index()
                data["x_values"] = [str(k) for k in counts.index.tolist()]
                data["y_values"] = [int(v)  for v in counts.values.tolist()]
            else:
                grouped = df.groupby(x_col)[y_col].mean().sort_index()
                data["x_values"] = [str(k)         for k in grouped.index.tolist()]
                data["y_values"] = [round(float(v), 4) for v in grouped.values.tolist()]

        elif chart_type == "histogram":
            if x_col not in df.columns:
                return None
            data["values"] = df[x_col].dropna().tolist()

        elif chart_type == "scatter":
            if x_col not in df.columns or y_col not in df.columns:
                return None
            sample = df[[x_col, y_col]].dropna().sample(
                min(500, len(df)), random_state=42
            )
            data["x_values"] = [float(v) for v in sample[x_col].tolist()]
            data["y_values"] = [float(v) for v in sample[y_col].tolist()]

        elif chart_type == "box":
            if x_col not in df.columns or y_col not in df.columns:
                return None
            groups = df.groupby(x_col)[y_col].apply(list).head(10)
            data["groups"] = {str(k): [float(i) for i in v] for k, v in groups.items()}

        else:
            return None  # Unknown chart type

    except Exception as exc:
        print(f"  Warning: could not fill data for '{chart_spec.get('title')}': {exc}")
        return None

    return data


# ─────────────────────────────────────────────
# 5.  Python-side chart guarantees (fallback)
# ─────────────────────────────────────────────

def ensure_default_charts(llm_charts, df, column_data):
    """
    After the LLM plan, guarantee obvious charts for uncovered columns:
      - Pie  : categorical col with ≤ 12 unique values not already bar/pie'd
      - Bar  : categorical col with ≤ 20 unique values not already bar/pie'd
      - Histogram : numeric col not already histogrammed

    Also handles the "lookup table" pattern: when each category appears
    exactly once in the DataFrame the numeric column IS the value, so it
    is used as y_col instead of "count" (prevents all-equal pie slices).
    """
    covered_bar_pie = {
        c.get("x_column") for c in llm_charts if c.get("type") in ("bar", "pie")
    }
    covered_hist = {
        c.get("x_column") for c in llm_charts if c.get("type") == "histogram"
    }

    cat_cols = [c for c in column_data if c.get("structural_type") == "categorical"]
    num_cols = [c for c in column_data if c.get("structural_type") == "numeric"]
    num_in_df = [c for c in num_cols if c["column"] in df.columns]

    extra = []

    for col in cat_cols:
        name   = col["column"]
        unique = col.get("unique_count", 0)

        if name in covered_bar_pie:
            continue

        # Decide y_col: if every category appears exactly once, use the
        # first numeric column as the value (e.g. payment_method → transactions)
        y_col = "count"
        if unique == len(df) and num_in_df:
            y_col = num_in_df[0]["column"]

        if unique <= 12:
            extra.append({
                "type":        "pie",
                "title":       f"Distribution of {name.replace('_', ' ').title()}",
                "x_column":    name,
                "y_column":    y_col,
                "description": f"Proportional breakdown of each category in {name}.",
            })
            covered_bar_pie.add(name)
        elif unique <= 20:
            extra.append({
                "type":        "bar",
                "title":       f"Count Distribution of {name.replace('_', ' ').title()}",
                "x_column":    name,
                "y_column":    y_col,
                "description": f"Frequency of each category in {name}.",
            })
            covered_bar_pie.add(name)

    for col in num_cols:
        name = col["column"]
        if name not in covered_hist and name in df.columns:
            extra.append({
                "type":        "histogram",
                "title":       f"Distribution of {name.replace('_', ' ').title()}",
                "x_column":    name,
                "description": f"Frequency distribution of values in {name}.",
            })
            covered_hist.add(name)

    return extra


# ─────────────────────────────────────────────
# 6.  Main orchestration
# ─────────────────────────────────────────────

def run_decision_maker(user_id):
    """
    Full pipeline:
      1. Discover files
      2. Load profiling JSON + cleaned CSV
      3. Extract column data
      4. Ask Gemini for a chart plan
      5. Fill chart data from DataFrame
      6. Save the chart plan JSON to the output directory
    Returns (filled_charts, output_dir).
    """
    api_key    = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Check backend/.env or analytics/.env")

    client = genai.Client(api_key=api_key)

    # --- discover files ---
    print(f"\n[1/5] Locating files for user: {user_id}")
    profiling_path, cleaned_path = find_user_files(user_id)
    print(f"      Profiling : {os.path.basename(profiling_path)}")
    print(f"      Cleaned   : {os.path.basename(cleaned_path)}")

    with open(profiling_path, "r", encoding="utf-8") as f:
        profiling = json.load(f)

    df = pd.read_csv(cleaned_path)
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns")

    # --- extract column data ---
    print("\n[2/5] Extracting column metadata for LLM context...")
    column_data = extract_column_data(df, profiling)
    print(f"      {len(column_data)} eligible columns found")

    # --- LLM chart plan ---
    print(f"\n[3/5] Asking Gemini ({model_name}) for chart plan...")
    raw_response = ask_llm_for_chart_plan(column_data, profiling, client, model_name)

    print("\n[4/5] Parsing LLM response...")
    try:
        plan   = parse_llm_response(raw_response)
        charts = plan.get("charts", [])
    except Exception as exc:
        print(f"      WARNING: Could not parse LLM response ({exc}). Raw output:")
        print(f"      {raw_response[:400]}")
        charts = []

    print(f"      LLM proposed {len(charts)} chart(s)")
    if len(charts) == 0:
        print(f"      Raw LLM output: {raw_response[:300]}")

    # Python-side fallback — always adds obvious charts the LLM may have missed
    extra = ensure_default_charts(charts, df, column_data)
    if extra:
        print(f"      Adding {len(extra)} Python-guaranteed chart(s) for uncovered columns:")
        for e in extra:
            print(f"        + {e['type'].upper():10s}  {e['title']}")
        charts = charts + extra

    # --- fill data ---
    print("\n[5/5] Computing chart data from DataFrame...")
    filled_charts = []
    for chart in charts:
        filled = fill_chart_data(chart, df)
        if filled:
            filled_charts.append(filled)
            print(f"      [OK]   {chart['type'].upper():10s}  {chart['title']}")
        else:
            print(f"      [SKIP] {chart['type'].upper():10s}  {chart['title']}  (column missing or incompatible)")

    # --- persist the plan for inspection ---
    output_dir = os.path.join(
        BASE_DIR, "storage", "output", user_id, "analytics"
    )
    os.makedirs(output_dir, exist_ok=True)

    plan_path = os.path.join(output_dir, "chart_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump({"user_id": user_id, "charts": filled_charts}, f, indent=2)
    print(f"\n      Chart plan saved → {plan_path}")

    return filled_charts, output_dir


# ─────────────────────────────────────────────
# 7.  LLM-driven dataset join planning
# ─────────────────────────────────────────────

def ask_llm_for_join_plan(datasets_meta: list, client, model_name: str) -> str:
    """
    Ask the LLM to read multiple profiling JSONs and produce a join plan.

    datasets_meta: list of {"filename": str, "profiling": dict, "columns": list[str]}
    Returns raw LLM response text (JSON string).
    """
    summaries = []
    for d in datasets_meta:
        prof = d["profiling"]
        col_names = d["columns"]
        col_types = {
            c["column_name"]: c.get("structural_type", "unknown")
            for c in prof.get("column_wise_summary", [])
        }
        summaries.append({
            "filename":    d["filename"],
            "rows":        prof.get("number_of_rows", 0),
            "columns":     col_names,
            "column_types": col_types,
        })

    prompt = f"""You are a data engineering expert. You are given metadata for {len(datasets_meta)} datasets
that share one or more column names (potential foreign keys / join keys).

Your task: produce a JSON join plan that specifies every meaningful way to combine
these datasets. Only suggest joins where BOTH datasets share at least one column name.

Dataset summaries:
{json.dumps(summaries, indent=2)}

Rules:
- "left_file"  and "right_file" must be exact filenames from the list above.
- "on" must be a list of column names that exist in BOTH datasets.
- "how" must be one of: "inner", "left", "right".
  - Prefer "inner" when the join key is a true identifier (id/key column).
  - Prefer "left"  when keeping all rows of the left file makes domain sense.
- "result_name" must be a short, descriptive snake_case name for the joined table
  (e.g. "orders_with_customers").
- Only include joins that would yield genuinely useful combined analytics.
- If more than 2 datasets exist, you may suggest multi-step joins (chain them).
  For a chain, set "left_file" to a previously defined "result_name".
- Do NOT hallucinate column names that are not in both datasets.

Return ONLY a valid JSON object — NO markdown fences, NO extra text:
{{
  "joins": [
    {{
      "left_file":   "filename_a",
      "right_file":  "filename_b",
      "on":          ["shared_column"],
      "how":         "inner",
      "result_name": "descriptive_joined_name",
      "description": "One sentence explaining what this join reveals"
    }}
  ]
}}

If no meaningful join is possible, return: {{"joins": []}}"""

    return _generate_with_model_fallback(client, model_name, prompt)


def execute_join_plan(join_plan: dict, named_dfs: dict) -> list:
    """
    Execute join plan produced by the LLM.

    join_plan  : parsed dict with key "joins"
    named_dfs  : {filename: pd.DataFrame} — includes individual frames and any
                 previously created joined frames (so chains work).

    Returns list of {"result_name": str, "description": str, "df": pd.DataFrame}
    """
    results = []
    frames  = dict(named_dfs)   # working copy – accumulate new frames here

    for spec in join_plan.get("joins", []):
        left_name   = spec.get("left_file",  "")
        right_name  = spec.get("right_file", "")
        on_cols     = spec.get("on",         [])
        how         = spec.get("how",        "inner")
        result_name = spec.get("result_name", f"{left_name}__{right_name}")
        description = spec.get("description", "")

        if not on_cols:
            continue

        left_df  = frames.get(left_name)
        right_df = frames.get(right_name)

        if left_df is None or right_df is None:
            continue

        # Validate every join key exists in both frames
        valid_on = [c for c in on_cols if c in left_df.columns and c in right_df.columns]
        if not valid_on:
            continue

        try:
            joined = left_df.merge(right_df, on=valid_on, how=how, suffixes=("_x", "_y"))
            if joined.empty:
                continue
            frames[result_name] = joined
            results.append({
                "result_name": result_name,
                "description": description,
                "df":          joined,
            })
        except Exception:
            continue

    return results

