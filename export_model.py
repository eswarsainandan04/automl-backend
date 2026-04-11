"""
export_model.py
==============
Export and batch-prediction APIs for trained models.

Endpoints:
- GET  /export/{session_id}/schema/{dataset_base}
- GET  /export/{session_id}/template/{dataset_base}
- POST /export/{session_id}/predict/{dataset_base}
- GET  /export/{session_id}/code/{dataset_base}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt
from pydantic import BaseModel, Field

from config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER
from data_preprocessing.supabase_storage import download_json, upload_file
from jwt_handler import ALGORITHM, SECRET_KEY

router = APIRouter(prefix="/export", tags=["Export"])
security = HTTPBearer()


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _email_from_token(token: str) -> str:
    try:
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _user_id_by_email(email: str) -> str:
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return str(row[0])


def _get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    return _user_id_by_email(_email_from_token(credentials.credentials))


# ── Subprocess helper (shared with history/automl patterns) ───────────────────

def _resolve_model_python() -> Path:
    for env_name in ("MODEL_SELECTION_PYTHON", "FLAML_PYTHON"):
        configured = os.getenv(env_name)
        if not configured:
            continue
        p = Path(configured).expanduser()
        if p.exists() and p.is_file():
            return p
    current = Path(sys.executable).expanduser()
    if current.exists() and current.is_file():
        return current
    raise FileNotFoundError("Model Python not found. Set MODEL_SELECTION_PYTHON.")


def _run_model_testing_subprocess(function_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Run a model_testing function in the dedicated ML subprocess."""
    python_exe = _resolve_model_python()
    backend_dir = Path(__file__).resolve().parent
    sentinel = "__MODEL_TESTING_JSON__"

    code = (
        "import json,sys;"
        "import model_testing;"
        "fn=getattr(model_testing, sys.argv[1]);"
        "kwargs=json.loads(sys.argv[2]);"
        "res=fn(**kwargs);"
        f"print('{sentinel}'+json.dumps(res))"
    )

    proc = subprocess.run(
        [str(python_exe), "-c", code, function_name, json.dumps(kwargs)],
        cwd=str(backend_dir),
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        raise RuntimeError(
            f"Model testing subprocess failed (function={function_name}).\n"
            f"stderr:\n{stderr_tail}\nstdout:\n{stdout_tail}"
        )

    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(sentinel):
            return json.loads(line[len(sentinel):].strip())

    raise RuntimeError(
        f"Model testing subprocess completed but returned no structured output for {function_name}."
    )


# ── Pydantic bodies ───────────────────────────────────────────────────────────

class ExportPredictBody(BaseModel):
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    save: bool = True
    filename: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/{session_id}/schema/{dataset_base}")
def export_schema(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Return the model input schema for export predictions."""
    try:
        data = _run_model_testing_subprocess(
            function_name="get_model_testing_schema_supabase",
            kwargs={
                "user_id": user_id,
                "session_id": session_id,
                "dataset_base": dataset_base,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return data


@router.get("/{session_id}/template/{dataset_base}")
def export_template(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Return a CSV template with the required header columns."""
    try:
        schema = _run_model_testing_subprocess(
            function_name="get_model_testing_schema_supabase",
            kwargs={
                "user_id": user_id,
                "session_id": session_id,
                "dataset_base": dataset_base,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    headers = schema.get("selected_features") or []
    if not headers:
        raise HTTPException(status_code=400, detail="No selected features available for template.")

    csv_text = ",".join(headers) + "\n"
    filename = f"{dataset_base}_template.csv"
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/{session_id}/predict/{dataset_base}")
def export_predict(
    session_id: str,
    dataset_base: str,
    body: ExportPredictBody,
    user_id: str = Depends(_get_current_user_id),
):
    """Run batch predictions and optionally save output CSV to Supabase."""
    if not body.rows:
        raise HTTPException(status_code=400, detail="No input rows provided.")

    try:
        result = _run_model_testing_subprocess(
            function_name="predict_batch_from_session_model",
            kwargs={
                "user_id": user_id,
                "session_id": session_id,
                "dataset_base": dataset_base,
                "rows": body.rows,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    predictions = result.get("predictions") or []
    probabilities = result.get("probabilities") or []
    rows = result.get("rows") or []
    feature_order = result.get("feature_order") or []

    if len(rows) != len(predictions):
        raise HTTPException(status_code=500, detail="Prediction count mismatch.")

    output_rows: List[Dict[str, Any]] = []
    for row, pred, prob in zip(rows, predictions, probabilities):
        out = dict(row)
        out["prediction"] = pred
        if prob is not None:
            out["probability"] = prob
        output_rows.append(out)

    cols = list(feature_order)
    cols.append("prediction")
    if any(p is not None for p in probabilities):
        cols.append("probability")

    df_out = pd.DataFrame(output_rows)
    if cols and all(c in df_out.columns for c in cols):
        df_out = df_out[cols]

    csv_bytes = df_out.to_csv(index=False).encode("utf-8")

    output_file = None
    if body.save:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = body.filename or f"{dataset_base}_predictions_{timestamp}.csv"
        path = f"output/{user_id}/{session_id}/{filename}"
        upload_file(path, csv_bytes, content_type="text/csv")
        output_file = {"path": path, "filename": filename}

    return {
        "dataset_base": dataset_base,
        "task": result.get("task"),
        "rows": output_rows,
        "csv": csv_bytes.decode("utf-8"),
        "output_file": output_file,
        "extra_columns": result.get("extra_columns", []),
        "feature_order": feature_order,
    }


@router.get("/{session_id}/code/{dataset_base}")
def export_code(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Return a simple, dataset-specific training + prediction Python script."""
    report_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_model_report.json"
    features_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_features.json"

    try:
        report = download_json(report_path)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model report not found for dataset '{dataset_base}'. ({exc})",
        )

    try:
        features_meta = download_json(features_path)
    except Exception:
        features_meta = {}

    target = report.get("target") or features_meta.get("target") or "target"
    task = (report.get("task") or features_meta.get("task") or "classification").lower()
    selected_features = report.get("selected_features") or features_meta.get("selected_features") or []
    feature_processing = report.get("feature_processing") or {}

    if not selected_features:
        raise HTTPException(status_code=400, detail="No selected features found for this dataset.")

    best_model = report.get("best_model") or report.get("hpo_model") or ""
    best_params = report.get("hpo_best_params") or {}
    ensemble = report.get("ensemble") or {}
    top_k_models = report.get("top_k_models") or ensemble.get("models") or []
    candidate_models = list(top_k_models)
    if not candidate_models and best_model:
        candidate_models = [best_model]

    used_models: List[str] = []
    seen_models = set()
    for model_name in candidate_models:
        if not model_name or model_name in seen_models:
            continue
        seen_models.add(model_name)
        used_models.append(model_name)

    if not used_models:
        raise HTTPException(status_code=400, detail="No trained model names found in report metadata.")

    hpo_models = report.get("hpo_models") or []
    params_map = {
        m.get("model_name"): (m.get("best_params") or {})
        for m in hpo_models
        if m.get("model_name")
    }
    if best_model and best_params and best_model not in params_map:
        params_map[best_model] = best_params
    used_params_map = {m: (params_map.get(m) or {}) for m in used_models}

    leaderboard = report.get("leaderboard") or []
    scaling_map = {
        row.get("model_name"): bool(row.get("scaling_required"))
        for row in leaderboard
        if row.get("model_name")
    }
    used_scaling_map = {m: bool(scaling_map.get(m, False)) for m in used_models}

    voting = ensemble.get("voting") or ("soft" if task == "classification" else "mean")
    use_ensemble = len(used_models) > 1
    needs_label_encoder = task == "classification" and any(
        m in {"XGBClassifier", "LGBMClassifier"} for m in used_models
    )

    model_specs: Dict[str, Dict[str, str]] = {
        "LogisticRegression": {"module": "sklearn.linear_model", "class": "LogisticRegression"},
        "SGDClassifier": {"module": "sklearn.linear_model", "class": "SGDClassifier"},
        "PassiveAggressiveClassifier": {"module": "sklearn.linear_model", "class": "PassiveAggressiveClassifier"},
        "KNeighborsClassifier": {"module": "sklearn.neighbors", "class": "KNeighborsClassifier"},
        "SVC": {"module": "sklearn.svm", "class": "SVC"},
        "LinearSVC": {"module": "sklearn.svm", "class": "LinearSVC"},
        "DecisionTreeClassifier": {"module": "sklearn.tree", "class": "DecisionTreeClassifier"},
        "RandomForestClassifier": {"module": "sklearn.ensemble", "class": "RandomForestClassifier"},
        "ExtraTreesClassifier": {"module": "sklearn.ensemble", "class": "ExtraTreesClassifier"},
        "GradientBoostingClassifier": {"module": "sklearn.ensemble", "class": "GradientBoostingClassifier"},
        "HistGradientBoostingClassifier": {"module": "sklearn.ensemble", "class": "HistGradientBoostingClassifier"},
        "AdaBoostClassifier": {"module": "sklearn.ensemble", "class": "AdaBoostClassifier"},
        "BaggingClassifier": {"module": "sklearn.ensemble", "class": "BaggingClassifier"},
        "GaussianNB": {"module": "sklearn.naive_bayes", "class": "GaussianNB"},
        "CategoricalNB": {"module": "sklearn.naive_bayes", "class": "CategoricalNB"},
        "MultinomialNB": {"module": "sklearn.naive_bayes", "class": "MultinomialNB"},
        "LinearRegression": {"module": "sklearn.linear_model", "class": "LinearRegression"},
        "Ridge": {"module": "sklearn.linear_model", "class": "Ridge"},
        "Lasso": {"module": "sklearn.linear_model", "class": "Lasso"},
        "ElasticNet": {"module": "sklearn.linear_model", "class": "ElasticNet"},
        "SGDRegressor": {"module": "sklearn.linear_model", "class": "SGDRegressor"},
        "PassiveAggressiveRegressor": {"module": "sklearn.linear_model", "class": "PassiveAggressiveRegressor"},
        "KNeighborsRegressor": {"module": "sklearn.neighbors", "class": "KNeighborsRegressor"},
        "DecisionTreeRegressor": {"module": "sklearn.tree", "class": "DecisionTreeRegressor"},
        "RandomForestRegressor": {"module": "sklearn.ensemble", "class": "RandomForestRegressor"},
        "ExtraTreesRegressor": {"module": "sklearn.ensemble", "class": "ExtraTreesRegressor"},
        "GradientBoostingRegressor": {"module": "sklearn.ensemble", "class": "GradientBoostingRegressor"},
        "HistGradientBoostingRegressor": {"module": "sklearn.ensemble", "class": "HistGradientBoostingRegressor"},
        "AdaBoostRegressor": {"module": "sklearn.ensemble", "class": "AdaBoostRegressor"},
        "BaggingRegressor": {"module": "sklearn.ensemble", "class": "BaggingRegressor"},
        "GaussianProcessRegressor": {"module": "sklearn.gaussian_process", "class": "GaussianProcessRegressor"},
        "XGBClassifier": {"module": "xgboost", "class": "XGBClassifier", "package": "xgboost"},
        "XGBRegressor": {"module": "xgboost", "class": "XGBRegressor", "package": "xgboost"},
        "LGBMClassifier": {"module": "lightgbm", "class": "LGBMClassifier", "package": "lightgbm"},
        "LGBMRegressor": {"module": "lightgbm", "class": "LGBMRegressor", "package": "lightgbm"},
        "CatBoostClassifier": {"module": "catboost", "class": "CatBoostClassifier", "package": "catboost"},
        "CatBoostRegressor": {"module": "catboost", "class": "CatBoostRegressor", "package": "catboost"},
    }

    unsupported_models = [m for m in used_models if m not in model_specs]
    if unsupported_models:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot generate code for unsupported model(s): {unsupported_models}",
        )

    import json as _json
    import pprint as _pprint

    def _py_literal(value: Any) -> str:
        return _pprint.pformat(value, width=100, sort_dicts=False)

    base_imports: List[str] = [
        "import pickle",
        "import pandas as pd",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.preprocessing import OneHotEncoder",
    ]

    if any(used_scaling_map.values()):
        base_imports.append("from sklearn.preprocessing import StandardScaler")

    if needs_label_encoder:
        base_imports.append("from sklearn.preprocessing import LabelEncoder")

    if task == "classification":
        base_imports.append("from sklearn.metrics import accuracy_score")
    else:
        base_imports.append("from sklearn.metrics import r2_score")

    if use_ensemble:
        if task == "classification":
            base_imports.append("from sklearn.ensemble import VotingClassifier")
        else:
            base_imports.append("from sklearn.ensemble import VotingRegressor")

    module_to_classes: Dict[str, List[str]] = {}
    optional_imports: Dict[str, Dict[str, Any]] = {}
    for model_name in used_models:
        spec = model_specs[model_name]
        module = spec["module"]
        cls_name = spec["class"]
        package = spec.get("package")
        if package:
            key = f"{package}:{module}"
            if key not in optional_imports:
                optional_imports[key] = {"package": package, "module": module, "classes": []}
            optional_imports[key]["classes"].append(cls_name)
        else:
            module_to_classes.setdefault(module, []).append(cls_name)

    model_import_lines: List[str] = []
    for module in sorted(module_to_classes.keys()):
        classes = sorted(set(module_to_classes[module]))
        model_import_lines.append(f"from {module} import {', '.join(classes)}")

    for key in sorted(optional_imports.keys()):
        spec = optional_imports[key]
        package = spec["package"]
        module = spec["module"]
        classes = sorted(set(spec["classes"]))
        class_csv = ", ".join(classes)
        model_import_lines.append("try:")
        model_import_lines.append(f"    from {module} import {class_csv}")
        model_import_lines.append("except Exception as exc:")
        model_import_lines.append(
            f"    raise ImportError(\"Install {package} to use model(s): {class_csv}\") from exc"
        )

    constructor_overrides = {
        "SVC": "SVC(probability=True)",
    }

    lines: List[str] = []
    lines.append("# AutoML Export: Simple training + prediction script")
    lines.append("# Generated from your session's trained model report.")
    lines.append("")

    for imp in base_imports:
        lines.append(imp)
    for imp in model_import_lines:
        lines.append(imp)

    lines.append("")
    lines.append(f"DATASET_BASE = {_py_literal(dataset_base)}")
    lines.append(f"TASK = {_py_literal(task)}")
    lines.append(f"TARGET = {_py_literal(target)}")
    lines.append(f"SELECTED_FEATURES = {_py_literal(selected_features)}")
    lines.append(f"FEATURE_PROCESSING = {_py_literal(feature_processing)}")
    lines.append(f"USED_MODELS = {_py_literal(used_models)}")
    lines.append(f"MODEL_PARAMS = {_py_literal(used_params_map)}")
    lines.append(f"MODEL_SCALING = {_py_literal(used_scaling_map)}")
    lines.append(f"VOTING = {_py_literal(voting)}")
    lines.append(f"NEEDS_LABEL_ENCODER = {_py_literal(needs_label_encoder)}")
    lines.append("data = 'Your-dataset-name.csv'      # Training dataset path")
    lines.append("predict_data = 'input.csv'          # New rows for prediction")
    lines.append("OUTPUT_PRED_CSV = 'predictions.csv'")
    lines.append("OUTPUT_MODEL_PKL = f'{DATASET_BASE}_trained_pipeline.pkl'")
    lines.append("")
    lines.append("MODEL_BUILDERS = {")
    for model_name in used_models:
        class_name = model_specs[model_name]["class"]
        ctor = constructor_overrides.get(model_name, f"{class_name}()")
        lines.append(f"    {_json.dumps(model_name)}: lambda: {ctor},")
    lines.append("}")
    lines.append("")
    lines.append("def make_estimator(model_name):")
    lines.append("    if model_name not in MODEL_BUILDERS:")
    lines.append("        raise ValueError(f'Unsupported model in export: {model_name}')")
    lines.append("    model = MODEL_BUILDERS[model_name]()")
    lines.append("    params = MODEL_PARAMS.get(model_name, {})")
    lines.append("    if params:")
    lines.append("        model.set_params(**params)")
    lines.append("    steps = []")
    lines.append("    if MODEL_SCALING.get(model_name, False):")
    lines.append("        steps.append(('scaler', StandardScaler(with_mean=False)))")
    lines.append("    steps.append(('model', model))")
    lines.append("    return Pipeline(steps)")
    lines.append("")
    lines.append("def _feature_groups():")
    lines.append("    categorical, boolean_cols, numeric = [], [], []")
    lines.append("    for col in SELECTED_FEATURES:")
    lines.append("        label = str(FEATURE_PROCESSING.get(col, ''))")
    lines.append("        if 'Cast to int' in label:")
    lines.append("            boolean_cols.append(col)")
    lines.append("        elif any(tok in label for tok in [")
    lines.append("            'OneHotEncoding', 'TargetEncoding', 'FrequencyEncoding',")
    lines.append("            'MEstimateEncoding', 'CatBoostEncoding', 'BinaryEncoder',")
    lines.append("        ]):")
    lines.append("            categorical.append(col)")
    lines.append("        else:")
    lines.append("            numeric.append(col)")
    lines.append("    return categorical, boolean_cols, numeric")
    lines.append("")
    lines.append("def _coerce_boolean_columns(df, columns):")
    lines.append("    bool_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}")
    lines.append("    out = df.copy()")
    lines.append("    for col in columns:")
    lines.append("        if col not in out.columns:")
    lines.append("            continue")
    lines.append("        s = out[col].astype(str).str.strip().str.lower().map(bool_map)")
    lines.append("        out[col] = pd.to_numeric(s, errors='coerce').fillna(0).astype('int8')")
    lines.append("    return out")
    lines.append("")
    lines.append("# 1) Train on your dataset")
    lines.append("df = pd.read_csv(data)")
    lines.append("required_cols = SELECTED_FEATURES + [TARGET]")
    lines.append("missing_train_cols = [c for c in required_cols if c not in df.columns]")
    lines.append("if missing_train_cols:")
    lines.append("    raise ValueError(f'Missing required columns in train.csv: {missing_train_cols}')")
    lines.append("")
    lines.append("df = df.dropna(subset=[TARGET])")
    lines.append("X = df[SELECTED_FEATURES].copy()")
    lines.append("y = df[TARGET].copy()")
    lines.append("")
    lines.append("categorical_features, boolean_features, numeric_features = _feature_groups()")
    lines.append("for c in categorical_features:")
    lines.append("    if c in X.columns:")
    lines.append("        X[c] = X[c].astype(str)")
    lines.append("X = _coerce_boolean_columns(X, boolean_features)")
    lines.append("")
    lines.append("transformers = []")
    lines.append("if numeric_features:")
    lines.append("    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])")
    lines.append("    transformers.append(('num', num_pipe, numeric_features))")
    lines.append("if boolean_features:")
    lines.append("    bool_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])")
    lines.append("    transformers.append(('bool', bool_pipe, boolean_features))")
    lines.append("if categorical_features:")
    lines.append("    cat_pipe = Pipeline([")
    lines.append("        ('imputer', SimpleImputer(strategy='most_frequent')),")
    lines.append("        ('onehot', OneHotEncoder(handle_unknown='ignore')),")
    lines.append("    ])")
    lines.append("    transformers.append(('cat', cat_pipe, categorical_features))")
    lines.append("")
    lines.append("preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')")
    lines.append("")
    lines.append("stratify = y if TASK == 'classification' and y.nunique() > 1 else None")
    lines.append("X_train, X_val, y_train, y_val = train_test_split(")
    lines.append("    X, y, test_size=0.2, random_state=42, stratify=stratify")
    lines.append(")")
    lines.append("")
    lines.append("label_encoder = None")
    lines.append("y_train_model = y_train")
    lines.append("y_val_model = y_val")
    lines.append("if TASK == 'classification' and NEEDS_LABEL_ENCODER:")
    lines.append("    label_encoder = LabelEncoder()")
    lines.append("    y_train_model = label_encoder.fit_transform(y_train)")
    lines.append("    y_val_model = label_encoder.transform(y_val)")
    lines.append("")
    lines.append("if len(USED_MODELS) > 1:")
    lines.append("    estimators = [(name, make_estimator(name)) for name in USED_MODELS]")
    lines.append("    if TASK == 'classification':")
    lines.append("        final_model = VotingClassifier(estimators=estimators, voting=VOTING)")
    lines.append("    else:")
    lines.append("        final_model = VotingRegressor(estimators=estimators)")
    lines.append("else:")
    lines.append("    final_model = make_estimator(USED_MODELS[0])")
    lines.append("")
    lines.append("pipeline = Pipeline([")
    lines.append("    ('preprocessor', preprocessor),")
    lines.append("    ('model', final_model),")
    lines.append("])")
    lines.append("")
    lines.append("pipeline.fit(X_train, y_train_model)")
    lines.append("val_pred = pipeline.predict(X_val)")
    lines.append("if TASK == 'classification':")
    lines.append("    print('Validation accuracy:', round(float(accuracy_score(y_val_model, val_pred)), 6))")
    lines.append("else:")
    lines.append("    print('Validation R2:', round(float(r2_score(y_val, val_pred)), 6))")
    lines.append("")
    lines.append("# Save trained pipeline")
    lines.append("with open(OUTPUT_MODEL_PKL, 'wb') as f:")
    lines.append("    pickle.dump({")
    lines.append("        'framework': 'sklearn',")
    lines.append("        'pipeline': pipeline,")
    lines.append("        'selected_features': SELECTED_FEATURES,")
    lines.append("        'target': TARGET,")
    lines.append("        'task': TASK,")
    lines.append("        'used_models': USED_MODELS,")
    lines.append("        'model_params': MODEL_PARAMS,")
    lines.append("        'label_encoder': label_encoder,")
    lines.append("    }, f)")
    lines.append("")
    lines.append("# 2) Predict on new rows")
    lines.append("pred_df = pd.read_csv(predict_data)")
    lines.append("missing_pred_cols = [c for c in SELECTED_FEATURES if c not in pred_df.columns]")
    lines.append("if missing_pred_cols:")
    lines.append("    raise ValueError(f'Missing required columns in input.csv: {missing_pred_cols}')")
    lines.append("")
    lines.append("pred_input = pred_df[SELECTED_FEATURES].copy()")
    lines.append("for c in categorical_features:")
    lines.append("    if c in pred_input.columns:")
    lines.append("        pred_input[c] = pred_input[c].astype(str)")
    lines.append("pred_input = _coerce_boolean_columns(pred_input, boolean_features)")
    lines.append("pred_output = pipeline.predict(pred_input)")
    lines.append("if TASK == 'classification' and label_encoder is not None:")
    lines.append("    pred_output = label_encoder.inverse_transform(pd.Series(pred_output).astype(int))")
    lines.append("pred_df['prediction'] = pred_output")
    lines.append("pred_df.to_csv(OUTPUT_PRED_CSV, index=False)")
    lines.append("print(f'Predictions saved to {OUTPUT_PRED_CSV}')")
    lines.append("print(f'Model saved to {OUTPUT_MODEL_PKL}')")

    return {"language": "python", "code": "\n".join(lines)}
