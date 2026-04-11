"""
model_building/models.py
========================
ML model imports, availability flags, constants, and model instantiation registry.

Imported by model_selection.py.
"""

from __future__ import annotations

from typing import Any, List, Tuple

# ── Sklearn: Text vectorisers ─────────────────────────────────────────────────
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer,
)

# ── Sklearn: Classification models ───────────────────────────────────────────
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB

# ── Sklearn: Regression models ───────────────────────────────────────────────
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    PassiveAggressiveRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor

# ── Third-party boosting libraries ───────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


# ── Re-export text vectoriser types for use in model_selection ───────────────
TEXT_VEC_TYPES = (TfidfVectorizer, CountVectorizer, HashingVectorizer)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_TEMPORAL_TYPES = frozenset({
    "date", "time", "datetime", "date/time",
    "date/datetime", "time/datetime", "date/time/datetime", "temporal",
})

_NEEDS_LABEL_ENCODING: frozenset = frozenset({
    "XGBClassifier",
    "LGBMClassifier",
})

_NEEDS_DENSE_INPUT: frozenset = frozenset({
    "LogisticRegression",
    "SGDClassifier",
    "KNeighborsClassifier",
    "SVC",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "AdaBoostClassifier",
    "BaggingClassifier",
    "GaussianNB",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SGDRegressor",
    "KNeighborsRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "AdaBoostRegressor",
    "BaggingRegressor",
    "GaussianProcessRegressor",
    "XGBRegressor",
    "LGBMRegressor",
    "CatBoostRegressor",
})

_MAX_TEXT_FEATURES: int = 4_096


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INSTANTIATION REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

import logging
logger = logging.getLogger("ModelSelection")


def _build_model_instance(model_name: str, task: str) -> Any:
    _clf_map = {
        "LogisticRegression":              lambda: LogisticRegression(n_jobs=-1),
        "SGDClassifier":                   lambda: SGDClassifier(random_state=42, n_jobs=-1),
        "PassiveAggressiveClassifier":     lambda: PassiveAggressiveClassifier(random_state=42, n_jobs=-1),
        "KNeighborsClassifier":            lambda: KNeighborsClassifier(n_jobs=-1),
        "SVC":                             lambda: SVC(probability=True, random_state=42),
        "LinearSVC":                       lambda: LinearSVC(random_state=42),
        "DecisionTreeClassifier":          lambda: DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier":          lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
        "ExtraTreesClassifier":            lambda: ExtraTreesClassifier(random_state=42, n_jobs=-1),
        "GradientBoostingClassifier":      lambda: GradientBoostingClassifier(random_state=42),
        "HistGradientBoostingClassifier":  lambda: HistGradientBoostingClassifier(random_state=42),
        "AdaBoostClassifier":              lambda: AdaBoostClassifier(random_state=42),
        "BaggingClassifier":               lambda: BaggingClassifier(random_state=42, n_jobs=-1),
        "GaussianNB":                      lambda: GaussianNB(),
        "CategoricalNB":                   lambda: CategoricalNB(),
        "MultinomialNB":                   lambda: MultinomialNB(),
    }
    _reg_map = {
        "LinearRegression":                lambda: LinearRegression(n_jobs=-1),
        "Ridge":                           lambda: Ridge(random_state=42),
        "Lasso":                           lambda: Lasso(random_state=42),
        "ElasticNet":                      lambda: ElasticNet(random_state=42),
        "SGDRegressor":                    lambda: SGDRegressor(random_state=42),
        "PassiveAggressiveRegressor":      lambda: PassiveAggressiveRegressor(random_state=42),
        "KNeighborsRegressor":             lambda: KNeighborsRegressor(n_jobs=-1),
        "DecisionTreeRegressor":           lambda: DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor":           lambda: RandomForestRegressor(random_state=42, n_jobs=-1),
        "ExtraTreesRegressor":             lambda: ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "GradientBoostingRegressor":       lambda: GradientBoostingRegressor(random_state=42),
        "HistGradientBoostingRegressor":   lambda: HistGradientBoostingRegressor(random_state=42),
        "AdaBoostRegressor":               lambda: AdaBoostRegressor(random_state=42),
        "BaggingRegressor":                lambda: BaggingRegressor(random_state=42, n_jobs=-1),
        "GaussianProcessRegressor":        lambda: GaussianProcessRegressor(random_state=42),
    }

    if _HAS_XGB:
        _clf_map["XGBClassifier"] = lambda: XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        )
        _reg_map["XGBRegressor"] = lambda: XGBRegressor(
            random_state=42, n_jobs=-1, verbosity=0,
        )

    if _HAS_LGBM:
        _clf_map["LGBMClassifier"] = lambda: LGBMClassifier(
            random_state=42, n_jobs=-1, verbose=-1,
        )
        _reg_map["LGBMRegressor"] = lambda: LGBMRegressor(
            random_state=42, n_jobs=-1, verbose=-1,
        )

    if _HAS_CATBOOST:
        _clf_map["CatBoostClassifier"] = lambda: CatBoostClassifier(
            random_seed=42, verbose=0,
        )
        _reg_map["CatBoostRegressor"] = lambda: CatBoostRegressor(
            random_seed=42, verbose=0,
        )

    registry = _clf_map if task == "classification" else _reg_map
    factory = registry.get(model_name)
    if factory is None:
        logger.warning("Model '%s' not available — skipping.", model_name)
        return None
    return factory()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL SELECTION: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def select_classification_models(dataset_rows: int, feature_types: str) -> List[Tuple[str, bool]]:
    
    # For small datasets, prefer simpler models that are less likely to overfit.
    if dataset_rows < 1_000:
        if feature_types == "numeric":
            return [
                ("LogisticRegression", True),
                ("KNeighborsClassifier", True),
                ("LinearSVC", False),
                ("PassiveAggressiveClassifier", False),
                ("DecisionTreeClassifier", False),
                ("AdaBoostClassifier", False),
                ("GaussianNB", False),
            ]
        if feature_types == "categorical":
            return [
                ("LogisticRegression", True),
                ("DecisionTreeClassifier", False),
                ("ExtraTreesClassifier", False),
                ("AdaBoostClassifier", False),
                ("BaggingClassifier", False),
                ("CategoricalNB", False),
            ]
        if feature_types == "text":
            return [
                ("LogisticRegression", True),
                ("LinearSVC", False),
                ("PassiveAggressiveClassifier", False),
                ("MultinomialNB", False),
            ]
        if feature_types in _TEMPORAL_TYPES:
            return [
                ("LogisticRegression", True),
                ("DecisionTreeClassifier", False),
                ("AdaBoostClassifier", False),
            ]
        return [
            ("RandomForestClassifier", False),
            ("LogisticRegression", True),
            ("GradientBoostingClassifier", False),
            ("AdaBoostClassifier", False),
        ]

    # For Medium datasets, include more complex models but still keep some simpler ones for comparison and interpretability.
    if dataset_rows < 10_000:
        if feature_types == "numeric":
            return [
                ("LogisticRegression", True),
                ("SGDClassifier", True),
                ("LinearSVC", False),
                ("PassiveAggressiveClassifier", False),
                ("RandomForestClassifier", False),
                ("ExtraTreesClassifier", False),
                ("GradientBoostingClassifier", False),
                ("HistGradientBoostingClassifier", False),
                ("AdaBoostClassifier", False),
            ]
        if feature_types == "categorical":
            return [
                ("LogisticRegression", True),
                ("SGDClassifier", True),
                ("RandomForestClassifier", False),
                ("ExtraTreesClassifier", False),
                ("HistGradientBoostingClassifier", False),
                ("BaggingClassifier", False),
            ]
        if feature_types == "text":
            return [
                ("LogisticRegression", True),
                ("SGDClassifier", True),
                ("LinearSVC", False),
                ("PassiveAggressiveClassifier", False),
                ("MultinomialNB", False),
            ]
        if feature_types in _TEMPORAL_TYPES:
            return [
                ("RandomForestClassifier", False),
                ("GradientBoostingClassifier", False),
                ("HistGradientBoostingClassifier", False),
                ("AdaBoostClassifier", False),
            ]
        return [
            ("RandomForestClassifier", False),
            ("HistGradientBoostingClassifier", False),
            ("GradientBoostingClassifier", False),
            ("AdaBoostClassifier", False),
            ("BaggingClassifier", False),
        ]
        
    # For large datasets, prioritize scalable models that can handle more data and complexity, while still including some linear models for baseline comparison.
    if feature_types == "numeric":
        return [
            ("SGDClassifier", True),
            ("PassiveAggressiveClassifier", False),
            ("LinearSVC", False),
            ("LogisticRegression", True),
            ("HistGradientBoostingClassifier", False),
            ("RandomForestClassifier", False),
            ("LGBMClassifier", False),
            ("CatBoostClassifier", False),
        ]
    if feature_types == "categorical":
        return [
            ("SGDClassifier", True),
            ("PassiveAggressiveClassifier", False),
            ("LogisticRegression", True),
            ("HistGradientBoostingClassifier", False),
            ("RandomForestClassifier", False),
            ("LGBMClassifier", False),
            ("CatBoostClassifier", False),
        ]
    if feature_types == "text":
        return [
            ("SGDClassifier", True),
            ("PassiveAggressiveClassifier", False),
            ("LinearSVC", False),
            ("LogisticRegression", True),
            ("MultinomialNB", False),
        ]
    if feature_types in _TEMPORAL_TYPES:
        return [
            ("HistGradientBoostingClassifier", False),
            ("RandomForestClassifier", False),
            ("AdaBoostClassifier", False),
            ("LGBMClassifier", False),
            ("CatBoostClassifier", False),
        ]
    return [
        ("HistGradientBoostingClassifier", False),
        ("RandomForestClassifier", False),
        ("SGDClassifier", True),
        ("PassiveAggressiveClassifier", False),
        ("LGBMClassifier", False),
        ("CatBoostClassifier", False),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL SELECTION: REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def select_regression_models(dataset_rows: int, feature_types: str) -> List[Tuple[str, bool]]:
    # For small datasets, prefer simpler models that are less likely to overfit.
    if dataset_rows < 1_000:
        if feature_types == "numeric":
            return [
                ("LinearRegression", True),
                ("Ridge", True),
                ("Lasso", True),
                ("ElasticNet", True),
                ("KNeighborsRegressor", True),
                ("DecisionTreeRegressor", False),
                ("AdaBoostRegressor", False),
                ("GaussianProcessRegressor", True),
            ]
        if feature_types == "categorical":
            return [
                ("Ridge", True),
                ("ElasticNet", True),
                ("DecisionTreeRegressor", False),
                ("ExtraTreesRegressor", False),
                ("BaggingRegressor", False),
            ]
        if feature_types == "text":
            return [
                ("Ridge", True),
                ("ElasticNet", True),
                ("SGDRegressor", True),
                ("PassiveAggressiveRegressor", False),
            ]
        if feature_types in _TEMPORAL_TYPES:
            return [
                ("Ridge", True),
                ("ElasticNet", True),
                ("DecisionTreeRegressor", False),
                ("AdaBoostRegressor", False),
            ]
        return [
            ("RandomForestRegressor", False),
            ("Ridge", True),
            ("ElasticNet", True),
            ("GradientBoostingRegressor", False),
            ("AdaBoostRegressor", False),
        ]

    # For Medium datasets, include more complex models but still keep some simpler ones for comparison and interpretability.
    if dataset_rows < 10_000:
        if feature_types == "numeric":
            return [
                ("Ridge", True),
                ("ElasticNet", True),
                ("SGDRegressor", True),
                ("PassiveAggressiveRegressor", False),
                ("RandomForestRegressor", False),
                ("GradientBoostingRegressor", False),
                ("HistGradientBoostingRegressor", False),
                ("AdaBoostRegressor", False),
            ]
        if feature_types == "categorical":
            return [
                ("Ridge", True),
                ("ElasticNet", True),
                ("RandomForestRegressor", False),
                ("ExtraTreesRegressor", False),
                ("HistGradientBoostingRegressor", False),
                ("BaggingRegressor", False),
            ]
        if feature_types == "text":
            return [
                ("Ridge", True),
                ("ElasticNet", True),
                ("SGDRegressor", True),
                ("PassiveAggressiveRegressor", False),
            ]
        if feature_types in _TEMPORAL_TYPES:
            return [
                ("RandomForestRegressor", False),
                ("GradientBoostingRegressor", False),
                ("HistGradientBoostingRegressor", False),
                ("AdaBoostRegressor", False),
            ]
        return [
            ("RandomForestRegressor", False),
            ("HistGradientBoostingRegressor", False),
            ("GradientBoostingRegressor", False),
            ("AdaBoostRegressor", False),
            ("BaggingRegressor", False),
        ]

    # For large datasets, prioritize scalable models that can handle more data and complexity, while still including some linear models for baseline comparison.
    if feature_types == "numeric":
        return [
            ("SGDRegressor", True),
            ("PassiveAggressiveRegressor", False),
            ("LinearRegression", True),
            ("ElasticNet", True),
            ("HistGradientBoostingRegressor", False),
            ("RandomForestRegressor", False),
            ("LGBMRegressor", False),
            ("CatBoostRegressor", False),
        ]
    if feature_types == "categorical":
        return [
            ("SGDRegressor", True),
            ("PassiveAggressiveRegressor", False),
            ("LinearRegression", True),
            ("ElasticNet", True),
            ("HistGradientBoostingRegressor", False),
            ("RandomForestRegressor", False),
            ("LGBMRegressor", False),
            ("CatBoostRegressor", False),
        ]
    if feature_types == "text":
        return [
            ("SGDRegressor", True),
            ("PassiveAggressiveRegressor", False),
            ("LinearRegression", True),
            ("Ridge", True),
            ("ElasticNet", True),
        ]
    if feature_types in _TEMPORAL_TYPES:
        return [
            ("HistGradientBoostingRegressor", False),
            ("RandomForestRegressor", False),
            ("AdaBoostRegressor", False),
            ("LGBMRegressor", False),
            ("CatBoostRegressor", False),
        ]
    return [
        ("HistGradientBoostingRegressor", False),
        ("RandomForestRegressor", False),
        ("SGDRegressor", True),
        ("PassiveAggressiveRegressor", False),
        ("LGBMRegressor", False),
        ("CatBoostRegressor", False),
    ]