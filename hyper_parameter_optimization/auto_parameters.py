from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

try:
    import optuna
except Exception as exc:  # pragma: no cover
    raise ImportError("Optuna is required for hyperparameter optimization.") from exc

try:
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
except Exception as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for hyperparameter optimization.") from exc


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("HPO")


_HAS_XGB = False
try:
    import xgboost  # noqa: F401
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

_HAS_LGBM = False
try:
    import lightgbm  # noqa: F401
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

_HAS_CATBOOST = False
try:
    import catboost  # noqa: F401
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False


def _is_supported(model_name: str) -> bool:
    supported = {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "AdaBoostClassifier",
        "AdaBoostRegressor",
        "BaggingClassifier",
        "BaggingRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "LogisticRegression",
        "SGDClassifier",
        "SGDRegressor",
        "PassiveAggressiveClassifier",
        "PassiveAggressiveRegressor",
        "KNeighborsClassifier",
        "KNeighborsRegressor",
        "SVC",
        "LinearSVC",
        "GaussianNB",
        "MultinomialNB",
        "CategoricalNB",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "CatBoostClassifier",
        "CatBoostRegressor",
    }
    if model_name in {"XGBClassifier", "XGBRegressor"}:
        return _HAS_XGB
    if model_name in {"LGBMClassifier", "LGBMRegressor"}:
        return _HAS_LGBM
    if model_name in {"CatBoostClassifier", "CatBoostRegressor"}:
        return _HAS_CATBOOST
    return model_name in supported


def get_search_space(trial: optuna.Trial, model_name: str, task: str) -> Optional[Dict[str, Any]]:
    if not _is_supported(model_name):
        return None

    if model_name in {"RandomForestClassifier", "RandomForestRegressor"}:
        params: Dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        if model_name == "RandomForestClassifier":
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        else:
            params["criterion"] = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
        return params

    if model_name in {"ExtraTreesClassifier", "ExtraTreesRegressor"}:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        return params

    if model_name == "LogisticRegression":
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
        # Keep a single static distribution for "penalty" across all trials.
        # Optuna disallows changing categorical choice sets for the same name.
        raw_penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

        # Normalize unsupported solver/penalty combinations to valid settings.
        if solver == "lbfgs":
            penalty = "l2"
        elif solver == "liblinear" and raw_penalty == "elasticnet":
            penalty = "l1"
        else:
            penalty = raw_penalty

        params = {
            "solver": solver,
            "penalty": penalty,
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            "max_iter": trial.suggest_int("max_iter", 400, 5000),
        }
        if solver == "saga" and penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return params

    if model_name in {"SGDClassifier", "SGDRegressor"}:
        params: Dict[str, Any] = {
            "alpha": trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "max_iter": trial.suggest_int("max_iter", 500, 6000),
        }
        if model_name == "SGDClassifier":
            params["loss"] = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber"])
        else:
            params["loss"] = trial.suggest_categorical(
                "loss",
                ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            )
            if params["loss"] in {"huber", "epsilon_insensitive", "squared_epsilon_insensitive"}:
                params["epsilon"] = trial.suggest_float("epsilon", 1e-4, 1.0, log=True)
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return params

    if model_name == "PassiveAggressiveClassifier":
        return {
            "C": trial.suggest_float("C", 1e-5, 20.0, log=True),
            "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
            "max_iter": trial.suggest_int("max_iter", 500, 6000),
        }

    if model_name == "PassiveAggressiveRegressor":
        return {
            "C": trial.suggest_float("C", 1e-5, 20.0, log=True),
            "loss": trial.suggest_categorical("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
            "epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 500, 6000),
        }

    if model_name in {"KNeighborsClassifier", "KNeighborsRegressor"}:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        }

    if model_name == "SVC":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        }

    if model_name == "LinearSVC":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
            "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
            "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        }

    if model_name in {"DecisionTreeClassifier", "DecisionTreeRegressor"}:
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    if model_name in {"AdaBoostClassifier", "AdaBoostRegressor"}:
        params: Dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 2.0, log=True),
        }
        if model_name == "AdaBoostRegressor":
            params["loss"] = trial.suggest_categorical("loss", ["linear", "square", "exponential"])
        return params

    if model_name in {"BaggingClassifier", "BaggingRegressor"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 10, 300),
            "max_samples": trial.suggest_float("max_samples", 0.4, 1.0),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "bootstrap_features": trial.suggest_categorical("bootstrap_features", [False, True]),
        }

    if model_name in {"GradientBoostingClassifier", "GradientBoostingRegressor"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    if model_name in {"HistGradientBoostingClassifier", "HistGradientBoostingRegressor"}:
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True),
        }

    if model_name == "GaussianNB":
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-7, log=True),
        }

    if model_name == "MultinomialNB":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        }

    if model_name == "CategoricalNB":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        }

    if model_name in {"Ridge", "Lasso", "ElasticNet"}:
        params: Dict[str, Any] = {
            "alpha": trial.suggest_float("alpha", 1e-5, 1e3, log=True),
        }
        if model_name in {"Lasso", "ElasticNet"}:
            params["max_iter"] = trial.suggest_int("max_iter", 1000, 12000)
        if model_name == "ElasticNet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.01, 0.99)
        return params

    if model_name in {"XGBClassifier", "XGBRegressor"} and _HAS_XGB:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        if task == "classification":
            params["eval_metric"] = "logloss"
        return params

    if model_name in {"LGBMClassifier", "LGBMRegressor"} and _HAS_LGBM:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return params

    if model_name in {"CatBoostClassifier", "CatBoostRegressor"} and _HAS_CATBOOST:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 400),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        }
        return params

    return None


def _score_or_default(scores: np.ndarray, default: float) -> float:
    if scores is None or len(scores) == 0:
        return default
    return float(np.mean(scores))


def _baseline_cv_score(
    model_instance: Any,
    X_train: Any,
    y_train: Any,
    scoring: str,
    cv: int,
) -> float:
    try:
        scores = cross_val_score(
            model_instance,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            error_score="raise",
            n_jobs=-1,
        )
        return _score_or_default(scores, float("-inf"))
    except Exception as exc:
        logger.warning("Baseline CV failed: %s", exc)
        return float("-inf")




def run_hpo(
    model_name: str,
    model_instance: Any,
    X_train: Any,
    y_train: Any,
    task: str,
    n_trials: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    scoring = "accuracy" if task == "classification" else "r2"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(),
    )

    if not _is_supported(model_name):
        logger.warning("HPO skipped: unsupported model '%s'.", model_name)
        baseline_score = _baseline_cv_score(model_instance, X_train, y_train, scoring, cv)
        return {
            "best_params": {},
            "best_score": float(baseline_score),
            "study": study,
            "status": "skipped",
            "n_trials": 0,
        }

    def objective(trial: optuna.Trial) -> float:
        params = get_search_space(trial, model_name, task)
        if params is None:
            logger.warning("HPO skipped: no search space for '%s'.", model_name)
            return float("-inf")

        try:
            estimator = clone(model_instance)
            clean_params = {k: v for k, v in params.items() if v is not None}
            estimator.set_params(**clean_params)

            scores = cross_val_score(
                estimator,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                error_score="raise",
                n_jobs=-1,
            )
            score = _score_or_default(scores, float("-inf"))

            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            logger.info("HPO trial=%d params=%s score=%.6f", trial.number, clean_params, score)
            return score

        except optuna.TrialPruned:
            logger.info("HPO trial=%d pruned", trial.number)
            raise
        except Exception as exc:
            logger.warning("HPO trial=%d failed: %s", trial.number, exc)
            print(f"[HPO] trial={trial.number} failed: {exc}")
            return float("-inf")

    try:
        study.optimize(objective, n_trials=n_trials)
    except Exception as exc:
        logger.warning("HPO optimization failed: %s", exc)
        print(f"[HPO] optimization failed: {exc}")

    best_params = study.best_params if len(study.trials) else {}
    best_score = study.best_value if len(study.trials) else float("-inf")

    status = "completed" if len(study.trials) else "failed"
    if not np.isfinite(best_score):
        baseline_score = _baseline_cv_score(model_instance, X_train, y_train, scoring, cv)
        if np.isfinite(baseline_score):
            best_score = baseline_score
            status = "skipped"

    return {
        "best_params": best_params,
        "best_score": float(best_score),
        "study": study,
        "status": status,
        "n_trials": len(study.trials),
    }
