"""
===========================================================
AutoML Scaling Engine (Updated)
===========================================================
Scales ALL numeric columns including datetime-derived columns.
Skips TF-IDF columns automatically.
"""

import os
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

import joblib
from autogluon.features.generators import AutoMLPipelineFeatureGenerator


# =========================================================
# Logging
# =========================================================

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("AutoMLScalingEngine")


# =========================================================
# Configuration
# =========================================================

class ScalingConfig:
    ENABLE_SKEW_DETECTION = True
    ENABLE_DECIMAL_SCALING = True
    ENABLE_ROBUST_SCALER = True
    PERSIST_SCALERS = True
    SCALER_VERSION = "3.0.0"


# =========================================================
# Metadata Manager
# =========================================================

class MetadataManager:

    def __init__(self, path: str):
        self.path = path
        self.data = self._load()

    def _load(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def _json_serializer(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    def update_scaling_summary(self, summary: Dict):
        self.data["scaling_summary"] = summary
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4, default=self._json_serializer)


# =========================================================
# Numeric Profiler
# =========================================================

class NumericProfiler:


    @staticmethod
    def safe(value):
        if pd.isna(value) or np.isinf(value):
            return 0.0
        return float(value)

    @staticmethod
    def profile(series: pd.Series) -> Dict[str, Any]:

        clean = series.dropna()

        if clean.empty:
            return {"empty": True}

        mean = clean.mean()
        std = clean.std()
        skew = clean.skew()
        kurt = clean.kurtosis()

        q75 = clean.quantile(0.75)
        q25 = clean.quantile(0.25)

        iqr = q75 - q25

        return {
            "mean": NumericProfiler.safe(mean),
            "std": NumericProfiler.safe(std),
            "min": NumericProfiler.safe(clean.min()),
            "max": NumericProfiler.safe(clean.max()),
            "skew": NumericProfiler.safe(skew),
            "kurtosis": NumericProfiler.safe(kurt),
            "iqr": NumericProfiler.safe(iqr)
        }

# =========================================================
# Scaling Strategies
# =========================================================

class BaseScalingStrategy:
    name = "base"

    def fit_transform(self, series: pd.Series):
        raise NotImplementedError

    def save(self, path):
        pass


class ZScoreScaling(BaseScalingStrategy):
    name = "zscore"

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, series):
        return self.scaler.fit_transform(series.values.reshape(-1,1)).flatten()

    def save(self, path):
        joblib.dump(self.scaler, path)


class MinMaxScaling(BaseScalingStrategy):
    name = "minmax"

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, series):
        return self.scaler.fit_transform(series.values.reshape(-1,1)).flatten()

    def save(self, path):
        joblib.dump(self.scaler, path)


class RobustScalingStrategy(BaseScalingStrategy):
    name = "robust"

    def __init__(self):
        self.scaler = RobustScaler()

    def fit_transform(self, series):
        return self.scaler.fit_transform(series.values.reshape(-1,1)).flatten()

    def save(self, path):
        joblib.dump(self.scaler, path)


class DecimalScalingStrategy(BaseScalingStrategy):
    name = "decimal"

    def fit_transform(self, series):

        max_abs = series.abs().max()

        if max_abs == 0:
            return series.values

        k = len(str(int(abs(max_abs))))
        return (series / (10 ** k)).values


# =========================================================
# Strategy Selector
# =========================================================

class ScalingStrategySelector:

    @staticmethod
    def select(series: pd.Series, profile: Dict[str, Any]):

        if profile.get("empty"):
            return None

        min_val = profile["min"]
        max_val = profile["max"]
        std_val = profile["std"]
        skew_val = profile["skew"]
        iqr = profile["iqr"]

        # already scaled
        if 0 <= min_val and max_val <= 1:
            return None

        # skew
        if ScalingConfig.ENABLE_SKEW_DETECTION and abs(skew_val) > 2:
            return RobustScalingStrategy()

        # huge values
        if ScalingConfig.ENABLE_DECIMAL_SCALING and abs(max_val) > 1e6:
            return DecimalScalingStrategy()

        # outliers
        if ScalingConfig.ENABLE_ROBUST_SCALER and iqr > std_val:
            return RobustScalingStrategy()

        # default
        if std_val > abs(profile["mean"]):
            return ZScoreScaling()

        return MinMaxScaling()


# =========================================================
# Scaling Engine
# =========================================================

class ScalingEngine:

    def __init__(self, userid: str):

        self.userid = userid

        self.meta_dir = f"storage/meta_data/{userid}"
        self.output_dir = f"storage/output/{userid}"

        self.scaler_store = os.path.join(self.output_dir, "scalers")

        os.makedirs(self.scaler_store, exist_ok=True)

    def run(self):

        profiling_files = [
            f for f in os.listdir(self.meta_dir)
            if f.endswith("_profiling.json")
        ]

        for file in profiling_files:
            self.process_file(file)

    def process_file(self, profiling_file):

        base_name = profiling_file.replace("_profiling.json","")

        metadata_path = os.path.join(self.meta_dir, profiling_file)

        encoded_path = os.path.join(self.output_dir, f"{base_name}_encoded.csv")

        scaling_output = os.path.join(self.output_dir, f"{base_name}_scaling.csv")

        if not os.path.exists(encoded_path):
            logger.warning(f"Encoded file missing for {base_name}")
            return

        logger.info(f"Processing {base_name}")

        metadata = MetadataManager(metadata_path)

        df = pd.read_csv(encoded_path)

        # =========================================================
        # AUTO DETECT NUMERIC COLUMNS (NEW LOGIC)
        # =========================================================

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # skip tfidf columns
        numeric_columns = [
            col for col in numeric_columns
            if "tfidf" not in col.lower()
        ]

        scaling_summary = {}
        scaled_columns = []

        for column in numeric_columns:

            try:

                logger.info(f"Scaling column: {column}")

                profile = NumericProfiler.profile(df[column])

                strategy = ScalingStrategySelector.select(df[column], profile)

                if strategy is None:

                    scaling_summary[column] = {
                        "method": "skipped_already_scaled"
                    }

                    continue

                df[column] = strategy.fit_transform(df[column])

                if ScalingConfig.PERSIST_SCALERS and hasattr(strategy, "save"):

                    scaler_path = os.path.join(
                        self.scaler_store,
                        f"{base_name}_{column}_scaler.pkl"
                    )

                    strategy.save(scaler_path)

                scaling_summary[column] = {
                    "method": strategy.name,
                    "profile": profile
                }

                scaled_columns.append(column)

            except Exception as e:

                logger.error(f"Error scaling {column}: {str(e)}")

                logger.error(traceback.format_exc())

        # =========================================================
        # AutoGluon Compatibility
        # =========================================================

        generator = AutoMLPipelineFeatureGenerator(
            enable_numeric_features=True,
            enable_categorical_features=False,
            enable_datetime_features=False,
            enable_text_special_features=False,
            enable_text_ngram_features=False,
        )

        df_scaled = generator.fit_transform(df)

        df_scaled.to_csv(scaling_output, index=False)

        scaling_metadata = {
            "scaling_version": ScalingConfig.SCALER_VERSION,
            "scaling_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_columns_scaled": len(scaled_columns),
            "scaled_columns": scaled_columns,
            "methods_used": list(
                set(v["method"] for v in scaling_summary.values())
            ),
            "column_wise_scaling": scaling_summary
        }

        metadata.update_scaling_summary(scaling_metadata)

        logger.info(f"Completed scaling for {base_name}")


# =========================================================
# Entry
# =========================================================

def main():

    userid = input("Enter user id: ").strip()

    if not userid:
        print("User ID required.")
        return

    engine = ScalingEngine(userid)

    engine.run()

    print("Scaling pipeline completed successfully.")


if __name__ == "__main__":
    main()