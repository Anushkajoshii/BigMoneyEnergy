# # app/models/model.py
# import os
# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import ElasticNet
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.pkl")

# def featurize(df: pd.DataFrame):
#     """
#     Expects df to have monthly_net_income, fixed_expenses, variable_expenses, debt_monthly.
#     Engineer features for a baseline model.
#     """
#     X = pd.DataFrame()
#     X["income"] = df["monthly_net_income"]
#     X["fixed"] = df["fixed_expenses"]
#     X["variable"] = df["variable_expenses"]
#     X["debt_monthly"] = df["debt_monthly"]
#     X["discretionary"] = X["income"] - (X["fixed"] + X["variable"] + X["debt_monthly"])
#     X["savings_rate"] = X["discretionary"] / (X["income"] + 1e-9)
#     return X.fillna(0)

# def train(df: pd.DataFrame, target_col="monthly_savings"):
#     """
#     Train a baseline ElasticNet model on df; df must contain features + target.
#     Saves model to MODEL_PATH.
#     """
#     X = featurize(df)
#     y = df[target_col]
#     pipeline = Pipeline([
#         ("scaler", StandardScaler()),
#         ("model", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=5000))
#     ])
#     pipeline.fit(X, y)
#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump(pipeline, f)
#     return pipeline

# def load_model():
#     if not os.path.exists(MODEL_PATH):
#         return None
#     with open(MODEL_PATH, "rb") as f:
#         return pickle.load(f)

# def predict(snapshot: dict):
#     """
#     Given a single snapshot dict, return predicted monthly savings.
#     """
#     model = load_model()
#     if model is None:
#         return None
#     df = pd.DataFrame([snapshot])
#     X = featurize(df)
#     pred = model.predict(X)[0]
#     return float(pred)


# app/models/model.py
# app/models/model.py
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODEL_POINTER: Path = BASE_DIR / "saved_model.pkl"
MODEL_ARCHIVE_DIR: Path = BASE_DIR / "model_archive"
METRICS_PATH: Path = BASE_DIR / "training_metrics.csv"
MODEL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Expect df to have monthly_net_income, fixed_expenses, variable_expenses, debt_monthly."""
    X = pd.DataFrame(index=df.index)
    X["income"] = df["monthly_net_income"]
    X["fixed"] = df["fixed_expenses"]
    X["variable"] = df["variable_expenses"]
    X["debt_monthly"] = df["debt_monthly"]
    X["discretionary"] = X["income"] - (X["fixed"] + X["variable"] + X["debt_monthly"])
    X["savings_rate"] = X["discretionary"] / (X["income"] + 1e-9)
    return X.fillna(0)


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=5000))
    ])


def _serialize_model(pipeline: Pipeline, version: str, created_at: str) -> bytes:
    payload = {"pipeline": pipeline, "metadata": {"version": version, "created_at": created_at}}
    return pickle.dumps(payload)


def save_model(pipeline: Pipeline, pointer_path: Optional[Path] = None, archive_dir: Optional[Path] = None) -> Path:
    """
    Save timestamped model to archive_dir and write pointer_path to latest.
    Uses module globals if explicit paths not provided (works with monkeypatch in tests).
    """
    pointer_path = Path(pointer_path) if pointer_path is not None else MODEL_POINTER
    archive_dir = Path(archive_dir) if archive_dir is not None else MODEL_ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)

    version = uuid.uuid4().hex[:8]
    created_at = datetime.utcnow().isoformat() + "Z"
    filename = f"model_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{version}.pkl"
    archive_path = archive_dir / filename

    with open(archive_path, "wb") as f:
        f.write(_serialize_model(pipeline, version, created_at))

    with open(pointer_path, "wb") as f:
        f.write(_serialize_model(pipeline, version, created_at))

    logger.info("Model archived to: %s", archive_path)
    logger.info("Model pointer updated: %s", pointer_path)
    return archive_path


def load_model(path: Optional[Path] = None) -> Optional[Pipeline]:
    """Load pipeline saved at pointer (or explicit path)."""
    path = Path(path) if path is not None else MODEL_POINTER
    if not path.exists():
        logger.debug("Model not found at %s", path)
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "pipeline" in payload:
            return payload["pipeline"]
        return payload
    except Exception as e:
        logger.error("Failed to load model from %s: %s", path, e)
        return None


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def train(
    df: pd.DataFrame,
    target_col: str = "monthly_savings",
    test_size: float = 0.2,
    save: bool = True,
    save_metrics: bool = False,
    metrics_path: Path = METRICS_PATH,
    cv: int = 5
) -> Tuple[Pipeline, Dict[str, float]]:
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    X_train, y_train = featurize(train_df), train_df[target_col]
    X_test, y_test = featurize(test_df), test_df[target_col]

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    # cross-val
    try:
        if len(df) >= cv:
            cv_scores = cross_val_score(pipeline, featurize(df), df[target_col], cv=cv, scoring="r2")
            metrics["cv_mean"] = float(cv_scores.mean())
            metrics["cv_std"] = float(cv_scores.std())
    except Exception as e:
        logger.warning("Cross-validation failed: %s", e)

    # save
    archive_path: Optional[Path] = None
    if save:
        archive_path = save_model(pipeline)
        logger.info("Saved trained model to archive: %s", archive_path)

    if save_metrics:
        record = {
            "model_path": str(archive_path) if archive_path else "",
            "n_samples": len(df),
            "test_size": test_size,
            "r2": metrics.get("r2"),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "cv_mean": metrics.get("cv_mean"),
            "cv_std": metrics.get("cv_std"),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        df_metrics = pd.DataFrame([record])
        if metrics_path.exists():
            df_metrics.to_csv(metrics_path, mode="a", header=False, index=False)
        else:
            df_metrics.to_csv(metrics_path, index=False)
        logger.info("Metrics appended to: %s", metrics_path)

    logger.info("Model evaluation: R2=%.3f MAE=%.2f RMSE=%.2f", metrics["r2"], metrics["mae"], metrics["rmse"])
    if "cv_mean" in metrics:
        logger.info("Cross-validation R² mean=%.3f std=%.3f", metrics["cv_mean"], metrics["cv_std"])

    return pipeline, metrics


def predict(snapshot: dict, model_path: Optional[Path] = None) -> Optional[float]:
    model = load_model(path=model_path)
    if model is None:
        logger.warning("No model found when calling predict()")
        return None
    df = pd.DataFrame([snapshot])
    X = featurize(df)
    pred = model.predict(X)[0]
    return float(pred)
