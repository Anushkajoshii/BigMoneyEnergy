# # tests/test_models.py
# import pandas as pd
# from app.data import generate_synthetic_users
# from app.models.model import train, load_model, predict

# def test_train_and_predict(tmp_path):
#     df = generate_synthetic_users(50)
#     # craft monthly_savings target as income - (fixed+variable+debt_monthly)
#     df["monthly_savings"] = df["monthly_net_income"] - (df["fixed_expenses"] + df["variable_expenses"] + df["debt_monthly"])
#     model = train(df)
#     assert model is not None
#     m = load_model()
#     assert m is not None
#     snapshot = df.iloc[0].to_dict()
#     pred = predict(snapshot)
#     assert isinstance(pred, float)


# tests/test_model.py
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# import module under test
import app.models.model as model_mod


def make_synthetic_df(n=200, seed=42):
    rng = np.random.RandomState(seed)
    income = rng.normal(50000, 10000, size=n).clip(min=5000)
    fixed = rng.normal(15000, 3000, size=n).clip(min=0)
    variable = rng.normal(8000, 2000, size=n).clip(min=0)
    debt = rng.normal(2000, 1000, size=n).clip(min=0)
    discretionary = income - (fixed + variable + debt)
    # simple target: some portion of discretionary plus noise
    monthly_savings = (discretionary * 0.4) + rng.normal(0, 1000, size=n)
    df = pd.DataFrame({
        "monthly_net_income": income,
        "fixed_expenses": fixed,
        "variable_expenses": variable,
        "debt_monthly": debt,
        "monthly_savings": monthly_savings
    })
    return df


@pytest.fixture
def tmp_model_dir(tmp_path, monkeypatch):
    """
    Create a temporary model directory and point the module constants to it
    so tests don't touch real model files.
    """
    temp_dir = tmp_path / "models"
    temp_dir.mkdir()
    # monkeypatch paths in module
    monkeypatch.setattr(model_mod, "MODEL_POINTER", temp_dir / "saved_model.pkl")
    monkeypatch.setattr(model_mod, "MODEL_ARCHIVE_DIR", temp_dir / "model_archive")
    monkeypatch.setattr(model_mod, "METRICS_PATH", temp_dir / "training_metrics.csv")
    model_mod.MODEL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    return temp_dir


def test_featurize_basic():
    df = pd.DataFrame({
        "monthly_net_income": [50000, 20000],
        "fixed_expenses": [15000, 8000],
        "variable_expenses": [8000, 4000],
        "debt_monthly": [2000, 1000]
    })
    X = model_mod.featurize(df)
    expected_cols = {"income", "fixed", "variable", "debt_monthly", "discretionary", "savings_rate"}
    assert set(X.columns) == expected_cols
    assert not X.isna().any().any()


def test_train_and_predict_roundtrip(tmp_model_dir):
    df = make_synthetic_df(n=300)
    # train with saving model into tmp dir
    pipeline, metrics = model_mod.train(df, save=True, save_metrics=True)
    assert "r2" in metrics and "mae" in metrics and "rmse" in metrics
    # pointer file created
    assert model_mod.MODEL_POINTER.exists()
    # archive dir populated
    assert any(model_mod.MODEL_ARCHIVE_DIR.iterdir())
    # now test predict on a sample snapshot
    sample = {
        "monthly_net_income": float(df["monthly_net_income"].iloc[0]),
        "fixed_expenses": float(df["fixed_expenses"].iloc[0]),
        "variable_expenses": float(df["variable_expenses"].iloc[0]),
        "debt_monthly": float(df["debt_monthly"].iloc[0])
    }
    pred = model_mod.predict(sample)
    assert isinstance(pred, float)


def test_train_raises_on_missing_target():
    df = pd.DataFrame({
        "monthly_net_income": [10000],
        "fixed_expenses": [2000],
        "variable_expenses": [1000],
        "debt_monthly": [500]
    })
    with pytest.raises(ValueError):
        model_mod.train(df, target_col="monthly_savings")
