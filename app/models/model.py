# app/models/model.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.pkl")

def featurize(df: pd.DataFrame):
    """
    Expects df to have monthly_net_income, fixed_expenses, variable_expenses, debt_monthly.
    Engineer features for a baseline model.
    """
    X = pd.DataFrame()
    X["income"] = df["monthly_net_income"]
    X["fixed"] = df["fixed_expenses"]
    X["variable"] = df["variable_expenses"]
    X["debt_monthly"] = df["debt_monthly"]
    X["discretionary"] = X["income"] - (X["fixed"] + X["variable"] + X["debt_monthly"])
    X["savings_rate"] = X["discretionary"] / (X["income"] + 1e-9)
    return X.fillna(0)

def train(df: pd.DataFrame, target_col="monthly_savings"):
    """
    Train a baseline ElasticNet model on df; df must contain features + target.
    Saves model to MODEL_PATH.
    """
    X = featurize(df)
    y = df[target_col]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=5000))
    ])
    pipeline.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict(snapshot: dict):
    """
    Given a single snapshot dict, return predicted monthly savings.
    """
    model = load_model()
    if model is None:
        return None
    df = pd.DataFrame([snapshot])
    X = featurize(df)
    pred = model.predict(X)[0]
    return float(pred)
