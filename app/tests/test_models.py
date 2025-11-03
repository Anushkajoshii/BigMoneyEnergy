# tests/test_models.py
import pandas as pd
from app.data import generate_synthetic_users
from app.models.model import train, load_model, predict

def test_train_and_predict(tmp_path):
    df = generate_synthetic_users(50)
    # craft monthly_savings target as income - (fixed+variable+debt_monthly)
    df["monthly_savings"] = df["monthly_net_income"] - (df["fixed_expenses"] + df["variable_expenses"] + df["debt_monthly"])
    model = train(df)
    assert model is not None
    m = load_model()
    assert m is not None
    snapshot = df.iloc[0].to_dict()
    pred = predict(snapshot)
    assert isinstance(pred, float)
