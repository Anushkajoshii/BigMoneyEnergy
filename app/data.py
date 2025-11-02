# app/data_generator.py
import pandas as pd
import numpy as np

def generate_synthetic_users(n=100, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        monthly_net_income = rng.normal(70000, 20000)  # in local currency; adjust for your market
        monthly_net_income = max(10000, monthly_net_income)
        fixed = rng.normal(20000, 5000)
        fixed = max(5000, fixed)
        variable = rng.normal(15000, 7000)
        variable = max(2000, variable)
        savings_balance = rng.normal(150000, 80000)
        savings_balance = max(0, savings_balance)
        emergency_target = (fixed + variable) * 3  # 3 months
        debt_total = rng.choice([0, 50000, 200000, 500000], p=[0.5, 0.2, 0.2, 0.1])
        debt_monthly = debt_total * 0.02  # crude proxy
        rows.append({
            "monthly_net_income": monthly_net_income,
            "fixed_expenses": fixed,
            "variable_expenses": variable,
            "savings_balance": savings_balance,
            "emergency_target": emergency_target,
            "debt_total": debt_total,
            "debt_monthly": debt_monthly
        })
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = generate_synthetic_users(10)
    print(df.head().to_string(index=False))
