# tests/test_engine.py
from app.engine import calculate_loan_monthly_payment, monte_carlo_purchase_risk

def test_loan_payment_zero_apr():
    p = calculate_loan_monthly_payment(100000, 50000, 0.0, 12)
    assert abs(p - 4166.6666) < 1.0  # principal 50k / 12

def test_monte_carlo_runs():
    snapshot = {
        "monthly_net_income": 60000,
        "fixed_expenses": 15000,
        "variable_expenses": 10000,
        "savings_balance": 100000,
        "emergency_target": 75000,
        "debt_monthly": 2000,
        "monthly_additional_savings": 0
    }
    out = monte_carlo_purchase_risk(snapshot, price=200000, down_payment=50000, apr=0.08, term_months=60, months_ahead=12, sims=200)
    assert "stats" in out
    assert 0.0 <= out["stats"]["prob_shortfall"] <= 1.0
