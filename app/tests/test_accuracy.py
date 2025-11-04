# app/tests/test_accuracy.py
import numpy as np
from app.engine import monte_carlo_purchase_risk

def test_monte_carlo_accuracy():
    """
    Check the Monte Carlo simulation behaves consistently and statistically sound.
    """
    snapshot = {
        "monthly_net_income": 70000,
        "variable_expenses": 10000,
        "fixed_expenses": 20000,
        "car_price": 1000000,
        "savings_balance" : 2000,
        "monthly_additional_savings": 5000,
        "emergency_target": 150000,
        "debt_monthly": 2000
    }

    price = 1000000       # ₹10L car
    down_payment = 200000 # ₹2L upfront
    apr = 0.09            # 9% annual interest rate
    term_months = 24
    months_ahead = 24
    sims = 500

    # Run multiple simulations to check stability
    probs = []
    for _ in range(5):
        result = monte_carlo_purchase_risk(
            snapshot,
            price,
            down_payment,
            apr,
            term_months,
            months_ahead,
            sims
        )
        assert "prob_shortfall" in result, "Expected prob_shortfall in output"
        assert 0 <= result["prob_shortfall"] <= 1, "Probability should be valid"
        probs.append(result["prob_shortfall"])

    # Check variation (stability)
    std_dev = np.std(probs)
    assert std_dev < 0.05, f"Simulation too unstable (std={std_dev:.3f})"

    # Optional: expected trend — higher expenses → higher shortfall
    snapshot_rich = snapshot.copy()
    snapshot_rich["fixed_expenses"] = 10000
    low_risk = monte_carlo_purchase_risk(
        snapshot_rich, price, down_payment, apr, term_months, months_ahead, sims
    )["prob_shortfall"]

    snapshot_broke = snapshot.copy()
    snapshot_broke["fixed_expenses"] = 60000
    high_risk = monte_carlo_purchase_risk(
        snapshot_broke, price, down_payment, apr, term_months, months_ahead, sims
    )["prob_shortfall"]

    assert high_risk > low_risk, (
        f"Model should reflect higher shortfall risk for higher expenses, "
        f"but got low_risk={low_risk:.2f}, high_risk={high_risk:.2f}"
    )

    # Print model "stability accuracy"
    mean_prob = np.mean(probs)
    stability_score = round((1 - std_dev) * 100, 2)
    print(f"\n✅ Monte Carlo stability score: {stability_score}% (avg shortfall={mean_prob:.2f})")
