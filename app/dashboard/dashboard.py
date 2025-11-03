# app/dashboard/dashboard.py
import numpy as np
import pandas as pd
from app.engine import monte_carlo_purchase_risk, calculate_loan_monthly_payment

def compare_buy_vs_wait(snapshot: dict, product_price: float, down_payment: float, apr: float, term_months: int,
                        horizon_months: int = 24, sims: int = 3000, extra_savings_options=None):
    """
    Returns a list of scenarios: buy_now and buy_after_n_months with extra savings options.
    extra_savings_options: list of extra monthly savings to test (e.g., [0, 5000, 10000])
    """
    if extra_savings_options is None:
        extra_savings_options = [0, 5000, 10000]

    scenarios = []

    # buy now scenario
    mc_now = monte_carlo_purchase_risk(snapshot, price=product_price, down_payment=down_payment, apr=apr, term_months=term_months,
                                      months_ahead=horizon_months, sims=sims)
    s_now = {"scenario": "buy_now", "extra_savings": 0, "stats": mc_now["stats"]}
    scenarios.append(s_now)

    # buy after N months with extra savings
    for extra in extra_savings_options:
        snap_copy = snapshot.copy()
        # simulate N months of adding extra savings (we will do simple deterministic projection: add extra*months to savings)
        for n in [3, 6, 9, 12]:
            snap_n = snap_copy.copy()
            snap_n["savings_balance"] = snap_n["savings_balance"] + extra * n
            # recompute down payment if user accumulates for down payment (we assume same down payment target)
            mc_later = monte_carlo_purchase_risk(snap_n, price=product_price, down_payment=down_payment, apr=apr, term_months=term_months,
                                                months_ahead=horizon_months, sims=max(1000, int(sims/3)))
            scenarios.append({
                "scenario": f"buy_after_{n}m",
                "months_wait": n,
                "extra_savings": extra,
                "stats": mc_later["stats"]
            })
    return scenarios
