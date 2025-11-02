# app/engine.py
import numpy as np
import pandas as pd

def rule_checks(snapshot: dict, loan_monthly_payment: float) -> dict:
    """
    Run quick rule-based checks and return dictionary with flags and messages.
    snapshot keys: monthly_net_income, fixed_expenses, variable_expenses, savings_balance, emergency_target, debt_monthly, debt_total
    """
    income = float(snapshot.get("monthly_net_income", 0))
    fixed = float(snapshot.get("fixed_expenses", 0))
    variable = float(snapshot.get("variable_expenses", 0))
    savings = float(snapshot.get("savings_balance", 0))
    emergency_target = float(snapshot.get("emergency_target", 0))
    debt_monthly = float(snapshot.get("debt_monthly", 0))

    essential = fixed + variable
    monthly_available = income - essential - debt_monthly
    if monthly_available < 0:
        monthly_available = 0

    # DTI: (existing debt payments + new loan payment) / income
    dti = (debt_monthly + loan_monthly_payment) / (income + 1e-9)

    messages = []
    safe = True

    # emergency fund check
    if savings < emergency_target:
        messages.append(f"Emergency fund below target: {savings:.0f} vs target {emergency_target:.0f}")
        safe = False

    # DTI threshold (example: 0.36)
    if dti > 0.36:
        messages.append(f"Debt-to-income ratio would be {dti:.2f} (> 0.36 recommended).")
        safe = False

    # monthly payment should not exceed, say, 15% of income
    if loan_monthly_payment / (income + 1e-9) > 0.15:
        messages.append(f"New loan monthly payment {loan_monthly_payment:.0f} is >15% of monthly income.")
        safe = False

    return {"safe": safe, "messages": messages, "dti": dti, "monthly_available": monthly_available}


def calculate_loan_monthly_payment(price: float, down_payment: float, apr: float, term_months: int) -> float:
    """Simple amortizing loan monthly payment formula."""
    principal = max(0.0, price - down_payment)
    if principal <= 0 or apr <= 0 or term_months <= 0:
        # handle zero-interest or fully paid downpayment
        if term_months <= 0:
            return 0.0
        return principal / term_months
    monthly_rate = apr / 12.0
    payment = (principal * monthly_rate) / (1 - (1 + monthly_rate) ** (-term_months))
    return float(payment)


def monte_carlo_purchase_risk(snapshot: dict,
                              price: float,
                              down_payment: float,
                              apr: float,
                              term_months: int,
                              months_ahead: int = 12,
                              sims: int = 3000,
                              income_volatility: float = 0.05,
                              expense_shock_prob: float = 0.05,
                              expense_shock_scale: float = 0.5):
    """
    Run Monte Carlo simulation to estimate probability of shortfall in next `months_ahead` months if purchase happens now.
    Returns dict with probability_of_shortfall, distribution of final savings, and summary stats.
    """
    income = float(snapshot.get("monthly_net_income", 0))
    fixed = float(snapshot.get("fixed_expenses", 0))
    variable = float(snapshot.get("variable_expenses", 0))
    savings = float(snapshot.get("savings_balance", 0))
    emergency_target = float(snapshot.get("emergency_target", 0))
    debt_monthly = float(snapshot.get("debt_monthly", 0))
    monthly_contribution = float(snapshot.get("monthly_additional_savings", 0))  # user action

    # monthly loan payment
    loan_payment = calculate_loan_monthly_payment(price, down_payment, apr, term_months)

    # precompute monthly essential expenses
    base_essential = fixed + variable

    rng = np.random.default_rng(seed=42)

    final_savings = np.zeros(sims)
    shortfall_flags = np.zeros(sims, dtype=bool)

    for sim in range(sims):
        s = savings
        # simulate month by month
        for m in range(months_ahead):
            # simulate income shock: income ~ N(income, income*income_volatility)
            inc = rng.normal(income, max(1e-6, income * income_volatility))
            inc = max(0.0, inc)

            # expense shock: with some small probability a big unexpected expense occurs
            if rng.random() < expense_shock_prob:
                shock = base_essential * expense_shock_scale * rng.random()
            else:
                shock = 0.0

            # monthly expenses
            essential = base_essential + shock

            # monthly net change = inc - essential - debt_monthly - loan_payment + monthly_contribution
            delta = inc - essential - debt_monthly - loan_payment + monthly_contribution

            s = s + delta
            # floor at large negative allowed (representing borrowing), but we just track shortfall
            if s < -1e6:
                s = -1e6

            # check early shortfall: if emergency fund (savings) falls below emergency_target at any month -> shortfall
            if s < emergency_target:
                shortfall_flags[sim] = True
                # we can break early if desired, but continue to build distribution
                # break

        final_savings[sim] = s

    prob_shortfall = float(np.mean(shortfall_flags))
    stats = {
        "prob_shortfall": prob_shortfall,
        "final_savings_mean": float(np.mean(final_savings)),
        "final_savings_median": float(np.median(final_savings)),
        "final_savings_p10": float(np.percentile(final_savings, 10)),
        "final_savings_p90": float(np.percentile(final_savings, 90)),
        "loan_payment": float(loan_payment)
    }
    return {"stats": stats, "final_savings": final_savings, "shortfall_flags": shortfall_flags}
