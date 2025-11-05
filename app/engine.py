# import numpy as np
# from app.utils import fan_chart_figure

# def calculate_loan_monthly_payment(price, down_payment, apr, term_months):
#     """Calculate EMI for loan-based purchases."""
#     loan_amount = price - down_payment
#     if loan_amount <= 0 or term_months <= 0:
#         return 0

#     # Handle zero-interest loans separately
#     if apr == 0:
#         return loan_amount / term_months

#     r = apr / 12 / 100
#     return loan_amount * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)


# def rule_checks(snapshot, monthly_payment):
#     """Basic affordability checks."""
#     dti = (snapshot["debt_monthly"] + monthly_payment) / snapshot["monthly_net_income"]
#     safe = dti < 0.36 and snapshot["savings_balance"] >= snapshot["emergency_target"]

#     messages = []
#     if dti >= 0.36:
#         messages.append(f"High Debt-to-Income ratio ({dti:.2f}). Try lowering EMI or debts.")
#     if snapshot["savings_balance"] < snapshot["emergency_target"]:
#         messages.append("Emergency fund below target.")

#     return {"safe": safe, "messages": messages, "dti": dti}



# def monte_carlo_purchase_risk(
#     snapshot,
#     price,
#     down_payment,
#     apr,
#     term_months,
#     months_ahead,
#     sims,
#     purchase_type="One-time purchase",
#     buy_in_months=0,
# ):
#     """Simulate future savings under uncertainty and compute probability of shortfall."""

#     np.random.seed(42)

#     # Extract snapshot safely
#     monthly_income = snapshot.get("monthly_net_income", 0)
#     fixed_exp = snapshot.get("fixed_expenses", 0)
#     var_exp = snapshot.get("variable_expenses", 0)
#     savings = snapshot.get("savings_balance", 0)
#     add_savings = snapshot.get("monthly_additional_savings", 0)
#     emergency_target = snapshot.get("emergency_target", 100000)
#     debt_monthly = snapshot.get("debt_monthly", 0)

#     # Loan setup
#     # loan_amount = max(price - down_payment, 0)
#     # if loan_amount > 0:
#     #     monthly_rate = apr / 12
#     #     loan_payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -term_months)
#     # else:
#     #     loan_payment = 0
#     loan_amount = max(price - down_payment, 0)
#     if loan_amount > 0 and term_months > 0:
#         if apr == 0:
#             loan_payment = loan_amount / term_months
#         else:
#             monthly_rate = apr / 12
#             loan_payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -term_months)
#     else:
#         loan_payment = 0



#     # Simulate monthly income and expenses variation
#     income_sd = 0.05 * monthly_income
#     var_exp_sd = 0.1 * var_exp

#     final_savings = np.zeros(sims)

#     for s in range(sims):
#         cash = savings
#         for m in range(months_ahead):
#             income = np.random.normal(monthly_income, income_sd)
#             expenses = np.random.normal(var_exp, var_exp_sd) + fixed_exp + debt_monthly

#             # Add savings and subtract loan if purchase has been made
#             cash += income - expenses
#             cash += add_savings

#             if m >= buy_in_months:
#                 cash -= loan_payment

#         final_savings[s] = cash
#     # Compute statistics
#     prob_shortfall = float(np.mean(final_savings < emergency_target))
#     stats = {
#         "final_savings_mean": float(np.mean(final_savings)),
#         "final_savings_median": float(np.median(final_savings)),
#         "final_savings_p10": float(np.percentile(final_savings, 10)),
#         "final_savings_p90": float(np.percentile(final_savings, 90)),
#     }

#     # Create a simple simulated trajectory matrix for visualization
#     sim_matrix = np.tile(final_savings, (months_ahead, 1)).T  # fake paths just for chart

#     # return {
#     #     "prob_shortfall": prob_shortfall,
#     #     "final_savings": final_savings,
#     #     "stats": stats,
#     #     "chart": fan_chart_figure(sim_matrix, months_ahead)
#     # }
#     return {
#     "sim_matrix": sim_matrix,  # shape: (sims, months_ahead)
#     "final_savings": final_savings,
#     "prob_shortfall": prob_shortfall,
#     "stats": stats
# }


# app/engine.py
import numpy as np
from typing import Dict, Any

def calculate_loan_monthly_payment(price: float, down_payment: float, apr: float, term_months: int) -> float:
    """Calculate EMI for loan-based purchases."""
    loan_amount = max(price - down_payment, 0.0)
    if loan_amount <= 0 or term_months <= 0:
        return 0.0
    if apr == 0:
        return loan_amount / term_months
    r = apr / 12 / 100.0
    return loan_amount * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)


def rule_checks(snapshot: Dict[str, float], monthly_payment: float) -> Dict[str, Any]:
    """Basic affordability checks."""
    monthly_income = float(snapshot.get("monthly_net_income", 0.0))
    debt_monthly = float(snapshot.get("debt_monthly", 0.0))
    savings_balance = float(snapshot.get("savings_balance", 0.0))
    emergency_target = float(snapshot.get("emergency_target", 0.0))

    dti = (debt_monthly + monthly_payment) / (monthly_income + 1e-9)
    safe = dti < 0.36 and savings_balance >= emergency_target

    messages = []
    if dti >= 0.36:
        messages.append(f"High Debt-to-Income ratio ({dti:.2f}). Try lowering EMI or debts.")
    if savings_balance < emergency_target:
        messages.append("Emergency fund below target.")

    return {"safe": safe, "messages": messages, "dti": dti}


def monte_carlo_purchase_risk(
    snapshot: Dict[str, float],
    price: float,
    down_payment: float,
    apr: float,
    term_months: int,
    months_ahead: int,
    sims: int,
    purchase_type: str = "One-time purchase",
    buy_in_months: int = 0,
) -> Dict[str, Any]:
    """
    Simulate future savings under uncertainty and compute probability of shortfall.

    Returns a dict with keys:
      - "stats": dict including "prob_shortfall" and other summary stats
      - "final_savings": numpy array of final savings for each sim
      - "sim_matrix": (sims, months_ahead) matrix (duplicated final_savings for compatibility)
      - "prob_shortfall": top-level float (kept for legacy/tests)
    """
    rng = np.random.default_rng(42)

    # Extract snapshot safely
    monthly_income = float(snapshot.get("monthly_net_income", 0.0))
    fixed_exp = float(snapshot.get("fixed_expenses", 0.0))
    var_exp = float(snapshot.get("variable_expenses", 0.0))
    savings = float(snapshot.get("savings_balance", 0.0))
    add_savings = float(snapshot.get("monthly_additional_savings", 0.0))
    emergency_target = float(snapshot.get("emergency_target", 0.0))
    debt_monthly = float(snapshot.get("debt_monthly", 0.0))

    # Loan setup
    loan_amount = max(price - down_payment, 0.0)
    if loan_amount > 0 and term_months > 0:
        if apr == 0:
            loan_payment = loan_amount / term_months
        else:
            monthly_rate = apr / 12
            loan_payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -term_months)
    else:
        loan_payment = 0.0

    # Simulate monthly income and expenses variation
    income_sd = max(1e-3, 0.05 * monthly_income)
    var_exp_sd = max(1e-3, 0.1 * var_exp)

    final_savings = np.zeros(sims)

    for s in range(sims):
        cash = savings
        for m in range(months_ahead):
            income = rng.normal(monthly_income, income_sd)
            variable = rng.normal(var_exp, var_exp_sd)
            outflow = fixed_exp + max(0.0, variable) + debt_monthly
            if m >= buy_in_months:
                outflow += loan_payment
            cash += income - outflow + add_savings
        final_savings[s] = cash

    # Compute statistics
    prob_shortfall = float(np.mean(final_savings < emergency_target))
    stats = {
        "prob_shortfall": prob_shortfall,
        "final_savings_mean": float(np.mean(final_savings)),
        "final_savings_median": float(np.median(final_savings)),
        "final_savings_p10": float(np.percentile(final_savings, 10)),
        "final_savings_p90": float(np.percentile(final_savings, 90)),
    }

    # Create a simple simulated trajectory matrix for visualization compatibility
    sim_matrix = np.tile(final_savings, (months_ahead if months_ahead > 0 else 1, 1)).T

    return {
        "sim_matrix": sim_matrix,          # shape: (sims, months_ahead)
        "final_savings": final_savings,    # raw simulations
        "stats": stats,                    # summary with prob_shortfall
        "prob_shortfall": prob_shortfall,  # top-level convenience key (tests expect this)
    }
