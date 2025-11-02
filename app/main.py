# app/main.py
import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ui import inputs_panel
from app.engine import calculate_loan_monthly_payment, rule_checks, monte_carlo_purchase_risk
from app.utils import fan_chart_figure
import pandas as pd

st.set_page_config(page_title="GenZ Finance Advisor (MVP)", layout="wide")

st.title("GenZ Finance Advisor — Can I buy a BMW? (MVP)")
st.write("This is a prototype advisory tool. Not a substitute for professional financial advice.")

# Collect inputs
inputs = inputs_panel()

snapshot = {
    "monthly_net_income": inputs["monthly_net_income"],
    "fixed_expenses": inputs["fixed_expenses"],
    "variable_expenses": inputs["variable_expenses"],
    "savings_balance": inputs["savings_balance"],
    "emergency_target": inputs["emergency_target"],
    "debt_monthly": inputs["debt_monthly"],
    "monthly_additional_savings": inputs["monthly_additional_savings"]
}

loan_payment = calculate_loan_monthly_payment(inputs["car_price"], inputs["down_payment"], inputs["apr"], inputs["term_months"])

st.header("Quick Eligibility Checks")
checks = rule_checks(snapshot, loan_payment)
if checks["safe"]:
    st.success("Basic rule checks passed ✅")
else:
    st.error("Basic rule checks flagged issues ❗")

for m in checks["messages"]:
    st.write("- " + m)

st.write(f"Estimated monthly loan payment: {loan_payment:,.0f}")

st.header("Monte Carlo risk simulation")
horizon_months = st.slider("Simulation horizon (months)", min_value=6, max_value=60, value=24, step=6)
sims = st.selectbox("Number of simulations", options=[1000, 3000, 5000], index=1)

with st.spinner("Running simulation..."):
    mc = monte_carlo_purchase_risk(snapshot,
                                   price=inputs["car_price"],
                                   down_payment=inputs["down_payment"],
                                   apr=inputs["apr"],
                                   term_months=inputs["term_months"],
                                   months_ahead=horizon_months,
                                   sims=sims)

prob = mc["stats"]["prob_shortfall"]
st.metric("Probability of emergency-fund shortfall within horizon", f"{prob*100:.1f}%")

col1, col2 = st.columns([2, 1])
with col1:
    fig = fan_chart_figure(mc["final_savings"], snapshot["savings_balance"],
                           title=f"Distribution of final savings after {horizon_months} months")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Simulation summary")
    s = mc["stats"]
    st.write(pd.DataFrame({
        "stat": ["prob_shortfall", "mean_final_savings", "median_final_savings", "p10", "p90", "loan_monthly_payment"],
        "value": [f"{s['prob_shortfall']:.2f}", f"{s['final_savings_mean']:.0f}", f"{s['final_savings_median']:.0f}",
                  f"{s['final_savings_p10']:.0f}", f"{s['final_savings_p90']:.0f}", f"{s['loan_payment']:.0f}"]
    }))

st.header("Recommendation")
threshold = st.slider("Acceptable shortfall probability threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
if prob <= threshold and checks["safe"]:
    st.success("Recommendation: You *can* consider buying now (according to this model).")
else:
    st.warning("Recommendation: Not advisable to buy now. Consider delaying or increasing savings.")
    # Show simple actionable suggestions
    deficit_note = ""
    if snapshot["savings_balance"] < snapshot["emergency_target"]:
        deficit_note += f"- Increase emergency funds by {snapshot['emergency_target'] - snapshot['savings_balance']:.0f}.\n"
    if checks["dti"] > 0.36:
        deficit_note += f"- Lower monthly debt or down payment larger to reduce DTI.\n"
    st.write("Suggested actions:")
    st.write(deficit_note or "Consider small changes: increase monthly savings, reduce discretionary spend, raise down payment.")

st.header("EDA (quick view)")
st.write("Monthly snapshot:")
st.table(pd.DataFrame([snapshot]).T.rename(columns={0:"value"}))
