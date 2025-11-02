# app/ui.py
import streamlit as st

def inputs_panel():
    st.sidebar.header("Your finances (monthly / balances)")
    monthly_net_income = st.sidebar.number_input("Monthly net income", value=70000.0, min_value=0.0, step=1000.0)
    fixed_expenses = st.sidebar.number_input("Monthly fixed expenses (rent etc.)", value=20000.0, min_value=0.0, step=500.0)
    variable_expenses = st.sidebar.number_input("Monthly variable expenses", value=15000.0, min_value=0.0, step=500.0)
    savings_balance = st.sidebar.number_input("Current liquid savings", value=150000.0, min_value=0.0, step=1000.0)
    emergency_target = st.sidebar.number_input("Emergency fund target (total)", value=90000.0, min_value=0.0, step=1000.0)
    debt_monthly = st.sidebar.number_input("Existing monthly debt payments", value=2000.0, min_value=0.0, step=100.0)

    st.sidebar.header("Car purchase details")
    car_price = st.sidebar.number_input("Car price (BMW)", value=2500000.0, min_value=0.0, step=10000.0)
    down_payment = st.sidebar.number_input("Planned down payment", value=500000.0, min_value=0.0, step=10000.0)
    apr = st.sidebar.number_input("Loan APR (decimal, e.g., 0.10 for 10%)", value=0.10, min_value=0.0, step=0.005)
    term_years = st.sidebar.number_input("Loan term (years)", value=5, min_value=1, max_value=7, step=1)
    monthly_additional_savings = st.sidebar.number_input("Planned extra monthly savings (if you commit)", value=0.0, min_value=0.0, step=500.0)

    return {
        "monthly_net_income": monthly_net_income,
        "fixed_expenses": fixed_expenses,
        "variable_expenses": variable_expenses,
        "savings_balance": savings_balance,
        "emergency_target": emergency_target,
        "debt_monthly": debt_monthly,
        "car_price": car_price,
        "down_payment": down_payment,
        "apr": apr,
        "term_months": int(term_years * 12),
        "monthly_additional_savings": monthly_additional_savings
    }
