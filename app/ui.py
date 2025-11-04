import streamlit as st

def inputs_panel(prefix=""):
    st.sidebar.header("Your Financial Snapshot")

    monthly_net_income = st.sidebar.number_input(
        "Monthly Net Income (₹)", value=100000, step=5000, key=f"{prefix}monthly_net_income"
    )
    fixed_expenses = st.sidebar.number_input(
        "Fixed Expenses (₹)", value=40000, step=2000, key=f"{prefix}fixed_expenses"
    )
    variable_expenses = st.sidebar.number_input(
        "Variable Expenses (₹)", value=20000, step=2000, key=f"{prefix}variable_expenses"
    )
    savings_balance = st.sidebar.number_input(
        "Current Savings (₹)", value=300000, step=10000, key=f"{prefix}savings_balance"
    )
    emergency_target = st.sidebar.number_input(
        "Target Emergency Fund (₹)", value=250000, step=10000, key=f"{prefix}emergency_target"
    )
    debt_monthly = st.sidebar.number_input(
        "Existing Monthly Debt (₹)", value=0, step=1000, key=f"{prefix}debt_monthly"
    )
    monthly_additional_savings = st.sidebar.number_input(
        "Monthly Savings Contribution (₹)", value=10000, step=1000, key=f"{prefix}monthly_additional_savings"
    )

    purchase_type = st.radio(
        "Select Purchase Type",
        ["One-time purchase", "EMI-based purchase", "Future plan"],
        horizontal=True,
        key=f"{prefix}purchase_type"
    )

    product_name = st.text_input(
        "Product name (e.g., BMW, iPhone, Gold)", "BMW", key=f"{prefix}product_name"
    )
    car_price = st.number_input(
        f"Price of {product_name} (₹)", value=1000000, step=10000, key=f"{prefix}car_price"
    )

    down_payment = 0
    apr = 0.0
    term_months = 0
    buy_in_months = 0

    if purchase_type == "EMI-based purchase":
        st.subheader("Loan / EMI Details")
        down_payment = st.number_input(
            "Down Payment (₹)", value=200000, step=10000, key=f"{prefix}down_payment"
        )
        apr = st.number_input(
            "Annual Interest Rate (%)", value=9.0, step=0.5, key=f"{prefix}apr"
        )
        term_months = st.number_input(
            "Loan Term (months)", value=36, step=6, key=f"{prefix}term_months"
        )

    elif purchase_type == "Future plan":
        st.subheader("Future Purchase Planning")
        buy_in_months = st.number_input(
            "Plan to buy in (months)", value=12, step=1, key=f"{prefix}buy_in_months"
        )

    return {
        "monthly_net_income": monthly_net_income,
        "fixed_expenses": fixed_expenses,
        "variable_expenses": variable_expenses,
        "savings_balance": savings_balance,
        "emergency_target": emergency_target,
        "debt_monthly": debt_monthly,
        "monthly_additional_savings": monthly_additional_savings,
        "purchase_type": purchase_type,
        "product_name": product_name,
        "car_price": car_price,
        "down_payment": down_payment,
        "apr": apr,
        "term_months": term_months,
        "buy_in_months": buy_in_months,
    }

