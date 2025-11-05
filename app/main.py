

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from app.ui import inputs_panel
from app.engine import calculate_loan_monthly_payment, rule_checks, monte_carlo_purchase_risk
from app.utils import fan_chart_figure, final_savings_histogram
from app.advisor_ai import get_ai_explanation

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="GenZ Finance Advisor", layout="wide")

st.title("💸 GenZ Finance Advisor — Should You Buy It?")
st.write("Simulate your financial health before making a purchase. Works for cars, luxury items, gadgets, or any goal!")

inputs = inputs_panel()

snapshot = {
    "monthly_net_income": inputs["monthly_net_income"],
    "fixed_expenses": inputs["fixed_expenses"],
    "variable_expenses": inputs["variable_expenses"],
    "savings_balance": inputs["savings_balance"],
    "emergency_target": inputs["emergency_target"],
    "debt_monthly": inputs["debt_monthly"],
    "monthly_additional_savings": inputs["monthly_additional_savings"],
}

# EMI calculation only if needed
loan_payment = 0
if inputs["purchase_type"] == "EMI-based purchase":
    loan_payment = calculate_loan_monthly_payment(
        inputs["car_price"], inputs["down_payment"], inputs["apr"], inputs["term_months"]
    )

# Quick check
st.header("🔍 Quick Affordability Check")
checks = rule_checks(snapshot, loan_payment)
if checks["safe"]:
    st.success("✅ Financial rules look good.")
else:
    st.warning("⚠️ Some financial stress indicators detected.")
for msg in checks["messages"]:
    st.write("- " + msg)

if inputs["purchase_type"] == "EMI-based purchase":
    st.write(f"Estimated EMI: ₹{loan_payment:,.0f}/month")

# Run simulation
st.header("🎲 Monte Carlo Risk Simulation")
horizon_months = st.slider("Simulation horizon (months)", 6, 60, 24, 6)
sims = st.selectbox("Number of simulations", [1000, 3000, 5000], index=1)

with st.spinner("Running financial simulation..."):
    mc = monte_carlo_purchase_risk(
        snapshot,
        price=inputs["car_price"],
        down_payment=inputs["down_payment"],
        apr=inputs["apr"],
        term_months=inputs["term_months"],
        months_ahead=horizon_months,
        sims=sims,
        purchase_type=inputs["purchase_type"],
        buy_in_months=inputs["buy_in_months"],
    )
    
st.subheader("📊 Monte Carlo Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Savings Growth Over Time")
    fan_chart_img = fan_chart_figure(mc["sim_matrix"], months_ahead=horizon_months)
    st.image(fan_chart_img, caption="Monte Carlo Savings Projection")

with col2:
    st.markdown("### Final Savings Distribution")
    hist_fig = final_savings_histogram(
        final_savings_array=mc["final_savings"],
        baseline_savings=snapshot.get("savings_balance", 0)
    )
    st.plotly_chart(hist_fig, width='stretch')



# Recommendation
st.header("🧠 Recommendation")
threshold = st.slider("Acceptable shortfall probability", 0.0, 1.0, 0.2, 0.05)
recommendation = ""
prob = mc["prob_shortfall"]
if prob <= threshold and checks["safe"]:
    recommendation = f"You can consider buying {inputs['product_name']} now 🎉"
    st.success(recommendation)
else:
    recommendation = f"Hold off on buying {inputs['product_name']} — build more buffer."
    st.warning(recommendation)
    st.write("- Increase monthly savings or delay purchase.")
    st.write("- Reduce price range or EMI duration for safer outcomes.")
ai_text = get_ai_explanation(
    snapshot,
    inputs['product_name'],
    {"prob_shortfall": mc["prob_shortfall"], "loan_payment": loan_payment, **mc["stats"]},
    recommendation
)

st.markdown("### 💬 AI Advisor’s Take")
st.write(ai_text)

st.header("📊 Financial Snapshot")
st.table(pd.DataFrame([snapshot]).T.rename(columns={0: "Value"}))

# -------------------------------------------------------------------
# 📄 REPORT DOWNLOAD SECTION
# -------------------------------------------------------------------
st.header("📄 Download Your Financial Report")

def generate_pdf_report(inputs, snapshot, mc, recommendation, prob):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("💸 GenZ Finance Advisor Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>Product:</b> {inputs['product_name']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Purchase Type:</b> {inputs['purchase_type']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Price:</b> ₹{inputs['car_price']:,.0f}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("📊 <b>Financial Snapshot</b>", styles["Heading2"]))
    data = [["Metric", "Value"]] + [[k.replace("_", " ").title(), f"₹{v:,.0f}" if isinstance(v, (int, float)) else v] for k, v in snapshot.items()]
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("🎲 <b>Monte Carlo Simulation Summary</b>", styles["Heading2"]))
    stats_table = Table([["Metric", "Value"]] + [[k, f"{v:.3f}" if isinstance(v, float) else v] for k, v in mc["stats"].items()])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>Probability of Shortfall:</b> {prob*100:.1f}%", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("🧠 <b>Recommendation</b>", styles["Heading2"]))
    elements.append(Paragraph(recommendation, styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

pdf_buffer = generate_pdf_report(inputs, snapshot, mc, recommendation, prob)
st.download_button(
    label="📥 Download Financial Report (PDF)",
    data=pdf_buffer,
    file_name=f"{inputs['product_name']}_finance_report.pdf",
    mime="application/pdf",
)

