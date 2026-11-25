# BigMoneyEnergy 
### Your AI that helps you flex adult money — responsibly.

**Live demo:** https://bigmoneyenergy.streamlit.app


**AI-powered Personal Finance Simulator — simulate purchases, test loan options, and get AI-driven “Buy vs Wait” recommendations in an interactive Streamlit app.**

---

## 🚀 Overview
BigMoneyEnergy helps users make smarter financial decisions by simulating future savings and risks using Monte Carlo simulations and Groq AI models. It analyzes income, expenses, and savings patterns to recommend whether to **Buy Now, Wait, or Save More**.

**Tech Stack:** Python, Streamlit, Groq API, NumPy, Pandas, Scikit-learn, Matplotlib, Plotly, ReportLab

---

## 🧠 Features
- **💸 Monte Carlo Simulations** — Forecast savings and financial shortfall probabilities.
- **🏦 EMI & Loan Modeling** — Compute EMI, total interest, and affordability scenarios.
- **🧮 AI Financial Advisor (Groq)** — Get personalized, conversational financial guidance.
- **📊 Visualization Dashboards** — Fan charts and histograms for risk visualization.
- **📄 PDF/Excel Reports** — Auto-generate financial summaries and recommendations.
- **⚙️ Offline Mode** — Works without API key using deterministic fallback advice.
- **🧾 Model Logging** — Saves model metrics (R² = 0.91, MAE = ₹860) with timestamps for reproducibility.

---

## 🧮 Setup & Run
### 1️⃣ Clone Repository
```bash
git clone https://github.com/Anushkajoshii/BigMoneyEnergy.git
cd BigMoneyEnergy
```

### 2️⃣ Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Run App
```bash
export GROQ_API_KEY="your_groq_key_here"
streamlit run streamlit.py
```

### 4️⃣ Run Tests
```bash
PYTHONPATH=. pytest -q
```

## Run locally (Docker)

Build:
```bash
docker build -t bigmoneyenergy .
```
Run:
```bash
docker run -p 8501:8501 bigmoneyenergy

# if .streamlit/secrets.toml for api hosted on streamlit
docker run --rm -p 8501:8501 \
  -v "$(pwd)/.streamlit:/home/appuser/.streamlit:ro" \
  bigmoneyenergy:local
```
Or with docker-compose:

```bash
docker-compose up --build
```

---

## 📈 Model Metrics Example
| Metric | Value |
|---------|--------|
| R² | 0.91 |
| MAE | ₹860 |
| RMSE | ₹1090 |
| CV Mean ± Std | 0.93 ± 0.01 |

---

## 🧰 Architecture Overview
### **1. UI Layer (Streamlit)**
- Collects user data (income, expenses, purchase type).
- Displays visualizations and affordability verdicts.
- Allows PDF/Excel downloads.

### **2. Engine Layer (Simulation + Logic)**
- `monte_carlo_purchase_risk()` simulates thousands of financial paths.
- `calculate_loan_monthly_payment()` handles EMI logic.
- `rule_checks()` applies heuristics for affordability and emergency fund readiness.

### **3. AI Layer (Groq Integration)**
- Connects to Groq’s LLM to generate natural-language insights.
- Fallback logic ensures deterministic advice offline.

### **4. Model Layer (ML & Persistence)**
- ElasticNet regression pipeline (StandardScaler + ElasticNet).
- Archives models as `.pkl` with timestamped metrics.

### **5. Visualization & Reporting**
- Matplotlib & Plotly for fan charts and risk histograms.
- ReportLab for exporting professional PDF summaries.

---

## 🧱 Architecture Diagram
```
        ┌──────────────────────────────┐
        │        Streamlit UI         │
        │  (User inputs & dashboard)  │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌──────────────────────────────┐
        │        Engine Layer          │
        │ Monte Carlo + EMI + Rules    │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌──────────────────────────────┐
        │        AI Advisor (Groq)     │
        │   LLM-based explanations     │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌──────────────────────────────┐
        │        Model Layer           │
        │ ElasticNet ML + Persistence  │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌──────────────────────────────┐
        │ Visualization & Reports      │
        │   Matplotlib / ReportLab     │
        └──────────────────────────────┘
```

---

## 📦 Sample Output
```
Model evaluation: R2=0.907 MAE=860.02 RMSE=1090.65
Cross-validation R² mean=0.932 std=0.005
Saved model: model_20251105T162536Z.pkl
```

---
**Author:** [Anushka Joshi](https://github.com/Anushkajoshii)  
**Repo:** [BigMoneyEnergy](https://github.com/Anushkajoshii/BigMoneyEnergy)  
