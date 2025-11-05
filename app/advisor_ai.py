
# app/advisor_ai.py
import os
import json
import textwrap
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

def format_prompt(snapshot, product_name, simulation_stats, recommendation):
    prompt = f"""
    You are a smart Gen-Z personal finance assistant. A user with the following snapshot asked about buying {product_name}.

    Snapshot:
    {json.dumps(snapshot, indent=2)}

    Simulation summary:
    {json.dumps(simulation_stats, indent=2)}

    Current recommendation: {recommendation}

    Provide a short, friendly explanation (3–5 bullets) about:
    - Why this recommendation was made  
    - Simple next steps (like increasing savings or waiting)  
    - Example timeline if they save ₹X more monthly  

    Keep it conversational, not too formal.
    """
    return prompt.strip()

def get_ai_explanation(snapshot, product_name, simulation_stats, recommendation):
    if not GROQ_API_KEY:
        # fallback text if API not set
        prob = simulation_stats.get("prob_shortfall", None)
        loan_payment = simulation_stats.get("loan_payment", None)
        bullets = [
            f"Probability of shortfall: {prob:.2%}" if prob else "Shortfall data unavailable.",
            f"Estimated monthly payment: ₹{loan_payment:,.0f}" if loan_payment else "",
            "Tip: boost savings, delay purchase, or adjust EMI duration.",
            "Example: saving ₹10,000 more each month builds ₹1.2L yearly buffer."
        ]
        return "Here’s a quick take:\n" + "\n".join(["- " + b for b in bullets])

    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = format_prompt(snapshot, product_name, simulation_stats, recommendation)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",  # or 'mixtral-8x7b' / 'gemma-7b'
            temperature=0.8,
            max_tokens=400
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI explanation unavailable — {e})"
