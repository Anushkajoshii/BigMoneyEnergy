
# # app/advisor_ai.py
# import os
# import json
# import textwrap
# import streamlit as st
# from groq import Groq
# from dotenv import load_dotenv
# load_dotenv()

# groq_api_key = st.secrets["GROQ_API_KEY"]

# GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# def format_prompt(snapshot, product_name, simulation_stats, recommendation):
#     prompt = f"""
#     You are a smart Gen-Z personal finance assistant. A user with the following snapshot asked about buying {product_name}.

#     Snapshot:
#     {json.dumps(snapshot, indent=2)}

#     Simulation summary:
#     {json.dumps(simulation_stats, indent=2)}

#     Current recommendation: {recommendation}

#     Provide a short, friendly explanation (3–5 bullets) about:
#     - Why this recommendation was made  
#     - Simple next steps (like increasing savings or waiting)  
#     - Example timeline if they save ₹X more monthly  

#     Keep it conversational, not too formal.
#     """
#     return prompt.strip()

# def get_ai_explanation(snapshot, product_name, simulation_stats, recommendation):
#     if not GROQ_API_KEY:
#         # fallback text if API not set
#         prob = simulation_stats.get("prob_shortfall", None)
#         loan_payment = simulation_stats.get("loan_payment", None)
#         bullets = [
#             f"Probability of shortfall: {prob:.2%}" if prob else "Shortfall data unavailable.",
#             f"Estimated monthly payment: ₹{loan_payment:,.0f}" if loan_payment else "",
#             "Tip: boost savings, delay purchase, or adjust EMI duration.",
#             "Example: saving ₹10,000 more each month builds ₹1.2L yearly buffer."
#         ]
#         return "Here’s a quick take:\n" + "\n".join(["- " + b for b in bullets])

#     try:
#         client = Groq(api_key=GROQ_API_KEY)
#         prompt = format_prompt(snapshot, product_name, simulation_stats, recommendation)
#         chat_completion = client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model="llama-3.3-70b-versatile",  # or 'mixtral-8x7b' / 'gemma-7b'
#             temperature=0.8,
#             max_tokens=400
#         )
#         return chat_completion.choices[0].message.content.strip()
#     except Exception as e:
#         return f"(AI explanation unavailable — {e})"


# app/advisor_ai.py
import os
import json
import textwrap
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str, required: bool = False):
    """
    Prefer environment variable, then fallback to streamlit secrets (if available).
    If required and missing, raise a clear error.
    """
    # 1) environment variable (best for containers/CI)
    val = os.environ.get(key)
    if val:
        return val

    # 2) streamlit secrets (works when running locally with .streamlit/secrets.toml)
    try:
        # st.secrets may raise if not present; use .get to be safe
        val = st.secrets.get(key) if hasattr(st, "secrets") else None
    except Exception:
        val = None

    if val:
        return val

    if required:
        raise RuntimeError(
            f"Missing secret: {key}. Provide it as an environment variable or in .streamlit/secrets.toml"
        )
    return None

# Read keys (GROQ is required for full AI explanation; adjust required flag as needed)
GROQ_API_KEY = get_secret("GROQ_API_KEY", required=False)
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", required=False)

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
    """
    Returns an AI-generated human-friendly explanation if GROQ_API_KEY is present,
    otherwise returns a deterministic fallback summary.
    """
    # If no API key, return a harmless fallback text (useful for local dev or CI)
    if not GROQ_API_KEY:
        prob = simulation_stats.get("prob_shortfall", None)
        loan_payment = simulation_stats.get("loan_payment", None)
        bullets = [
            f"Probability of shortfall: {prob:.2%}" if (prob is not None) else "Shortfall data unavailable.",
            f"Estimated monthly payment: ₹{loan_payment:,.0f}" if (loan_payment is not None) else "",
            "Tip: boost savings, delay purchase, or adjust EMI duration.",
            "Example: saving ₹10,000 more each month builds ~₹1.2L yearly buffer."
        ]
        # filter empty strings
        bullets = [b for b in bullets if b]
        return "Here’s a quick take:\n" + "\n".join(["- " + b for b in bullets])

    # If we have a key, call Groq safely
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = format_prompt(snapshot, product_name, simulation_stats, recommendation)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",  # or other model you prefer
            temperature=0.8,
            max_tokens=400
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        # Return a friendly fallback with the error for debugging
        return f"(AI explanation unavailable — {e})"
