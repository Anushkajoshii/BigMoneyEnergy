# app/advisor_ai.py
"""AI advisor: a short, friendly explanation of the recommendation.

The app must always show useful advice, even when the LLM is unavailable
(no key, rate limit, network error). In those cases we fall back to a
rule-based summary built from the simulation results.
"""
import json
import logging
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.1-8b-instant"  # small and fast: fewer tokens per call, fewer rate-limit hits


def get_secret(key: str, required: bool = False):
    """Read a secret from the environment first, then Streamlit secrets."""
    val = os.environ.get(key)
    if val:
        return val
    try:
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


GROQ_API_KEY = get_secret("GROQ_API_KEY", required=False)
GROQ_MODEL = get_secret("GROQ_MODEL", required=False) or DEFAULT_MODEL


def format_prompt(snapshot, product_name, simulation_stats, recommendation):
    return f"""
You are a friendly personal finance assistant for young professionals in India.
A user with the following snapshot asked about buying {product_name}.

Snapshot:
{json.dumps(snapshot, indent=2, default=float)}

Simulation summary:
{json.dumps(simulation_stats, indent=2, default=float)}

Current recommendation: {recommendation}

Give a short, friendly explanation in 3 to 5 bullets:
- why this recommendation was made
- simple next steps (for example, saving more or waiting)
- an example timeline if they save a bit more each month

Keep it conversational. Use rupees. Do not invent numbers that are not in the data above.
""".strip()


def fallback_summary(simulation_stats):
    """Rule-based advice used whenever the LLM is unavailable."""
    prob = simulation_stats.get("prob_shortfall")
    loan_payment = simulation_stats.get("loan_payment")
    bullets = []
    if prob is not None:
        bullets.append(f"Chance your savings fall short in the simulation: {prob:.0%}.")
    if loan_payment:
        bullets.append(f"Estimated EMI: ₹{loan_payment:,.0f} a month.")
    bullets += [
        "To lower the risk: save a little more each month, wait a few months, or pick a cheaper option.",
        "For example, saving ₹10,000 more each month builds about ₹1.2 lakh of extra buffer in a year.",
    ]
    return "\n".join("- " + b for b in bullets)


def get_ai_explanation(snapshot, product_name, simulation_stats, recommendation):
    """Return (text, used_ai). Never raises and never shows raw errors to the user."""
    if not GROQ_API_KEY:
        return fallback_summary(simulation_stats), False
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY, max_retries=1, timeout=20)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": format_prompt(snapshot, product_name, simulation_stats, recommendation)}],
            model=GROQ_MODEL,
            temperature=0.6,
            max_tokens=350,
        )
        text = (response.choices[0].message.content or "").strip()
        return (text, True) if text else (fallback_summary(simulation_stats), False)
    except Exception as exc:  # rate limit, network, bad key: log it, show the fallback
        logger.warning("AI advisor unavailable, using fallback: %s", exc)
        return fallback_summary(simulation_stats), False
