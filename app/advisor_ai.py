# app/advisor_ai.py
import os
import json
import textwrap

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

def format_prompt(snapshot, product_name, simulation_stats, recommendation):
    prompt = f"""
    You are a helpful personal finance advisor. A user with the following snapshot asked about buying {product_name}.

    Snapshot:
    {json.dumps(snapshot, indent=2)}

    Simulation summary:
    {json.dumps(simulation_stats, indent=2)}

    Current recommendation: {recommendation}

    Provide a short, friendly explanation (3-5 bullets) why this recommendation was made, simple action steps the user could take (monthly savings amounts, delays), and an example plan showing months-to-go if they save X more per month.

    Keep it conversational and Gen-Z friendly.
    """
    return prompt

def get_ai_explanation(snapshot, product_name, simulation_stats, recommendation):
    # If no API key, return a deterministic text
    if not OPENAI_API_KEY:
        # fallback: build a crisp human explanation
        prob = simulation_stats.get("prob_shortfall", None)
        loan_payment = simulation_stats.get("loan_payment", None)
        bullets = [
            f"Probability of shortfall within horizon: {prob:.2%}" if prob is not None else "No probability available.",
            f"Estimated monthly payment would be {loan_payment:,.0f}." if loan_payment is not None else "",
            "Actionable: increase monthly savings, raise down payment, shorten loan term, or postpone purchase.",
            "Example: save an extra ₹10,000/month — this will add ₹120,000/year to your savings and likely reduce shortfall risk."
        ]
        text = "Here’s a quick take:\n\n" + "\n".join(["- " + b for b in bullets])
        return textwrap.dedent(text)

    # If API key exists — call OpenAI (this is a placeholder; you can swap to official openai package)
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        prompt = format_prompt(snapshot, product_name, simulation_stats, recommendation)
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list().data else "gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.8
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"AI explanation unavailable (error: {e}).\n\nFallback:\n" + get_ai_explanation.__wrapped__(snapshot, product_name, simulation_stats, recommendation)
