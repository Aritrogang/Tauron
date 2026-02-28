"""
backend/llm_engine.py

Local LLM wrapper using Ollama's REST API.

Setup (one-time):
    brew install ollama
    ollama serve &          # starts daemon on localhost:11434
    ollama pull mistral     # downloads mistral:7b-instruct Q4_K_M (~4.1 GB)

Why Ollama over llama-cpp-python:
    - M4 Pro Metal acceleration is pre-compiled in Ollama — no fragile CMAKE flags
    - Daemon model: LLM crash doesn't kill the FastAPI server
    - Single install command vs. compile-time dependencies
    See claude.md for full rationale.

Inference parameters:
    temperature=0.2   — low variance: consistent, predictable alerts (not creative)
    num_predict=60    — hard cap at ~60 tokens (~1 sentence), keeps latency < 2s on M4

Fallback:
    If Ollama is unreachable (e.g., daemon not started), _fallback_alert() returns
    a template-formatted string. The API never returns a 500 due to LLM failure.
"""

import httpx
import logging

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 30.0  # seconds — generous for demo conditions

# ---------------------------------------------------------------------------
# Prompts — versioned here for documentation and hackathon judging (see claude.md)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an AI assistant for dairy farmers. "
    "Convert structured cow health data into ONE clear, actionable sentence. "
    "Use plain English. No veterinary jargon. "
    "Be specific: name the cow, name the risk, name the action. "
    "Never say 'I' or 'the model' — speak directly: 'Isolate #47...' not 'You should isolate...'. "
    "Maximum 25 words."
)


def _build_user_prompt(xai_json: dict) -> str:
    """
    Converts the structured XAI intermediate dict into an LLM user prompt.

    This function is separate from the HTTP call so prompt text can be tuned
    independently without touching the Ollama connection logic.

    Args:
        xai_json: dict with keys cow_id, risk_score, top_feature, feature_delta, top_edge
    """
    cow = xai_json["cow_id"]
    risk = xai_json["risk_score"]
    feature = xai_json["top_feature"].replace("_", " ")
    delta = xai_json["feature_delta"]
    edge_from = xai_json["top_edge"]["from"]
    edge_to = xai_json["top_edge"]["to"]
    edge_weight = xai_json["top_edge"]["weight"]

    delta_str = f"{delta:+.0%}" if delta != 0.0 else "baseline"

    return (
        f"Cow ID: #{cow}\n"
        f"Risk score: {risk:.0%}\n"
        f"Top risk factor: {feature} (change from baseline: {delta_str})\n"
        f"Highest-risk contact: #{edge_from} shared space with #{edge_to} "
        f"(connection strength: {edge_weight:.0%})\n"
        f"Generate a one-sentence farmer alert."
    )


def _fallback_alert(xai_json: dict) -> str:
    """
    Template-based fallback used when Ollama is unreachable.
    Ensures the API always returns a usable string.
    """
    cow = xai_json["cow_id"]
    feature = xai_json["top_feature"].replace("_", " ")
    contact = xai_json["top_edge"]["to"]
    return (
        f"Check #{cow}: {feature} detected, shared space with #{contact}. "
        "Inspect immediately."
    )


async def generate_alert(xai_json: dict) -> str:
    """
    Main entry point. Calls Ollama's local API and returns a farmer alert string.

    Always returns a string — never raises. Falls back to template on any error.

    Args:
        xai_json: structured dict from xai_bridge.build_xai_json()

    Returns:
        Plain-English one-sentence alert (e.g. "Isolate #47: milk drop detected...")
    """
    prompt = _build_user_prompt(xai_json)

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 60,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            alert_text = result["response"].strip()
            logger.info(
                "LLM alert for cow %d: %s...",
                xai_json["cow_id"],
                alert_text[:50],
            )
            return alert_text

    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.warning("Ollama unreachable (%s) — using template fallback", e)
        return _fallback_alert(xai_json)
    except Exception as e:
        logger.error("Unexpected LLM error: %s — using template fallback", e)
        return _fallback_alert(xai_json)
