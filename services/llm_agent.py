"""
LLM agent helpers for advisor/predictor narrative output.

Safe-by-default:
- Disabled unless ENABLE_AGENT=true in .env
- Falls back to None on any API error/timeout
- LRU cache reduces repeated API calls
- Supports three providers: anthropic, openai, nvidia

.env configuration:
    ENABLE_AGENT=true

    # Pick one provider:
    AGENT_PROVIDER=anthropic          # recommended — fast, cheap Haiku
    ANTHROPIC_API_KEY=sk-ant-...

    AGENT_PROVIDER=openai
    OPENAI_API_KEY=sk-...

    AGENT_PROVIDER=nvidia             # free tier, slower
    NVIDIA_API_KEY=...

    # Optional overrides:
    AGENT_MODEL=claude-haiku-4-5      # override default model for provider
    AGENT_TIMEOUT_SECONDS=15
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import threading
from collections import OrderedDict
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Provider endpoints ────────────────────────────────────────────────────────
OPENAI_URL   = "https://api.openai.com/v1/chat/completions"
NVIDIA_URL   = "https://integrate.api.nvidia.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# ── Default models per provider ───────────────────────────────────────────────
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai":    "gpt-4.1-mini",
    "nvidia":    "moonshotai/kimi-k2.5",
}

# ── Timeouts ──────────────────────────────────────────────────────────────────
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_READ_TIMEOUT    = 15
DEFAULT_HARD_TIMEOUT    = 20

# ── LRU cache ─────────────────────────────────────────────────────────────────
CACHE_MAX = 256
_CACHE: OrderedDict[str, str] = OrderedDict()
_LAST_ERROR: str | None = None


def _set_last_error(msg: str | None) -> None:
    global _LAST_ERROR
    _LAST_ERROR = msg


def _env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def agent_enabled() -> bool:
    return _env_true("ENABLE_AGENT", "false")


def agent_provider() -> str:
    p = os.getenv("AGENT_PROVIDER", "anthropic").strip().lower()
    return p if p in {"anthropic", "openai", "nvidia"} else "anthropic"


def agent_model() -> str:
    """Return the model to use — respects AGENT_MODEL env override."""
    override = os.getenv("AGENT_MODEL", "").strip()
    if override:
        return override
    return DEFAULT_MODELS[agent_provider()]


def _cache_get(key: str) -> str | None:
    if key not in _CACHE:
        return None
    _CACHE.move_to_end(key)
    return _CACHE[key]


def _cache_set(key: str, value: str) -> None:
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    if len(_CACHE) > CACHE_MAX:
        _CACHE.popitem(last=False)


def _make_cache_key(prefix: str, payload: dict, model: str) -> str:
    body = json.dumps({"prefix": prefix, "model": model, "payload": payload},
                      sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(body.encode()).hexdigest()


# ── Anthropic API call ────────────────────────────────────────────────────────

def _call_anthropic(system_prompt: str, user_prompt: str,
                    max_tokens: int, model: str, timeout: tuple) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        _set_last_error("Missing ANTHROPIC_API_KEY")
        return None

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type":      "application/json",
    }
    body = {
        "model":      model,
        "max_tokens": max_tokens,
        "system":     system_prompt,
        "messages":   [{"role": "user", "content": user_prompt}],
    }

    resp_box: dict[str, requests.Response] = {}
    err_box:  dict[str, Exception]         = {}

    def _do_post():
        try:
            resp_box["resp"] = requests.post(
                ANTHROPIC_URL, headers=headers, json=body, timeout=timeout
            )
        except Exception as e:
            err_box["err"] = e

    hard = float(os.getenv("AGENT_HARD_TIMEOUT_SECONDS", str(DEFAULT_HARD_TIMEOUT)))
    t = threading.Thread(target=_do_post, daemon=True)
    t.start()
    t.join(max(1.0, hard))

    if t.is_alive():
        _set_last_error(f"Hard timeout after {hard:.0f}s")
        return None
    if "err" in err_box:
        _set_last_error(f"Request error: {err_box['err']}")
        return None

    resp = resp_box["resp"]
    if resp.status_code != 200:
        _set_last_error(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    data = resp.json()
    content = data.get("content", [])
    for block in content:
        if block.get("type") == "text":
            return block["text"].strip()

    _set_last_error("No text block in Anthropic response")
    return None


# ── OpenAI-compatible API call (openai + nvidia) ──────────────────────────────

def _call_openai_compat(system_prompt: str, user_prompt: str,
                        max_tokens: int, model: str,
                        endpoint: str, api_key: str,
                        timeout: tuple) -> str | None:
    if not api_key:
        _set_last_error(f"Missing API key for endpoint {endpoint}")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    body: dict[str, Any] = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": 0.2,
        "stream":      False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    # Kimi: disable extended thinking for speed
    if "kimi" in model.lower():
        body["thinking"] = {"type": "disabled"}

    resp_box: dict[str, requests.Response] = {}
    err_box:  dict[str, Exception]         = {}

    def _do_post():
        try:
            resp_box["resp"] = requests.post(
                endpoint, headers=headers, json=body, timeout=timeout
            )
        except Exception as e:
            err_box["err"] = e

    hard = float(os.getenv("AGENT_HARD_TIMEOUT_SECONDS", str(DEFAULT_HARD_TIMEOUT)))
    t = threading.Thread(target=_do_post, daemon=True)
    t.start()
    t.join(max(1.0, hard))

    if t.is_alive():
        _set_last_error(f"Hard timeout after {hard:.0f}s")
        return None
    if "err" in err_box:
        _set_last_error(f"Request error: {err_box['err']}")
        return None

    resp = resp_box["resp"]
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        _set_last_error(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        _set_last_error(f"No choices in response: {str(data)[:200]}")
        return None

    text = choices[0].get("message", {}).get("content", "")
    if isinstance(text, list):
        text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
    return text.strip() if text else None


# ── Unified chat completion ───────────────────────────────────────────────────

def _chat_completion(system_prompt: str, user_prompt: str,
                     max_tokens: int = 300) -> str | None:
    if not agent_enabled():
        return None

    _set_last_error(None)
    provider = agent_provider()
    model    = agent_model()

    connect_t = float(os.getenv("AGENT_CONNECT_TIMEOUT_SECONDS", str(DEFAULT_CONNECT_TIMEOUT)))
    read_t    = float(os.getenv("AGENT_READ_TIMEOUT_SECONDS",    str(DEFAULT_READ_TIMEOUT)))
    timeout   = (connect_t, read_t)

    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, max_tokens, model, timeout)

    elif provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        return _call_openai_compat(system_prompt, user_prompt, max_tokens,
                                   model, NVIDIA_URL, api_key, timeout)
    else:  # openai
        api_key = os.getenv("OPENAI_API_KEY", "")
        return _call_openai_compat(system_prompt, user_prompt, max_tokens,
                                   model, OPENAI_URL, api_key, timeout)


# ── Public functions ──────────────────────────────────────────────────────────

def build_advisor_agent_plan(
    city: str,
    row: dict[str, Any],
    shap_drivers: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    strengths: list[str],
    weaknesses: list[str],
) -> str | None:
    if not agent_enabled():
        return None

    payload = {
        "city": city,
        "superhost_probability": row.get("superhost_probability"),
        "response_rate":         row.get("host_response_rate_clean"),
        "amenity_count":         row.get("amenity_count"),
        "top_drivers":           shap_drivers[:6],
        "recommendations":       recommendations[:5],
        "strengths":             strengths[:5],
        "weaknesses":            weaknesses[:5],
    }
    key = _make_cache_key("advisor_plan", payload, agent_model())
    cached = _cache_get(key)
    if cached:
        return cached

    system_prompt = (
        "You are an Airbnb host operations strategist. "
        "Write concise, practical recommendations in Markdown."
    )
    user_prompt = (
        "Create a compact action plan in Markdown with:\n"
        "1) A one-line diagnosis\n"
        "2) A 'Next 7 days' action list (3 bullets using - not checkboxes)\n"
        "3) A 'Next 30 days' action list (3 bullets using - not checkboxes)\n"
        "4) Budget guidance (short)\n\n"
        f"Input data:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    text = _chat_completion(system_prompt, user_prompt, max_tokens=300)
    if text:
        _cache_set(key, text)
    return text


def build_advisor_agent_answer(
    city: str,
    question: str,
    listing_context: dict[str, Any],
) -> str | None:
    if not agent_enabled() or not question.strip():
        return None

    payload = {"city": city, "question": question.strip(),
               "listing_context": listing_context}
    key = _make_cache_key("advisor_qa", payload, agent_model())
    cached = _cache_get(key)
    if cached:
        return cached

    system_prompt = (
        "You are an Airbnb host operations strategist. "
        "Answer the user's question with concise, practical guidance grounded in the listing data. "
        "Use Markdown. If data is missing, state the limitation briefly."
    )
    user_prompt = (
        f"User question:\n{question.strip()}\n\n"
        f"Listing context:\n{json.dumps(listing_context, ensure_ascii=False)}\n\n"
        "Response format:\n"
        "- One-line direct answer\n"
        "- 2-4 concrete actions\n"
        "- Optional risk/watchout line if relevant"
    )
    text = _chat_completion(system_prompt, user_prompt, max_tokens=380)
    if text:
        _cache_set(key, text)
    return text


def build_investor_agent_brief(
    city: str,
    form_values: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    if not agent_enabled():
        return None

    payload = {
        "city":                  city,
        "predicted_price":       result.get("predicted_price"),
        "nbhd_median":           result.get("nbhd_median"),
        "pct_vs_neighbourhood":  result.get("pct_vs_nbhd"),
        "amenity_gaps":          result.get("amenity_gaps", [])[:4],
        "top_drivers":           result.get("drivers", [])[:5],
        "summary_input": {
            "neighbourhood":  form_values.get("neighbourhood"),
            "property_type":  form_values.get("property_type"),
            "accommodates":   form_values.get("accommodates"),
            "bedrooms":       form_values.get("bedrooms"),
        },
    }
    key = _make_cache_key("predictor_brief", payload, agent_model())
    cached = _cache_get(key)
    if cached:
        return cached

    system_prompt = (
        "You are a short-term rental investment analyst. "
        "Return concise, decision-oriented output in Markdown."
    )
    user_prompt = (
        "Write a concise investment brief with:\n"
        "1) Price stance (1 sentence)\n"
        "2) Top 3 actions ranked by expected impact\n"
        "3) A caution note about downside risk\n\n"
        f"Input data:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    text = _chat_completion(system_prompt, user_prompt, max_tokens=260)
    if text:
        _cache_set(key, text)
    return text


def build_predictor_whatif_plan(
    city: str,
    budget: float,
    base_price: float,
    scenarios: list[dict[str, Any]],
) -> str | None:
    if not agent_enabled():
        return None

    payload = {"city": city, "budget": budget,
               "base_price": base_price, "scenarios": scenarios[:3]}
    key = _make_cache_key("predictor_whatif", payload, agent_model())
    cached = _cache_get(key)
    if cached:
        return cached

    system_prompt = (
        "You are a short-term rental investment analyst. "
        "Return concise decision-oriented Markdown."
    )
    user_prompt = (
        "Given the ranked what-if scenarios, provide:\n"
        "1) Best option and why (1-2 lines)\n"
        "2) 7-day implementation checklist (3 bullets)\n"
        "3) One downside risk to monitor\n\n"
        f"Input data:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    text = _chat_completion(system_prompt, user_prompt, max_tokens=260)
    if text:
        _cache_set(key, text)
    return text


def build_investor_agent_answer(
    city: str,
    question: str,
    investment_context: dict[str, Any],
) -> str | None:
    if not agent_enabled() or not question.strip():
        return None

    payload = {"city": city, "question": question.strip(),
               "investment_context": investment_context}
    key = _make_cache_key("investor_qa", payload, agent_model())
    cached = _cache_get(key)
    if cached:
        return cached

    system_prompt = (
        "You are a short-term rental investment analyst. "
        "Answer in concise, decision-oriented Markdown grounded in the provided context."
    )
    user_prompt = (
        f"User question:\n{question.strip()}\n\n"
        f"Context:\n{json.dumps(investment_context, ensure_ascii=False)}\n\n"
        "Response format:\n"
        "- One-line stance\n"
        "- 2-4 prioritized actions with rationale\n"
        "- One downside risk to monitor"
    )
    text = _chat_completion(system_prompt, user_prompt, max_tokens=340)
    if text:
        _cache_set(key, text)
    return text


def agent_status_text() -> str:
    if not agent_enabled():
        return "Agent disabled — set `ENABLE_AGENT=true` in .env"
    provider = agent_provider()
    model    = agent_model()
    key_map  = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "nvidia":    "NVIDIA_API_KEY",
    }
    key_name = key_map[provider]
    api_key  = os.getenv(key_name, "")
    if not api_key:
        return f"Agent enabled but `{key_name}` is missing"
    if _LAST_ERROR:
        return f"Agent: {provider} / {model} — last error: {_LAST_ERROR}"
    return f"Agent ready: {provider} / {model}"
