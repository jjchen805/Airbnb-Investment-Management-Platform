"""
Tests for services/llm_agent.py

These tests mock all network calls — no real API keys required.
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.llm_agent import (
    _env_true,
    agent_enabled,
    agent_provider,
    agent_model,
    _cache_get,
    _cache_set,
    _make_cache_key,
    _CACHE,
    DEFAULT_MODELS,
    agent_status_text,
    build_advisor_agent_plan,
    build_advisor_agent_answer,
    build_investor_agent_brief,
    build_investor_agent_answer,
    build_predictor_whatif_plan,
)


# ── _env_true ─────────────────────────────────────────────────────────────────

class TestEnvTrue:
    def test_true_string(self):
        with patch.dict(os.environ, {"FLAG": "true"}):
            assert _env_true("FLAG") is True

    def test_yes_string(self):
        with patch.dict(os.environ, {"FLAG": "yes"}):
            assert _env_true("FLAG") is True

    def test_1_string(self):
        with patch.dict(os.environ, {"FLAG": "1"}):
            assert _env_true("FLAG") is True

    def test_false_string(self):
        with patch.dict(os.environ, {"FLAG": "false"}):
            assert _env_true("FLAG") is False

    def test_missing_uses_default(self):
        env = {k: v for k, v in os.environ.items() if k != "FLAG"}
        with patch.dict(os.environ, env, clear=True):
            assert _env_true("FLAG", "true") is True
            assert _env_true("FLAG", "false") is False


# ── agent_enabled / agent_provider / agent_model ──────────────────────────────

class TestAgentConfig:
    def test_agent_disabled_by_default(self):
        with patch.dict(os.environ, {"ENABLE_AGENT": "false"}):
            assert agent_enabled() is False

    def test_agent_enabled(self):
        with patch.dict(os.environ, {"ENABLE_AGENT": "true"}):
            assert agent_enabled() is True

    def test_default_provider_is_anthropic(self):
        env = {k: v for k, v in os.environ.items() if k != "AGENT_PROVIDER"}
        with patch.dict(os.environ, env, clear=True):
            assert agent_provider() == "anthropic"

    def test_provider_openai(self):
        with patch.dict(os.environ, {"AGENT_PROVIDER": "openai"}):
            assert agent_provider() == "openai"

    def test_provider_invalid_falls_back_to_anthropic(self):
        with patch.dict(os.environ, {"AGENT_PROVIDER": "unknown_llm"}):
            assert agent_provider() == "anthropic"

    def test_default_model_for_anthropic(self):
        with patch.dict(os.environ, {"AGENT_PROVIDER": "anthropic"},
                        clear=False):
            os.environ.pop("AGENT_MODEL", None)
            assert agent_model() == DEFAULT_MODELS["anthropic"]

    def test_model_override(self):
        with patch.dict(os.environ, {"AGENT_MODEL": "claude-opus-4-5"}):
            assert agent_model() == "claude-opus-4-5"


# ── LRU cache ─────────────────────────────────────────────────────────────────

class TestLRUCache:
    def setup_method(self):
        _CACHE.clear()

    def test_miss_returns_none(self):
        assert _cache_get("nonexistent") is None

    def test_set_and_get(self):
        _cache_set("k1", "hello")
        assert _cache_get("k1") == "hello"

    def test_overwrite(self):
        _cache_set("k1", "first")
        _cache_set("k1", "second")
        assert _cache_get("k1") == "second"

    def test_eviction_at_max(self):
        from services.llm_agent import CACHE_MAX
        for i in range(CACHE_MAX + 5):
            _cache_set(f"key_{i}", f"val_{i}")
        assert len(_CACHE) <= CACHE_MAX

    def test_lru_evicts_oldest(self):
        from services.llm_agent import CACHE_MAX
        # Fill to max
        for i in range(CACHE_MAX):
            _cache_set(f"key_{i}", f"val_{i}")
        # Access key_0 to make it recently used
        _cache_get("key_0")
        # Add one more to trigger eviction — key_1 should be evicted (oldest unused)
        _cache_set("new_key", "new_val")
        assert _cache_get("key_0") == "val_0"   # recently used — should survive
        assert _cache_get("new_key") == "new_val"


# ── _make_cache_key ───────────────────────────────────────────────────────────

class TestMakeCacheKey:
    def test_returns_string(self):
        key = _make_cache_key("test", {"a": 1}, "model-x")
        assert isinstance(key, str)

    def test_same_inputs_same_key(self):
        k1 = _make_cache_key("prefix", {"x": 1}, "model")
        k2 = _make_cache_key("prefix", {"x": 1}, "model")
        assert k1 == k2

    def test_different_payload_different_key(self):
        k1 = _make_cache_key("prefix", {"x": 1}, "model")
        k2 = _make_cache_key("prefix", {"x": 2}, "model")
        assert k1 != k2

    def test_different_prefix_different_key(self):
        k1 = _make_cache_key("a", {}, "model")
        k2 = _make_cache_key("b", {}, "model")
        assert k1 != k2

    def test_different_model_different_key(self):
        k1 = _make_cache_key("prefix", {}, "model-a")
        k2 = _make_cache_key("prefix", {}, "model-b")
        assert k1 != k2


# ── agent_status_text ─────────────────────────────────────────────────────────

class TestAgentStatusText:
    def test_disabled_message(self):
        with patch.dict(os.environ, {"ENABLE_AGENT": "false"}):
            assert "disabled" in agent_status_text().lower()

    def test_missing_key_message(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "ENABLE_AGENT", "AGENT_PROVIDER")}
        env["ENABLE_AGENT"] = "true"
        env["AGENT_PROVIDER"] = "anthropic"
        with patch.dict(os.environ, env, clear=True):
            assert "missing" in agent_status_text().lower()

    def test_ready_message(self):
        with patch.dict(os.environ, {
            "ENABLE_AGENT": "true",
            "AGENT_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-fake",
        }):
            status = agent_status_text()
            assert "ready" in status.lower()
            assert "anthropic" in status.lower()


# ── Public functions return None when agent disabled ──────────────────────────

class TestPublicFunctionsDisabled:
    """When ENABLE_AGENT=false all public functions must return None immediately."""

    def setup_method(self):
        os.environ["ENABLE_AGENT"] = "false"

    def teardown_method(self):
        os.environ.pop("ENABLE_AGENT", None)

    def test_build_advisor_plan_returns_none(self):
        result = build_advisor_agent_plan("sf", {}, [], [], [], [])
        assert result is None

    def test_build_advisor_answer_returns_none(self):
        result = build_advisor_agent_answer("sf", "some question", {})
        assert result is None

    def test_build_investor_brief_returns_none(self):
        result = build_investor_agent_brief("sf", {}, {})
        assert result is None

    def test_build_investor_answer_returns_none(self):
        result = build_investor_agent_answer("sf", "some question", {})
        assert result is None

    def test_build_whatif_plan_returns_none(self):
        result = build_predictor_whatif_plan("sf", 1000.0, 200.0, [])
        assert result is None

    def test_empty_question_returns_none(self):
        os.environ["ENABLE_AGENT"] = "true"
        result = build_advisor_agent_answer("sf", "   ", {})
        assert result is None


# ── Public functions: cache hit skips API call ────────────────────────────────

class TestCacheHit:
    """If the answer is cached, no HTTP request should be made."""

    def setup_method(self):
        _CACHE.clear()
        os.environ["ENABLE_AGENT"] = "true"
        os.environ["AGENT_PROVIDER"] = "anthropic"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake"

    def teardown_method(self):
        _CACHE.clear()
        for k in ("ENABLE_AGENT", "AGENT_PROVIDER", "ANTHROPIC_API_KEY"):
            os.environ.pop(k, None)

    def test_advisor_plan_uses_cache(self):
        payload = {
            "city": "sf",
            "superhost_probability": 0.8,
            "response_rate": 100,
            "amenity_count": 10,
            "top_drivers": [],
            "recommendations": [],
            "strengths": [],
            "weaknesses": [],
        }
        key = _make_cache_key("advisor_plan", payload, agent_model())
        _cache_set(key, "cached response")

        with patch("services.llm_agent.requests.post") as mock_post:
            result = build_advisor_agent_plan("sf", {
                "superhost_probability": 0.8,
                "host_response_rate_clean": 100,
                "amenity_count": 10,
            }, [], [], [], [])
            mock_post.assert_not_called()
        assert result == "cached response"


# ── Public functions: successful API call ─────────────────────────────────────

def _mock_anthropic_response(text: str) -> MagicMock:
    """Build a fake requests.Response for Anthropic."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": text}]
    }
    return mock_resp


class TestSuccessfulAPICalls:
    def setup_method(self):
        _CACHE.clear()
        os.environ["ENABLE_AGENT"] = "true"
        os.environ["AGENT_PROVIDER"] = "anthropic"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake"

    def teardown_method(self):
        _CACHE.clear()
        for k in ("ENABLE_AGENT", "AGENT_PROVIDER", "ANTHROPIC_API_KEY"):
            os.environ.pop(k, None)

    def test_advisor_plan_returns_text(self):
        with patch("services.llm_agent.requests.post",
                   return_value=_mock_anthropic_response("## Plan\n- Do X")):
            result = build_advisor_agent_plan(
                "sf",
                {"superhost_probability": 0.7, "host_response_rate_clean": 90,
                 "amenity_count": 8},
                [], [], [], []
            )
        assert result == "## Plan\n- Do X"

    def test_advisor_plan_is_cached_after_call(self):
        with patch("services.llm_agent.requests.post",
                   return_value=_mock_anthropic_response("cached plan")):
            build_advisor_agent_plan("sf", {}, [], [], [], [])
        assert len(_CACHE) == 1

    def test_investor_brief_returns_text(self):
        with patch("services.llm_agent.requests.post",
                   return_value=_mock_anthropic_response("## Brief")):
            result = build_investor_agent_brief("nyc", {"neighbourhood": "SoHo"}, {})
        assert result == "## Brief"

    def test_investor_answer_returns_text(self):
        with patch("services.llm_agent.requests.post",
                   return_value=_mock_anthropic_response("Good investment.")):
            result = build_investor_agent_answer("chicago", "Is this a good deal?", {})
        assert result == "Good investment."

    def test_whatif_plan_returns_text(self):
        with patch("services.llm_agent.requests.post",
                   return_value=_mock_anthropic_response("Add a hot tub.")):
            result = build_predictor_whatif_plan("sf", 5000.0, 200.0, [])
        assert result == "Add a hot tub."


# ── HTTP error handling ───────────────────────────────────────────────────────

class TestAPIErrorHandling:
    def setup_method(self):
        _CACHE.clear()
        os.environ["ENABLE_AGENT"] = "true"
        os.environ["AGENT_PROVIDER"] = "anthropic"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake"

    def teardown_method(self):
        _CACHE.clear()
        for k in ("ENABLE_AGENT", "AGENT_PROVIDER", "ANTHROPIC_API_KEY"):
            os.environ.pop(k, None)

    def test_http_error_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch("services.llm_agent.requests.post", return_value=mock_resp):
            result = build_advisor_agent_answer("sf", "What should I do?", {})
        assert result is None

    def test_network_exception_returns_none(self):
        import requests as req
        with patch("services.llm_agent.requests.post",
                   side_effect=req.exceptions.ConnectionError("network down")):
            result = build_investor_agent_brief("sf", {}, {})
        assert result is None
