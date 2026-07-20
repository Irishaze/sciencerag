"""Tests for LLM/embedding backend config (sciencerag/common/config.py)."""

from sciencerag.common import config


def test_default_llm_model_is_deepseek(monkeypatch):
    monkeypatch.delenv("SCIENCERAG_LLM_MODEL", raising=False)
    assert config.get_llm_model() == "deepseek/deepseek-chat"


def test_default_embedding_model_is_local_sentence_transformer(monkeypatch):
    monkeypatch.delenv("SCIENCERAG_EMBEDDING_MODEL", raising=False)
    assert config.get_embedding_model() == "st-all-MiniLM-L6-v2"


def test_llm_model_is_overridable_via_env(monkeypatch):
    monkeypatch.setenv("SCIENCERAG_LLM_MODEL", "openai/gpt-4o-mini")
    assert config.get_llm_model() == "openai/gpt-4o-mini"


def test_embedding_model_is_overridable_via_env(monkeypatch):
    monkeypatch.setenv("SCIENCERAG_EMBEDDING_MODEL", "text-embedding-3-small")
    assert config.get_embedding_model() == "text-embedding-3-small"
