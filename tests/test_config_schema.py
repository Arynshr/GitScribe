import pytest
from pydantic import ValidationError

from gitscribe.core.config_schema import GitScribeConfig

VALID_RAW = {
    "llm": {"model": "llama-3.3-70b-versatile", "fallback_model": "llama-3.1-8b-instant"},
}


def test_valid_config_loads_with_defaults():
    cfg = GitScribeConfig(**VALID_RAW)
    assert cfg.llm.model == "llama-3.3-70b-versatile"
    assert cfg.retrieval.min_prs == 3
    assert cfg.risk_classifier.trivial_threshold == 0.15


def test_missing_required_llm_model_raises():
    with pytest.raises(ValidationError):
        GitScribeConfig(llm={"fallback_model": "llama-3.1-8b-instant"})


def test_temperature_out_of_range_raises():
    with pytest.raises(ValidationError):
        GitScribeConfig(llm={**VALID_RAW["llm"], "temperature": 5.0})


def test_trivial_threshold_out_of_range_raises():
    with pytest.raises(ValidationError):
        GitScribeConfig(**VALID_RAW, risk_classifier={"trivial_threshold": 1.5})


def test_as_dict_matches_node_access_pattern():
    cfg = GitScribeConfig(**VALID_RAW)
    d = cfg.as_dict()
    assert d["llm"]["model"] == "llama-3.3-70b-versatile"
    assert d["retrieval"]["max_prs"] == 10
