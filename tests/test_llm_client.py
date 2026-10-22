
from unittest.mock import patch

import pytest

from src.llm_client import generate_answer


def test_default_provider_uses_ollama():
    with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
        with patch(
            "src.llm_client.ask_ollama",
            return_value="Local response",
        ) as mock_ollama:
            result = generate_answer("Test prompt")

    assert result == "Local response"
    mock_ollama.assert_called_once_with("Test prompt")


def test_hosted_provider_returns_answer():
    with patch.dict(
        "os.environ",
        {
            "LLM_PROVIDER": "hosted",
            "LLM_API_KEY": "test-key",
        },
    ):
        with patch("src.llm_client.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "choices": [
                    {"message": {"content": "Hosted response"}}
                ]
            }

            result = generate_answer("Test prompt")

    assert result == "Hosted response"
    mock_post.assert_called_once()


def test_hosted_provider_requires_api_key():
    with patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "hosted"},
    ):
        with patch.dict("os.environ", {"LLM_API_KEY": ""}):
            with pytest.raises(RuntimeError):
                generate_answer("Test prompt")


def test_unknown_provider_is_rejected():
    with patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "unknown"},
    ):
        with pytest.raises(ValueError):
            generate_answer("Test prompt")
