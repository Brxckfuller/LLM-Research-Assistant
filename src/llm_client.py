
import os
import requests

from src.ollama_client import ask_ollama


def generate_answer(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        return ask_ollama(prompt)

    if provider == "hosted":
        api_key = os.getenv("LLM_API_KEY")
        api_url = os.getenv(
            "LLM_API_URL",
            "https://api.openai.com/v1/chat/completions",
        )
        model = os.getenv("LLM_MODEL", "gpt-4.1-mini")

        if not api_key:
            raise RuntimeError("LLM_API_KEY is not configured.")

        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_completion_tokens": 650,
            },
            timeout=120,
        )

        response.raise_for_status()

        return (
            response.json()["choices"][0]["message"]["content"]
            or ""
        ).strip()

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
