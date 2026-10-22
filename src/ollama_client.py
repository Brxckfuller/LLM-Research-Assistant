import os
import json
import requests
import logging

logger = logging.getLogger("api.rag_service")

DEFAULT_MODEL = "llama3.1:8b"

# Local development:
# http://localhost:11434/api/chat
#
# Docker:
# http://host.docker.internal:11434/api/chat

OLLAMA_CHAT_URL = os.getenv(
    "OLLAMA_CHAT_URL",
    "http://localhost:11434/api/chat",
)


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
    "temperature": 0.1,
    "top_p": 0.85,
    "num_predict": 650,
},
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=600,
        )

        response.raise_for_status()

        data = response.json()

        logger.info(
            "ollama_inference "
            "model=%s "
            "prompt_tokens=%s "
            "generated_tokens=%s "
            "load_seconds=%.3f "
            "prompt_eval_seconds=%.3f "
            "eval_seconds=%.3f",
            model,
            data.get("prompt_eval_count"),
            data.get("eval_count"),
            data.get("load_duration", 0) / 1e9,
            data.get("prompt_eval_duration", 0) / 1e9,
            data.get("eval_duration", 0) / 1e9,
        )

        return data.get("message", {}).get("content", "").strip()

    except requests.exceptions.ConnectionError:
        return (
            "Ollama is not running. "
            "Open a terminal and run `ollama serve`, then try again."
        )

    except Exception as e:
        return f"Error calling Ollama: {e}"


def stream_ollama(prompt: str, model: str = DEFAULT_MODEL):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "temperature": 0.1,
            "top_p": 0.85,
        },
    }

    try:
        with requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            stream=True,
            timeout=600,
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line.decode("utf-8"))

                if "message" in chunk:
                    content = chunk["message"].get("content", "")

                    if content:
                        yield content

                if chunk.get("done"):
                    break

    except requests.exceptions.ConnectionError:
        yield (
            "Ollama is not running. "
            "Open a terminal and run `ollama serve`, then try again."
        )

    except Exception as e:
        yield f"Error calling Ollama: {e}"

