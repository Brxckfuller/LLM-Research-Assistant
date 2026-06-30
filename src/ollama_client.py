import requests


DEFAULT_MODEL = "llama3.1:8b"


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.85,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()

    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Open a terminal and run `ollama serve`, then try again."

    except Exception as e:
        return f"Error calling Ollama: {e}"


def stream_ollama(prompt: str, model: str = DEFAULT_MODEL):
    url = "http://localhost:11434/api/chat"

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
        with requests.post(url, json=payload, stream=True, timeout=600) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                data = response.json() if False else None

                import json
                chunk = json.loads(line.decode("utf-8"))

                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                    if content:
                        yield content

                if chunk.get("done"):
                    break

    except requests.exceptions.ConnectionError:
        yield "Ollama is not running. Open a terminal and run `ollama serve`, then try again."

    except Exception as e:
        yield f"Error calling Ollama: {e}" 

