import os
import requests


CHAT_APP_URL = os.getenv("CHAT_APP_URL", "http://chat-app:8001")


def get_api_response(query, session_id=None, model=None):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    data = {
        "query": query,
        "model": model,
    }

    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post(
            f"{CHAT_APP_URL}/chat",
            headers=headers,
            json=data,
            timeout=120,
        )

        if response.status_code == 200:
            return response.json()

        return {
            "error": (
                f"API request failed with status code "
                f"{response.status_code}: {response.text}"
            ),
            "is_done": True,
            "response": None,
        }

    except requests.exceptions.ConnectionError as e:
        return {
            "error": f"Backend nicht erreichbar: {e}",
            "is_done": True,
            "response": None,
        }

    except requests.exceptions.Timeout:
        return {
            "error": "Backend-Zeitüberschreitung.",
            "is_done": True,
            "response": None,
        }

    except ValueError as e:
        return {
            "error": f"Backend hat keine gültige JSON-Antwort geliefert: {e}",
            "is_done": True,
            "response": None,
        }

    except Exception as e:
        return {
            "error": f"Unerwarteter Fehler: {e}",
            "is_done": True,
            "response": None,
        }