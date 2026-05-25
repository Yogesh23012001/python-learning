"""Multi-turn conversation by passing history explicitly."""

from __future__ import annotations

from api.config import get_settings
from google import genai


def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    chat = client.chats.create(model=settings.gemini_default_model)

    response = chat.send_message("My name is Yogesh. I am learning AI.")
    print("Turn 1:", response.text)
    print()

    response = chat.send_message("What was my name again?")
    print("Turn 2:", response.text)
    print()

    response = chat.send_message("In one short sentence, what's a tip for learning AI?")
    print("Turn 3:", response.text)


if __name__ == "__main__":
    main()
