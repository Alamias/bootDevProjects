import argparse
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    messages = [
        types.Content(
            role="user",
            parts=[types.Part(text=args.user_prompt)],
        )
    ]

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
    )

    if args.verbose:
        print(f"User prompt: {args.user_prompt}\n")
        if response.usage_metadata:
            um = response.usage_metadata
            print(
                f"Prompt tokens: {um.prompt_token_count}\n"
                f"Response tokens: {um.candidates_token_count}\n"
            )
        else:
            print("No usage metadata found\n")

    print(response.text)


if __name__ == "__main__":
    main()
