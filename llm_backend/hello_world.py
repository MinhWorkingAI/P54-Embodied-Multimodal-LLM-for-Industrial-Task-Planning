"""
hello_world.py
--------------
Basic connection test for the llm_backend module.
Run this first to confirm your API key and chosen backend are working
before running the full parser.

Usage:
    python hello_world.py
    LLM_BACKEND=gemini python hello_world.py
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

def main():
    backend = os.getenv("LLM_BACKEND", "openai").lower().strip()
    print(f"Testing backend: {backend.upper()}")
    print("-" * 40)

    try:
        from backends import get_llm
        llm = get_llm()
    except EnvironmentError as e:
        print(f"  Setup error: {e}")
        print("  Check your .env file and make sure the API key is set.")
        sys.exit(1)

    if backend == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
    elif backend == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    elif backend == "deepseek":
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    else:
        model = "unknown"

    print(f"Model    : {model}")
    print("Sending test message...\n")

    try:
        response = llm.invoke([
            HumanMessage(content=(
                "You are a robot assistant. "
                "Reply with exactly one sentence confirming you are ready."
            ))
        ])
        print(f"Response : {response.content}")
        print(f"\n  {backend.upper()} connection successful! You're ready to use the LLM module.")

    except Exception as e:
        print(f"  Connection failed: {e}")
        print("  Check your API key and internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()