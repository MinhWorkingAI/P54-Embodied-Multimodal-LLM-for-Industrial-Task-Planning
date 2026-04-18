"""
backends/__init__.py
--------------------
Factory that returns the correct LLM based on the LLM_BACKEND env var.

Usage:
    from backends import get_llm
    llm = get_llm()   # reads LLM_BACKEND from environment
"""

import os

SUPPORTED_BACKENDS = ("openai", "gemini", "deepseek")


def get_llm():
    """
    Return an initialised LangChain LLM based on the LLM_BACKEND env var.

    LLM_BACKEND=openai    -> openai_backend.build_llm()   (default)
    LLM_BACKEND=gemini    -> gemini_backend.build_llm()
    LLM_BACKEND=deepseek  -> deepseek_backend.build_llm()

    Raises:
        ValueError: If LLM_BACKEND is set to an unsupported value.
    """
    backend = os.getenv("LLM_BACKEND", "openai").lower().strip()

    if backend == "openai":
        from .openai_backend import build_llm  
    elif backend == "gemini":
        from .gemini_backend import build_llm  
    elif backend == "deepseek":
        from .deepseek_backend import build_llm
    else:
        raise ValueError(
            f"Unsupported LLM_BACKEND='{backend}'. "
            f"Choose one of: {SUPPORTED_BACKENDS}"
        )

    return build_llm()