"""
Utility functions — LLM clients via LangChain.
"""

import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.config import (
    GROQ_API_KEY, GROQ_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
)


def get_groq_llm(temperature: float = 0.1, max_tokens: int = 2000) -> ChatGroq:
    """Get a LangChain ChatGroq instance."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env file")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_openrouter_llm(temperature: float = 0.1, max_tokens: int = 2000) -> ChatOpenAI:
    """Get a LangChain ChatOpenAI instance pointing to OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in .env file")
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm(provider: str = "groq", **kwargs):
    """Get LLM instance by provider name."""
    if provider == "groq":
        return get_groq_llm(**kwargs)
    elif provider == "openrouter":
        return get_openrouter_llm(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def safe_float(value, default=0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0) -> int:
    """Safely convert a value to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default