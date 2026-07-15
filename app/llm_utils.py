from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

LLMBackend = Literal["cloud", "local"]

def get_llm(backend: LLMBackend = "cloud") -> BaseChatModel:
    if backend == "cloud":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required.")
        model = os.getenv("OPENAI_MODEL", "gpt-5.4-2026-03-05")
        return ChatOpenAI(model=model, temperature=0.2)

    if backend == "local":
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return ChatOllama(model=model, base_url=host, temperature=0.2)

    raise ValueError(f"Unknown backend: {backend!r} (expected 'cloud' or 'local')")