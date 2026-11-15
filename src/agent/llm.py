import os


def get_llm():
    """Builds the chat model from env vars. Provider SDKs are imported lazily
    so switching providers later doesn't mean installing every SDK up front,
    and tests never need this at all (they inject a fake model)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # a local model can fall into a repetition loop and generate for tens of minutes, so both the
        # length of one reply and the wait for it are capped (a capped case is recorded as a failure)
        return ChatOllama(
            model=model or "qwen3:8b",
            base_url=os.getenv("OLLAMA_BASE_URL"),
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "3000")),
            client_kwargs={"timeout": float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))},
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4o-mini")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or "claude-3-5-haiku-latest")

    raise ValueError(
        f"unsupported LLM_PROVIDER '{provider}' (supported: ollama, openai, anthropic)"
    )
