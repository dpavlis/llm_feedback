import logging
import re
import threading
from typing import Optional

import tiktoken

from openai import OpenAI

from app.config import settings
from app.models.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

THINKING_DISABLED_INSTRUCTION = (
    "Thinking mode is disabled. Provide only the final answer without chain-of-thought, "
    "internal reasoning, or <think> blocks."
)
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


class OpenAIProvider(BaseLLMProvider):
    """LLM provider using the OpenAI API (supports fine-tuned models)."""

    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._loaded = False
        self.lock = threading.Lock()

    def load_model(self) -> None:
        """Initialize the OpenAI client."""
        if self._loaded:
            logger.warning("OpenAI provider already initialized, skipping")
            return

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when using the 'openai' provider. "
                "Set it in your .env file or environment."
            )

        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._client = OpenAI(**client_kwargs)
        self._loaded = True

        logger.info(f"OpenAI provider initialized with model: {settings.model_name}")
        if settings.openai_base_url:
            logger.info(f"Using custom base URL: {settings.openai_base_url}")

    def unload_model(self) -> None:
        """Release the OpenAI client."""
        self._client = None
        self._loaded = False
        logger.info("OpenAI provider unloaded")

    def generate_response(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Generate a response using the OpenAI Chat Completions API.

        Note: top_k and repetition_penalty are not supported by the OpenAI API
        and are silently ignored.
        """
        if not self._loaded or self._client is None:
            raise RuntimeError("OpenAI provider not initialized. Call load_model() first.")

        max_new_tokens = max_new_tokens if max_new_tokens is not None else settings.max_response_tokens
        temperature = temperature if temperature is not None else settings.temperature
        top_p = top_p if top_p is not None else settings.top_p

        # Prepend system prompt if configured
        if settings.system_prompt:
            messages = [{"role": "system", "content": settings.system_prompt}] + messages

        if not settings.enable_thinking_mode:
            messages = [{"role": "system", "content": THINKING_DISABLED_INSTRUCTION}] + messages

        with self.lock:
            response = self._client.chat.completions.create(
                model=settings.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        content = response.choices[0].message.content or ""
        if not settings.enable_thinking_mode:
            content = THINK_TAG_RE.sub("", content)

        return content.strip()

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens for the given conversation messages."""
        if not messages:
            return 0

        if settings.system_prompt:
            messages = [{"role": "system", "content": settings.system_prompt}] + messages

        try:
            encoding = tiktoken.encoding_for_model(settings.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        # Defaults based on OpenAI token counting guidance for chat models.
        tokens_per_message = 3
        tokens_per_name = 1

        if settings.model_name == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1

        total_tokens = 0
        for message in messages:
            total_tokens += tokens_per_message
            for key, value in message.items():
                total_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    total_tokens += tokens_per_name

        total_tokens += 3
        return total_tokens

    def count_token_breakdown(self, messages: list[dict[str, str]]) -> dict[str, int]:
        """Count tokens by role for the given conversation messages."""
        if not messages:
            return {"system": 0, "user": 0, "assistant": 0, "total": 0}

        if settings.system_prompt:
            messages = [{"role": "system", "content": settings.system_prompt}] + messages

        try:
            encoding = tiktoken.encoding_for_model(settings.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        system_tokens = 0
        user_tokens = 0
        assistant_tokens = 0
        for message in messages:
            content = str(message.get("content", ""))
            token_count = len(encoding.encode(content))
            role = message.get("role")
            if role == "system":
                system_tokens += token_count
            elif role == "user":
                user_tokens += token_count
            elif role == "assistant":
                assistant_tokens += token_count

        total_tokens = self.count_tokens(messages)
        return {
            "system": system_tokens,
            "user": user_tokens,
            "assistant": assistant_tokens,
            "total": total_tokens,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return settings.model_name
