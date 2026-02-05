import logging

from app.config import settings, LLMProvider
from app.models.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


def create_llm_provider() -> BaseLLMProvider:
    """Create the appropriate LLM provider based on configuration.

    Returns:
        An instance of BaseLLMProvider.

    Raises:
        ValueError: If the configured provider is not recognized.
    """
    if settings.llm_provider == LLMProvider.huggingface:
        from app.models.huggingface_provider import HuggingFaceProvider
        logger.info("Creating HuggingFace provider")
        return HuggingFaceProvider()

    elif settings.llm_provider == LLMProvider.openai:
        from app.models.openai_provider import OpenAIProvider
        logger.info("Creating OpenAI provider")
        return OpenAIProvider()

    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
