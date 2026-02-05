import abc
from typing import Optional


class BaseLLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    def load_model(self) -> None:
        """Initialize the provider / load resources. Called once at startup."""
        ...

    @abc.abstractmethod
    def unload_model(self) -> None:
        """Release resources. Called at shutdown."""
        ...

    @abc.abstractmethod
    def generate_response(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate a response given conversation history.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.

        Returns:
            The generated assistant response text.
        """
        ...

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Whether the provider is ready to generate."""
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """The name/identifier of the active model."""
        ...
