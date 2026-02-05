from enum import Enum

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class LLMProvider(str, Enum):
    """Supported LLM provider backends."""
    huggingface = "huggingface"
    openai = "openai"


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables or .env file.

    All settings can be overridden by setting environment variables with the same name
    (case-insensitive). For example: PORT=9000 or MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
    """

    # ===================
    # Application Settings
    # ===================
    app_name: str = Field(
        default="CloverDX LLM Chat",
        description="Application name displayed in the UI"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, auto-reload)"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )

    # ===================
    # Server Settings
    # ===================
    host: str = Field(
        default="0.0.0.0",
        description="Host address to bind the server to"
    )
    port: int = Field(
        default=8000,
        description="Port number for the HTTP server"
    )
    workers: int = Field(
        default=1,
        description="Number of Uvicorn worker processes (use 1 for GPU inference)"
    )

    # ===================
    # Provider Settings
    # ===================
    llm_provider: LLMProvider = Field(
        default=LLMProvider.huggingface,
        description="LLM provider to use: 'huggingface' (local model) or 'openai' (API)"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (required when llm_provider=openai)"
    )
    openai_base_url: Optional[str] = Field(
        default=None,
        description="Custom OpenAI-compatible API base URL (for Azure, vLLM, etc.)"
    )

    # ===================
    # Model Settings
    # ===================
    model_name: str = Field(
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        description="Model name: HuggingFace model ID, or OpenAI model name (e.g., 'gpt-4o-mini', 'ft:gpt-4o-mini:org:custom:id')"
    )
    model_path: Optional[Path] = Field(
        default=None,
        description="Local path to model files (overrides model_name if set)"
    )
    model_device: Optional[str] = Field(
        default=None,
        description="Device to run model on: 'cuda', 'cuda:0', 'cuda:1', 'mps', 'cpu', or None for auto-detect"
    )
    cuda_visible_devices: Optional[str] = Field(
        default=None,
        description="Comma-separated list of CUDA device IDs to make visible (e.g., '0,1' or '2')"
    )
    model_dtype: str = Field(
        default="auto",
        description="Model precision: 'auto', 'float16', 'bfloat16', 'float32', or '4bit', '8bit' for quantization"
    )
    max_model_len: int = Field(
        default=4096,
        description="Maximum context length for the model (input + output tokens)"
    )
    max_response_tokens: int = Field(
        default=1024,
        description="Maximum tokens to generate in a single response"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, higher = more random)"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        description="Top-k sampling parameter (0 = disabled)"
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Repetition penalty (1.0 = no penalty, higher = less repetition)"
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading models with custom code from HuggingFace"
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Load model in 8-bit quantization (requires bitsandbytes)"
    )
    load_in_4bit: bool = Field(
        default=False,
        description="Load model in 4-bit quantization (requires bitsandbytes)"
    )

    # ===================
    # Session Settings
    # ===================
    session_timeout_hours: int = Field(
        default=24,
        description="Hours before inactive sessions expire"
    )
    cookie_name: str = Field(
        default="llm_session",
        description="Name of the session cookie"
    )
    cookie_secure: bool = Field(
        default=False,
        description="Require HTTPS for session cookies (enable in production)"
    )

    # ===================
    # Persistence Settings
    # ===================
    data_dir: Path = Field(
        default=Path("data/conversations"),
        description="Directory for storing conversation logs"
    )

    # ===================
    # System Prompt
    # ===================
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to prepend to all conversations"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow both UPPER_CASE and lower_case env vars
        case_sensitive = False


settings = Settings()
