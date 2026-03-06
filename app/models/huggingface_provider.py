import os
import threading
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.config import settings
from app.models.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """LLM provider using local HuggingFace Transformers models."""

    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.lock = threading.Lock()
        self.device: Optional[str] = None
        self._loaded = False
        self._model_path: Optional[str] = None
        self._system_role_supported: bool = True
        self._resolved_model_name: Optional[str] = None

        # Apply CUDA device restriction if configured
        self._configure_cuda_devices()

    def _configure_cuda_devices(self) -> None:
        """Set CUDA_VISIBLE_DEVICES if configured."""
        if settings.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
            logger.info(f"Set CUDA_VISIBLE_DEVICES={settings.cuda_visible_devices}")

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if settings.model_device:
            return settings.model_device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_model_path(self) -> str:
        """Get the model path (local path or HuggingFace model name)."""
        if settings.model_path and settings.model_path.exists():
            return str(settings.model_path)
        return settings.model_name

    def _get_torch_dtype(self) -> torch.dtype:
        """Determine the torch dtype based on configuration."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        if settings.model_dtype in dtype_map:
            return dtype_map[settings.model_dtype]

        # Auto-detect based on device
        if self.device in ("cuda", "mps"):
            # Check if bfloat16 is supported on CUDA
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _check_system_role_support(self) -> bool:
        """Return True if the tokenizer's chat template accepts a system role."""
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}],
                tokenize=False,
                add_generation_prompt=False,
            )
            logger.info("Chat template supports system role")
            return True
        except Exception:
            logger.info("Chat template does not support system role; system prompt will be merged into first user message")
            return False

    def _apply_system_prompt(self, messages: list[dict], system_prompt: str) -> list[dict]:
        """Prepend the system prompt, merging it into the first user message if needed."""
        if self._system_role_supported:
            return [{"role": "system", "content": system_prompt}] + messages
        # Merge into the first user turn
        messages = list(messages)
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                messages[i] = {"role": "user", "content": f"{system_prompt}\n\n{msg['content']}"}
                return messages
        # Fallback: no user message found, prepend as a user turn
        return [{"role": "user", "content": system_prompt}] + messages

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization config if enabled."""
        if settings.load_in_4bit or settings.model_dtype == "4bit":
            logger.info("Using 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif settings.load_in_8bit or settings.model_dtype == "8bit":
            logger.info("Using 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None

    def load_model(self) -> None:
        """Load the model and tokenizer. Called once at startup."""
        if self._loaded:
            logger.warning("Model already loaded, skipping reload")
            return

        self.device = self._detect_device()
        self._model_path = self._get_model_path()

        logger.info(f"Loading model from: {self._model_path}")
        logger.info(f"Target device: {self.device}")

        if torch.cuda.is_available():
            logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=settings.trust_remote_code,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect whether this model's chat template supports the system role
        self._system_role_supported = self._check_system_role_support()

        # Determine dtype and quantization
        torch_dtype = self._get_torch_dtype()
        quantization_config = self._get_quantization_config()

        logger.info(f"Using dtype: {torch_dtype}")

        # Build model loading kwargs
        model_kwargs = {
            "trust_remote_code": settings.trust_remote_code,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        # Use device_map for CUDA, manual placement otherwise
        if self.device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"
        elif self.device == "mps":
            # MPS doesn't support device_map, load to CPU then move
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            **model_kwargs,
        )

        # Move to device if not using device_map
        if not self.device.startswith("cuda"):
            self.model.to(self.device)

        self.model.eval()
        self._loaded = True

        # Resolve the effective model name from the loaded tokenizer so that
        # a MODEL_PATH-only config shows the real model identifier, not the default.
        self._resolved_model_name = self.tokenizer.name_or_path
        logger.info(f"Resolved model name: {self._resolved_model_name}")

        # Log memory usage for CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i} memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        logger.info("Model loaded successfully")

    def unload_model(self) -> None:
        """Unload the model to free memory. Called at shutdown."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False

        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")

    def generate_response(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate a response given a conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Roles should be 'user' or 'assistant'.
            max_new_tokens: Maximum tokens to generate (default from settings)
            temperature: Sampling temperature (default from settings)
            top_p: Top-p sampling parameter (default from settings)

        Returns:
            The generated assistant response text.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_new_tokens = max_new_tokens if max_new_tokens is not None else settings.max_response_tokens
        temperature = temperature if temperature is not None else settings.temperature
        top_p = top_p if top_p is not None else settings.top_p

        # Prepend system prompt if configured
        if settings.system_prompt:
            messages = self._apply_system_prompt(messages, settings.system_prompt)

        # Use lock to ensure thread-safe inference
        with self.lock:
            # Apply chat template (Qwen2.5 supports this natively)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.max_model_len - max_new_tokens,  # Leave room for response
            )

            # Move inputs to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Build generation kwargs
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Only use sampling if temperature > 0
            if temperature > 0:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": settings.top_k if settings.top_k > 0 else None,
                    "repetition_penalty": settings.repetition_penalty,
                })
            else:
                gen_kwargs["do_sample"] = False

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            # Decode only the new tokens (response)
            response_ids = outputs[0][input_ids.shape[1] :]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            return response.strip()

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens for the given conversation messages."""
        if not self._loaded or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not messages:
            return 0

        if settings.system_prompt:
            messages = self._apply_system_prompt(messages, settings.system_prompt)

        with self.lock:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = self.tokenizer(text, add_special_tokens=False)

        input_ids = inputs.get("input_ids", [])
        if isinstance(input_ids, torch.Tensor):
            return int(input_ids.shape[-1])
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)

    def count_token_breakdown(self, messages: list[dict[str, str]]) -> dict[str, int]:
        """Count tokens by role for the given conversation messages."""
        if not self._loaded or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        system_tokens = 0
        if settings.system_prompt:
            system_tokens = len(self.tokenizer.encode(settings.system_prompt))

        user_tokens = 0
        assistant_tokens = 0
        for message in messages:
            content = str(message.get("content", ""))
            token_count = len(self.tokenizer.encode(content))
            if message.get("role") == "user":
                user_tokens += token_count
            elif message.get("role") == "assistant":
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
        """Check if the model is loaded."""
        return self._loaded

    @property
    def model_name(self) -> str:
        """Get the effective model name.

        Returns the name derived from the loaded model when MODEL_NAME was not
        explicitly configured, falling back to the settings value otherwise.
        """
        return self._resolved_model_name or settings.model_name
