"""
The Nexus Core - Configuration Management

Loads settings from environment variables. If a .env file exists in the
project root, it is parsed automatically (python-dotenv if available,
otherwise a built-in minimal parser is used).

Usage:
    from nexus_core_config import config

    print(config.nexus_data_path)
    print(config.llm_provider)

To reload config (e.g. in tests):
    from nexus_core_config import load_config
    my_config = load_config("/path/to/custom.env")
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_env_file(env_path: Path) -> None:
    """Load a .env file into os.environ without overwriting existing vars."""
    if not env_path.exists():
        return

    # Try python-dotenv first (better handling of edge cases)
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    # Built-in minimal parser (stdlib only)
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Type-safe helpers
# ---------------------------------------------------------------------------

def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(
            f"Config error: {key} must be an integer, got {val!r}"
        )


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(
            f"Config error: {key} must be a float, got {val!r}"
        )


def _env_optional(key: str) -> Optional[str]:
    """Return None if the env var is missing or empty."""
    val = os.environ.get(key, "").strip()
    return val if val else None


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NexusConfig:
    """
    Immutable configuration object for The Nexus Core.
    All values are loaded once at startup from environment variables.
    """

    # --- Paths --------------------------------------------------------------
    nexus_data_path: str

    # --- LLM Provider -------------------------------------------------------
    llm_provider: str           # openai | anthropic | ollama | none
    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str]
    ollama_base_url: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    # --- RAG Parameters -----------------------------------------------------
    search_top_k: int
    context_max_tokens: int
    dedup_similarity_threshold: float
    topic_change_threshold: float

    # --- Quality Thresholds -------------------------------------------------
    quality_high_threshold: float
    quality_med_threshold: float
    quality_low_threshold: float

    # --- Brain-Like AI ------------------------------------------------------
    recursive_max_depth: int
    recursive_reflection_threshold: float
    memory_stm_capacity: int
    memory_stm_retention_minutes: int

    # --- Logging ------------------------------------------------------------
    log_level: str

    def validate(self) -> None:
        """Raise ValueError for any invalid configuration."""
        valid_providers = {"openai", "anthropic", "ollama", "none"}
        if self.llm_provider not in valid_providers:
            raise ValueError(
                f"LLM_PROVIDER must be one of {valid_providers}, "
                f"got {self.llm_provider!r}"
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic"
            )
        if not (0.0 <= self.llm_temperature <= 2.0):
            raise ValueError(
                f"LLM_TEMPERATURE must be between 0.0 and 2.0, "
                f"got {self.llm_temperature}"
            )
        if not (0.0 <= self.dedup_similarity_threshold <= 1.0):
            raise ValueError(
                f"DEDUP_SIMILARITY_THRESHOLD must be 0.0-1.0, "
                f"got {self.dedup_similarity_threshold}"
            )
        if not (0.0 <= self.recursive_reflection_threshold <= 1.0):
            raise ValueError(
                f"RECURSIVE_REFLECTION_THRESHOLD must be 0.0-1.0, "
                f"got {self.recursive_reflection_threshold}"
            )
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {valid_log_levels}, "
                f"got {self.log_level!r}"
            )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(env_file: Optional[str] = None) -> NexusConfig:
    """
    Load configuration from environment variables.

    Args:
        env_file: Path to a .env file. Defaults to '.env' in the same
                  directory as this module. Pass an empty string "" to
                  skip .env loading entirely (useful in tests).

    Returns:
        A validated, immutable NexusConfig instance.
    """
    if env_file is None:
        env_file = str(Path(__file__).parent / ".env")

    if env_file:
        _load_env_file(Path(env_file))

    cfg = NexusConfig(
        # Paths
        nexus_data_path=_env_str("NEXUS_DATA_PATH", "./nexus_data"),

        # LLM
        llm_provider=_env_str("LLM_PROVIDER", "none").lower(),
        openai_api_key=_env_optional("OPENAI_API_KEY"),
        anthropic_api_key=_env_optional("ANTHROPIC_API_KEY"),
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_model=_env_str("LLM_MODEL", ""),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.7),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 2048),

        # RAG
        search_top_k=_env_int("SEARCH_TOP_K", 5),
        context_max_tokens=_env_int("CONTEXT_MAX_TOKENS", 4096),
        dedup_similarity_threshold=_env_float("DEDUP_SIMILARITY_THRESHOLD", 0.85),
        topic_change_threshold=_env_float("TOPIC_CHANGE_THRESHOLD", 0.5),

        # Quality thresholds
        quality_high_threshold=_env_float("QUALITY_HIGH_THRESHOLD", 0.8),
        quality_med_threshold=_env_float("QUALITY_MED_THRESHOLD", 0.6),
        quality_low_threshold=_env_float("QUALITY_LOW_THRESHOLD", 0.4),

        # Brain AI
        recursive_max_depth=_env_int("RECURSIVE_MAX_DEPTH", 3),
        recursive_reflection_threshold=_env_float(
            "RECURSIVE_REFLECTION_THRESHOLD", 0.8
        ),
        memory_stm_capacity=_env_int("MEMORY_STM_CAPACITY", 20),
        memory_stm_retention_minutes=_env_int("MEMORY_STM_RETENTION_MINUTES", 30),

        # Logging
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )

    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Module-level singleton - import this in all other modules
# ---------------------------------------------------------------------------

config: NexusConfig = load_config()


# ---------------------------------------------------------------------------
# Logging setup  (runs once when this module is first imported)
# ---------------------------------------------------------------------------

def configure_logging(cfg: NexusConfig = config) -> None:
    """
    Apply cfg.log_level to the root Python logger.

    Called automatically when this module is imported.  Can be called again
    after load_config() in tests or scripts that need a custom level.
    """
    level = getattr(logging, cfg.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,   # override any handlers already attached (e.g. by uvicorn)
    )


configure_logging()
