"""BugInsight AI — Configuration Loader.

Provides a unified interface to load and access the master config.yaml.
All modules should import config from here rather than parsing YAML directly.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve project root: walk up from this file until we find configs/
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


class Config:
    """Singleton-style configuration manager for BugInsight AI.

    Loads ``configs/config.yaml`` once and provides dictionary-style and
    attribute-style access to all settings.  Paths stored in the YAML are
    resolved relative to the project root so the codebase is portable across
    local machines and cloud environments.
    """

    _instance: Optional["Config"] = None
    _data: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self, config_path: Optional[str] = None) -> None:
        """Load the YAML config file into memory."""
        if config_path is None:
            config_path = str(_THIS_DIR / "config.yaml")
        config_path = Path(config_path).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            self._data = yaml.safe_load(fh)

        logger.info("Configuration loaded from %s", config_path)

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a config-relative path against the project root."""
        return (_PROJECT_ROOT / relative_path).resolve()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a nested value using dot-notation.

        Example::

            cfg.get("models.codebert.learning_rate")  # -> 2e-5

        Args:
            dotted_key: Dot-separated key path.
            default: Value returned when the key is missing.

        Returns:
            The configuration value or *default*.
        """
        keys = dotted_key.split(".")
        node: Any = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def get_path(self, dotted_key: str) -> Path:
        """Like :meth:`get` but resolves the value as a project-relative path.

        Args:
            dotted_key: Dot-separated key whose value is a relative path string.

        Returns:
            Resolved :class:`~pathlib.Path`.

        Raises:
            KeyError: If the key does not exist in the config.
        """
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"Config key not found: {dotted_key}")
        return self._resolve_path(str(value))

    @property
    def project_root(self) -> Path:
        """Return the resolved project root directory."""
        return _PROJECT_ROOT

    @property
    def seed(self) -> int:
        """Return the primary reproducibility seed."""
        return int(self._data.get("seed", 42))

    @property
    def seeds(self) -> List[int]:
        """Return the list of seeds for multi-seed experiments."""
        return self._data.get("seeds_for_experiments", [42])

    @property
    def num_classes(self) -> int:
        """Return the number of severity classes."""
        return int(self.get("dataset.num_classes", 4))

    @property
    def label_order(self) -> List[str]:
        """Return the ordered list of severity labels."""
        return self.get("dataset.label_order", ["Critical", "Major", "Minor", "Trivial"])

    @property
    def raw(self) -> Dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._data

    def __repr__(self) -> str:
        return f"Config(keys={list(self._data.keys())})"

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing with different configs)."""
        cls._instance = None
        cls._data = {}


def load_config(config_path: Optional[str] = None) -> Config:
    """Convenience function to load and return the global :class:`Config`.

    Args:
        config_path: Optional path to a YAML config file.  When ``None``,
            the default ``configs/config.yaml`` is used.

    Returns:
        The loaded :class:`Config` singleton.
    """
    Config.reset()
    return Config(config_path)
