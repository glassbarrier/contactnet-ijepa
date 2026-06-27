"""
Configuration loading utilities.

Supports YAML config files with command-line overrides.
All paths are read from config — nothing is hardcoded.
"""

import os
import yaml
import argparse
from typing import Any, Optional


def load_yaml(path: str) -> dict:
    """Load a single YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: Optional[dict] = None) -> dict:
    """
    Deep-merge override into base recursively.
    override values take precedence.
    """
    if override is None:
        return base

    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def flatten_config(cfg: dict, prefix: str = "") -> dict:
    """Flatten nested dict to dot-separated keys for CLI override."""
    flat = {}
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, full_key))
        else:
            flat[full_key] = value
    return flat


def apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply command-line overrides of the form 'key=value' or 'section.key=value'.
    Supports basic types: int, float, bool, str, None.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")

        key, value_str = override.split("=", 1)
        value = _parse_value(value_str)
        _set_nested(cfg, key, value)

    return cfg


def _parse_value(s: str) -> Any:
    """Parse a string value to its inferred type."""
    s = s.strip()

    if s.lower() == "none":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    try:
        return int(s)
    except ValueError:
        pass

    try:
        return float(s)
    except ValueError:
        pass

    # Remove quotes if present
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]

    return s


def _set_nested(cfg: dict, key: str, value: Any):
    """Set a value in nested dict using dot-separated key."""
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cfg:
            cfg[part] = {}
        cfg = cfg[part]
    cfg[parts[-1]] = value


def load_config(
    config_path: str,
    default_path: Optional[str] = None,
    cli_overrides: Optional[list[str]] = None,
) -> dict:
    """
    Load configuration with optional default and CLI overrides.

    Args:
        config_path: Path to main config YAML.
        default_path: Optional base config to merge under main config.
        cli_overrides: List of 'key=value' strings from CLI.

    Returns:
        Merged configuration dict.
    """
    # Expand user path (~)
    config_path = os.path.expanduser(config_path)

    if default_path:
        default_path = os.path.expanduser(default_path)
        cfg = merge_configs(load_yaml(default_path), load_yaml(config_path))
    else:
        cfg = load_yaml(config_path)

    if cli_overrides:
        cfg = apply_cli_overrides(cfg, cli_overrides)

    return cfg


def create_arg_parser(description: str = "") -> argparse.ArgumentParser:
    """Create standard argument parser with config and override support."""
    parser = argparse.ArgumentParser(
        description=description or "I-JEPA for Contact Network Components",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override",
        "-o",
        type=str,
        nargs="*",
        default=[],
        help="Config overrides as key=value pairs (e.g., data.batch_size=64)",
    )
    return parser