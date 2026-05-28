from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def dict_to_namespace(data: dict[str, Any]) -> SimpleNamespace:
    """Convert nested dictionaries to SimpleNamespace for dot access."""
    converted: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            converted[key] = dict_to_namespace(value)
        elif isinstance(value, list):
            converted[key] = [dict_to_namespace(item) if isinstance(item, dict) else item for item in value]
        else:
            converted[key] = value
    return SimpleNamespace(**converted)


def namespace_to_dict(obj: Any) -> Any:
    """Convert SimpleNamespace recursively back to dictionaries."""
    if isinstance(obj, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, list):
        return [namespace_to_dict(item) for item in obj]
    return obj


def load_config(config_path: str | Path) -> SimpleNamespace:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return dict_to_namespace(data)


def save_config(config: SimpleNamespace, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(namespace_to_dict(config), file, sort_keys=False)


def config_arg_parser(description: str) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML config file.",
    )
    return parser
