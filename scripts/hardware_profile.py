#!/usr/bin/env python3
"""Load one audited hardware profile without adding a YAML dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "configs/hardware"
SCHEMA_VERSION = "shopping-hardware-profile-v1"


def profile_path(name_or_path: str | Path) -> Path:
    candidate = Path(name_or_path)
    if not candidate.suffix and candidate.parent == Path("."):
        candidate = PROFILE_DIR / f"{candidate.name}.yaml"
    elif not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def load_hardware_profile(name_or_path: str | Path) -> dict:
    """Read the JSON-compatible YAML profile and validate its public schema."""
    path = profile_path(name_or_path)
    if not path.is_file():
        raise ValueError(f"hardware profile does not exist: {path}")
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid hardware profile {path}: {exc}") from exc
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"hardware profile must use {SCHEMA_VERSION}: {path}")
    if profile.get("name") != path.stem:
        raise ValueError(f"hardware profile name must match filename: {path}")
    grpo = profile.get("grpo")
    sft = profile.get("sft")
    if not isinstance(grpo, dict) or not isinstance(sft, dict):
        raise ValueError("hardware profile requires grpo and sft objects")
    overrides = grpo.get("hydra_overrides")
    environment = grpo.get("environment")
    arguments = sft.get("arguments")
    if not isinstance(overrides, list) or not all(
        isinstance(item, str) for item in overrides
    ):
        raise ValueError("grpo.hydra_overrides must be a list of strings")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, (str, int, float, bool))
        for key, value in environment.items()
    ):
        raise ValueError("grpo.environment must contain scalar environment values")
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) for item in arguments
    ):
        raise ValueError("sft.arguments must be a list of strings")
    profile["path"] = str(path)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("sft-args", "grpo-overrides", "grpo-environment"),
    )
    parser.add_argument("profile")
    args = parser.parse_args()
    try:
        profile = load_hardware_profile(args.profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.stage == "sft-args":
        values = profile["sft"]["arguments"]
    elif args.stage == "grpo-overrides":
        values = profile["grpo"]["hydra_overrides"]
    else:
        values = [
            f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in profile["grpo"]["environment"].items()
        ]
    print("\n".join(values))


if __name__ == "__main__":
    main()
