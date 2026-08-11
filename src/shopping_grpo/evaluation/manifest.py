"""Reproducibility manifest helpers for offline evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from shopping_grpo.evaluation.contracts import (
    CONTRACT_VERSION,
    JUDGE_SCHEMA_VERSION,
    RUBRIC_SCHEMA_VERSION,
)
from shopping_grpo.evaluation.metrics import DETERMINISTIC_METRICS_VERSION
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION,
    TRAJECTORY_JUDGE_PROMPT_VERSION,
)
from shopping_grpo.evaluation.results import EVALUATION_RESULT_VERSION
from shopping_grpo.evaluation.trajectory import NORMALIZED_TRAJECTORY_VERSION


RUN_MANIFEST_VERSION = "shopping-evaluation-run-manifest-v1"
_SECRET_FIELD_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "access_token",
    "client_secret",
    "secret",
    "password",
}


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_secrets(value: object, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _SECRET_FIELD_NAMES or normalized.endswith(
                "_api_key"
            ):
                raise ValueError(f"{path}.{key} is a forbidden secret field")
            _reject_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")


def build_run_manifest(
    *,
    run_id: str,
    actor: Mapping,
    task_manifest: Mapping,
    environment: Mapping,
    protocol: Mapping,
    code: Mapping,
    judge: Mapping,
    outputs: Mapping | None = None,
    created_at: str | None = None,
) -> dict:
    """Build a versioned manifest while refusing to serialize credentials."""

    manifest = {
        "schema_version": RUN_MANIFEST_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "run_id": str(run_id),
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat(),
        "actor": deepcopy(dict(actor)),
        "task_manifest": deepcopy(dict(task_manifest)),
        "environment": deepcopy(dict(environment)),
        "protocol": deepcopy(dict(protocol)),
        "code": deepcopy(dict(code)),
        "judge": deepcopy(dict(judge)),
        "outputs": deepcopy(dict(outputs or {})),
        "artifact_schemas": {
            "rubric": RUBRIC_SCHEMA_VERSION,
            "normalized_trajectory": NORMALIZED_TRAJECTORY_VERSION,
            "deterministic_metrics": DETERMINISTIC_METRICS_VERSION,
            "judge": JUDGE_SCHEMA_VERSION,
            "per_task_evaluation": EVALUATION_RESULT_VERSION,
        },
        "prompt_versions": {
            "rubric_curator": RUBRIC_CURATOR_PROMPT_VERSION,
            "trajectory_judge": TRAJECTORY_JUDGE_PROMPT_VERSION,
        },
    }
    _reject_secrets(manifest)
    return manifest
