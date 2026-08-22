"""Fail-closed validation guards for frozen semantic Discovery arm plumbing."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import torch

from research_programs.planner.semantic_feedback_readiness import (
    DonorUnit,
    select_wrong_semantic_donor,
)


class SemanticDiscoveryValidationError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def validate_a3r_training_equivalence(
    *,
    a3_config: dict[str, Any],
    a3r_config: dict[str, Any],
    a3_training_row_sha256: str,
    a3r_training_row_sha256: str,
) -> str:
    """Require exact A3/A3r training identity except the frozen latent target block."""
    if a3_config.get("variant") != "A3" or a3r_config.get("variant") != "A3r":
        raise SemanticDiscoveryValidationError("A3/A3r variant identity mismatch")
    if a3_training_row_sha256 != a3r_training_row_sha256:
        raise SemanticDiscoveryValidationError("A3r training examples/order differ from A3")

    normalized_a3 = copy.deepcopy(a3_config)
    normalized_a3r = copy.deepcopy(a3r_config)
    normalized_a3["variant"] = "A3"
    normalized_a3r["variant"] = "A3"
    normalized_a3.pop("a3r", None)
    normalized_a3r.pop("a3r", None)
    if normalized_a3 != normalized_a3r:
        raise SemanticDiscoveryValidationError(
            "A3r config differs from A3 outside frozen random-code target identity"
        )
    return _digest(
        {
            "normalized_training_config": normalized_a3,
            "training_row_sha256": a3_training_row_sha256,
        }
    )


def validate_normalized_float32_feedback(value: torch.Tensor) -> None:
    if value.dtype != torch.float32:
        raise SemanticDiscoveryValidationError("foreign semantic z must be float32")
    if value.shape[-1] != 384:
        raise SemanticDiscoveryValidationError("foreign semantic z must be 384-dimensional")
    if not torch.isfinite(value).all():
        raise SemanticDiscoveryValidationError("foreign semantic z must be finite")
    norms = torch.linalg.vector_norm(value, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0):
        raise SemanticDiscoveryValidationError("foreign semantic z must be l2-normalized")


def select_frozen_wrong_semantic_donor(
    target: DonorUnit, candidates: list[DonorUnit]
) -> tuple[DonorUnit | None, str]:
    donor = select_wrong_semantic_donor(target, candidates)
    if donor is None:
        return None, "NOT_EVALUATED_DONOR_UNAVAILABLE"
    return donor, "EVALUATED"
