"""Fail-closed validation guards for frozen semantic Discovery arm plumbing."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from research_programs.planner.semantic_feedback_readiness import (
    DonorUnit,
    select_wrong_semantic_donor,
)

from .model import LockedPlanner, TaskEncoding


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


class A3rPlanner(LockedPlanner):
    """Separate trainable A3-shaped planner for frozen random-code supervision.

    The historical LockedPlanner source remains byte-identical.  A3r uses the
    exact A3 parameter inventory/activation mask while carrying a separate
    Discovery identity and checkpoint namespace.
    """

    discovery_variant = "A3r"

    def __init__(self, seed: int = 17):
        super().__init__(seed, "A3")


class FrozenSameCheckpointA3(LockedPlanner):
    """A3-only intervention wrapper with no new parameters or retraining path."""

    INTERVENTIONS = frozenset({"ZERO", "FOREIGN", "WRONG_SEMANTIC_DONOR"})

    def __init__(self, seed: int = 17):
        super().__init__(seed, "A3")

    def forward(
        self,
        encoded: TaskEncoding,
        actions: torch.Tensor,
        arg1: torch.Tensor,
        arg2: torch.Tensor,
        *,
        semantic_feedback: torch.Tensor | None = None,
        semantic_intervention: str | None = None,
        foreign_semantic_feedback: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        if semantic_intervention not in self.INTERVENTIONS:
            raise SemanticDiscoveryValidationError("explicit frozen semantic intervention required")
        if semantic_intervention in {"FOREIGN", "WRONG_SEMANTIC_DONOR"}:
            if not isinstance(foreign_semantic_feedback, torch.Tensor):
                raise SemanticDiscoveryValidationError("foreign semantic feedback required")
            validate_normalized_float32_feedback(foreign_semantic_feedback)
        elif foreign_semantic_feedback is not None:
            raise SemanticDiscoveryValidationError("foreign semantic feedback forbidden for ZERO")

        memory = self.encode(encoded)
        steps = actions.shape[1]
        pos = torch.arange(steps, device=actions.device)
        previous = torch.cat([torch.full_like(actions[:, :1], 4), actions[:, :-1]], 1)
        action_part = F.embedding(previous, self.concept_packer.previous_action_embedding.weight)
        action_part[:, 0] = self.concept_packer.bos_embedding
        refs = memory[:, encoded.ref_slot_positions]
        prev1 = torch.cat([torch.zeros_like(arg1[:, :1]), arg1[:, :-1]], 1)
        prev2 = torch.cat([torch.zeros_like(arg2[:, :1]), arg2[:, :-1]], 1)
        r1 = refs.gather(1, prev1[..., None].expand(-1, -1, 256))
        r2 = refs.gather(1, prev2[..., None].expand(-1, -1, 256))
        r1 = r1 * (previous != 4)[..., None]
        r2 = r2 * ((previous == 1) | (previous == 3))[..., None]
        x = action_part + self.concept_packer.step_position_embedding.weight[pos]
        x = x + F.linear(r1, self.concept_packer.previous_arg1_projection.weight)
        x = x + F.linear(r2, self.concept_packer.previous_arg2_projection.weight)

        if semantic_feedback is None:
            semantic_feedback = torch.zeros((*x.shape[:2], 384), device=x.device)
        projection_input = semantic_feedback
        if semantic_intervention in {"FOREIGN", "WRONG_SEMANTIC_DONOR"}:
            assert foreign_semantic_feedback is not None
            if foreign_semantic_feedback.shape != semantic_feedback.shape:
                raise SemanticDiscoveryValidationError("foreign semantic feedback shape mismatch")
            projection_input = foreign_semantic_feedback

        projected_semantic = self.project_semantic(projection_input).clone()
        projected_semantic[:, 0] = 0
        semantic_component = projected_semantic
        if semantic_intervention == "ZERO":
            # Frozen intervention surface: after exact A3 projection + normalization.
            semantic_component = torch.zeros_like(projected_semantic)

        x = x + semantic_component
        x = F.layer_norm(
            x,
            (256,),
            self.concept_packer.output_norm.weight,
            self.concept_packer.output_norm.bias,
            1e-5,
        )
        x = x + self.planner_decoder.position_embedding.weight[pos]
        for layer in self.planner_decoder.layers.children():
            y = F.layer_norm(x, (256,), layer.norm1.weight, layer.norm1.bias, 1e-5)
            x = x + self._attention(y, y, layer.self_attn, causal=True)
            y = F.layer_norm(x, (256,), layer.norm2.weight, layer.norm2.bias, 1e-5)
            x = x + self._attention(y, memory, layer.cross_attn, key_mask=encoded.attention_mask)
            y = F.layer_norm(x, (256,), layer.norm3.weight, layer.norm3.bias, 1e-5)
            x = x + self._ffn(y, layer.ffn)
        x = F.layer_norm(
            x,
            (256,),
            self.planner_decoder.final_norm.weight,
            self.planner_decoder.final_norm.bias,
            1e-5,
        )
        action_logits = F.linear(x, self.heads.action.weight, self.heads.action.bias)
        arg1_logits = (x @ self.heads.arg1_pointer.weight) @ refs.transpose(1, 2)
        arg2_logits = (x @ self.heads.arg2_pointer.weight) @ refs.transpose(1, 2)
        z_semantic = self.latent(x)
        return SimpleNamespace(
            action=action_logits,
            arg1=arg1_logits,
            arg2=arg2_logits,
            hidden=x,
            z_semantic=z_semantic,
            projected_semantic=projected_semantic,
            semantic_component=semantic_component,
            semantic_intervention=semantic_intervention,
        )


def select_frozen_wrong_semantic_donor(
    target: DonorUnit, candidates: list[DonorUnit]
) -> tuple[DonorUnit | None, str]:
    donor = select_wrong_semantic_donor(target, candidates)
    if donor is None:
        return None, "NOT_EVALUATED_DONOR_UNAVAILABLE"
    return donor, "EVALUATED"