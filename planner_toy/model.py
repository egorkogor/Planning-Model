"""Contract-locked PyTorch A2 model used by the non-confirmatory toy slice."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

SEED = 17
TORCH_VERSION = "2.12.0"
INVENTORY = Path(__file__).parents[1] / "docs/architecture/planner_module_inventory_v1.yaml"


def validate_torch_runtime() -> None:
    version = torch.__version__.split("+", 1)[0]
    if version != TORCH_VERSION:
        raise RuntimeError(f"PyTorch {TORCH_VERSION} required, found {torch.__version__}")


class _Node(nn.Module):
    pass


def _install(root: nn.Module, name: str, shape: list[int]) -> None:
    node = root
    parts = name.split(".")
    for part in parts[:-1]:
        if not hasattr(node, part):
            node.add_module(part, _Node())
        node = getattr(node, part)
    node.register_parameter(parts[-1], nn.Parameter(torch.empty(shape, dtype=torch.float32)))


def _inventory() -> dict:
    with INVENTORY.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class LockedA2(nn.Module):
    """A2 whose 177 state tensors are constructed directly from the locked inventory."""

    def __init__(self, seed: int = SEED):
        super().__init__()
        validate_torch_runtime()
        contract = _inventory()
        self.inventory = contract["tensors"]
        for item in self.inventory:
            _install(self, item["name"], item["shape"])
        self.reset_parameters(seed)
        active = {x["name"] for x in self.inventory if "A2" in x["active_arms"]}
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in active)
        self.active_names = frozenset(active)

    def reset_parameters(self, seed: int) -> None:
        by_name = {x["name"]: x for x in self.inventory}
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                item = by_name[name]
                payload = b"planner-init-v1\0" + seed.to_bytes(8, "big") + b"\0" + name.encode()
                tensor_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
                    2**63 - 1
                )
                rng = np.random.Generator(np.random.PCG64(tensor_seed))
                kind = item["parameter_type"]
                if kind in {"linear_weight", "bilinear_weight"}:
                    fan_out, fan_in = item["shape"]
                    limit = math.sqrt(6.0 / (fan_in + fan_out))
                    value = rng.uniform(-limit, limit, item["shape"]).astype("<f4")
                elif kind in {"embedding_weight", "standalone_parameter"}:
                    value = rng.normal(0.0, 0.02, item["shape"]).astype("<f4")
                elif kind == "layer_norm_weight":
                    value = np.ones(item["shape"], dtype="<f4")
                else:
                    value = np.zeros(item["shape"], dtype="<f4")
                parameter.copy_(torch.from_numpy(value.copy()))
                if name in {
                    "task_encoder.token_embedding.weight",
                    "planner_decoder.a1_token_embedding.weight",
                }:
                    parameter[0].zero_()

    @staticmethod
    def _attention(x, memory, module, causal=False):
        q = F.linear(x, module.q_proj.weight)
        k = F.linear(memory, module.k_proj.weight)
        v = F.linear(memory, module.v_proj.weight)
        batch, q_len, _ = q.shape
        k_len = k.shape[1]
        q = q.view(batch, q_len, 8, 32).transpose(1, 2)
        k = k.view(batch, k_len, 8, 32).transpose(1, 2)
        v = v.view(batch, k_len, 8, 32).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(32)
        if causal:
            mask = torch.ones(q_len, k_len, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        values = torch.softmax(scores.float(), -1).to(v.dtype) @ v
        return F.linear(values.transpose(1, 2).reshape(batch, q_len, 256), module.out_proj.weight)

    @staticmethod
    def _ffn(x, module):
        return F.linear(
            F.gelu(F.linear(x, module.linear1.weight, module.linear1.bias)),
            module.linear2.weight,
            module.linear2.bias,
        )

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = F.embedding(token_ids, self.task_encoder.token_embedding.weight)
        x = x + self.task_encoder.position_embedding.weight[positions]
        # These normative embeddings describe segment/argument metadata. Toy canonical
        # encoding uses segment and argument-position zero, while retaining a real path.
        x = x + self.task_encoder.segment_embedding.weight[0]
        x = x + self.task_encoder.predicate_argument_position_embedding.weight[0]
        for layer in self.task_encoder.layers.children():
            y = F.layer_norm(x, (256,), layer.norm1.weight, layer.norm1.bias, 1e-5)
            x = x + self._attention(y, y, layer.self_attn)
            y = F.layer_norm(x, (256,), layer.norm2.weight, layer.norm2.bias, 1e-5)
            x = x + self._ffn(y, layer.ffn)
        return F.layer_norm(
            x, (256,), self.task_encoder.final_norm.weight, self.task_encoder.final_norm.bias, 1e-5
        )

    def forward(
        self, token_ids: torch.Tensor, actions: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor
    ) -> SimpleNamespace:
        memory = self.encode(token_ids)
        steps = actions.shape[1]
        pos = torch.arange(steps, device=actions.device)
        previous = torch.cat([torch.full_like(actions[:, :1], 4), actions[:, :-1]], 1)
        action_part = F.embedding(previous, self.concept_packer.previous_action_embedding.weight)
        action_part[:, 0] = self.concept_packer.bos_embedding
        refs = memory[:, : token_ids.shape[1]]
        prev1 = torch.cat([torch.zeros_like(arg1[:, :1]), arg1[:, :-1]], 1)
        prev2 = torch.cat([torch.zeros_like(arg2[:, :1]), arg2[:, :-1]], 1)
        r1 = refs.gather(1, prev1[..., None].expand(-1, -1, 256))
        r2 = refs.gather(1, prev2[..., None].expand(-1, -1, 256))
        r1[:, 0].zero_()
        r2[:, 0].zero_()
        x = action_part + self.concept_packer.step_position_embedding.weight[pos]
        x = x + F.linear(r1, self.concept_packer.previous_arg1_projection.weight)
        x = x + F.linear(r2, self.concept_packer.previous_arg2_projection.weight)
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
            x = x + self._attention(y, memory, layer.cross_attn)
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
        return SimpleNamespace(action=action_logits, arg1=arg1_logits, arg2=arg2_logits)


def canonical_token_ids(row: dict) -> torch.Tensor:
    """Stable 40-symbol canonical encoding with object refs in leading positions."""
    blocks = row["blocks"]
    ids = list(range(1, len(blocks) + 1))
    symbols = {"HAND_EMPTY": 8, "ON_TABLE": 9, "CLEAR": 10, "ON": 11, "HOLDING": 12, "GOAL": 13}
    for section in ("initial", "goal"):
        if section == "goal":
            ids.append(symbols["GOAL"])
        for fact in row[section]:
            ids.append(symbols[fact[0]])
            ids.extend(blocks.index(arg) + 1 for arg in fact[1:])
    return torch.tensor([ids], dtype=torch.long)
