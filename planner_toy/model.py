"""Contract-locked PyTorch A2 model used by the non-confirmatory toy slice."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

SEED = 17
TORCH_VERSION = "2.12.0"
SOURCE_INVENTORY = Path(__file__).parents[1] / "docs/architecture/planner_module_inventory_v1.yaml"


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
    inventory = SOURCE_INVENTORY
    if not inventory.is_file():
        inventory = Path(str(files("planner_toy").joinpath("planner_module_inventory_v1.yaml")))
    with inventory.open(encoding="utf-8") as handle:
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
    def _attention(x, memory, module, causal=False, key_mask=None):
        q = F.linear(x, module.q_proj.weight)
        k = F.linear(memory, module.k_proj.weight)
        v = F.linear(memory, module.v_proj.weight)
        batch, q_len, _ = q.shape
        k_len = k.shape[1]
        q = q.view(batch, q_len, 8, 32).transpose(1, 2)
        k = k.view(batch, k_len, 8, 32).transpose(1, 2)
        v = v.view(batch, k_len, 8, 32).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(32)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
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

    def encode(self, encoded: TaskEncoding) -> torch.Tensor:
        token_ids = encoded.token_ids
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = F.embedding(token_ids, self.task_encoder.token_embedding.weight)
        x = x + self.task_encoder.position_embedding.weight[positions]
        x = x + F.embedding(encoded.segment_ids, self.task_encoder.segment_embedding.weight)
        x = x + F.embedding(
            encoded.argument_position_ids,
            self.task_encoder.predicate_argument_position_embedding.weight,
        )
        for layer in self.task_encoder.layers.children():
            y = F.layer_norm(x, (256,), layer.norm1.weight, layer.norm1.bias, 1e-5)
            x = x + self._attention(y, y, layer.self_attn, key_mask=encoded.attention_mask)
            y = F.layer_norm(x, (256,), layer.norm2.weight, layer.norm2.bias, 1e-5)
            x = x + self._ffn(y, layer.ffn)
        return F.layer_norm(
            x, (256,), self.task_encoder.final_norm.weight, self.task_encoder.final_norm.bias, 1e-5
        )

    def forward(
        self, encoded: TaskEncoding, actions: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor
    ) -> SimpleNamespace:
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


@dataclass(frozen=True)
class TaskEncoding:
    token_ids: torch.Tensor
    segment_ids: torch.Tensor
    argument_position_ids: torch.Tensor
    attention_mask: torch.Tensor
    ref_slot_positions: tuple[int, ...]


TOKENS = {
    "PAD": 0,
    "BOS": 1,
    "EOS": 2,
    "DOMAIN_BLOCKS": 3,
    "LEDGER": 4,
    "STATE": 5,
    "GOAL": 6,
    "PRED_OPEN": 7,
    "PRED_CLOSE": 8,
    "TYPE_BLOCK": 9,
    "ON": 10,
    "ON_TABLE": 11,
    "CLEAR": 12,
    "HOLDING": 13,
    "HAND_EMPTY": 14,
}


def canonical_task_encoding(row: dict) -> TaskEncoding:
    """Exact task-encoding/1.6 sequence, metadata and ledger REF positions."""
    ids: list[int] = []
    segments: list[int] = []
    arguments: list[int] = []
    refs: list[int] = []

    def add(token: int, segment: int, argument: int = 0) -> None:
        ids.append(token)
        segments.append(segment)
        arguments.append(argument)

    add(TOKENS["BOS"], 0)
    add(TOKENS["DOMAIN_BLOCKS"], 0)
    add(TOKENS["LEDGER"], 1)
    for index, _ in enumerate(row["blocks"]):
        refs.append(len(ids))
        add(32 + index, 1)
        add(TOKENS["TYPE_BLOCK"], 1)
    for marker, section, segment in (("STATE", "initial", 2), ("GOAL", "goal", 3)):
        add(TOKENS[marker], segment)
        for fact in sorted(row[section], key=tuple):
            add(TOKENS["PRED_OPEN"], segment)
            add(TOKENS[fact[0]], segment, 1)
            for index, ref in enumerate(fact[1:]):
                add(32 + row["blocks"].index(ref), segment, 2 + index)
            add(TOKENS["PRED_CLOSE"], segment)
    add(TOKENS["EOS"], 0)
    length = len(ids)
    if length > 192:
        raise ValueError("task encoding exceeds locked length")
    padding = 192 - length
    ids += [0] * padding
    segments += [0] * padding
    arguments += [0] * padding
    return TaskEncoding(
        torch.tensor([ids]),
        torch.tensor([segments]),
        torch.tensor([arguments]),
        torch.tensor([[True] * length + [False] * padding]),
        tuple(refs),
    )
