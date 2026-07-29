"""Deterministic, CPU-only training and evidence for the toy A2 arm."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from .canonical import canonical_bytes
from .model import SEED, LockedA2, canonical_task_encoding

ACTIONS = {"PICK_UP": 0, "UNSTACK": 1, "PUT_DOWN": 2, "STACK": 3, "END": 4}


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def labels(row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    plan = row["oracle_work_plan"]
    action, arg1, arg2 = [], [], []
    for step in plan:
        action.append(ACTIONS[step[0]])
        arg1.append(row["blocks"].index(step[1]) if len(step) > 1 else 0)
        arg2.append(row["blocks"].index(step[2]) if len(step) > 2 else 0)
    padding = 17 - len(action)
    return (
        torch.tensor([action + [ACTIONS["END"]] * padding]),
        torch.tensor([arg1 + [0] * padding]),
        torch.tensor([arg2 + [0] * padding]),
    )


def train(row: dict, output: Path, *, config: dict, config_hash: str) -> tuple[LockedA2, dict]:
    """Run real AdamW updates and persist reproducible checkpoints/evidence."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    steps = config["training"]["steps"]
    model = LockedA2(SEED).cpu()
    output.mkdir(parents=True, exist_ok=True)
    initial = output / "initialization.pt"
    torch.save(model.state_dict(), initial)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    active = [parameter for parameter in model.parameters() if parameter.requires_grad]
    training = config["training"]
    optimizer = torch.optim.AdamW(
        active,
        lr=training["learning_rate"],
        betas=tuple(training["adamw_betas"]),
        eps=training["eps"],
        weight_decay=training["weight_decay"],
    )
    action, arg1, arg2 = labels(row)
    valid_steps = len(row["oracle_work_plan"])
    encoded = canonical_task_encoding(row)
    losses = []
    gradient_norm = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(encoded, action, arg1, arg2)
        flat_action = action[:, :valid_steps].flatten()
        loss = F.cross_entropy(logits.action[:, :valid_steps].flatten(0, 1), flat_action)
        has_arg1 = flat_action != ACTIONS["END"]
        has_arg2 = (flat_action == ACTIONS["UNSTACK"]) | (flat_action == ACTIONS["STACK"])
        if has_arg1.any():
            loss = loss + F.cross_entropy(
                logits.arg1[:, :valid_steps].flatten(0, 1)[has_arg1],
                arg1[:, :valid_steps].flatten()[has_arg1],
            )
        if has_arg2.any():
            loss = loss + F.cross_entropy(
                logits.arg2[:, :valid_steps].flatten(0, 1)[has_arg2],
                arg2[:, :valid_steps].flatten()[has_arg2],
            )
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(active, training["gradient_clip_norm"])
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    trained = output / "trained.pt"
    torch.save(model.state_dict(), trained)
    dormant = [name for name, p in model.named_parameters() if not p.requires_grad]
    active_changed = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and not torch.equal(before[name], p.detach())
    ]
    dormant_equal = all(torch.equal(before[name], model.state_dict()[name]) for name in dormant)
    dormant_grad_none = all(dict(model.named_parameters())[name].grad is None for name in dormant)
    named = dict(model.named_parameters())
    active_gradient_evidence = {
        name: {
            "finite": bool(torch.isfinite(parameter.grad).all()),
            "nonzero": bool(torch.any(parameter.grad != 0)),
        }
        for name, parameter in named.items()
        if parameter.requires_grad
    }
    optimizer_active_names = {
        name
        for name, parameter in named.items()
        if parameter.requires_grad and parameter in optimizer.state
    }
    optimizer_state_complete = optimizer_active_names == set(model.active_names)
    optimizer_state_finite_nonzero = optimizer_state_complete and all(
        torch.isfinite(optimizer.state[named[name]]["exp_avg"]).all()
        and bool(torch.any(optimizer.state[named[name]]["exp_avg"] != 0))
        and torch.isfinite(optimizer.state[named[name]]["exp_avg_sq"]).all()
        and bool(torch.any(optimizer.state[named[name]]["exp_avg_sq"] != 0))
        for name in model.active_names
    )
    optimizer_nonzero = any(
        state
        and any(
            torch.is_tensor(v) and v.numel() and bool(torch.any(v != 0)) for v in state.values()
        )
        for state in optimizer.state.values()
    )
    optimizer_evidence = {
        "schema_version": "toy-optimizer-evidence/1.0",
        "config_hash": config_hash,
        "active_parameter_names": sorted(model.active_names),
        "optimizer_state_parameter_names": sorted(optimizer_active_names),
        "active_gradient_evidence": active_gradient_evidence,
        "state_matches_active_set": optimizer_state_complete,
        "state_all_finite_nonzero": optimizer_state_finite_nonzero,
    }
    optimizer_path = output / "optimizer-evidence.json"
    optimizer_path.write_bytes(canonical_bytes(optimizer_evidence) + b"\n")
    report = {
        "schema_version": "toy-a2-training/1.0",
        "config_hash": config_hash,
        "dataset_hash": config["dataset_hash"],
        "training_task_id": config["training_task_id"],
        "training_task_hash": config["training_task_hash"],
        "inventory_sha256": config["inventory_sha256"],
        "task_encoding_sha256": config["task_encoding_sha256"],
        "runtime": config["runtime"],
        "code_commit": config["code_commit"],
        "seed": SEED,
        "torch_version": torch.__version__,
        "device": "cpu",
        "steps": steps,
        "tensor_count": len(model.state_dict()),
        "active_tensor_count": len(model.active_names),
        "dormant_tensor_count": len(dormant),
        "active_changed_count": len(active_changed),
        "active_grad_count": sum(p.grad is not None for p in active),
        "active_gradient_evidence": active_gradient_evidence,
        "active_gradients_all_finite_nonzero": all(
            row["finite"] and row["nonzero"] for row in active_gradient_evidence.values()
        ),
        "dormant_grad_none": dormant_grad_none,
        "dormant_byte_equal": dormant_equal,
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": [0.9, 0.95],
        "optimizer_nonzero_state": optimizer_nonzero,
        "optimizer_active_state_count": len(optimizer_active_names),
        "optimizer_state_matches_active_set": optimizer_state_complete,
        "optimizer_state_all_finite_nonzero": optimizer_state_finite_nonzero,
        "gradient_norm": gradient_norm,
        "losses": losses,
        # Content hashes deliberately exclude container serialization metadata.
        "initialization_sha256": state_dict_sha256(before),
        "trained_sha256": state_dict_sha256(model.state_dict()),
        "initialization_file_sha256": file_sha256(initial),
        "trained_file_sha256": file_sha256(trained),
        "optimizer_evidence_path": "model/optimizer-evidence.json",
        "optimizer_evidence_hash": file_sha256(optimizer_path),
    }
    (output / "training-report.json").write_bytes(canonical_bytes(report) + b"\n")
    return model, report
