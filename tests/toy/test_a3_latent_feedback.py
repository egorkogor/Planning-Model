from __future__ import annotations

import torch

from planner_toy.dataset import generate
from planner_toy.model import LockedPlanner, canonical_task_encoding
from planner_toy.semantic import DIMENSION, target_for_step, targets
from planner_toy.training import labels, train


def _case():
    row = next(row for row in generate()["train"] if len(row["oracle_work_plan"]) > 1)
    return row, canonical_task_encoding(row), labels(row)


def test_latent_shape_normalization_determinism_and_hidden_sensitivity() -> None:
    row, encoded, values = _case()
    first = LockedPlanner(17, "A3").eval()
    second = LockedPlanner(17, "A3").eval()
    feedback = torch.cat([torch.zeros(1, 1, DIMENSION), targets(row)[:, :-1]], 1)
    with torch.no_grad():
        a = first(encoded, *values, semantic_feedback=feedback)
        b = second(encoded, *values, semantic_feedback=feedback)
        changed = first.latent(a.hidden + torch.linspace(0, 1, 256))
    assert a.z_semantic.shape == (1, 17, DIMENSION)
    assert torch.isfinite(a.z_semantic).all()
    assert torch.allclose(a.z_semantic.norm(dim=-1), torch.ones(1, 17), atol=1e-6)
    assert torch.equal(a.z_semantic, b.z_semantic)
    assert not torch.equal(a.z_semantic, changed)


def test_feedback_projection_and_causal_variant_sensitivity() -> None:
    _, encoded, values = _case()
    zero = torch.zeros(1, 17, DIMENSION)
    foreign = zero.clone()
    foreign[:, 1] = target_for_step(["STACK", "b0", "b1"])
    states = {}
    for variant in ("A2", "A3", "A4"):
        model = LockedPlanner(17, variant).eval()
        with torch.no_grad():
            baseline = model(encoded, *values, semantic_feedback=zero)
            changed = model(encoded, *values, semantic_feedback=foreign)
        states[variant] = (baseline, changed)
    a3, changed = states["A3"]
    assert a3.projected_semantic.shape == (1, 17, 256)
    assert torch.equal(a3.semantic_component[:, 0], torch.zeros(1, 256))
    assert torch.equal(a3.hidden[:, 0], changed.hidden[:, 0])  # future z cannot affect the past
    assert not torch.allclose(a3.hidden[:, 1], changed.hidden[:, 1])
    for variant in ("A2", "A4"):
        baseline, changed = states[variant]
        assert torch.equal(baseline.hidden, changed.hidden)
    assert states["A4"][0].projected_semantic is not None
    assert torch.isfinite(states["A4"][0].projected_semantic).all()
    assert torch.count_nonzero(states["A4"][0].semantic_component) == 0


def test_inventory_masks_and_a2_dormancy() -> None:
    models = {variant: LockedPlanner(17, variant) for variant in ("A2", "A3", "A4")}
    assert len({tuple(model.state_dict()) for model in models.values()}) == 1
    assert not models["A2"].heads.latent.weight.requires_grad
    assert models["A3"].heads.latent.weight.requires_grad
    assert models["A4"].semantic.latent_feedback.linear1.weight.requires_grad


def test_toy_targets_are_limited_normalized_and_deterministic() -> None:
    step = ["UNSTACK", "b0", "b1"]
    assert torch.equal(target_for_step(step), target_for_step(step))
    assert target_for_step(step).shape == (DIMENSION,)
    assert torch.allclose(target_for_step(step).norm(), torch.tensor(1.0), atol=1e-6)
    assert not torch.equal(target_for_step(step), target_for_step(["PUT_DOWN", "b0"]))


def test_a3_and_a4_training_gradient_policy(tmp_path) -> None:
    from planner_toy.canonical import canonical_bytes
    from planner_toy.e2e import _config, file_hash

    row = _case()[0]
    for variant in ("A3", "A4"):
        config = _config(row, generate(), variant=variant)
        config["training"]["steps"] = 1
        path = tmp_path / variant / "config.json"
        path.parent.mkdir()
        path.write_bytes(canonical_bytes(config) + b"\n")
        model, report = train(row, tmp_path / variant / "model", config=config,
                              config_hash=file_hash(path))
        assert report["component_losses"]["semantic"] >= 0
        assert report["latent_norm_mean"] == 1.0
        assert model.heads.latent.weight.grad is not None
        assert torch.isfinite(model.heads.latent.weight.grad).all()
        if variant == "A3":
            assert model.semantic.latent_feedback.linear1.weight.grad is not None
        else:
            assert model.semantic.latent_feedback.linear1.weight.grad is None
