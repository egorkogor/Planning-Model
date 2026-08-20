from __future__ import annotations

import hashlib

import torch

from planner_toy import a2_gradient_clipping as producer


def test_gradient_items_preserve_no_grad_without_materializing_zero():
    active = torch.nn.Parameter(torch.tensor([1.0]))
    inactive = torch.nn.Parameter(torch.tensor([2.0]))
    active.grad = torch.tensor([3.0])
    assert inactive.grad is None

    items = producer._gradient_items([("active", active), ("inactive", inactive)])

    assert [name for name, _ in items] == ["active", "inactive"]
    assert items[0][1] is not None
    assert items[1][1] is None
    assert producer._gradient_activity(items) == [
        {"index": 0, "name": "active", "state": "GRAD"},
        {"index": 1, "name": "inactive", "state": "NO_GRAD"},
    ]
    assert inactive.grad is None


def test_no_grad_marker_is_distinct_from_real_zero_gradient():
    no_grad = [("weight", None)]
    zero_grad = [("weight", torch.zeros(1, dtype=torch.float32))]

    assert producer._named_gradient_sha256(no_grad) != producer._named_gradient_sha256(zero_grad)
    assert producer._global_l2_norm(no_grad) == 0.0
    assert producer._global_l2_norm(
        [("inactive", None), ("active", torch.tensor([3.0, 4.0]))]
    ) == 5.0


def test_gradient_hash_v1_1_exact_wire_framing():
    items = [
        ("inactive", None),
        ("active", torch.tensor([[1.0, -2.0]], dtype=torch.float32)),
    ]
    digest = hashlib.sha256()
    digest.update(b"a2-named-gradients-exact/1.1\0")

    name = b"inactive"
    digest.update(len(name).to_bytes(8, "big") + name)
    marker = b"NO_GRAD"
    digest.update(len(marker).to_bytes(8, "big") + marker)

    name = b"active"
    digest.update(len(name).to_bytes(8, "big") + name)
    marker = b"GRAD"
    digest.update(len(marker).to_bytes(8, "big") + marker)
    tensor = items[1][1].detach().cpu().contiguous()
    dtype = str(tensor.dtype).encode("ascii")
    digest.update(len(dtype).to_bytes(8, "big") + dtype)
    digest.update(tensor.ndim.to_bytes(8, "big"))
    for dimension in tensor.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    data = tensor.numpy().tobytes()
    digest.update(len(data).to_bytes(8, "big") + data)

    assert producer.GRADIENT_HASH_VERSION == "a2-named-gradients-exact/1.1"
    assert producer._named_gradient_sha256(items) == "sha256:" + digest.hexdigest()


def test_gradient_hash_v1_0_is_not_current_alias():
    assert producer.GRADIENT_HASH_VERSION != "a2-named-gradients-exact/1.0"
    assert producer.CLIPPING_CONTRACT["gradient_hash_version"] == producer.GRADIENT_HASH_VERSION
    assert producer.CLIPPING_CONTRACT["gradient_evidence_commitment_version"] == (
        "a2-gradient-evidence-commitment/1.1"
    )
