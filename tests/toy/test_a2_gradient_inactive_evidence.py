from __future__ import annotations

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
    assert inactive.grad is None


def test_no_grad_marker_is_distinct_from_real_zero_gradient():
    no_grad = [("weight", None)]
    zero_grad = [("weight", torch.zeros(1, dtype=torch.float32))]

    assert producer._named_gradient_sha256(no_grad) != producer._named_gradient_sha256(zero_grad)
    assert producer._global_l2_norm(no_grad) == 0.0
    assert producer._global_l2_norm(
        [("inactive", None), ("active", torch.tensor([3.0, 4.0]))]
    ) == 5.0
