import math
import pytest
import torch
import torch.nn.functional as F

from GrooveModel.Loss import UncertaintyWeightedMultiTaskLoss


def _seed():
    torch.manual_seed(0)


@pytest.fixture
def basic_tasks():
    # one of each type
    return {
        "semantic": {"type": "ce"},
        "depth_mse": {"type": "mse"},
        "depth_mae": {"type": "mae"},
    }


def test_init_and_params(basic_tasks):
    loss = UncertaintyWeightedMultiTaskLoss(basic_tasks)
    # parameters exist and require grad
    for name in basic_tasks:
        assert name in loss.log_vars
        assert loss.log_vars[name].requires_grad
    # repr is informative
    r = repr(loss)
    assert "semantic:ce" in r and "depth_mse:mse" in r and "depth_mae:mae" in r


def test_get_sigmas_matches_formula(basic_tasks):
    # pick non-zero s values to verify formula
    init = {"semantic": -0.7, "depth_mse": 0.4, "depth_mae": 1.5}
    loss = UncertaintyWeightedMultiTaskLoss(basic_tasks, init_log_vars=init)
    sigmas = loss.get_sigmas()
    for k, s in init.items():
        expected = math.sqrt(math.exp(s))
        assert pytest.approx(sigmas[k], rel=1e-6, abs=1e-6) == expected


def test_forward_shapes_and_scalar_total(basic_tasks):
    _seed()
    loss = UncertaintyWeightedMultiTaskLoss(basic_tasks, reduction="mean")

    # CE: logits (N, C, H, W) and targets (N, H, W)
    N, C, H, W = 2, 3, 4, 5
    logits = torch.randn(N, C, H, W, requires_grad=True)
    targets_ce = torch.randint(0, C, (N, H, W))

    # MSE / MAE: predictions & targets (N, D)
    D = 6
    pred_mse = torch.randn(N, D, requires_grad=True)
    targ_mse = torch.randn(N, D)
    pred_mae = torch.randn(N, D, requires_grad=True)
    targ_mae = torch.randn(N, D)

    outputs = {"semantic": logits, "depth_mse": pred_mse, "depth_mae": pred_mae}
    targets = {"semantic": targets_ce, "depth_mse": targ_mse, "depth_mae": targ_mae}

    total, details = loss(outputs, targets, diagnostics=False)
    assert isinstance(total, torch.Tensor)
    assert total.ndim == 1  # scalar
    assert details is None

    # backprop sanity
    total.backward()
    # some grads should exist
    assert logits.grad is not None
    assert pred_mse.grad is not None
    assert pred_mae.grad is not None


def test_weight_formulas_match_manual(basic_tasks):
    _seed()
    # Fix s values to check exact closed-form
    init = {"semantic": 0.0, "depth_mse": 0.3, "depth_mae": -0.8}
    loss = UncertaintyWeightedMultiTaskLoss(basic_tasks, reduction="mean", init_log_vars=init)

    # Small tensors for clarity
    N, C = 3, 4
    logits = torch.randn(N, C, requires_grad=True)
    targets_ce = torch.tensor([0, 1, 2])

    pred_mse = torch.tensor([[0.2, -1.0], [0.3, 0.1], [1.2, -0.7]], requires_grad=True)
    targ_mse = torch.tensor([[0.0, -1.0], [0.0, 0.0], [1.0, -1.0]])

    pred_mae = torch.tensor([[1.0, -2.0], [0.0, 3.0], [-1.0, 1.5]], requires_grad=True)
    targ_mae = torch.tensor([[0.0, -1.0], [0.5, 1.0], [-1.5, 1.0]])

    outputs = {"semantic": logits, "depth_mse": pred_mse, "depth_mae": pred_mae}
    targets = {"semantic": targets_ce, "depth_mse": targ_mse, "depth_mae": targ_mae}

    total, _ = loss(outputs, targets, diagnostics=False)

    # Manual pieces
    s_ce = torch.tensor([init["semantic"]])
    s_mse = torch.tensor([init["depth_mse"]])
    s_mae = torch.tensor([init["depth_mae"]])

    ce_base = F.cross_entropy(logits, targets_ce, reduction="mean")
    mse_base = F.mse_loss(pred_mse, targ_mse, reduction="mean")
    mae_base = F.l1_loss(pred_mae, targ_mae, reduction="mean")

    manual = (
        torch.exp(-s_ce) * ce_base + s_ce
        + 0.5 * torch.exp(-s_mse) * mse_base + 0.5 * s_mse
        + torch.exp(-0.5 * s_mae) * mae_base + 0.5 * s_mae
    )
    assert torch.allclose(total, manual.squeeze(), rtol=1e-6, atol=1e-6)


def test_reduction_sum(basic_tasks):
    _seed()
    loss = UncertaintyWeightedMultiTaskLoss(
        basic_tasks,
        reduction="sum",
        init_log_vars={"semantic": 0.0, "depth_mse": 0.0, "depth_mae": 0.0},
    )

    N, C, H, W = 2, 3, 2, 2
    logits = torch.randn(N, C, H, W, requires_grad=True)
    targets_ce = torch.randint(0, C, (N, H, W))
    pred_mse = torch.randn(N, 5, requires_grad=True)
    targ_mse = torch.randn(N, 5)
    pred_mae = torch.randn(N, 5, requires_grad=True)
    targ_mae = torch.randn(N, 5)

    outputs = {"semantic": logits, "depth_mse": pred_mse, "depth_mae": pred_mae}
    targets = {"semantic": targets_ce, "depth_mse": targ_mse, "depth_mae": targ_mae}

    total, _ = loss(outputs, targets, diagnostics=False)

    ce  = F.cross_entropy(logits, targets_ce, reduction="sum")
    mse = F.mse_loss(pred_mse, targ_mse, reduction="sum")
    mae = F.l1_loss(pred_mae, targ_mae, reduction="sum")

    # s = 0  => CE + 0.5 * MSE + 1.0 * MAE
    manual = ce + 0.5 * mse + 1.0 * mae

    assert torch.allclose(total.squeeze(), manual, rtol=1e-6, atol=1e-6)


def test_ce_kwargs_ignore_index_and_weight(basic_tasks):
    _seed()
    C = 4
    class_weights = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float32)
    ce_kwargs = {"semantic": {"ignore_index": 255, "weight": class_weights}}

    loss = UncertaintyWeightedMultiTaskLoss(basic_tasks, reduction="mean", init_log_vars={"semantic": 0.0}, ce_kwargs=ce_kwargs)

    # Make a tiny batch where one label is ignored
    logits = torch.randn(3, C, requires_grad=True)
    targets_ce = torch.tensor([0, 1, 255])  # last one ignored

    # dummy regression tasks (won't affect the assertion; just present)
    pred_mse = torch.zeros(3, 2, requires_grad=True)
    targ_mse = torch.zeros(3, 2)
    pred_mae = torch.zeros(3, 2, requires_grad=True)
    targ_mae = torch.zeros(3, 2)

    outputs = {"semantic": logits, "depth_mse": pred_mse, "depth_mae": pred_mae}
    targets = {"semantic": targets_ce, "depth_mse": targ_mse, "depth_mae": targ_mae}

    total, _ = loss(outputs, targets, diagnostics=False)

    # Manual CE part with the same kwargs
    ce_only = F.cross_entropy(logits, targets_ce, reduction="mean", ignore_index=255, weight=class_weights)

    # Since s=0 for CE and all regression outputs/targets equal, the other terms are 0.5*s (s=0) + 0 = 0
    # so total should equal CE + 0.5*MSE + 0.5*MAE = CE + 0 + 0
    assert torch.allclose(total, ce_only, rtol=1e-6, atol=1e-6)