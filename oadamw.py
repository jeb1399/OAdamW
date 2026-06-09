# OAdamW: StableAdamW + Muon (Newton-Schulz orthogonalisation)
#
# StableAdamW — MIT License — Copyright (c) 2023-present Benjamin Warner
# Muon / Newton-Schulz iteration — MIT License — Copyright (c) 2024 Kosson et al.
#   NS quintic coefficients from the Modular-Muon codebase
#
# Kahan summation inspired by torchdistX `AnyPrecisionAdamW`
#   BSD 3-Clause — Copyright (c) Meta Platforms, Inc. and affiliates
#
# Triton kernels inspired by:
#   AdamW-Triton-PyTorch — MIT — Copyright (c) 2024 Less Wright
#   lion-pytorch — MIT — Copyright (c) 2023 Phil Wang
#
# Design:
#   Parameter groups marked with `"muon": True` apply Newton-Schulz orthogonalisation
#   to each gradient before it enters the Adam update.  The orthogonalised gradient is
#   the same shape as the original (no rank reduction), so moment tensors stay
#   full-rank and Kahan summation still applies for float16/bfloat16 parameters.
#
#   The critical insight vs the previous GaLore design:
#   - GaLore needed a `projector=` argument threaded through _single_param_oadamw
#     so that project_back() could be called inside the function, complicating every
#     code path (two Kahan branches, two standard branches).
#   - With Muon, orthogonalisation is a pure pre-processing step.  _step_muon_param
#     calls NS → then delegates to the SAME _single_param_oadamw as every other
#     parameter, with no extra arguments.  The function is back to clean StableAdamW.
#
#   foreach / Triton are genuinely incompatible with per-parameter NS pre-processing
#   (NS reads a tensor, foreach kernels require all tensors batched by dtype/device).
#   Muon groups therefore always use the single-param path.
#
#   Other choices preserved from StableAdamW:
#   - Debiased betas: mathematically equivalent to standard bias correction but
#     computed in the update rule without a separate division step.
#   - beta2 default 0.99: correct default when RMS stabilisation is active.
#   - Decoupled / fully-decoupled weight decay: strictly more flexible.

from collections.abc import Callable, Iterable
from typing import Any
from dataclasses import asdict

import torch
from torch import Tensor
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

from optimi.optimizer import OptimiOptimizer
from optimi.utils import (
    HAS_TRITON,
    TORCH_TO_TRITON_DTYPE,
    _default_to_triton,
    _device_guard,
    _get_triton_block_size,
    debias_beta,
)

__all__ = ["OAdamW", "oadamw", "MuonProjector"]


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalisation (Muon)
# ---------------------------------------------------------------------------

class MuonProjector:
    """Orthogonalises gradient updates via Newton-Schulz iteration.

    NS iteration drives a matrix toward its polar factor — the closest matrix
    whose singular values are all 1 — using only matrix multiplications.  This
    replaces SVD-based low-rank projection (GaLore) with a cheaper operation that
    preserves the full dimensionality of the gradient.

    Differences from GaLoreProjector:
    - No rank parameter: the output has the same shape as the input gradient.
      Moment tensors (exp_avg, exp_avg_sq) remain full-rank.
    - No SVD: each NS step costs O(m² · n) where m = min(rows, cols), versus
      O(m · n · min(m, n)) for a full SVD.
    - Stable with the optimised quintic coefficients (a, b, c): converges in
      5 iterations for singular values in [0, 1] after spectral normalisation.

    Args:
        ns_steps: Newton-Schulz iterations per call (default: 5)
        update_proj_gap: Steps between NS recomputations; 1 = every step (default: 1)
        scale: Multiplicative factor applied on project_back (default: 1.0)
        verbose: Print debug info (default: False)
    """

    def __init__(
        self,
        ns_steps: int = 5,
        update_proj_gap: int = 1,
        scale: float = 1.0,
        verbose: bool = False,
    ):
        self.ns_steps = ns_steps
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.verbose = verbose
        self._cached: Tensor | None = None

    def project(self, grad: Tensor, iter: int) -> Tensor:
        """Return the Newton-Schulz orthogonalisation of grad.

        The result is cached and reused for ``update_proj_gap`` steps before the
        next NS pass (default: recompute every step).
        """
        if self._cached is None or iter % self.update_proj_gap == 0:
            self._cached = self._newton_schulz(grad)
        return self._cached

    def project_back(self, grad: Tensor) -> Tensor:
        """Scale-only pass-through — NS does not reduce dimensions."""
        return grad * self.scale

    def _newton_schulz(self, G: Tensor) -> Tensor:
        """Newton-Schulz orthogonalisation with optimised quintic coefficients.

        Iterates X ← a·X + (b·A + c·A²)·X  where A = X·Xᵀ and
        (a, b, c) = (3.4445, -4.7750, 2.0315) are chosen so that the degree-5
        polynomial maps singular values in [0, 1] to ≈ 1 after 5 steps.

        Always operates on the shorter-axis orientation so that the Gram matrix
        A is at most (min_dim × min_dim), keeping memory and FLOPs small.
        """
        a, b, c = 3.4445, -4.7750, 2.0315
        orig_dtype = G.dtype

        # Operate on the wider dimension so A = X Xᵀ is as small as possible
        transposed = G.shape[0] > G.shape[1]
        if transposed:
            G = G.T

        # float32 throughout for numerical stability; spectral normalisation
        X = G.float()
        X = X / (X.norm() + 1e-7)

        for _ in range(self.ns_steps):
            A = X @ X.T
            X = a * X + (b * A + c * A @ A) @ X

        if transposed:
            X = X.T

        return X.to(orig_dtype)


# ---------------------------------------------------------------------------
# Triton state restoration after load_state_dict
# ---------------------------------------------------------------------------

def _restore_triton_scratch_state(optim: OptimiOptimizer):
    "Restore or create scratch to fp32 after potentially cast to low precision by load_state_dict."
    for group in optim.param_groups:
        if group.get("triton") and "muon" not in group:
            for p in group["params"]:
                state = optim.state[p]
                if "mean_square" in state:
                    state["mean_square"] = state["mean_square"].to(dtype=torch.float32, device=p.device)
                else:
                    state["mean_square"] = torch.zeros(1, dtype=torch.float32, device=p.device)


# ---------------------------------------------------------------------------
# Model-aware parameter grouping
# ---------------------------------------------------------------------------

def _auto_group_model(model: torch.nn.Module) -> list[dict]:
    """Partition a model's trainable parameters into Muon and standard groups.

    2-D+ tensors (weight matrices of linear layers, embeddings, etc.) go into the
    Muon group so Newton-Schulz orthogonalisation is applied to them.  1-D tensors
    (biases, layer-norm scales/shifts) go into the standard StableAdamW group.

    This is the grouping that runs automatically when an ``nn.Module`` is passed to
    ``OAdamW``.  Pass explicit parameter-group dicts if you need a different split.
    """
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    adam_params  = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups: list[dict] = []
    if muon_params:
        groups.append({"params": muon_params, "muon": True})
    if adam_params:
        groups.append({"params": adam_params})
    return groups


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------

class OAdamW(OptimiOptimizer):
    """OAdamW optimizer: StableAdamW with optional Muon gradient orthogonalisation.

    Pass an ``nn.Module`` to get automatic parameter grouping: 2-D+ weight matrices
    receive Newton-Schulz orthogonalisation (Muon) and 1-D parameters (biases, norms)
    receive standard StableAdamW.  Pass an iterable of tensors or explicit parameter-
    group dicts for full manual control.

    Standard parameter groups use the full StableAdamW stack: debiased betas, per-tensor
    RMS learning-rate stabilisation, Kahan summation for low-precision training, and
    foreach / Triton implementations for throughput.

    Muon parameter groups (``"muon": True`` in the group dict, or auto-assigned when
    a model is passed) additionally apply Newton-Schulz orthogonalisation to each
    gradient before computing Adam moments.  The orthogonalised gradient has the same
    shape as the original so moment tensors stay full-rank and Kahan summation still
    fires for float16/bfloat16 parameters.

    Usage::

        # simplest — auto-groups the model
        optimizer = OAdamW(model, lr=5e-6)

        # explicit groups for custom splits
        optimizer = OAdamW([
            {"params": proj_weights, "muon": True},
            {"params": other_params},
        ], lr=5e-6)

    Args:
        params: ``nn.Module`` (auto-grouped), iterable of tensors, or list of group dicts
        lr: Learning rate
        betas: Coefficients for gradient and squared-gradient moving averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient (default: 1e-2)
        eps: Added to denominator to improve numerical stability (default: 1e-6)
        decouple_lr: Apply fully decoupled weight decay instead of decoupled weight decay
            (default: False)
        max_lr: Maximum scheduled learning rate. Required when ``decouple_lr`` is True
            (default: None)
        kahan_sum: Enables Kahan summation for low-precision parameters.
            Auto-enables for float16/bfloat16 when unspecified (default: None)
        foreach: Enables the foreach implementation for non-Muon parameters. Auto-selects
            when unspecified (default: None)
        triton: Enables Triton kernels for non-Muon parameters. Auto-selects when
            unspecified (default: None)
        gradient_release: Fuses optimizer step with backward pass (default: False)

    Muon group-level keys (add to parameter group dict):
        muon (bool): Presence and truth of this key enables Muon for the group
        ns_steps (int): Newton-Schulz iterations per gradient (default: 5)
        update_proj_gap (int): Steps between NS recomputations; 1 = every step (default: 1)
        scale (float): Scale factor applied after NS (default: 1.0)
    """

    def __init__(
        self,
        params: torch.nn.Module | Iterable[Tensor] | Iterable[dict],
        training_args: None = None,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 1e-2,
        eps: float = 1e-6,
        decouple_lr: bool = False,
        max_lr: float | None = None,
        kahan_sum: bool | None = None,
        foreach: bool | None = None,
        triton: bool | None = None,
        gradient_release: bool = False,
    ):
        if isinstance(params, torch.nn.Module):
            params = _auto_group_model(params)
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]=}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]=}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps=}")

        if training_args is not None:
            if training_args.learning_rate is not None:
                lr = training_args.learning_rate
            if training_args.weight_decay is not None:
                weight_decay = training_args.weight_decay

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            decouple_lr=decouple_lr,
            max_lr=max_lr,
            kahan_sum=kahan_sum,
            foreach=foreach,
            triton=triton,
            gradient_release=gradient_release,
            setup=False,
        )
        print(defaults)
        super().__init__(params, defaults)
        self.register_load_state_dict_post_hook(_restore_triton_scratch_state)

    # ------------------------------------------------------------------
    # State initialisation helpers
    # ------------------------------------------------------------------

    def _init_state(self, group: dict[str, Any], state: dict[Tensor, Any], param: Tensor):
        """Initialise state for a non-Muon parameter."""
        if "kahan_comp" not in state:
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["eps_sq"] = torch.tensor(group["eps"] ** 2, dtype=param.dtype, device=param.device)

            if (group["kahan_sum"] or group["kahan_sum"] is None) and param.dtype in [torch.float16, torch.bfloat16]:
                state["kahan_comp"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                group["kahan_sum"] = True
            elif group["triton"]:
                state["kahan_comp"] = torch.zeros(1, dtype=torch.uint8, device=param.device)
            else:
                state["kahan_comp"] = None

            if group["triton"]:
                state["mean_square"] = torch.zeros(1, dtype=torch.float32, device=param.device)

            if group["gradient_release"]:
                state["step"] = torch.tensor(0, dtype=torch.int32)

    def _init_muon_state(self, group: dict[str, Any], state: dict[Tensor, Any], param: Tensor):
        """Initialise full-rank state for a Muon parameter.

        Identical to _init_state minus Triton scratch and gradient-release step
        counter — neither is applicable to the Muon single-param path.
        """
        if "kahan_comp" not in state:
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["eps_sq"] = torch.tensor(group["eps"] ** 2, dtype=param.dtype, device=param.device)
            if (group["kahan_sum"] or group["kahan_sum"] is None) and param.dtype in [torch.float16, torch.bfloat16]:
                state["kahan_comp"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                group["kahan_sum"] = True
            else:
                state["kahan_comp"] = None

    def _init_group(
        self,
        group: dict[str, Any],
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        eps_sqs: list[Tensor],
        kahan_comps: list[Tensor],
        mean_squares: list[Tensor],
    ):
        """Batch-initialise state for all non-Muon parameters in a group."""
        if not group["setup"]:
            group["setup"] = True
            group["step"] = torch.tensor(0, dtype=torch.int32)

            if group["triton"] is None and group["foreach"] is None:
                group["triton"] = _default_to_triton(params)

        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)
            state = self.state[p]

            self._init_state(group, state, p)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            eps_sqs.append(state["eps_sq"])
            kahan_comps.append(state["kahan_comp"])

            if group["triton"]:
                mean_squares.append(state["mean_square"])

    # ------------------------------------------------------------------
    # Muon single-parameter step
    # ------------------------------------------------------------------

    def _step_muon_param(self, group: dict[str, Any], param: Tensor):
        """Step for a single Muon parameter.

        Applies Newton-Schulz orthogonalisation to the gradient (for 2-D parameters),
        then delegates to the same _single_param_oadamw core used by every
        non-Muon parameter.  No extra arguments are needed in the core function —
        NS is a pure pre-processing step, not a bookend around the update.

        1-D parameters (biases, layer norms) in a Muon group bypass NS and receive
        the raw gradient so that the optimizer behaves sensibly for all shapes.
        """
        if param.grad is None:
            return

        state = self.state[param]

        if "step" not in state:
            state["step"] = torch.tensor(0, dtype=torch.int32)
        if "projector" not in state:
            state["projector"] = MuonProjector(
                ns_steps=group.get("ns_steps", 5),
                update_proj_gap=group.get("update_proj_gap", 1),
                scale=group.get("scale", 1.0),
            )

        state["step"].add_(1)
        step_int = state["step"].item()

        # NS orthogonalisation only makes sense for matrices; skip for 1-D params
        grad = state["projector"].project(param.grad, step_int) if param.ndim >= 2 else param.grad

        self._init_muon_state(group, state, param)

        beta1_hat = debias_beta(group["beta1"], step_int)
        beta2_hat = debias_beta(group["beta2"], step_int)

        # Delegate to the exact same core as every non-Muon parameter.
        # The only difference is `grad` above: for 2-D params it is NS(raw_grad).
        _single_param_oadamw(
            param=param,
            grad=grad,
            exp_avg=state["exp_avg"],
            exp_avg_sq=state["exp_avg_sq"],
            eps_sq=state["eps_sq"],
            kahan_comp=state["kahan_comp"],
            lr=group["lr"],
            beta1_comp=1 - beta1_hat,
            beta2_hat=beta2_hat,
            beta2_comp=1 - beta2_hat,
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            decouple_lr=group["decouple_lr"],
            max_lr=group["max_lr"],
            kahan_sum=group["kahan_sum"] or False,
            update_parameters=True,
        )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable | None = None, param: Tensor | None = None):
        """Performs a single optimization step on the whole model or individual parameter.

        Args:
            closure: A closure which reevaluates the model and returns the loss.
                Incompatible with performing an optimization step on a single ``param``.
            param: An individual parameter to perform a fused optimization step during
                the backward pass. Requires ``gradient_release=True`` and model hooks
                created with ``register_gradient_release``. Incompatible with ``closure``.
                Not supported for Muon parameter groups.
        """
        loss = None
        if closure is not None and param is None:
            with torch.enable_grad():
                loss = closure()

        if param is None:
            for group in self.param_groups:
                if group.get("muon"):
                    # Muon groups: NS pre-processing + single-param Adam (no foreach/Triton)
                    for p in group["params"]:
                        self._step_muon_param(group, p)
                    continue

                # Non-Muon groups: full StableAdamW batch path
                params, grads, exp_avgs, exp_avg_sqs, eps_sqs, kahan_comps, mean_squares = (
                    [], [], [], [], [], [], []
                )
                self._init_group(
                    group=group,
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    eps_sqs=eps_sqs,
                    kahan_comps=kahan_comps,
                    mean_squares=mean_squares,
                )
                oadamw(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    eps_sqs=eps_sqs,
                    kahan_comps=kahan_comps,
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    step=group["step"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
                    kahan_sum=group["kahan_sum"],
                    foreach=group["foreach"],
                    triton=group["triton"],
                    gradient_release=False,
                    optimizer_accumulation=False,
                    mean_squares=mean_squares,
                )
        else:
            # Gradient-release mode (non-Muon parameters only)
            state = self.state[param]
            group = state["group"]
            self._init_state(group, state, param)

            if group["triton"]:
                oadamw(
                    params=param,
                    grads=param.grad,
                    exp_avgs=state["exp_avg"],
                    exp_avg_sqs=state["exp_avg_sq"],
                    eps_sqs=state["eps_sq"],
                    kahan_comps=state["kahan_comp"],
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    step=state["step"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
                    kahan_sum=group["kahan_sum"],
                    foreach=False,
                    triton=True,
                    gradient_release=True,
                    optimizer_accumulation=self._optimizer_accumulation,
                    mean_squares=state["mean_square"],
                )
            else:
                oadamw(
                    params=param,
                    grads=param.grad,
                    exp_avgs=state["exp_avg"],
                    exp_avg_sqs=state["exp_avg_sq"],
                    eps_sqs=state["eps_sq"],
                    kahan_comps=state["kahan_comp"],
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    step=state["step"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
                    kahan_sum=group["kahan_sum"],
                    foreach=False,
                    triton=False,
                    gradient_release=True,
                    optimizer_accumulation=self._optimizer_accumulation,
                )

        return loss


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def oadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    eps_sqs: list[Tensor],
    kahan_comps: list[Tensor | None] | None = None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: Tensor,
    decouple_lr: bool = False,
    max_lr: float | None = None,
    kahan_sum: bool = False,
    foreach: bool = False,
    triton: bool = False,
    gradient_release: bool = False,
    optimizer_accumulation: bool = False,
    mean_squares: list[Tensor] | None = None,
):
    """Functional API to apply a OAdamW optimisation step (non-Muon path).

    See ``OAdamW`` for full documentation.  For Muon parameter groups, the
    step is managed internally by ``OAdamW._step_muon_param``.

    Args:
        params: Parameters to update
        grads: Parameter gradients
        exp_avgs: Gradient moving averages
        exp_avg_sqs: Squared gradient moving averages
        eps_sqs: Squared epsilon term tensors
        kahan_comps: Kahan summation compensations
        lr: Learning rate
        beta1: Gradient moving average coefficient
        beta2: Squared gradient moving average coefficient
        weight_decay: Weight decay coefficient
        eps: Added to denominator to improve numerical stability
        step: Step counter used for bias correction
        decouple_lr: Apply fully decoupled weight decay
        max_lr: Maximum scheduled learning rate for ``decouple_lr``
        kahan_sum: Enables Kahan summation for low precision parameters
        foreach: Enables the faster foreach implementation
        triton: Enables Triton support for the optimizer
        gradient_release: Fuses optimizer step as part of the parameter's backward pass
        optimizer_accumulation: Accumulate gradients into state during gradient release step
        mean_squares: RMS calculation scratch tensor for Triton kernel
    """
    step.add_(1)
    step_int = step.item()
    beta1_hat = debias_beta(beta1, step_int)
    beta1_comp = 1 - beta1_hat
    beta2_hat = debias_beta(beta2, step_int)
    beta2_comp = 1 - beta2_hat

    if kahan_comps is None:
        kahan_comps = [None] * len(params)

    if gradient_release:
        if triton:
            func = _single_param_triton_oadamw
        elif foreach:
            raise ValueError(f"Gradient release {gradient_release=} and foreach {foreach=} cannot be used together")
        else:
            func = _single_param_oadamw
    else:
        if triton:
            func = _triton_oadamw
        elif foreach:
            func = _foreach_oadamw
        else:
            func = _single_oadamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        eps_sqs,
        kahan_comps,
        lr=lr,
        beta1_hat=beta1_hat,
        beta1_comp=beta1_comp,
        beta2_hat=beta2_hat,
        beta2_comp=beta2_comp,
        weight_decay=weight_decay,
        eps=eps,
        decouple_lr=decouple_lr,
        max_lr=max_lr,
        kahan_sum=kahan_sum,
        update_parameters=(not optimizer_accumulation),
        mean_squares=mean_squares,
    )


# ---------------------------------------------------------------------------
# Low-level implementations (non-Muon)
# ---------------------------------------------------------------------------

def _single_oadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    eps_sqs: list[Tensor],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    decouple_lr: bool,
    max_lr: float | None,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    for i, param in enumerate(params):
        _single_param_oadamw(
            param=param,
            grad=grads[i],
            exp_avg=exp_avgs[i],
            exp_avg_sq=exp_avg_sqs[i],
            eps_sq=eps_sqs[i],
            kahan_comp=kahan_comps[i],
            lr=lr,
            beta1_comp=beta1_comp,
            beta2_hat=beta2_hat,
            beta2_comp=beta2_comp,
            weight_decay=weight_decay,
            eps=eps,
            decouple_lr=decouple_lr,
            max_lr=max_lr,
            kahan_sum=kahan_sum,
            update_parameters=update_parameters,
        )


def _single_param_oadamw(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    eps_sq: Tensor,
    kahan_comp: Tensor | None,
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    decouple_lr: bool,
    max_lr: float | None,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    """Single-parameter StableAdamW update core.

    Shared by both the non-Muon batch path and the Muon per-parameter path.
    For Muon parameters, ``grad`` is the NS-orthogonalised gradient; for all
    other parameters it is the raw gradient.  No other difference exists between
    the two callers.
    """
    # Update gradient moving averages with debiased betas
    exp_avg.lerp_(grad, weight=beta1_comp)
    exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=beta2_comp)

    if update_parameters:
        # Per-tensor RMS stabilisation
        rms = grad.pow(2).div_(exp_avg_sq.maximum(eps_sq)).mean().sqrt()
        lr = lr / max(1, rms.item())

        if weight_decay != 0:
            if decouple_lr:
                weight_decay = 1 - (lr / max_lr) * weight_decay
            else:
                weight_decay = 1 - lr * weight_decay
            param.mul_(weight_decay)

        if kahan_sum and param.dtype in [torch.float16, torch.bfloat16]:
            # Reuse grad as a temp buffer for old-param snapshot
            kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr)
            grad.copy_(param.detach())
            param.add_(kahan_comp)
            kahan_comp.add_(grad.sub_(param))
        else:
            param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr)


def _foreach_oadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    eps_sqs: list[Tensor],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    decouple_lr: bool,
    max_lr: float | None,
    kahan_sum: bool = False,
    **kwargs,
):
    grouped_tensors = _group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, eps_sqs, kahan_comps]
    )
    for (_, dtype), (
        (dev_params, dev_grads, dev_exp_avgs, dev_exp_avg_sqs, dev_eps_sqs, dev_kahan_comps),
        _,
    ) in grouped_tensors.items():
        do_kahan_sum = kahan_sum and dtype in [torch.float16, torch.bfloat16]

        torch._foreach_lerp_(dev_exp_avgs, dev_grads, weight=beta1_comp)
        torch._foreach_mul_(dev_exp_avg_sqs, scalar=beta2_hat)
        torch._foreach_addcmul_(dev_exp_avg_sqs, dev_grads, dev_grads, value=beta2_comp)

        # Compute per-parameter RMS stabilisation terms, reusing dev_grads as a buffer
        max_exp_avg_sqs = torch._foreach_maximum(dev_exp_avg_sqs, other=dev_eps_sqs)
        torch._foreach_pow_(dev_grads, exponent=2)
        torch._foreach_div_(dev_grads, max_exp_avg_sqs)
        del max_exp_avg_sqs

        if weight_decay != 0:
            neg_lrs, new_wds = [], []
            for r in dev_grads:
                neg_lrs.append(-lr / max(1, r.mean().sqrt().item()))
                if decouple_lr:
                    new_wds.append(1 + (neg_lrs[-1] / max_lr) * weight_decay)
                else:
                    new_wds.append(1 + neg_lrs[-1] * weight_decay)
            torch._foreach_mul_(dev_params, scalars=new_wds)
        else:
            neg_lrs = [-lr / max(1, r.mean().sqrt().item()) for r in dev_grads]

        # Adam denominator (reuse dev_grads as buffer)
        torch._foreach_copy_(dev_grads, dev_exp_avg_sqs)
        torch._foreach_sqrt_(dev_grads)
        torch._foreach_add_(dev_grads, eps)

        if do_kahan_sum:
            torch._foreach_addcdiv_(dev_kahan_comps, dev_exp_avgs, dev_grads, scalars=neg_lrs)
            torch._foreach_copy_(dev_grads, dev_params)
            torch._foreach_add_(dev_params, dev_kahan_comps, alpha=1)
            torch._foreach_sub_(dev_grads, dev_params, alpha=1)
            torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)
        else:
            torch._foreach_addcdiv_(dev_params, dev_exp_avgs, dev_grads, scalars=neg_lrs)


# ---------------------------------------------------------------------------
# Triton kernels (non-Muon path, only compiled when Triton is available)
# ---------------------------------------------------------------------------

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _oadamw_exp_avg_kernel(
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        mean_square_ptr,
        eps,
        beta1_hat,
        beta1_comp,
        beta2_hat,
        beta2_comp,
        n_elements,
        update_parameters: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        param_dtype: tl.constexpr = tl.float32,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # All intermediate computation in float32 regardless of parameter dtype
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)

        exp_avg = tl.fma(exp_avg, beta1_hat, beta1_comp * grad)
        exp_avg_sq = tl.fma(exp_avg_sq, beta2_hat, beta2_comp * grad * grad)

        if update_parameters:
            square = tl.where(mask, (grad * grad) / tl.maximum(exp_avg_sq, eps * eps), 0.0)
            block_sum = tl.sum(square, axis=0, dtype=tl.float32) / n_elements
            tl.atomic_add(mean_square_ptr, block_sum)

        tl.store(exp_avg_ptr + offsets, tl.cast(exp_avg, param_dtype), mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, tl.cast(exp_avg_sq, param_dtype), mask=mask)

    @triton.jit
    def _oadamw_update_kernel(
        param_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        kahan_ptr,
        mean_square_ptr,
        lr,
        weight_decay,
        eps,
        max_lr,
        do_weight_decay: tl.constexpr,
        kahan_sum: tl.constexpr,
        decouple_lr: tl.constexpr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        param = tl.load(param_ptr + offsets, mask=mask)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)

        mean_square = tl.load(mean_square_ptr)
        lr = lr / tl.maximum(1.0, tl.sqrt(mean_square))

        if do_weight_decay:
            if decouple_lr:
                weight_decay = 1.0 - (lr / max_lr) * weight_decay
            else:
                weight_decay = 1.0 - lr * weight_decay
            param = tl.cast(param * weight_decay, param.dtype)

        if kahan_sum:
            kahan_comp = tl.load(kahan_ptr + offsets, mask=mask).to(tl.float32)
            kahan_comp = kahan_comp - (lr * exp_avg / (tl.sqrt(exp_avg_sq) + eps))
            prev_param = param
            param = param + tl.cast(kahan_comp, param.dtype)
            kahan_comp = kahan_comp + prev_param.to(tl.float32) - param.to(tl.float32)
            tl.store(kahan_ptr + offsets, tl.cast(kahan_comp, param.dtype), mask=mask)
        else:
            param = param + tl.cast((-lr * exp_avg / (tl.sqrt(exp_avg_sq) + eps)), param.dtype)

        tl.store(param_ptr + offsets, param, mask=mask)

    def _triton_oadamw(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        eps_sqs: list[Tensor],
        kahan_comps: list[Tensor | None],
        *,
        mean_squares: list[Tensor],
        lr: float,
        beta1_hat: float,
        beta1_comp: float,
        beta2_hat: float,
        beta2_comp: float,
        weight_decay: float,
        eps: float,
        decouple_lr: bool,
        max_lr: float | None = None,
        kahan_sum: bool = False,
        **kwargs,
    ):
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            kahan_comp = kahan_comps[i]
            mean_square = mean_squares[i]

            n_elements = param.numel()
            block_size = _get_triton_block_size(n_elements)
            grid = (triton.cdiv(n_elements, block_size),)

            with _device_guard(param):
                _oadamw_exp_avg_kernel[grid](
                    grad_ptr=grad,
                    exp_avg_ptr=exp_avg,
                    exp_avg_sq_ptr=exp_avg_sq,
                    mean_square_ptr=mean_square,
                    eps=eps,
                    beta1_hat=beta1_hat,
                    beta1_comp=beta1_comp,
                    beta2_hat=beta2_hat,
                    beta2_comp=beta2_comp,
                    update_parameters=True,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                    param_dtype=TORCH_TO_TRITON_DTYPE[param.dtype],
                )
                _oadamw_update_kernel[grid](
                    param_ptr=param,
                    exp_avg_ptr=exp_avg,
                    exp_avg_sq_ptr=exp_avg_sq,
                    kahan_ptr=kahan_comp,
                    mean_square_ptr=mean_square,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    max_lr=max_lr,
                    do_weight_decay=weight_decay != 0.0,
                    kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                    decouple_lr=decouple_lr,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                )
                mean_square.zero_()

    def _single_param_triton_oadamw(
        param: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        eps_sq: Tensor,
        kahan_comp: Tensor | None,
        *,
        mean_squares: Tensor,
        lr: float,
        beta1_hat: float,
        beta1_comp: float,
        beta2_hat: float,
        beta2_comp: float,
        weight_decay: float,
        eps: float,
        decouple_lr: bool,
        max_lr: float | None = None,
        kahan_sum: bool = False,
        update_parameters: bool = True,
        **kwargs,
    ):
        n_elements = param.numel()
        block_size = _get_triton_block_size(n_elements)
        grid = (triton.cdiv(n_elements, block_size),)

        with _device_guard(param):
            _oadamw_exp_avg_kernel[grid](
                grad_ptr=grad,
                exp_avg_ptr=exp_avg,
                exp_avg_sq_ptr=exp_avg_sq,
                mean_square_ptr=mean_squares,
                eps=eps,
                beta1_hat=beta1_hat,
                beta1_comp=beta1_comp,
                beta2_hat=beta2_hat,
                beta2_comp=beta2_comp,
                update_parameters=update_parameters,
                n_elements=n_elements,
                BLOCK_SIZE=block_size,
                param_dtype=TORCH_TO_TRITON_DTYPE[param.dtype],
            )
            if update_parameters:
                _oadamw_update_kernel[grid](
                    param_ptr=param,
                    exp_avg_ptr=exp_avg,
                    exp_avg_sq_ptr=exp_avg_sq,
                    kahan_ptr=kahan_comp,
                    mean_square_ptr=mean_squares,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    max_lr=max_lr,
                    do_weight_decay=weight_decay != 0.0,
                    kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                    decouple_lr=decouple_lr,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                )
                mean_squares.zero_()
