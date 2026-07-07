"""Monte-Carlo estimator for the averaged J-lens Jacobians.

論文定義(transformer-circuits.pub/2026/workspace):
    J_l = E[ ∂h_final,t' / ∂h_l,t ]    對 t ≤ t'、與語料 prompt 取平均

精確 Jacobian 需要 d_model 次 backward per (prompt, t'),不可行。
改用隨機投影蒙地卡羅:對 u ~ N(0, I),E[u ⊗ (uᵀ J)] = J。
每次 backward 以 scalar = u · h_final[t'] 反傳,一次得到所有層、
所有 t ≤ t' 的 VJP(uᵀ ∂h_final,t'/∂h_l,t),累積外積 u ⊗ Σ_t vjp_t。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch

from .model_adapter import ModelBundle


@dataclass
class EstimatorState:
    """fp32 累加器:sums[l] = Σ u ⊗ vjp,n_pairs = 已累積的 (t, t') 對數。"""

    sums: list[torch.Tensor]
    n_pairs: int = 0
    n_prompts: int = 0

    @classmethod
    def zeros(cls, num_layers: int, d_model: int, device: torch.device) -> "EstimatorState":
        return cls(sums=[torch.zeros(d_model, d_model, dtype=torch.float32, device=device) for _ in range(num_layers)])

    def jacobians(self) -> list[torch.Tensor]:
        if self.n_pairs == 0:
            raise ValueError("尚未累積任何樣本")
        return [s / self.n_pairs for s in self.sums]


def accumulate_prompt(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    state: EstimatorState,
    num_targets: int = 8,
    generator: torch.Generator | None = None,
) -> None:
    """對單一 prompt 累積 MC 樣本。

    input_ids: [1, seq]。取樣 num_targets 個目標位置 t',每個配一個新的隨機 u。
    hidden_states = (embeddings, layer_1..layer_L);J-lens 的「層 l 激活」取
    layer_1..layer_L(共 num_layers 個,對齊 state.sums)。
    """
    hs = bundle.hidden_states(input_ids, with_grad=True)
    layers = hs[1:]  # layer_1 .. layer_L
    h_final = hs[-1]
    seq_len = input_ids.shape[1]
    d = bundle.d_model

    n = min(num_targets, seq_len)
    t_primes = torch.randperm(seq_len, generator=generator)[:n].tolist()

    for i, t_prime in enumerate(sorted(t_primes)):
        u = torch.randn(d, generator=generator, dtype=torch.float32).to(bundle.device)
        scalar = (h_final[0, t_prime].float() * u).sum()
        retain = i < n - 1
        grads = torch.autograd.grad(scalar, layers, retain_graph=retain, allow_unused=False)
        for l, g in enumerate(grads):
            # g: [1, seq, d];因果性使 t > t' 的梯度為 0,直接對全序列求和即 Σ_{t≤t'}
            vjp_sum = g[0].float().sum(dim=0)
            state.sums[l] += torch.outer(u, vjp_sum)
        state.n_pairs += t_prime + 1
    state.n_prompts += 1


def estimate_jacobians(
    bundle: ModelBundle,
    prompts: Iterable[str],
    max_tokens: int = 256,
    num_targets: int = 8,
    seed: int = 0,
    progress: Callable[[int], None] | None = None,
    state: EstimatorState | None = None,
) -> EstimatorState:
    """跑完整段語料,回傳(或續算)累加器狀態。"""
    if state is None:
        state = EstimatorState.zeros(bundle.num_layers, bundle.d_model, bundle.device)
    generator = torch.Generator().manual_seed(seed + state.n_prompts)

    for prompt in prompts:
        ids = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids
        if ids.shape[1] < 2:
            continue
        accumulate_prompt(bundle, ids.to(bundle.device), state, num_targets=num_targets, generator=generator)
        if progress is not None:
            progress(state.n_prompts)
    return state
