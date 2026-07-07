"""Precompute orchestration: corpus -> MC estimation -> checkpoints -> final Jacobians."""

from __future__ import annotations

from pathlib import Path

from . import cache
from .corpus import build_corpus
from .jacobian import EstimatorState, estimate_jacobians
from .model_adapter import ModelBundle

CORPUS_PATH = cache.DEFAULT_CACHE_DIR / "corpus.jsonl"


def precompute(
    bundle: ModelBundle,
    num_prompts: int = 1000,
    num_targets: int = 8,
    max_tokens: int = 256,
    checkpoint_every: int = 25,
    corpus_path: Path | None = None,
    progress: dict | None = None,
) -> None:
    """跑(或續跑)一個模型的平均 Jacobian 預計算。

    progress: 可選的共享 dict,server 用它回報 {done, total, state}。
    """
    prompts = build_corpus(corpus_path or CORPUS_PATH, num_prompts=num_prompts)
    state = cache.load_checkpoint(bundle.model_id, bundle.device)
    if state is None:
        state = EstimatorState.zeros(bundle.num_layers, bundle.d_model, bundle.device)
    start = state.n_prompts
    if progress is not None:
        progress.update({"state": "running", "done": start, "total": num_prompts})
    if start >= num_prompts:
        cache.save_state(bundle.model_id, state, final=True)
        if progress is not None:
            progress["state"] = "ready"
        return

    def on_progress(n_prompts_done: int) -> None:
        if progress is not None:
            progress["done"] = n_prompts_done
        if n_prompts_done % checkpoint_every == 0:
            cache.save_state(bundle.model_id, state)  # estimate_jacobians 就地更新 state

    estimate_jacobians(
        bundle,
        prompts[start:num_prompts],
        max_tokens=max_tokens,
        num_targets=num_targets,
        progress=on_progress,
        state=state,
    )
    cache.save_state(bundle.model_id, state, final=True)
    if progress is not None:
        progress.update({"state": "ready", "done": state.n_prompts})
