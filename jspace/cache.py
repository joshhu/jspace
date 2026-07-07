"""Disk cache for averaged Jacobians (safetensors) + resumable estimator checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .jacobian import EstimatorState

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "jspace"


def model_cache_dir(model_id: str, cache_dir: Path | None = None) -> Path:
    root = cache_dir or DEFAULT_CACHE_DIR
    return root / model_id.replace("/", "--")


def save_state(model_id: str, state: EstimatorState, cache_dir: Path | None = None, final: bool = False) -> Path:
    """存 checkpoint(sums + 計數);final=True 時另存平均後的 J 矩陣供讀出使用。"""
    d = model_cache_dir(model_id, cache_dir)
    d.mkdir(parents=True, exist_ok=True)

    tensors = {f"sum_{l}": s.cpu() for l, s in enumerate(state.sums)}
    save_file(tensors, str(d / "checkpoint.safetensors"))
    meta = {"n_pairs": state.n_pairs, "n_prompts": state.n_prompts, "num_layers": len(state.sums)}
    (d / "checkpoint.json").write_text(json.dumps(meta))

    if final:
        jac = {f"J_{l}": j.cpu() for l, j in enumerate(state.jacobians())}
        save_file(jac, str(d / "jacobians.safetensors"))
        (d / "meta.json").write_text(json.dumps({"model_id": model_id, **meta}))
    return d


def load_checkpoint(model_id: str, device: torch.device, cache_dir: Path | None = None) -> EstimatorState | None:
    d = model_cache_dir(model_id, cache_dir)
    ckpt, meta_p = d / "checkpoint.safetensors", d / "checkpoint.json"
    if not (ckpt.exists() and meta_p.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    tensors = load_file(str(ckpt))
    sums = [tensors[f"sum_{l}"].to(device) for l in range(meta["num_layers"])]
    return EstimatorState(sums=sums, n_pairs=meta["n_pairs"], n_prompts=meta["n_prompts"])


def load_jacobians(model_id: str, device: torch.device, cache_dir: Path | None = None) -> list[torch.Tensor] | None:
    d = model_cache_dir(model_id, cache_dir)
    # 若存在譜收縮去噪版(大模型 MC 樣本不足時後處理產生)優先使用
    for name in ("jacobians_denoised.safetensors", "jacobians.safetensors"):
        path = d / name
        if path.exists():
            tensors = load_file(str(path))
            return [tensors[f"J_{l}"].to(device) for l in range(len(tensors))]
    return None


def cache_status(model_id: str, cache_dir: Path | None = None) -> dict:
    d = model_cache_dir(model_id, cache_dir)
    meta_p, ckpt_p = d / "meta.json", d / "checkpoint.json"
    if meta_p.exists():
        return {"state": "ready", **json.loads(meta_p.read_text())}
    if ckpt_p.exists():
        return {"state": "partial", **json.loads(ckpt_p.read_text())}
    return {"state": "missing"}
