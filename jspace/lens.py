"""Readout engine: J-lens and logit-lens token distributions per (layer, position)."""

from __future__ import annotations

import torch

from .model_adapter import ModelBundle


@torch.no_grad()
def readout(
    bundle: ModelBundle,
    text: str,
    jacobians: list[torch.Tensor] | None,
    top_k: int = 8,
    max_tokens: int = 128,
) -> dict:
    """回傳每 (層, 位置) 的 top-k token 與機率。

    J-lens:  softmax(W_U norm(J_l h_l,t))   (需要預計算的 jacobians)
    logit lens: softmax(W_U norm(h_l,t))
    輸出 grid 形狀 [num_layers][seq_len],層索引 1..L(layer_1 起算)。
    """
    enc = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc.input_ids.to(bundle.device)
    hs = bundle.hidden_states(input_ids, with_grad=False)
    layers = hs[1:]  # layer_1 .. layer_L,對齊 jacobians 索引

    tokens = [bundle.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    result: dict = {
        "model_id": bundle.model_id,
        "tokens": tokens,
        "num_layers": bundle.num_layers,
        "truncated": len(tokens) >= max_tokens,
        "lenses": {},
    }

    def top_tokens(h_stack: torch.Tensor) -> list[list[dict]]:
        # h_stack: [seq, d] -> 每位置 top-k
        probs = torch.softmax(bundle.unembed(h_stack.to(bundle.model.dtype)).float(), dim=-1)
        vals, idxs = probs.topk(top_k, dim=-1)
        out = []
        for pos in range(h_stack.shape[0]):
            out.append(
                [
                    {"token": bundle.tokenizer.decode([idxs[pos, k].item()]), "p": round(vals[pos, k].item(), 6)}
                    for k in range(top_k)
                ]
            )
        return out

    logit_grid = [top_tokens(layer[0]) for layer in layers]
    result["lenses"]["logit"] = logit_grid

    if jacobians is not None:
        j_grid = []
        for l, layer in enumerate(layers):
            h = layer[0].float()  # [seq, d]
            jh = h @ jacobians[l].T  # (J h)ᵀ 對每個位置
            j_grid.append(top_tokens(jh))
        result["lenses"]["jlens"] = j_grid

    return result
