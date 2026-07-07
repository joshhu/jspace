"""Load a HuggingFace causal LM and expose the pieces the J-lens needs.

The J-lens needs, for any decoder-only transformer:
  - the residual stream after each layer (``output_hidden_states=True``)
  - the final norm applied before unembedding
  - the unembedding matrix W_U (lm_head)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class UnsupportedArchitectureError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    model_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    final_norm: nn.Module
    lm_head: nn.Module
    num_layers: int
    d_model: int
    device: torch.device

    @torch.no_grad()
    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """norm + W_U: map residual-stream vectors [..., d_model] to logits [..., vocab]."""
        return self.lm_head(self.final_norm(h))

    def hidden_states(self, input_ids: torch.Tensor, with_grad: bool = False):
        """Forward pass returning the tuple of residual streams.

        Returns (embeddings, layer_1, ..., layer_L) each of shape [batch, seq, d_model],
        全部是「未套 final norm 的原始 residual stream」。transformers 的
        hidden_states[-1] 已套過 final norm,因此用 pre-hook 抓 norm 的輸入取代之
        (J-lens 定義從 pre-norm 的 final residual stream 反傳,讀出時才套 norm)。
        權重已凍結,若要對激活求梯度必須從 inputs_embeds 建圖。
        """
        captured: list[torch.Tensor] = []

        def grab_norm_input(module: nn.Module, args: tuple) -> None:
            captured.append(args[0])

        handle = self.final_norm.register_forward_pre_hook(grab_norm_input)
        try:
            if with_grad:
                with torch.enable_grad():
                    embeds = self.model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
                    out = self.model(inputs_embeds=embeds, output_hidden_states=True, use_cache=False)
            else:
                with torch.no_grad():
                    out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        finally:
            handle.remove()
        if not captured:
            raise UnsupportedArchitectureError("final norm 在 forward 中未被呼叫,無法取得 final residual stream")
        return (*out.hidden_states[:-1], captured[0])


def _find_final_norm(model: PreTrainedModel) -> nn.Module:
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    if base is None:
        raise UnsupportedArchitectureError(f"無法定位 {type(model).__name__} 的 base model")
    for attr in ("norm", "final_layernorm", "ln_f"):
        norm = getattr(base, attr, None)
        if isinstance(norm, nn.Module):
            return norm
    raise UnsupportedArchitectureError(
        f"無法在 {type(base).__name__} 找到 final norm(嘗試過 norm/final_layernorm/ln_f)"
    )


def load_model(model_id: str, device: str | None = None, dtype: torch.dtype | None = None) -> ModelBundle:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)  # J-lens 只對「激活」求梯度,凍結權重省記憶體

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise UnsupportedArchitectureError(f"{model_id} 沒有 output embeddings(非 causal LM?)")

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if num_layers is None or d_model is None:
        raise UnsupportedArchitectureError(f"{model_id} 的 config 缺少層數/hidden size 資訊")

    return ModelBundle(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        final_norm=_find_final_norm(model),
        lm_head=lm_head,
        num_layers=num_layers,
        d_model=d_model,
        device=torch.device(device),
    )
