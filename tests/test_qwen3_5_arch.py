"""qwen3_5(Qwen3.6 系列)混合 linear/full attention 架構的相容性測試。"""

import torch
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from jspace.jacobian import EstimatorState, accumulate_prompt
from jspace.lens import readout
from jspace.model_adapter import ModelBundle

from tests.conftest import CharTokenizer


def make_tiny_qwen35() -> ModelBundle:
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128, head_dim=16,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    )
    model = Qwen3_5ForCausalLM(cfg).eval()
    model.requires_grad_(False)
    return ModelBundle(
        model_id="tiny-qwen3_5", model=model, tokenizer=CharTokenizer(128),
        final_norm=model.model.norm, lm_head=model.get_output_embeddings(),
        num_layers=4, d_model=64, device=torch.device("cpu"),
    )


def test_hidden_states_are_pre_norm():
    b = make_tiny_qwen35()
    ids = torch.randint(0, 128, (1, 6))
    hs = b.hidden_states(ids)
    assert len(hs) == b.num_layers + 1
    with torch.no_grad():
        logits = b.model(input_ids=ids).logits
        # 最後一個 hidden state 必須是 pre-norm:套 norm+unembed 後等於真實 logits
        assert torch.allclose(b.unembed(hs[-1]), logits, atol=1e-4)
        # 若已是 post-norm,直接 unembed 就會等於 logits(不應如此)
        assert not torch.allclose(b.lm_head(hs[-1]), logits, atol=1e-4)


def test_gradients_flow_through_linear_attention():
    b = make_tiny_qwen35()
    ids = torch.randint(0, 128, (1, 8))
    state = EstimatorState.zeros(b.num_layers, b.d_model, b.device)
    gen = torch.Generator().manual_seed(0)
    accumulate_prompt(b, ids, state, num_targets=4, generator=gen)
    assert state.n_pairs > 0
    for s in state.sums:
        assert torch.isfinite(s).all()
        assert s.abs().sum() > 0  # linear attention 層的梯度不應為零


def test_readout_works():
    b = make_tiny_qwen35()
    result = readout(b, "hello", jacobians=None, top_k=3)
    assert len(result["lenses"]["logit"]) == 4
