import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from jspace.model_adapter import ModelBundle


class CharTokenizer:
    """離線測試用:每個字元一個 token(id = ord % vocab)。"""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = [ord(c) % self.vocab_size for c in text]
        if truncation and max_length:
            ids = ids[:max_length]

        class Enc:
            pass

        enc = Enc()
        enc.input_ids = torch.tensor([ids], dtype=torch.long)
        return enc

    def decode(self, ids):
        return "".join(chr(i) if 32 <= i < 127 else f"<{i}>" for i in ids)


def make_tiny_bundle(d: int = 32, layers: int = 2, vocab: int = 128, seed: int = 0) -> ModelBundle:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        hidden_size=d,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vocab,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg).eval()
    model.requires_grad_(False)
    return ModelBundle(
        model_id="tiny-test",
        model=model,
        tokenizer=CharTokenizer(vocab),
        final_norm=model.model.norm,
        lm_head=model.get_output_embeddings(),
        num_layers=layers,
        d_model=d,
        device=torch.device("cpu"),
    )


@pytest.fixture(scope="session")
def tiny_bundle() -> ModelBundle:
    return make_tiny_bundle()
