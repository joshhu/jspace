"""快取 round-trip 與 lens 讀出的行為驗證。"""

import torch

from jspace import cache
from jspace.jacobian import EstimatorState
from jspace.lens import readout


def test_cache_roundtrip(tiny_bundle, tmp_path):
    state = EstimatorState.zeros(tiny_bundle.num_layers, tiny_bundle.d_model, tiny_bundle.device)
    for s in state.sums:
        s.copy_(torch.randn_like(s))
    state.n_pairs, state.n_prompts = 10, 3

    cache.save_state("test/model", state, cache_dir=tmp_path, final=True)

    loaded = cache.load_checkpoint("test/model", tiny_bundle.device, cache_dir=tmp_path)
    assert loaded.n_pairs == 10 and loaded.n_prompts == 3
    assert all(torch.allclose(a, b) for a, b in zip(loaded.sums, state.sums))

    jac = cache.load_jacobians("test/model", tiny_bundle.device, cache_dir=tmp_path)
    assert len(jac) == tiny_bundle.num_layers
    assert torch.allclose(jac[0], state.sums[0] / 10)

    assert cache.cache_status("test/model", cache_dir=tmp_path)["state"] == "ready"
    assert cache.cache_status("no/such", cache_dir=tmp_path)["state"] == "missing"


def test_readout_shapes_and_probs(tiny_bundle):
    result = readout(tiny_bundle, "hello world", jacobians=None, top_k=5)
    grid = result["lenses"]["logit"]
    n_tok = len(result["tokens"])
    assert len(grid) == tiny_bundle.num_layers
    assert all(len(row) == n_tok for row in grid)
    cell = grid[0][0]
    assert len(cell) == 5
    ps = [e["p"] for e in cell]
    assert ps == sorted(ps, reverse=True)
    assert 0 < sum(ps) <= 1.0 + 1e-6
    assert "jlens" not in result["lenses"]


def test_identity_jacobian_matches_logit_lens(tiny_bundle):
    """J = I 時 J-lens 應退化為 logit lens(讀出管線一致性)。"""
    eye = [torch.eye(tiny_bundle.d_model) for _ in range(tiny_bundle.num_layers)]
    result = readout(tiny_bundle, "abc", jacobians=eye, top_k=3)
    assert result["lenses"]["jlens"] == result["lenses"]["logit"]


def test_last_layer_matches_model_next_token(tiny_bundle):
    """最後一層 logit lens 讀出必須等於模型真實 next-token 分佈。"""
    enc = tiny_bundle.tokenizer("hello", return_tensors="pt")
    with torch.no_grad():
        true_logits = tiny_bundle.model(input_ids=enc.input_ids).logits
    true_top = true_logits[0, -1].softmax(-1).argmax().item()

    result = readout(tiny_bundle, "hello", jacobians=None, top_k=1)
    last_layer_last_pos = result["lenses"]["logit"][-1][-1][0]
    assert last_layer_last_pos["token"] == tiny_bundle.tokenizer.decode([true_top])
