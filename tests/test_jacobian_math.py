"""數學正確性:MC 估計器必須收斂到解析(精確)平均 Jacobian。"""

import torch

from jspace.jacobian import EstimatorState, accumulate_prompt


def exact_avg_jacobians(bundle, input_ids):
    """逐基向量 backward 的精確平均 Jacobian(僅測試用,d 小才可行)。"""
    hs = bundle.hidden_states(input_ids, with_grad=True)
    layers = hs[1:]
    h_final = layers[-1]
    seq = input_ids.shape[1]
    d = bundle.d_model
    totals = [torch.zeros(d, d) for _ in layers]
    n_pairs = 0
    for t_prime in range(seq):
        for i in range(d):
            grads = torch.autograd.grad(h_final[0, t_prime, i], layers, retain_graph=True)
            for l, g in enumerate(grads):
                totals[l][i] += g[0].sum(dim=0)
        n_pairs += t_prime + 1
    return [t / n_pairs for t in totals], n_pairs


def test_mc_estimator_converges_to_exact_jacobian(tiny_bundle):
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (1, 6))
    exact, _ = exact_avg_jacobians(tiny_bundle, ids)

    state = EstimatorState.zeros(tiny_bundle.num_layers, tiny_bundle.d_model, tiny_bundle.device)
    gen = torch.Generator().manual_seed(42)
    for _ in range(400):
        accumulate_prompt(tiny_bundle, ids, state, num_targets=6, generator=gen)
    est = state.jacobians()

    for l in range(tiny_bundle.num_layers):
        rel_err = (est[l] - exact[l]).norm() / exact[l].norm()
        cos = torch.nn.functional.cosine_similarity(est[l].flatten(), exact[l].flatten(), dim=0)
        assert cos > 0.97, f"layer {l}: cosine {cos:.3f} 太低,估計器有偏"
        assert rel_err < 0.30, f"layer {l}: 相對誤差 {rel_err:.3f} 未收斂"


def test_mc_estimate_improves_with_samples(tiny_bundle):
    """樣本數增加誤差應下降(驗證是收斂而非碰巧)。"""
    torch.manual_seed(2)
    ids = torch.randint(0, 128, (1, 5))
    exact, _ = exact_avg_jacobians(tiny_bundle, ids)

    errs = []
    for n_iter in (10, 640):
        state = EstimatorState.zeros(tiny_bundle.num_layers, tiny_bundle.d_model, tiny_bundle.device)
        gen = torch.Generator().manual_seed(7)
        for _ in range(n_iter):
            accumulate_prompt(tiny_bundle, ids, state, num_targets=5, generator=gen)
        est = state.jacobians()
        errs.append(sum((est[l] - exact[l]).norm() / exact[l].norm() for l in range(len(exact))))
    assert errs[1] < errs[0] / 3, f"誤差未隨樣本數下降:{errs}"


def test_pair_counting(tiny_bundle):
    """n_pairs 應等於 Σ(t'+1)(因果性:每個 t' 有 t'+1 個來源位置)。"""
    ids = torch.randint(0, 128, (1, 4))
    state = EstimatorState.zeros(tiny_bundle.num_layers, tiny_bundle.d_model, tiny_bundle.device)
    gen = torch.Generator().manual_seed(0)
    accumulate_prompt(tiny_bundle, ids, state, num_targets=4, generator=gen)
    assert state.n_pairs == 1 + 2 + 3 + 4
    assert state.n_prompts == 1
