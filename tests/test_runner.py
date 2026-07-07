"""precompute 編排的功能測試:checkpoint 觸發、續算、完成狀態。

回歸:on_progress 曾在 estimate_jacobians 返回前引用未定義的結果變數,
導致第一次 checkpoint(第 checkpoint_every 條 prompt)即崩潰。
"""

import json

from jspace import cache
from jspace.runner import precompute


def _write_corpus(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"text": f"prompt number {i} " * 30}) + "\n")


def test_precompute_checkpoints_and_completes(tiny_bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", tmp_path)
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, 6)

    progress: dict = {}
    precompute(
        tiny_bundle,
        num_prompts=6,
        num_targets=2,
        max_tokens=32,
        checkpoint_every=2,  # 會在完成前觸發多次 checkpoint(回歸點)
        corpus_path=corpus,
        progress=progress,
    )
    assert progress["state"] == "ready"
    assert progress["done"] == 6
    assert cache.cache_status(tiny_bundle.model_id, cache_dir=tmp_path)["state"] == "ready"
    jac = cache.load_jacobians(tiny_bundle.model_id, tiny_bundle.device, cache_dir=tmp_path)
    assert len(jac) == tiny_bundle.num_layers


def test_precompute_resumes_from_checkpoint(tiny_bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", tmp_path)
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, 4)

    precompute(tiny_bundle, num_prompts=2, num_targets=2, max_tokens=32,
               checkpoint_every=1, corpus_path=corpus)
    ckpt = cache.load_checkpoint(tiny_bundle.model_id, tiny_bundle.device, cache_dir=tmp_path)
    assert ckpt.n_prompts == 2

    # 擴大目標續跑:應從第 2 條之後接續,不重算
    progress: dict = {}
    precompute(tiny_bundle, num_prompts=4, num_targets=2, max_tokens=32,
               checkpoint_every=1, corpus_path=corpus, progress=progress)
    ckpt2 = cache.load_checkpoint(tiny_bundle.model_id, tiny_bundle.device, cache_dir=tmp_path)
    assert ckpt2.n_prompts == 4
    assert progress["state"] == "ready"
