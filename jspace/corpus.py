"""Pretraining-like prompt corpus for Jacobian averaging.

論文用 ~1000 條 pretraining 分佈的 prompt。這裡用 FineWeb(英文)串流取樣,
多語模型可混入 FineWeb-2 中文(cmn_Hani)。取樣結果落地成 jsonl,
確保續算時語料順序穩定。
"""

from __future__ import annotations

import json
from pathlib import Path


def _stream_texts(dataset: str, name: str | None, n: int, min_chars: int = 200) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset, name=name, split="train", streaming=True)
    texts: list[str] = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) >= min_chars:
            texts.append(text)
        if len(texts) >= n:
            break
    return texts


def build_corpus(
    path: Path,
    num_prompts: int = 1000,
    zh_fraction: float = 0.2,
) -> list[str]:
    """下載並落地語料;已存在且數量足夠時直接重用。"""
    if path.exists():
        texts = [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
        if len(texts) >= num_prompts:
            return texts[:num_prompts]

    n_zh = int(num_prompts * zh_fraction)
    n_en = num_prompts - n_zh
    texts = _stream_texts("HuggingFaceFW/fineweb", "sample-10BT", n_en)
    if n_zh > 0:
        try:
            texts += _stream_texts("HuggingFaceFW/fineweb-2", "cmn_Hani", n_zh)
        except Exception:
            # 中文語料抓不到就退回全英文,不讓預計算整個失敗
            texts += _stream_texts("HuggingFaceFW/fineweb", "sample-10BT", n_zh + len(texts))[len(texts):]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in texts[:num_prompts]:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return texts[:num_prompts]
