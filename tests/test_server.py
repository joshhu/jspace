"""API 功能測試:以 tiny model 替換真實模型載入。"""

import pytest
import torch
from fastapi.testclient import TestClient

import jspace.server as server
from tests.conftest import make_tiny_bundle


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "load_model", lambda mid, **kw: make_tiny_bundle())
    monkeypatch.setattr(server.cache, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(server, "_current", None)
    monkeypatch.setattr(server, "_jacobians", None)
    return TestClient(server.app)


def test_list_models(client):
    data = client.get("/api/models").json()
    assert len(data["models"]) >= 1
    assert all("state" in m for m in data["models"])


def test_analyze_logit_lens_only(client):
    r = client.post("/api/analyze", json={"model_id": "tiny", "text": "hi there", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["jlens_available"] is False
    assert "logit" in data["lenses"]
    assert len(data["tokens"]) == len("hi there")


def test_analyze_with_precomputed_jacobians(client, tmp_path):
    from jspace import cache as c
    from jspace.jacobian import EstimatorState

    bundle = make_tiny_bundle()
    state = EstimatorState.zeros(bundle.num_layers, bundle.d_model, bundle.device)
    for s in state.sums:
        s.copy_(torch.eye(bundle.d_model))
    state.n_pairs = 1
    c.save_state("tiny", state, cache_dir=tmp_path, final=True)

    r = client.post("/api/analyze", json={"model_id": "tiny", "text": "hi", "top_k": 3})
    data = r.json()
    assert data["jlens_available"] is True
    assert "jlens" in data["lenses"]


def test_analyze_empty_text_rejected(client):
    r = client.post("/api/analyze", json={"model_id": "tiny", "text": "   "})
    assert r.status_code == 400
