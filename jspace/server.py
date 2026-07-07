"""FastAPI server: model registry, analyze endpoint, background precompute."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import cache
from .lens import readout
from .model_adapter import ModelBundle, load_model
from .runner import precompute

DEFAULT_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "HuggingFaceTB/SmolLM2-360M",
]

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="jspace")

_lock = threading.Lock()  # 一次只載一個模型(UMA 記憶體有限)
_current: ModelBundle | None = None
_jacobians: list[torch.Tensor] | None = None
_precompute_progress: dict[str, dict] = {}


def _get_bundle(model_id: str) -> ModelBundle:
    global _current, _jacobians
    if _current is None or _current.model_id != model_id:
        if _current is not None:
            del _current
            _jacobians = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _current = load_model(model_id)
        _jacobians = None
    if _jacobians is None:
        _jacobians = cache.load_jacobians(model_id, _current.device)
    return _current


class AnalyzeRequest(BaseModel):
    model_id: str
    text: str
    top_k: int = 8
    max_tokens: int = 128


class PrecomputeRequest(BaseModel):
    model_id: str
    num_prompts: int = 1000


@app.get("/api/models")
def list_models() -> dict:
    models = []
    for mid in DEFAULT_MODELS:
        status = cache.cache_status(mid)
        prog = _precompute_progress.get(mid)
        if prog and prog.get("state") in ("running", "starting"):
            status = {**status, **prog, "state": "computing"}
        models.append({"model_id": mid, **status})
    return {"models": models}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    if not req.text.strip():
        raise HTTPException(400, "輸入文字不可為空")
    with _lock:
        try:
            bundle = _get_bundle(req.model_id)
        except Exception as e:
            raise HTTPException(400, f"模型載入失敗:{e}")
        result = readout(bundle, req.text, _jacobians, top_k=req.top_k, max_tokens=req.max_tokens)
    result["jlens_available"] = "jlens" in result["lenses"]
    return result


@app.post("/api/precompute")
def start_precompute(req: PrecomputeRequest) -> dict:
    prog = _precompute_progress.get(req.model_id)
    if prog and prog.get("state") == "running":
        return prog
    progress: dict = {"state": "starting", "done": 0, "total": req.num_prompts}
    _precompute_progress[req.model_id] = progress

    def run() -> None:
        global _jacobians
        try:
            # 載入獨立副本,避免佔住 _lock 數小時導致 analyze 無法回應
            bundle = load_model(req.model_id)
            precompute(bundle, num_prompts=req.num_prompts, progress=progress)
            with _lock:
                if _current is not None and _current.model_id == req.model_id:
                    _jacobians = cache.load_jacobians(req.model_id, _current.device)
        except Exception as e:
            traceback.print_exc()
            progress.update({"state": "error", "error": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return progress


@app.get("/api/precompute/status")
def precompute_status(model_id: str) -> dict:
    prog = _precompute_progress.get(model_id)
    if prog:
        return prog
    return cache.cache_status(model_id)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
