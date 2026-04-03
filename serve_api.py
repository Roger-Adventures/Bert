from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import ModerationPipeline


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
REPORT_DIR = BASE_DIR / "reports"


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待审核文本")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="批量待审核文本")


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def create_app(model_dir: str | None = None) -> FastAPI:
    resolved_model_dir = model_dir or os.getenv("MODEL_DIR", "artifacts/moderation_macbert")
    moderation_pipeline = ModerationPipeline(model_dir=resolved_model_dir)
    app = FastAPI(title="中文内容审核决策原型", version="0.3.0")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    if REPORT_DIR.exists():
        app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_loaded": moderation_pipeline.model_loaded,
            "device": moderation_pipeline.device,
            "model_dir": resolved_model_dir,
            "load_error": moderation_pipeline.load_error,
            "policy_version": moderation_pipeline.policy_version,
            "python_executable": sys.executable,
            "python_version": sys.version.split(" [")[0],
            "torch_available": module_available("torch"),
            "transformers_available": module_available("transformers"),
            "docs_url": app.docs_url,
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, object]:
        return moderation_pipeline.predict(request.text)

    @app.post("/batch_predict")
    def batch_predict(request: BatchPredictRequest) -> list[dict[str, object]]:
        return moderation_pipeline.batch_predict(request.texts)

    return app


app = create_app()
