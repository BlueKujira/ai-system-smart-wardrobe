from contextlib import asynccontextmanager
import uuid, time, joblib, pandas as pd
import os
from fastapi import FastAPI, HTTPException
from app.schemas import CNNRequest, CNNResult, CNNAttributes
from app.service import CNNService 
from app.logger import get_json_logger

logger = get_json_logger("wardrobe_api")
MODEL_VERSION = "v1.0.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.service = CNNService(
            model_path="model/fashion_model.keras",        
            mappings_path="model/mappings.json",
            preprocessor_path="artifacts/preprocessor.joblib"
        )
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        app.state.service = None
    yield
app = FastAPI(title="Умный гардероб", version=MODEL_VERSION, lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok" if app.state.service is not None else "degraded", 
        "model_version": MODEL_VERSION
    }

@app.post("/predict", response_model=CNNResult)
def predict(req: CNNRequest):
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())
    source = "model"
    confidence = 0.5
    
    try:
        if app.state.service is None:
            raise RuntimeError("Модель не инициализирована")
        
        result = app.state.service.predict(req)
        attrs = result.attributes
        confidence = result.confidence
        
    except Exception as exc:
        source = "fallback_rule"
        logger.warning(f"Сработал fallback: {str(exc)}")
        
        attrs = {
            "gender": 3, "masterCategory": 1, "subCategory": 38, 
            "articleType": 132, "baseColour": 44, "season": 2, 
            "usage": 0, "weather": 2
        }

    latency = round((time.perf_counter() - start_time) * 1000, 2)
    
    logger.info(
        "inference", 
        extra={"extra_data": {
            "trace_id": trace_id, 
            "image_path": req.image_path,
            "source": source, 
            "latency_ms": latency
        }}
    )

    return CNNResult(
        trace_id=trace_id,
        source=source,
        attributes=CNNAttributes(**attrs),
        confidence=confidence,
        model_version=MODEL_VERSION,
        needs_human_review=(confidence < 0.5)
    )