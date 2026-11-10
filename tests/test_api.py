from fastapi.testclient import TestClient
import sys
import os
from app.app import app 

client = TestClient(app)

def test_predict_success():
    payload = {"image_path": "tests/test_image1.jpg"}
    res = client.post("/predict", json=payload)
    
    assert res.status_code == 200
    data = res.json()
    
    assert "trace_id" in data
    assert "source" in data
    assert data["source"] in ["model", "fallback_rule"]

def test_validation_error():
    payload = {"image_path": ""}
    res = client.post("/predict", json=payload)
    
    assert res.status_code == 422

def test_fallback_activation():
    if not hasattr(app.state, "service"):
        app.state.service = "dummy_service"
    
    original_service = app.state.service
    app.state.service = None
    
    payload = {"image_path": "tests/test_image1.jpg"}
    res = client.post("/predict", json=payload)
    
    assert res.status_code == 200
    data = res.json()
    assert data["source"] == "fallback_rule"

    app.state.service = original_service