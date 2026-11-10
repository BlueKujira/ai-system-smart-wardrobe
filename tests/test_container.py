import requests
import os

BASE_URL = "http://localhost:8000"

def test_container_health():
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model_version" in data

def test_container_predict_real():
    test_image_path = "tests/test_image1.jpg" 
    
    assert os.path.exists(test_image_path), f"Картинка {test_image_path} не найдена!"

    payload = {"image_path": test_image_path}
    res = requests.post(f"{BASE_URL}/predict", json=payload)
    
    assert res.status_code == 200, f"Ошибка инференса: {res.text}"
    data = res.json()
    
    assert "attributes" in data
    assert "confidence" in data
    assert "model_version" in data
    assert "trace_id" in data           
    assert "source" in data             

    attrs = data["attributes"]
    assert 0 <= attrs["gender"] <= 4
    assert 0 <= attrs["season"] <= 3
    assert 0.0 <= data["confidence"] <= 1.0

def test_container_predict_bad_input():
    payload = {"image_path": "non_existent.jpg"}
    res = requests.post(f"{BASE_URL}/predict", json=payload)
    
    assert res.status_code == 200
    assert res.json()["source"] == "fallback_rule"