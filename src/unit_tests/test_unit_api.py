import pytest
import json
import math
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from api import app  # Импортируем API

client = TestClient(app)  # Создаём тестового клиента

# Получаем пути до тестовых json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)) ) 
TESTS_DIR = os.path.join(BASE_DIR, "tests", "test_data")

@pytest.mark.parametrize("test_file", [
    os.path.join(TESTS_DIR, "test_0.json"),
    os.path.join(TESTS_DIR, "test_1.json")
])
def test_unit_api(test_file):
    """Unit тестирование имитационного API"""
    # Открываем тест
    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    # Проверяем health-check
    health_response = client.get('/')
    assert health_response.status_code == 200
    
    # Проверяем POST-запрос predict
    predict_response = client.post("/predict", json=test_data["X"])
    assert predict_response.status_code == 200
    # Проверяем, что в ответе есть предсказание
    assert "prediction" in predict_response.json()

    # Ожидаемое и полученное предсказание
    y_true = test_data["y"]["prediction"]
    y_pred = predict_response.json()["prediction"]

    # Проверяем, что предсказание близко к ожидаемому (допуск ±1%)
    assert math.isclose(y_pred, y_true, rel_tol=0.01)
