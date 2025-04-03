from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from predict import PipelinePredictor

class CarFeatures(BaseModel):
    Doors: int
    Year: int
    Owner_Count: int
    Brand: str
    Model: str
    Fuel_Type: str
    Transmission: str
    Engine_Size: float
    Mileage: float

class CarPriceAPI:
    def __init__(self):
        """Инициализация API и зависимостей."""
        self.app = FastAPI()
        self.predictor = PipelinePredictor()
        self._register_routes()

    def _register_routes(self):
        """Регистрация маршрутов API."""
        @self.app.get('/')
        def health_check():
            return {'health_check': 'OK'}

        @self.app.post("/predict")
        def predict(features: CarFeatures):
            input_data = pd.DataFrame([features.model_dump()])
            prediction = self.predictor.predict(input_data)
            return {"prediction": prediction[0]}

    def get_app(self):
        """Возвращает экземпляр FastAPI приложения."""
        return self.app

# Создаем экземпляр API
api = CarPriceAPI()
app = api.get_app()