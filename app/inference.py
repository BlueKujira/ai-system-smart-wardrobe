import os
import json
import numpy as np
import tensorflow as tf
from .schemas import ModelUnavailableError

class CNNModel:
    MODEL_VERSION = '1.0.0'
    
    def __init__(self, model_path: str, mappings_path: str):
        if not os.path.exists(model_path):
            raise ModelUnavailableError(f'Модель не найдена: {model_path}')
        if not os.path.exists(mappings_path):
            raise ModelUnavailableError(f'Маппинги не найдены: {mappings_path}')
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            
            with open(mappings_path, 'r', encoding='utf-8') as f:
                self.mappings = json.load(f)
            
            self.attributes = list(self.mappings.keys())
        except Exception as e:
            raise ModelUnavailableError(f'Ошибка загрузки модели: {e}')
    
    def predict(self, img_array: np.ndarray) -> dict:
        try:
            preds = self.model.predict(img_array, verbose=0)
            return {
                attr: int(np.argmax(preds[attr][0]))
                for attr in self.attributes
            }
        except Exception as e:
            raise ModelUnavailableError(f'Ошибка предсказания: {e}')
    
    def predict_with_confidence(self, img_array: np.ndarray):
        try:
            preds = self.model.predict(img_array, verbose=0)
            
            attrs = {}
            confidences = []
            
            for attr in self.attributes:
                probs = preds[attr][0]
                idx = int(np.argmax(probs))
                attrs[attr] = idx
                confidences.append(float(probs[idx]))
            
            avg_conf = sum(confidences) / len(confidences)
            return attrs, avg_conf
        except Exception as e:
            raise ModelUnavailableError(f'Ошибка предсказания: {e}')
    
    @property
    def version(self) -> str:
        return self.MODEL_VERSION