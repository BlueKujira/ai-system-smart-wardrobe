from app.schemas import CNNRequest, CNNAttributes, CNNResult, InvalidInputError, ModelUnavailableError
from app.preprocessing import load_and_preprocess
from app.inference import CNNModel
import joblib

CONFIDENCE_T = 0.50

SEASON_TO_WEATHER = {
    0: 0,  # Fall - Cloudy
    1: 1,  # Spring - Rainy
    2: 2,  # Summer - Sunny
    3: 3,  # Winter - Snowy
}

class CNNService:
    def __init__(self, model_path, mappings_path, preprocessor_path):
        self.model = CNNModel(model_path, mappings_path)
        self.preprocessor = joblib.load(preprocessor_path)
    
    def predict(self, request: CNNRequest):
        image_size = self.preprocessor.get("image_size", 96)

        img_array = load_and_preprocess(request.image_path, image_size)
        
        attrs, confidence = self.model.predict_with_confidence(img_array)
        
        if 'weather' not in attrs:
            season = attrs.get('season', 0)
            attrs['weather'] = SEASON_TO_WEATHER.get(season, 0)
        
        prediction = CNNAttributes(**attrs)
        needs_review = confidence < CONFIDENCE_T
        
        return CNNResult(
            trace_id="test-trace-id",
            source="model", 
            attributes=prediction,
            confidence=round(confidence, 4),
            model_version=self.model.version,
            needs_human_review=needs_review,
        )