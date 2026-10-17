from .schemas import CNNRequest, CNNAttributes, CNNResult, InvalidInputError, ModelUnavailableError
from .preprocessing import load_and_preprocess
from .inference import CNNModel

CONFIDENCE_T = 0.50

SEASON_TO_WEATHER = {
    0: 0,  # Summer - Sunny
    1: 1,  # Fall - Cloudy
    2: 3,  # Winter - Snowy
    3: 2,  # Spring - Rainy
}

class CNNService:
    def __init__(self, model_path: str, mappings_path: str):
        self.model = CNNModel(model_path, mappings_path)
    
    def predict(self, request: CNNRequest):
        img_array = load_and_preprocess(request.image_path)

        attrs, confidence = self.model.predict_with_confidence(img_array)
        
        if 'weather' not in attrs:
            season = attrs.get('season', 0)
            attrs['weather'] = SEASON_TO_WEATHER.get(season, 0)
        
        prediction = CNNAttributes(**attrs)
        needs_review = confidence < CONFIDENCE_T
        
        return CNNResult(
            attributes=prediction,
            confidence=round(confidence, 4),
            model_version=self.model.version,
            needs_human_review=needs_review,
        )