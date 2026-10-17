import pytest
from PIL import Image
from app import CNNRequest, CNNResult, CNNService, InvalidInputError, ModelUnavailableError

MODEL_PATH = "model/fashion_model.keras"
MAPPINGS_PATH = "model/mappings.json"

def create_test_image(tmp_path, name, color):
    img = Image.new('RGB', (300, 300), color=color)
    path = tmp_path / name
    img.save(path)
    return str(path)

def test_valid_input(tmp_path):
    blue_image = create_test_image(tmp_path, 'blue.jpg', (50, 100, 200))
    
    service = CNNService(MODEL_PATH, MAPPINGS_PATH)
    
    request = CNNRequest(image_path=blue_image)
    result = service.predict(request)
    
    assert result.model_version == '1.0.0'
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.needs_human_review, bool)
    
    attrs = result.attributes
    assert 0 <= attrs.gender <= 4
    assert 0 <= attrs.masterCategory <= 6
    assert 0 <= attrs.articleType <= 142
    assert 0 <= attrs.weather <= 3

def test_different_inputs(tmp_path):
    blue_image = create_test_image(tmp_path, 'blue.jpg', (50, 100, 200))
    red_image = create_test_image(tmp_path, 'red.jpg', (200, 50, 50))
    
    service = CNNService(MODEL_PATH, MAPPINGS_PATH)

    result_blue = service.predict(CNNRequest(image_path=blue_image))
    result_red = service.predict(CNNRequest(image_path=red_image))
    
    assert result_blue.model_version == result_red.model_version
    
    attrs_blue = result_blue.attributes
    attrs_red = result_red.attributes

    assert (
        attrs_blue.gender != attrs_red.gender
        or attrs_blue.articleType != attrs_red.articleType
        or attrs_blue.baseColour != attrs_red.baseColour
    )

def test_missing_file():
    service = CNNService(MODEL_PATH, MAPPINGS_PATH)
    
    request = CNNRequest(image_path='')
    
    with pytest.raises(InvalidInputError):
        service.predict(request)

def test_empty_path():
    with pytest.raises(Exception):
        CNNRequest(image_path='')

def test_model_missing():
    with pytest.raises(ModelUnavailableError):
        CNNService(
            '',
            '',
        )