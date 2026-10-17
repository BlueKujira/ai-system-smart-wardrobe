import os
import numpy as np
from PIL import Image
from .schemas import InvalidInputError

IMAGE_SIZE = 96

def load_and_preprocess(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise InvalidInputError('Файл не найден')
    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.reshape(img_array, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
        return img_array
    except Exception as e:
        raise InvalidInputError('Ошибка обработки изображения')