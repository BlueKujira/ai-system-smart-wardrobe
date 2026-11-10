import os
import numpy as np
from PIL import Image
from app.schemas import InvalidInputError

def load_and_preprocess(image_path, image_size):
    if not os.path.exists(image_path):
        raise InvalidInputError('Файл не найден')
    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((image_size, image_size))
        img_array = np.array(img) / 255.0
        img_array = np.reshape(img_array, (1, image_size, image_size, 3))
        return img_array
    except Exception:
        raise InvalidInputError('Ошибка обработки изображения')