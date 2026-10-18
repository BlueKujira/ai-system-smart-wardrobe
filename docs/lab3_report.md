# Лабораторная работа № 3

## Тема проекта
Умный гардероб - автоматический подбор и рекомендация одежды по фотографии.

## Реализованный компонент
**CNN** — Компонент анализа изображений. Принимает путь к фотографии предмета одежды, возвращает набор из 8 атрибутов: пол, категория, подкатегория, тип изделия, цвет, сезон, назначение, погода с оценкой уверенности.

## Контракты
```python
class CNNRequest(BaseModel):
    image_path: str = Field(
        min_length=1,
        max_length=500,
        description="Путь до файла"
    ) 

class CNNAttributes(BaseModel):
    gender: int = Field(ge=0, le=4)
    masterCategory: int = Field(ge=0, le=6)
    subCategory: int = Field(ge=0, le=44)
    articleType: int = Field(ge=0, le=142)
    baseColour: int = Field(ge=0, le=45)
    season: int = Field(ge=0, le=3)
    usage: int = Field(ge=0, le=8)
    weather: int = Field(ge=0, le=3)

class CNNResult(BaseModel):
    attributes: CNNAttributes
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
    needs_human_review: bool
```
## Используемая модель
**Модель** сверточная нейронная сеть CNN на базе TensorFlow. Источники тестовых данных Fashion Product Images Dataset

## Установки зависимостей и локального запуска
Скачать веса модели и положить в папку model

Запустить тесты

pytest tests/test_cnn.py

| Вход | Ожидаемое поведение |
|---|---|
| Путь к валидному изображению | CNNResult с 8 атрибутами, увереность, версия модели |
| Путь к другому изображению | Метаданные совпадают, хотя бы один атрибут отличается |
| Пустая строка | Ошибка проверки входа |
| Несуществующий путь к файлу | Ошибка с указанием пути |
| Пустой путь к модели | Ошибка доступа к модели |