from pydantic import BaseModel, Field

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

class ComponentError(Exception):
    pass

class InvalidInputError(ComponentError):
    pass

class ModelUnavailableError(ComponentError):
    pass