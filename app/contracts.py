import pandera as pa
from pandera import Column, Check, DataFrameSchema

RawWardrobeSchema = DataFrameSchema({
    "id": Column(int, nullable=False),
    "ageGroup": Column(str, nullable=True),
    "gender": Column(str, Check.isin(["Women", "Men", "Boys", "Girls", "Unisex"]), nullable=False),
    "masterCategory": Column(str, nullable=False),
    "subCategory": Column(str, nullable=False),
    "articleType": Column(str, nullable=False),
    "baseColour": Column(str, nullable=True),
    "season": Column(str, Check.isin(["Fall", "Spring", "Summer", "Winter"]), nullable=True),
    "usage": Column(str, nullable=True),
}, strict=True)