from pydantic import BaseModel


class ReconstructResponse(BaseModel):
    image: str  # base64-encoded PNG


class SearchResult(BaseModel):
    image: str        # base64-encoded PNG
    attributes: dict  # {attr_name: int (0 or 1)}
    distance: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class InterpolateResponse(BaseModel):
    frames: list[str]  # list of base64-encoded PNGs


class GenerateResponse(BaseModel):
    image: str  # base64-encoded PNG


class TSNEResponse(BaseModel):
    coords: list[list[float]]        # [[x, y], ...]
    attributes: list[list[int]]      # [[0,1,...], ...] — 40 binary labels per image
    attribute_names: list[str]       # 40 CelebA attribute names
    image_indices: list[int]         # CelebA image indices (for click-to-show)
