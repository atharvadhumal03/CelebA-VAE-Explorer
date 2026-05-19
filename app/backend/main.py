import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.backend import inference, search
from app.backend.schemas import (
    GenerateResponse,
    InterpolateResponse,
    ReconstructResponse,
    SearchResponse,
    TSNEResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load_model()
    search.load_search_artifacts()
    yield


app = FastAPI(title="LatentLens", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")


@app.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct(image: UploadFile = File(...)):
    pil = _read_image(image)
    return ReconstructResponse(image=inference.reconstruct(pil))


@app.post("/search", response_model=SearchResponse)
async def search_faces(image: UploadFile = File(...)):
    pil = _read_image(image)
    mu = inference.encode(pil).cpu().numpy()
    results = search.search(mu, k=5)
    return SearchResponse(results=results)


@app.post("/interpolate", response_model=InterpolateResponse)
async def interpolate(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    steps: int = 8,
):
    pil_a = _read_image(image_a)
    pil_b = _read_image(image_b)
    frames = inference.interpolate_faces(pil_a, pil_b, steps=steps)
    return InterpolateResponse(frames=frames)


@app.get("/generate", response_model=GenerateResponse)
async def generate():
    return GenerateResponse(image=inference.generate_face())


@app.get("/tsne", response_model=TSNEResponse)
async def tsne():
    data = search.get_tsne_data()
    return TSNEResponse(**data)


# Serve the React frontend — must be mounted LAST so API routes take precedence
try:
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
except RuntimeError:
    pass  # frontend/dist not built yet (dev mode)
