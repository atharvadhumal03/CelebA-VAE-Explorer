import base64
import io
import json
import os

import faiss
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from src.visualize import CELEBA_ATTR_NAMES


_index: faiss.IndexFlatL2 | None = None
_latent_vectors: np.ndarray | None = None
_attributes: np.ndarray | None = None
_tsne_data: dict | None = None


def _load_artifacts():
    global _index, _latent_vectors, _attributes, _tsne_data
    if _index is not None:
        return

    repo_id = os.environ["HF_MODEL_REPO"]

    index_path   = hf_hub_download(repo_id=repo_id, filename="faiss_index.bin",     repo_type="model")
    latents_path = hf_hub_download(repo_id=repo_id, filename="latent_vectors.npy",  repo_type="model")
    attrs_path   = hf_hub_download(repo_id=repo_id, filename="attributes.npy",      repo_type="model")
    tsne_path    = hf_hub_download(repo_id=repo_id, filename="tsne_data.json",       repo_type="model")

    _index          = faiss.read_index(index_path)
    _latent_vectors = np.load(latents_path).astype(np.float32)
    _attributes     = np.load(attrs_path)
    with open(tsne_path) as f:
        _tsne_data = json.load(f)


def load_search_artifacts():
    """Called at FastAPI startup."""
    _load_artifacts()


def search(query_mu: np.ndarray, k: int = 5) -> list[dict]:
    """
    query_mu : [1, D] float32 numpy array (from inference.encode)
    Returns list of k dicts with keys: image (base64 PNG), attributes (dict), distance (float).
    """
    _load_artifacts()
    distances, indices = _index.search(query_mu.astype(np.float32), k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        attr_vec = _attributes[idx]
        attr_dict = {name: int(val) for name, val in zip(CELEBA_ATTR_NAMES, attr_vec)}

        # Latent vector → decode to image via inference module to avoid circular import
        from app.backend.inference import load_model, _tensor_to_b64, DEVICE
        import torch
        model = load_model()
        z = torch.tensor(_latent_vectors[idx], device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            img_tensor = model.decode(z)
        b64 = _tensor_to_b64(img_tensor)

        results.append({"image": b64, "attributes": attr_dict, "distance": float(dist)})

    return results


def get_tsne_data() -> dict:
    """Return the precomputed t-SNE payload for the /tsne endpoint."""
    _load_artifacts()
    return _tsne_data
