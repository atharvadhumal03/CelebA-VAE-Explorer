import base64
import io
import os

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from src.model import VAE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: VAE | None = None

_PREPROCESS = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_model() -> VAE:
    """Download best_model.pt from HF Hub, load once, cache in module state."""
    global _model
    if _model is not None:
        return _model

    repo_id = os.environ["HF_MODEL_REPO"]
    local_path = hf_hub_download(repo_id=repo_id, filename="best_model.pt", repo_type="model")

    ckpt = torch.load(local_path, map_location=DEVICE)
    cfg = ckpt["config"]
    model = VAE(latent_dim=cfg["model"]["latent_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _model = model
    return _model


def _tensor_to_b64(t: torch.Tensor) -> str:
    """Convert [1, 3, H, W] tensor in [-1, 1] to base64 PNG string."""
    img = t.squeeze(0)
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image to [1, 3, 128, 128] tensor on DEVICE."""
    return _PREPROCESS(pil.convert("RGB")).unsqueeze(0).to(DEVICE)


def reconstruct(pil: Image.Image) -> str:
    """Encode and decode a face. Returns base64 PNG."""
    model = load_model()
    x = _pil_to_tensor(pil)
    with torch.no_grad():
        recon, _, _ = model(x)
    return _tensor_to_b64(recon)


def encode(pil: Image.Image) -> torch.Tensor:
    """Return the mu vector [1, latent_dim] for a PIL image."""
    model = load_model()
    x = _pil_to_tensor(pil)
    with torch.no_grad():
        return model.encode(x)


def generate_face() -> str:
    """Sample one face from N(0, I). Returns base64 PNG."""
    model = load_model()
    with torch.no_grad():
        img = model.sample(1, DEVICE)
    return _tensor_to_b64(img)


def interpolate_faces(pil_a: Image.Image, pil_b: Image.Image, steps: int = 8) -> list[str]:
    """Linearly interpolate between two faces. Returns list of base64 PNGs."""
    from src.interpolate import interpolate
    model = load_model()
    ta = _pil_to_tensor(pil_a)
    tb = _pil_to_tensor(pil_b)
    frames = interpolate(model, ta, tb, steps=steps, device=DEVICE)
    return [_tensor_to_b64(f) for f in frames]
