import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
from omegaconf import OmegaConf

from src.dataset import get_dataloaders
from src.evaluate import encode_dataset
from src.model import VAE


def build_index(latents: np.ndarray) -> faiss.IndexFlatL2:
    D = latents.shape[1]
    index = faiss.IndexFlatL2(D)
    index.add(latents.astype(np.float32))
    return index


def save_index(index: faiss.IndexFlatL2, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: str) -> faiss.IndexFlatL2:
    return faiss.read_index(path)


def search(index: faiss.IndexFlatL2, query_latent: np.ndarray, k: int = 5):
    """
    query_latent : [D] or [1, D]
    Returns distances [k] and indices [k] into the indexed dataset.
    """
    q = query_latent.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(q, k)
    return distances[0], indices[0]


def encode_single(model: VAE, image_tensor: torch.Tensor, device) -> np.ndarray:
    """image_tensor: [1, 3, H, W] normalised to [-1, 1]."""
    model.eval()
    with torch.no_grad():
        mu = model.encode(image_tensor.to(device))
    return mu.cpu().numpy().squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents", default=None, help="Path to precomputed latent_vectors.npy")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(latent_dim=cfg.model.latent_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.latents and Path(args.latents).exists():
        latents = np.load(args.latents)
        print(f"Loaded precomputed latents: {latents.shape}")
    else:
        print("Encoding val dataset to build FAISS index...")
        _, val_loader, _ = get_dataloaders(
            root=cfg.data.root,
            image_size=cfg.data.image_size,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
        latents, attrs = encode_dataset(model, val_loader, device)
        out_dir = Path("outputs/latents")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "latent_vectors.npy", latents)
        np.save(out_dir / "attributes.npy", attrs)
        print(f"Saved latents to outputs/latents/")

    index = build_index(latents)
    index_path = Path("outputs/faiss/faiss_index.bin")
    save_index(index, index_path)
    print(f"FAISS index built ({index.ntotal} vectors) and saved to {index_path}")


if __name__ == "__main__":
    main()
