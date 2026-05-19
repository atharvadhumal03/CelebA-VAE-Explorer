import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torchvision.utils import make_grid


CELEBA_ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def save_reconstruction_grid(originals: torch.Tensor, reconstructions: torch.Tensor, path: Path, nrow: int = 8):
    """
    originals, reconstructions : [N, 3, H, W] in [-1, 1]
    Saves a side-by-side grid: top row = originals, bottom row = reconstructions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = torch.cat([originals, reconstructions], dim=0)
    grid = make_grid(combined * 0.5 + 0.5, nrow=nrow, padding=2)
    img = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(nrow * 1.5, 3))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_interpolation_strip(frames: list[torch.Tensor], path: Path):
    """frames: list of [1, 3, H, W] tensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    strip = torch.cat(frames, dim=0)
    grid = make_grid(strip * 0.5 + 0.5, nrow=len(frames), padding=2)
    img = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(len(frames) * 1.5, 1.5))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compute_tsne(latents: np.ndarray, perplexity: float = 30, n_iter: int = 1000) -> np.ndarray:
    """Returns [N, 2] 2D coordinates."""
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=1)
    return tsne.fit_transform(latents)


def save_tsne_plot(coords: np.ndarray, attrs: np.ndarray, attr_idx: int, out_path: Path):
    attr_name = CELEBA_ATTR_NAMES[attr_idx]
    labels = attrs[:, attr_idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for val, colour, label in [(0, "#aec6cf", f"no {attr_name}"), (1, "#ff6961", attr_name)]:
        mask = labels == val
        plt.scatter(coords[mask, 0], coords[mask, 1], c=colour, s=1, alpha=0.5, label=label)
    plt.legend(markerscale=6, loc="best")
    plt.title(f"t-SNE coloured by {attr_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_tsne_json(coords: np.ndarray, attrs: np.ndarray, image_indices: list, out_path: Path):
    """Serialise t-SNE data for the FastAPI /tsne endpoint."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coords": coords.tolist(),
        "attributes": attrs.tolist(),
        "attribute_names": CELEBA_ATTR_NAMES,
        "image_indices": image_indices,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents",    required=True, help="Path to latent_vectors.npy")
    parser.add_argument("--attributes", required=True, help="Path to attributes.npy")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--n-iter",     type=int,   default=1000)
    args = parser.parse_args()

    latents = np.load(args.latents)
    attrs   = np.load(args.attributes)
    N = len(latents)
    print(f"Computing t-SNE on {N} latent vectors...")

    coords = compute_tsne(latents, perplexity=args.perplexity, n_iter=args.n_iter)

    out_dir = Path("outputs/tsne")
    # Export JSON for the backend
    export_tsne_json(coords, attrs, list(range(N)), out_dir / "tsne_data.json")
    print(f"Saved tsne_data.json to {out_dir}/")

    # Save plots for a selection of salient attributes
    salient = ["Smiling", "Male", "Blond_Hair", "Young", "Eyeglasses", "Bald"]
    for attr_name in salient:
        if attr_name in CELEBA_ATTR_NAMES:
            idx = CELEBA_ATTR_NAMES.index(attr_name)
            save_tsne_plot(coords, attrs, idx, out_dir / f"tsne_{attr_name}.png")
    print(f"Saved t-SNE plots to {out_dir}/")


if __name__ == "__main__":
    main()
