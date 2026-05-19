import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchmetrics.image.fid import FrechetInceptionDistance

from src.dataset import get_dataloaders
from src.model import VAE


def compute_fid(model: VAE, real_loader, n_samples: int, device) -> float:
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    model.eval()

    # Real images
    real_count = 0
    for imgs, _ in real_loader:
        if real_count >= n_samples:
            break
        imgs = imgs.to(device)
        # FID expects [0, 1] uint8 or float; unnormalize from [-1,1]
        imgs_01 = (imgs * 0.5 + 0.5).clamp(0, 1)
        fid.update(imgs_01, real=True)
        real_count += imgs.size(0)

    # Generated images
    gen_count = 0
    batch_size = real_loader.batch_size
    with torch.no_grad():
        while gen_count < n_samples:
            n = min(batch_size, n_samples - gen_count)
            fake = model.sample(n, device)
            fake_01 = (fake * 0.5 + 0.5).clamp(0, 1)
            fid.update(fake_01, real=False)
            gen_count += n

    return float(fid.compute())


def compute_nn_accuracy(latents: np.ndarray, attrs: np.ndarray, n_queries: int = 500, k: int = 5) -> dict:
    """
    latents : [N, D]  — encoded mu vectors
    attrs   : [N, 40] — binary attribute labels
    Returns per-attribute accuracy and macro average.
    """
    import faiss

    N, D = latents.shape
    n_queries = min(n_queries, N)
    query_idx = np.random.choice(N, n_queries, replace=False)

    index = faiss.IndexFlatL2(D)
    index.add(latents.astype(np.float32))

    # k+1 because the query itself will be the nearest neighbour
    _, I = index.search(latents[query_idx].astype(np.float32), k + 1)
    # Drop the first result (self)
    I = I[:, 1:]

    n_attrs = attrs.shape[1]
    per_attr_acc = np.zeros(n_attrs)
    for a in range(n_attrs):
        query_labels    = attrs[query_idx, a]           # [n_queries]
        neighbor_labels = attrs[I.flatten(), a].reshape(n_queries, k)  # [n_queries, k]
        match = (neighbor_labels == query_labels[:, None]).mean(axis=1)
        per_attr_acc[a] = match.mean()

    macro_avg = per_attr_acc.mean()
    return {"per_attr_acc": per_attr_acc, "macro_avg": float(macro_avg)}


def encode_dataset(model: VAE, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Encode all images in loader, return (latents [N, D], attrs [N, 40])."""
    model.eval()
    all_mu, all_attrs = [], []
    with torch.no_grad():
        for imgs, attrs in loader:
            imgs = imgs.to(device)
            mu = model.encode(imgs)
            all_mu.append(mu.cpu().numpy())
            all_attrs.append(attrs.numpy())
    return np.concatenate(all_mu), np.concatenate(all_attrs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(latent_dim=cfg.model.latent_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader, test_loader = get_dataloaders(
        root=cfg.data.root,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    loader = val_loader if args.split == "val" else test_loader

    print("Encoding dataset...")
    latents, attrs = encode_dataset(model, loader, device)
    print(f"Encoded {len(latents)} images to latent dim {latents.shape[1]}")

    print("Computing FID...")
    fid_score = compute_fid(model, loader, n_samples=cfg.evaluation.fid_n_samples, device=device)
    print(f"FID: {fid_score:.2f}")

    print("Computing nearest-neighbour semantic accuracy...")
    nn_results = compute_nn_accuracy(latents, attrs, n_queries=cfg.evaluation.nn_n_queries)
    print(f"NN macro accuracy: {nn_results['macro_avg']:.4f}")

    # Save latents for downstream use (search index, t-SNE)
    out_dir = Path("outputs/latents")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "latent_vectors.npy", latents)
    np.save(out_dir / "attributes.npy", attrs)
    print(f"Saved latents and attributes to {out_dir}/")


if __name__ == "__main__":
    main()
