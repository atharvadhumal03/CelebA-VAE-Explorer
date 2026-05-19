import argparse
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import OmegaConf

from src.dataset import get_dataloaders
from src.loss import vae_loss
from src.model import VAE


# ── W&B setup ────────────────────────────────────────────────────────────────

def init_wandb(cfg, resume_run_id: str | None = None):
    """Try live W&B; fall back to offline automatically."""
    import wandb

    # Check connectivity before initialising
    try:
        socket.setdefaulttimeout(5)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        mode = "online"
    except OSError:
        mode = "offline"
        os.environ["WANDB_MODE"] = "offline"
        print("[W&B] No connectivity detected — switching to offline mode.")

    init_kwargs = dict(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=mode,
    )
    if resume_run_id:
        init_kwargs["id"] = resume_run_id
        init_kwargs["resume"] = "must"

    run = wandb.init(**init_kwargs)
    return run


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, cfg, ckpt_dir: Path, wandb_run_id: str | None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "wandb_run_id": wandb_run_id,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, path)

    # Keep best_model.pt separately
    best_path = ckpt_dir / "best_model.pt"
    if not best_path.exists():
        torch.save(torch.load(path), best_path)
    else:
        prev_best = torch.load(best_path, map_location="cpu")
        if val_loss < prev_best["val_loss"]:
            torch.save(torch.load(path), best_path)

    # Prune old rolling checkpoints, keep last N
    keep_n = cfg.checkpointing.keep_last_n
    rolling = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
    for old in rolling[:-keep_n]:
        old.unlink()

    return path


def load_checkpoint(path: str, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["val_loss"], ckpt.get("wandb_run_id")


# ── Epoch runners ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, cfg, device, train: bool):
    model.train(train)
    total_loss = recon_sum = kl_sum = 0.0
    n_batches = 0

    with torch.set_grad_enabled(train):
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            recon, mu, logvar = model(imgs)
            loss, r_loss, kl_loss = vae_loss(recon, imgs, mu, logvar, beta=cfg.model.beta)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            recon_sum  += r_loss.item()
            kl_sum     += kl_loss.item()
            n_batches  += 1

    return total_loss / n_batches, recon_sum / n_batches, kl_sum / n_batches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Smoke-test overrides
    if args.smoke_test:
        cfg.training.epochs = cfg.smoke_test.epochs
        print(f"[smoke-test] Running {cfg.training.epochs} epochs on {cfg.smoke_test.n_images} images.")

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # Data
    smoke = args.smoke_test
    train_loader, val_loader, _ = get_dataloaders(
        root=cfg.data.root,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        smoke_test=smoke,
        smoke_n=cfg.smoke_test.n_images,
    )

    # Model, optimizer, scheduler
    model = VAE(latent_dim=cfg.model.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, betas=tuple(cfg.training.betas))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=cfg.training.scheduler_factor, patience=cfg.training.scheduler_patience
    )

    start_epoch = 0
    best_val_loss = float("inf")
    wandb_run_id = None

    # Resume
    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        start_epoch, best_val_loss, wandb_run_id = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1  # next epoch
        print(f"[train] Resumed at epoch {start_epoch}, best val loss so far: {best_val_loss:.6f}")

    # W&B
    run = init_wandb(cfg, resume_run_id=wandb_run_id)
    wandb_run_id = run.id if run else None

    import wandb

    ckpt_dir = Path(cfg.checkpointing.output_dir)

    # Verify checkpoint/resume works in smoke test (epoch 0 → save → load → continue)
    if args.smoke_test and start_epoch == 0:
        print("[smoke-test] Checkpoint/resume verification enabled.")

    for epoch in range(start_epoch, cfg.training.epochs):
        train_loss, train_recon, train_kl = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        val_loss,   val_recon,   val_kl   = run_epoch(model, val_loader,   optimizer, cfg, device, train=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # GPU utilisation
        gpu_util = 0.0
        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization(device)

        print(
            f"Epoch {epoch:04d} | "
            f"train {train_loss:.4f} (r={train_recon:.4f} kl={train_kl:.4f}) | "
            f"val {val_loss:.4f} (r={val_recon:.4f} kl={val_kl:.4f}) | "
            f"lr={current_lr:.2e}"
        )

        wandb.log({
            "epoch": epoch,
            "train/total_loss": train_loss,
            "train/recon_loss": train_recon,
            "train/kl_loss":    train_kl,
            "val/total_loss":   val_loss,
            "val/recon_loss":   val_recon,
            "val/kl_loss":      val_kl,
            "lr":               current_lr,
            "gpu_util":         gpu_util,
        }, step=epoch)

        # Checkpoint every epoch unconditionally
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, cfg, ckpt_dir, wandb_run_id)

        # Smoke test: after epoch 0, verify resume works, then continue to epoch 1
        if args.smoke_test and epoch == 0:
            saved = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pt"
            print(f"[smoke-test] Verifying resume from {saved} ...")
            load_checkpoint(str(saved), model, optimizer, scheduler, device)
            print("[smoke-test] Resume OK — checkpoint loaded successfully.")

    print("[train] Done.")
    wandb.finish()


if __name__ == "__main__":
    main()
