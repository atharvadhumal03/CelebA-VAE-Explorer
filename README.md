# LatentLens

A Variational Autoencoder (VAE) trained on CelebA that learns a structured semantic latent space of faces.

## What it does

- **Reconstruct** — encode a face image and decode it back; see how much detail the latent space preserves
- **Search** — find the top-5 most visually and semantically similar faces from CelebA using FAISS
- **Explore** — navigate an interactive t-SNE map of the latent space, coloured by face attributes (smiling, age, hair colour, etc.)
- **Generate / Interpolate** — sample random faces from the prior, or smoothly interpolate between two faces through latent space

## Status

> Work in progress. This README will be updated at project completion.

## Tech Stack

- **Model:** PyTorch — CNN-based VAE, 128-dim latent space
- **Dataset:** CelebA (~202k face images, 40 binary attributes)
- **Similarity search:** FAISS
- **Experiment tracking:** Weights & Biases
- **Backend:** FastAPI
- **Frontend:** React
- **Deployment:** Docker on HuggingFace Spaces

## Project Structure

```
celeba-vae-explorer/
├── src/          # Model, dataset, training, evaluation, search, interpolation
├── scripts/      # SLURM job scripts for HPC
├── app/
│   ├── backend/  # FastAPI app and inference logic
│   └── frontend/ # React app (4 tabs)
├── configs/      # Hyperparameter configs
├── notebooks/    # EDA and quick experiments
├── docs/         # Architecture, pipeline, deployment, and integration docs
└── outputs/      # Checkpoints, generated images (not committed)
```

## Docs

- [Architecture](docs/architecture.md) — VAE model design and layer details
- [Pipeline](docs/pipeline.md) — Training, evaluation, smoke testing, and ablations
- [Frontend](docs/frontend.md) — React app and API contracts
- [Deployment](docs/deployment.md) — Docker, HuggingFace Spaces, and model hosting
- [Amalgam](docs/amalgam.md) — How all subsystems connect end-to-end

## Training

All training runs on the Northeastern HPC cluster (SLURM). See [docs/pipeline.md](docs/pipeline.md) for the full workflow.

```bash
ssh dhumal.a@login.explorer.northeastern.edu
cd celeba-vae-explorer && git pull
conda activate <env>
sbatch scripts/smoke_test.sh   # verify pipeline first
sbatch scripts/train.sh        # full training run
```

## Local Development

Local machines are for code editing only — no model runs locally.

```bash
# Backend
cd app/backend && uvicorn main:app --reload

# Frontend
cd app/frontend && npm install && npm run dev
```