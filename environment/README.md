# Formal experiment environment

Create the Linux/CUDA environment from the repository root:

```bash
conda env create -f environment/conda-linux-64.yml
conda activate publication-cem
python scripts/prefetch_evaluation_assets.py
python scripts/capture_environment.py
python scripts/verify_release.py
```

The formal run used Python 3.11, PyTorch 2.2.2, torchvision 0.17.2 and CUDA
12.1. Do not use the macOS development interpreter for reported latency or GPU
memory measurements. Create `evaluation_assets.json` before launching attacks
so all workers use identical LPIPS and FaceNet weights.
