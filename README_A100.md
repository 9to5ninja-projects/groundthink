# GroundThink A100 Training Guide

## 1. Setup on Vast.ai / RunPod
Use a PyTorch 2.1+ Template (e.g., RunPod Pytorch 2.1 or vast.ai generic).

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Running the Training
Run the training script directly. It is pre-configured for A100 80GB.
```bash
python train_groundthink_a100.py
```

## 3. Configuration
- **Batch Size**: 8 (Safe for 80GB VRAM)
- **Gradient Accumulation**: 16 (Effective Batch Size ~128)
- **Context Length**: 2048
- **Precision**: BF16 (Native)

## 4. Checkpoints
Checkpoints are saved to `checkpoints/groundthink_1B_A100/` every 50 optimizer steps.
Every checkpoint is a full `.pt` file compatible with `torch.load()`.
