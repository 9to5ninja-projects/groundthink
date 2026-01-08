# GroundThink V2: Scientific Run Instructions

## 1. Setup on A100 (Vast.ai)

```bash
# SSH into the machine (from your local VS Code or Terminal)
ssh root@<ip_address> -p <port>
```

```bash
# On the Remote Server:
cd /root/
git clone https://github.com/9to5ninja-projects/groundthink.git
cd groundthink

# Install Dependencies
pip install -r requirements.txt
pip install tensorboard
```

## 2. Dataset Preparation (The "Editor's Cut")

This step downloads FineWeb/PG19/TinyStories, filters out "AI Slop" and Refusals, and saves a clean `arrow` dataset.

```bash
python3 prepare_v2_dataset.py
```
*Expected: Creates `groundthink_v2_dataset` directory (~1.2GB).*

## 3. Training (125M / Hybrid)

Runs the training loop with:
- Split Optimizer (1.8x LR for Mamba projections)
- Gradient Clipping (1.0)
- "Secret Code" Recall Probe every 200 steps
- Checkpoints every 500 steps

```bash
python3 train_groundthink_v2_125m.py
```

## 4. Monitoring

To watch the training loss and generation in real-time without downloading logs, use the separate verify script (optional) or just watch the terminal output which now includes `TPS`, `Loss`, and `Probe` outputs.

```bash
# In a separate terminal
watch -n 10 "tail -n 20 logs/groundthink_v2_small/events*" 
# OR just trust the main terminal output
```
