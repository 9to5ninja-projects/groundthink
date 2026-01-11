# Use NVIDIA's PyTorch container as base (CUDA, cuDNN, PyTorch pre-installed)
# This avoids 99% of "Linux Nightmare" driver issues.
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace

# Install remaining python dependencies
# (Torch, NumPy, etc. are already in the base image)
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    bitsandbytes \
    accelerate \
    wandb

# Copy your project files into the container
COPY . /workspace/

# Set the default command to run your training script
CMD ["python", "groundthink/train_groundthink_a100.py"]
