# RunPod Serverless - Qwen-Image Deployment
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="Qwen-Image model RunPod Serverless deployment"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/runpod-volume/qwen_image
ENV HF_HOME=/runpod-volume/qwen_image
ENV TORCH_HOME=/runpod-volume/torch_cache
ENV CUDA_VISIBLE_DEVICES=0

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create volume mount point
RUN mkdir -p /runpod-volume

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import runpod; print(f'RunPod: {runpod.__version__}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# RunPod entrypoint
CMD ["python", "-u", "handler.py"]