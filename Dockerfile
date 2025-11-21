# RunPod Serverless - Qwen-Image Deployment
# Base image: CUDA 11.8 + Python 3.10
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

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY handler.py .
COPY runpod.toml .
COPY src/ ./src/

# Create volume mount point
RUN mkdir -p /runpod-volume

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import runpod; print(f'RunPod: {runpod.__version__}')"

# Health check script
RUN echo '#!/bin/bash\npython -c "import torch; print(f\"CUDA Available: {torch.cuda.is_available()}\")"' > /health_check.sh && \
    chmod +x /health_check.sh

# Expose port (optional, for local testing)
EXPOSE 8000

# RunPod entrypoint
CMD ["python", "-u", "handler.py"]