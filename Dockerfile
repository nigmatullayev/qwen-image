### RunPod Serverless - Qwen-Image Deployment
##FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
##
### Metadata
##LABEL maintainer="your-email@example.com"
##LABEL description="Qwen-Image model RunPod Serverless deployment"
##
### Environment variables
##ENV PYTHONUNBUFFERED=1
##ENV DEBIAN_FRONTEND=noninteractive
##ENV TRANSFORMERS_CACHE=/runpod-volume/qwen_image
##ENV HF_HOME=/runpod-volume/qwen_image
##ENV TORCH_HOME=/runpod-volume/torch_cache
##ENV CUDA_VISIBLE_DEVICES=0
##
### Working directory
##WORKDIR /app
##
### System dependencies
##RUN apt-get update && apt-get install -y \
##    git \
##    wget \
##    curl \
##    vim \
##    build-essential \
##    && apt-get clean \
##    && rm -rf /var/lib/apt/lists/*
##
### Copy requirements first (cache optimization)
##COPY requirements.txt .
##
### Install Python dependencies
##RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
##    pip install --no-cache-dir -r requirements.txt
##
### Copy all project files
##COPY . .
##
### Create volume mount point
##RUN mkdir -p /runpod-volume
##
### Verify installation
##RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" && \
##    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
##    python -c "import runpod; print(f'RunPod: {runpod.__version__}')"
##
### Health check
##HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
##    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
##
### RunPod entrypoint
##CMD ["python", "-u", "handler.py"]
#
## ============================================
## STAGE 1: Builder - Install dependencies
## ============================================
#FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 AS builder
#
#ENV PIP_NO_CACHE_DIR=1
#ENV PYTHONDONTWRITEBYTECODE=1
#
#WORKDIR /build
#
## Install dependencies in a separate location
#COPY requirements.txt .
#RUN pip install --no-cache-dir --target=/install -r requirements.txt && \
#    find /install -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
#    find /install -type f -name "*.pyc" -delete && \
#    find /install -type f -name "*.pyo" -delete
#
## ============================================
## STAGE 2: Runtime - Minimal final image
## ============================================
#FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
#
#ENV PYTHONUNBUFFERED=1
#ENV TRANSFORMERS_CACHE=/runpod-volume/qwen_image
#ENV HF_HOME=/runpod-volume/qwen_image
#ENV TORCH_HOME=/runpod-volume/torch_cache
#
#WORKDIR /app
#
## Copy only installed packages from builder
#COPY --from=builder /install /usr/local/lib/python3.10/site-packages/
#
## Copy application code
#COPY handler.py .
#COPY src/ ./src/
#
## Create volume mount
#RUN mkdir -p /runpod-volume
#
## Final cleanup
#RUN apt-get autoremove -y && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache
#
## Verify
#RUN python -c "import torch, transformers, runpod"
#
#CMD ["python", "-u", "handler.py"]

# RunPod Serverless - Qwen-Image (No src/ directory)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="Qwen-Image RunPod Serverless (standalone)"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/runpod-volume/qwen_image
ENV HF_HOME=/runpod-volume/qwen_image
ENV TORCH_HOME=/runpod-volume/torch_cache
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy only handler.py (standalone version)
COPY handler.py .

# Create volume mount
RUN mkdir -p /runpod-volume

# Cleanup
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# Verify installation
RUN python -c "import torch, transformers, runpod; print('Dependencies OK')"

# RunPod entrypoint
CMD ["python", "-u", "handler.py"]