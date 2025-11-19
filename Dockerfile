FROM runpod/pytorch:2.1.2-py3.10-cuda12.1.1-devel

WORKDIR /workspace

# Install system tools
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install dependencies
RUN pip install "diffusers[torch]" transformers accelerate safetensors
RUN pip install pillow runpod
RUN pip install huggingface-hub hf-transfer --upgrade

# Use network volume for HuggingFace cache
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

# Copy handler
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
