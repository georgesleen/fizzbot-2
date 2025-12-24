FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

COPY pyproject.toml uv.lock* /workspace/
RUN uv sync
RUN uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

COPY . /workspace
