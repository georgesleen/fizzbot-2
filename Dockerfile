FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    git \
    make \
    pkg-config \
    libssl-dev \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

WORKDIR /workspace

COPY pyproject.toml uv.lock* /workspace/
RUN uv sync

COPY . /workspace

WORKDIR /workspace/discord-bot

CMD ["cargo", "run", "--release"]
