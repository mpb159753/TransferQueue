FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if any needed for ray or others)
# Ray often needs some basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    rsync \
    ssh \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    "ray[default]" \
    "tensordict>=0.10.0" \
    "pyzmq" \
    "hydra-core" \
    "numpy<2.0.0" \
    "msgspec" \
    "psutil" \
    "torch" 
    # Torch is likely needed for tensordict, though not explicitly in the small requirements.txt above? 
    # tensordict usually depends on torch. Let's include torch just in case, or relie on tensordict to pull it.
    # Usually tensordict requires torch.

# Clean up
RUN rm -rf /root/.cache/pip
