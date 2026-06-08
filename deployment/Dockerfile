FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements-docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.11.0 torchvision==0.26.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-docker.txt

# Copy source code
COPY . .

# Warm-up: Pre-download model weights during build to fix 'Cold-Start Latency'
RUN python scripts/warmup.py

# Expose ports for Streamlit and FastAPI
EXPOSE 8501
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run local dev server
CMD ["streamlit", "run", "src/ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
