# PrimordiumEvolv Phase 0 - MIT SEAL + Ray 2.4.0 + Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV RAY_DISABLE_IMPORT_WARNING=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MIT SEAL from specific commit
RUN pip install git+https://github.com/mit-seal/seal-rl.git@8fe0c2e || \
    echo "MIT SEAL installation failed - using fallback implementation"

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY docs/ ./docs/

# Make scripts executable
RUN chmod +x scripts/*.py

# Create output directories
RUN mkdir -p runs logs

# Expose ports for Ray and WandB
EXPOSE 8265 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ray, torch, numpy; print('OK')" || exit 1

# Default command
CMD ["python", "scripts/run_baseline.py", "--help"]