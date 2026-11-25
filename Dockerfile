# Lightweight Python image — no CUDA/PyTorch needed for ONNX CPU inference
FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OpenCV (used by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first so ultralytics does not pull ~5GB of CUDA/NVIDIA wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision && \
    pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements.txt

# Copy application source
COPY . .

# Ensure model, log, and shared backend cache directories exist
RUN mkdir -p /app/models /app/logs /backend/.models_cache

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
