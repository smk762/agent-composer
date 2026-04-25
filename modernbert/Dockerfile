FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CUDA-enabled PyTorch first, then the rest
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch>=2.5.0 && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 7998
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7998"]
