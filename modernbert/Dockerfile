FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Pin torch to the latest cu124 build (2.6.0+cu124); cu124 index tops out there.
# PyPI now carries cu130+ which requires CUDA 13.0 / driver 570+ — pinning avoids that.
# torch is excluded from requirements.txt to prevent a subsequent PyPI reinstall.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        "torch==2.6.0+cu124" && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 7998
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7998"]
