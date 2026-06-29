FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- Dependencias del sistema mínimas ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- PyTorch CPU-only ----
# Optimización clave: instalamos torch/torchvision desde el índice CPU de PyTorch.

RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ---- Resto de dependencias ----
# torch/torchvision ya quedan satisfechos arriba, pip no los reinstala.
# Copiamos requirements primero para aprovechar el cache de capas de Docker:
# si solo cambia el código, esta capa no se reconstruye.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Código y modelo entrenado ----
COPY src/ ./src/
COPY models/ ./models/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV MODEL_DIR=models \
    MODEL_FILENAME=best_model.pth \
    CONFIDENCE_THRESHOLD=0.85

EXPOSE 8000

# Arranque del servicio (sin --reload: esto es runtime de producción, no desarrollo).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
