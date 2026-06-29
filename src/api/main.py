import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import HealthResponse, PredictResponse
from src.business.rules import apply_business_decision
from src.config import Config
from src.model.inference import (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, Predictor,
                                 get_device)

config = Config()

# Estado de la app: el predictor se carga en el arranque y se reutiliza.
state = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: carga única del modelo en memoria
    state["predictor"] = Predictor()
    yield
    # shutdown: liberamos la referencia
    state["predictor"] = None


app = FastAPI(
    title="Plant Health API",
    description="Clasificación binaria sano/enfermo sobre imagen o video agrícola.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health():
    """Verifica que el servicio está vivo y el modelo cargado."""
    predictor = state["predictor"]
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        device=str(get_device()),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    responses={400: {"description": "Archivo inválido o no soportado"}},
)
async def predict(file: UploadFile = File(...)):
    """Recibe una imagen o video y devuelve la predicción + decisión de negocio.

    El tipo se detecta por la extensión del archivo. Imágenes se procesan en
    memoria; los videos se guardan en un temporal porque OpenCV requiere un path.
    """
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No se recibió ningún archivo.")

    suffix = Path(file.filename).suffix.lower()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    if suffix in IMAGE_EXTENSIONS:
        try:
            pil = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Imagen inválida o corrupta.")
        result = predictor.predict_image(pil)

    elif suffix in VIDEO_EXTENSIONS:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            result = predictor.predict_video(tmp_path)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo procesar el video: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:
        raise HTTPException(
            status_code=400,
            detail=(f"Extensión no soportada: {suffix or 'desconocida'}. "
                    f"Imágenes: {sorted(IMAGE_EXTENSIONS)} | "
                    f"Videos: {sorted(VIDEO_EXTENSIONS)}"),
        )

    return apply_business_decision(result, config.confidence_threshold)
