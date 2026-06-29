from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ---- bloque compartido ----
class BusinessDecision(BaseModel):
    action: str = Field(..., description="approve | flag_for_review | manual_review")
    reason: str = Field(..., description="Justificación legible de la decisión")
    threshold_applied: float = Field(..., description="Umbral de confianza aplicado")


# ---- respuesta de IMAGEN ----
class Prediction(BaseModel):
    label: str
    is_healthy: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class Probabilities(BaseModel):
    healthy: float = Field(..., ge=0.0, le=1.0)
    diseased: float = Field(..., ge=0.0, le=1.0)


class ImageMetadata(BaseModel):
    # protected_namespaces=(): permite el campo 'model_version' sin warning en Pydantic v2
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    inference_time_ms: float


class ImageResponse(BaseModel):
    input_type: Literal["image"]
    prediction: Prediction
    probabilities: Probabilities
    business_decision: BusinessDecision
    metadata: ImageMetadata


# ---- respuesta de VIDEO ----
class VideoSummary(BaseModel):
    verdict: str
    is_healthy: bool
    diseased_ratio: float = Field(..., ge=0.0, le=1.0)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)


class PerClassFrames(BaseModel):
    healthy: int
    diseased: int


class VideoMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    inference_time_ms: float
    total_frames: int
    fps: float


class VideoResponse(BaseModel):
    input_type: Literal["video"]
    frames_analyzed: int
    frame_interval: int
    summary: VideoSummary
    business_decision: BusinessDecision
    per_class_frames: PerClassFrames
    metadata: VideoMetadata


# ---- unión discriminada de la respuesta de /predict ----
PredictResponse = Annotated[
    Union[ImageResponse, VideoResponse],
    Field(discriminator="input_type"),
]


# ---- infraestructura ----
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    device: str
