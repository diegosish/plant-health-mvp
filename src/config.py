"""Configuración centralizada del proyecto.

Toda la parametrización se lee de variables de entorno (con defaults sensatos),
lo que permite cambiar el comportamiento sin tocar el código: clave para operar
el MVP en distintos entornos (local, CI, nube).
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ---- Datos ----
    data_dir: str = os.getenv("DATA_DIR", "data/PlantVillage")
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
    val_split: float = float(os.getenv("VAL_SPLIT", "0.15"))
    test_split: float = float(os.getenv("TEST_SPLIT", "0.15"))
    max_per_class: int = int(os.getenv("MAX_PER_CLASS", "0"))  # 0 = usar todas las imágenes

    # ---- Entrenamiento ----
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    epochs: int = int(os.getenv("EPOCHS", "15"))
    lr: float = float(os.getenv("LEARNING_RATE", "0.001"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.0001"))
    freeze_epochs: int = int(os.getenv("FREEZE_EPOCHS", "3"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))
    seed: int = int(os.getenv("SEED", "42"))

    # ---- Modelo ----
    num_classes: int = 2  # 0 = sano, 1 = enfermo
    model_name: str = os.getenv("MODEL_NAME", "efficientnet-b0")
    pretrained: bool = os.getenv("PRETRAINED", "true").lower() == "true"
    model_dir: str = os.getenv("MODEL_DIR", "models")
    model_filename: str = os.getenv("MODEL_FILENAME", "best_model.pth")

    # ---- MLflow ----
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT", "plant-health-binary")

    # ---- Reglas de negocio ----
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))

    @property
    def model_path(self) -> str:
        return str(Path(self.model_dir) / self.model_filename)
