import sys
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.config import Config
from src.model.architecture import build_model
from src.model.dataset import CLASS_NAMES, build_transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    """Carga el modelo una sola vez y expone predicción de imagen y video.

    Pensado para instanciarse al arrancar la API (carga única de pesos) y
    reutilizarse en cada request, evitando recargar el modelo por petición.
    """

    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[torch.device] = None):
        config = Config()
        self.model_path = model_path or config.model_path
        self.device = device or get_device()

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {self.model_path}. "
                "Entrena primero con: python -m src.model.train"
            )

        # weights_only=False: es nuestro propio checkpoint y guarda metadatos
        # (config, class_names) además de los pesos. El try/except cubre
        # versiones de PyTorch que no aceptan el argumento.
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.model_path, map_location=self.device)

        self.class_names = ckpt.get("class_names", CLASS_NAMES)
        ckpt_config = ckpt.get("config", {})
        self.image_size = ckpt_config.get("image_size", config.image_size)
        self.model_version = ckpt_config.get("model_name", config.model_name)

        # Reconstruimos la arquitectura SIN pesos de ImageNet (cargamos los nuestros).
        self.model = build_model(num_classes=2, pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        # Mismo preprocessing que en evaluación -> evita train/serve skew.
        _, self.transform = build_transforms(self.image_size)

    # ---------- helpers internos ----------
    def _to_pil(self, source: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            # se asume array RGB (los frames de OpenCV se convierten antes de llamar)
            return Image.fromarray(source).convert("RGB")
        return Image.open(source).convert("RGB")  # path

    @torch.no_grad()
    def _predict_tensor(self, pil_image: Image.Image):
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        return idx, float(probs[idx].item()), float(probs[1].item())

    # ---------- API pública ----------
    def predict_image(self, source) -> Dict:
        start = time.time()
        try:
            pil = self._to_pil(source)
        except Exception as e:
            raise RuntimeError(f"No se pudo leer la imagen: {e}")

        idx, confidence, prob_diseased = self._predict_tensor(pil)
        elapsed_ms = round((time.time() - start) * 1000, 1)

        return {
            "input_type": "image",
            "prediction": {
                "label": self.class_names[idx],
                "is_healthy": idx == 0,
                "confidence": round(confidence, 4),
            },
            "probabilities": {
                self.class_names[0]: round(1 - prob_diseased, 4),
                self.class_names[1]: round(prob_diseased, 4),
            },
            "metadata": {
                "model_version": self.model_version,
                "inference_time_ms": elapsed_ms,
            },
        }

    def predict_video(self, video_path: Union[str, Path],
                      frame_interval: int = 15,
                      max_frames: int = 64,
                      diseased_threshold: float = 0.5) -> Dict:
        """Clasifica un video muestreando frames y agregando los resultados.

        frame_interval: procesa 1 de cada N frames (muestreo temporal).
        max_frames: tope de frames a analizar (evita videos largos costosos).
        diseased_threshold: ratio de frames enfermos para declarar el video enfermo.
        """
        import cv2  # import local: solo se necesita en la ruta de video

        start = time.time()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        counts = {0: 0, 1: 0}
        confidences = []
        analyzed = 0
        frame_idx = 0

        while analyzed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV entrega BGR
                idx, confidence, _ = self._predict_tensor(self._to_pil(rgb))
                counts[idx] += 1
                confidences.append(confidence)
                analyzed += 1
            frame_idx += 1

        cap.release()

        if analyzed == 0:
            raise RuntimeError("No se pudo analizar ningún frame del video.")

        diseased_ratio = counts[1] / analyzed
        verdict_idx = 1 if diseased_ratio >= diseased_threshold else 0
        elapsed_ms = round((time.time() - start) * 1000, 1)

        return {
            "input_type": "video",
            "frames_analyzed": analyzed,
            "frame_interval": frame_interval,
            "summary": {
                "verdict": self.class_names[verdict_idx],
                "is_healthy": verdict_idx == 0,
                "diseased_ratio": round(diseased_ratio, 4),
                "avg_confidence": round(float(np.mean(confidences)), 4),
            },
            "per_class_frames": {
                self.class_names[0]: counts[0],
                self.class_names[1]: counts[1],
            },
            "metadata": {
                "model_version": self.model_version,
                "inference_time_ms": elapsed_ms,
                "total_frames": total_frames,
                "fps": round(fps, 2),
            },
        }

    def predict_file(self, path: Union[str, Path]) -> Dict:
        """Detecta por extensión si la entrada es imagen o video y enruta."""
        suffix = Path(path).suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return self.predict_video(path)
        if suffix in IMAGE_EXTENSIONS:
            return self.predict_image(path)
        raise ValueError(
            f"Extensión no soportada: {suffix}. "
            f"Imágenes: {sorted(IMAGE_EXTENSIONS)} | Videos: {sorted(VIDEO_EXTENSIONS)}"
        )


def main():
    import json
    if len(sys.argv) < 2:
        print("Uso: python -m src.model.inference <ruta_imagen_o_video>")
        sys.exit(1)

    predictor = Predictor()
    result = predictor.predict_file(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
