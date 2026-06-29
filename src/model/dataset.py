import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Convención de clases binarias
CLASS_NAMES: Dict[int, str] = {0: "healthy", 1: "diseased"}
HEALTHY_KEYWORD = "healthy"

# Normalización ImageNet: usamos pesos preentrenados en ImageNet, así que las
# imágenes deben normalizarse con las mismas estadísticas con que se entrenó.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_samples(root_dir: str, max_per_class: int = 0) -> Tuple[List[Tuple[str, int]], Dict]:
    """Recorre subcarpetas y asigna etiqueta binaria.

    Regla del colapso:
        - nombre de carpeta contiene 'healthy' -> 0 (sano)
        - cualquier otra                        -> 1 (enfermo)

    max_per_class: si > 0, toma como máximo N imágenes por carpeta (submuestreo
    reproducible). Útil para entrenar en CPU sin usar las ~54k imágenes completas.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {root}")

    rng = random.Random(42)  # submuestreo reproducible, independiente del seed global
    samples: List[Tuple[str, int]] = []
    per_folder: Dict[str, Dict] = {}
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        label = 0 if HEALTHY_KEYWORD in class_dir.name.lower() else 1
        imgs = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
        if max_per_class > 0 and len(imgs) > max_per_class:
            imgs = rng.sample(imgs, max_per_class)
        for p in imgs:
            samples.append((str(p), label))
        per_folder[class_dir.name] = {"label": label, "count": len(imgs)}

    if not samples:
        raise RuntimeError(
            f"No se encontraron imágenes en {root}. "
            "Verifica que DATA_DIR apunta a la carpeta que contiene las clases de PlantVillage."
        )
    return samples, per_folder


def class_distribution(samples: List[Tuple[str, int]]) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    for _, label in samples:
        counts[label] += 1
    return counts


def stratified_split(samples, val_split, test_split, seed):
    """Split train/val/test estratificado por etiqueta (preserva proporción enfermo/sano)."""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    p_tv, p_test, y_tv, y_test = train_test_split(
        paths, labels, test_size=test_split, stratify=labels, random_state=seed
    )
    val_relative = val_split / (1.0 - test_split)
    p_train, p_val, y_train, y_val = train_test_split(
        p_tv, y_tv, test_size=val_relative, stratify=y_tv, random_state=seed
    )
    return (
        list(zip(p_train, y_train)),
        list(zip(p_val, y_val)),
        list(zip(p_test, y_test)),
    )


def compute_class_weights(samples) -> torch.Tensor:
    """Pesos inversos a la frecuencia, para compensar el desbalance en la loss.

    Al colapsar PlantVillage quedan muchas más imágenes 'enfermo' que 'sano',
    por lo que penalizamos más los errores en la clase minoritaria.
    """
    counts = class_distribution(samples)
    total = sum(counts.values())
    weights = [total / (2 * counts[c]) if counts[c] > 0 else 0.0 for c in (0, 1)]
    return torch.tensor(weights, dtype=torch.float)


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class PlantDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:  # imagen corrupta / formato inválido
            raise RuntimeError(f"No se pudo leer la imagen {path}: {e}")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(config):
    """Orquesta: descubre datos -> split estratificado -> dataloaders + metadatos."""
    set_seed(config.seed)
    samples, per_folder = build_samples(config.data_dir, config.max_per_class)
    train_s, val_s, test_s = stratified_split(
        samples, config.val_split, config.test_split, config.seed
    )
    train_tf, eval_tf = build_transforms(config.image_size)

    train_loader = DataLoader(
        PlantDataset(train_s, train_tf), batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        PlantDataset(val_s, eval_tf), batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        PlantDataset(test_s, eval_tf), batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    info = {
        "n_train": len(train_s),
        "n_val": len(val_s),
        "n_test": len(test_s),
        "train_dist": class_distribution(train_s),
        "val_dist": class_distribution(val_s),
        "test_dist": class_distribution(test_s),
        "class_weights": compute_class_weights(train_s).tolist(),
        "per_folder": per_folder,
    }
    return train_loader, val_loader, test_loader, info
