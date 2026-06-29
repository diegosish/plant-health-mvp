import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.config import Config
from src.model.architecture import build_model, set_backbone_trainable
from src.model.dataset import CLASS_NAMES, create_dataloaders, set_seed


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(labels, preds, probs):
    labels, preds, probs = np.array(labels), np.array(preds), np.array(probs)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall_diseased": recall_score(labels, preds, pos_label=1, zero_division=0),
        "recall_healthy": recall_score(labels, preds, pos_label=0, zero_division=0),
    }
    metrics["roc_auc"] = (
        roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else float("nan")
    )
    return metrics


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train() if train else model.eval()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return all_labels, all_preds, all_probs


def plot_confusion_matrix(labels, preds, out_path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([CLASS_NAMES[0], CLASS_NAMES[1]])
    ax.set_yticklabels([CLASS_NAMES[0], CLASS_NAMES[1]])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (test)")
    thr = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Dispositivo: {device}")

    train_loader, val_loader, test_loader, info = create_dataloaders(config)
    print("Distribución de datos:")
    print(json.dumps({k: v for k, v in info.items() if k != "per_folder"},
                     indent=2, ensure_ascii=False))

    model = build_model(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=True,  # warmup: arrancamos solo entrenando la cabeza
    ).to(device)

    class_weights = torch.tensor(info["class_weights"], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Param groups: backbone con LR menor, cabeza con LR mayor. Controlamos el
    # warmup/fine-tuning con requires_grad (no recreamos el optimizer -> el
    # scheduler sigue siendo válido).
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": config.lr / 10},
        {"params": model.classifier.parameters(), "lr": config.lr},
    ], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    best_score = -1.0
    with mlflow.start_run():
        mlflow.log_params({
            "model": config.model_name, "pretrained": config.pretrained,
            "image_size": config.image_size, "batch_size": config.batch_size,
            "epochs": config.epochs, "lr": config.lr,
            "weight_decay": config.weight_decay, "freeze_epochs": config.freeze_epochs,
            "class_weights": info["class_weights"], "train_dist": info["train_dist"],
        })

        for epoch in range(1, config.epochs + 1):
            if epoch == config.freeze_epochs + 1:
                print("Descongelando backbone para fine-tuning...")
                set_backbone_trainable(model, True)

            train_m = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
            val_m = run_epoch(model, val_loader, criterion, optimizer, device, scaler, False)
            scheduler.step()

            print(f"[{epoch}/{config.epochs}] "
                  f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} "
                  f"val_f1={val_m['f1']:.4f} val_recall_diseased={val_m['recall_diseased']:.4f}")

            mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()}, step=epoch)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()}, step=epoch)

            # priorizamos recall de enfermo, con F1 como balance
            score = 0.5 * val_m["recall_diseased"] + 0.5 * val_m["f1"]
            if score > best_score:
                best_score = score
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": vars(config),
                    "class_names": CLASS_NAMES,
                    "val_metrics": val_m,
                }, config.model_path)
                print(f"  -> Nuevo mejor modelo guardado (score={score:.4f})")

        # ---- Evaluación final en TEST con el mejor checkpoint ----
        ckpt = torch.load(config.model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_m = run_epoch(model, test_loader, criterion, optimizer, device, scaler, False)
        print("\n=== TEST ===")
        print(json.dumps(test_m, indent=2))
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

        labels, preds, _ = collect_predictions(model, test_loader, device)
        cm_path = Path(config.model_dir) / "confusion_matrix.png"
        plot_confusion_matrix(labels, preds, cm_path)
        mlflow.log_artifact(str(cm_path))

        report = classification_report(
            labels, preds, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]], zero_division=0
        )
        report_path = Path(config.model_dir) / "classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))
        print(report)

        mlflow.log_artifact(config.model_path)

    print(f"\nModelo final en: {config.model_path}")


if __name__ == "__main__":
    main()
