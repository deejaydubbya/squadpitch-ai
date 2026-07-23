from __future__ import annotations

# mypy: ignore-errors
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from squadpitch_ai.brand_quality.dataset import BrandQualityDataset, split_dataset
from squadpitch_ai.brand_quality.models import QUALITY_LABELS

try:  # pragma: no cover - exercised only when optional ml extra is installed.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    vocab_size: int = 1024
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 1e-3
    patience: int = 3
    mixed_precision: bool = False


def train_pytorch_quality_model(
    dataset: BrandQualityDataset,
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError("PyTorch is not installed. Install squadpitch-ai[ml] to train.")

    cfg = config or TrainingConfig()
    set_reproducible_seed(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_dataset(dataset)
    train_loader = DataLoader(
        BrandQualityTorchDataset(splits["train"], cfg.vocab_size),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        BrandQualityTorchDataset(splits["validation"], cfg.vocab_size), batch_size=cfg.batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrandQualityClassifier(cfg.vocab_size, len(QUALITY_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

    best_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler, cfg)
        validation_loss = validate_epoch(model, validation_loader, loss_fn, device)
        scheduler.step(validation_loss)
        history.append({"epoch": epoch, "trainLoss": train_loss, "validationLoss": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "datasetVersion": dataset.dataset_version,
                    "labels": list(QUALITY_LABELS),
                },
                output_dir / "brand_quality_model.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break
    report = {
        "modelVersion": "brand-quality-neural-shadow.v1",
        "datasetVersion": dataset.dataset_version,
        "device": str(device),
        "mixedPrecision": cfg.mixed_precision and device.type == "cuda",
        "bestValidationLoss": best_loss,
        "history": history,
        "checkpoint": str(output_dir / "brand_quality_model.pt"),
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class BrandQualityTorchDataset(Dataset):  # type: ignore[misc]
    def __init__(self, examples: list[Any], vocab_size: int) -> None:
        self.examples = examples
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        example = self.examples[index]
        features = torch.zeros(self.vocab_size + 3, dtype=torch.float32)
        for token in example.sanitized_text.lower().split():
            bucket = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.vocab_size
            features[bucket] += 1.0
        features[self.vocab_size] = min(len(example.sanitized_text) / 500.0, 1.0)
        features[self.vocab_size + 1] = 1.0 if example.accepted else 0.0
        features[self.vocab_size + 2] = 1.0 if example.corrected else 0.0
        labels = torch.tensor(
            [example.labels.get(label, 0) for label in QUALITY_LABELS], dtype=torch.float32
        )
        return features, labels


class BrandQualityClassifier(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, input_dim: int, output_dim: int) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is not installed")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

    def forward(self, features: Any) -> Any:
        return self.network(features)


def train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: Any,
    scaler: Any,
    cfg: TrainingConfig,
) -> float:
    model.train()
    losses = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision and device.type == "cuda"):
            logits = model(features)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


def validate_epoch(model: Any, loader: Any, loss_fn: Any, device: Any) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            losses.append(float(loss_fn(logits, labels).detach().cpu()))
    return sum(losses) / max(1, len(losses))


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
