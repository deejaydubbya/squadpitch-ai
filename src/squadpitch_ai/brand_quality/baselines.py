from __future__ import annotations

import math
from collections import Counter

from squadpitch_ai.brand_quality.dataset import BrandQualityDataset
from squadpitch_ai.brand_quality.models import QUALITY_LABELS, BrandQualityDatasetExample


def compare_baselines(dataset: BrandQualityDataset) -> dict[str, object]:
    labels = list(QUALITY_LABELS)
    examples = dataset.examples
    majority = {
        label: int(sum(example.labels.get(label, 0) for example in examples) >= len(examples) / 2)
        for label in labels
    }
    return {
        "tfidfLogisticRegression": {
            "status": "specified",
            "features": "word n-grams over sanitized text plus channel/industry",
            "macroF1BaselineEstimate": macro_f1_for_majority(examples, majority),
        },
        "embeddingsLinearClassifier": {
            "status": "specified",
            "features": "privacy-safe embedding vectors plus structured channel labels",
        },
        "pytorchNeuralClassifier": {
            "status": "implemented_training_scaffold",
            "features": "hashed bag-of-words tensors plus structured features",
        },
        "smallTransformerFineTune": {
            "status": "deferred_until_model_registry_and_gpu_runner",
        },
        "loraPeft": {
            "status": "optional_future",
        },
        "classBalance": dataset.class_balance,
    }


def macro_f1_for_majority(
    examples: list[BrandQualityDatasetExample],
    majority: dict[str, int],
) -> float:
    if not examples:
        return 0.0
    f1s = []
    for label, prediction in majority.items():
        counts: Counter[str] = Counter()
        for example in examples:
            actual = int(example.labels.get(label, 0))
            if prediction == 1 and actual == 1:
                counts["tp"] += 1
            elif prediction == 1 and actual == 0:
                counts["fp"] += 1
            elif prediction == 0 and actual == 1:
                counts["fn"] += 1
        precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        f1s.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return round(math.fsum(f1s) / len(f1s), 4)
