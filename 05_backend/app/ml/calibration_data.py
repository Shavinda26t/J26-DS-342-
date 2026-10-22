"""Leakage-free conformal calibration data for Component 2.

The conformal intervals in :mod:`app.ml.conformal` are only valid if their
calibration residuals come from drives that are **independent of the drives
used for final evaluation**.  Historically the backend seeded them from
``results/digital_twin_rul/heldout_rul_trajectory_results.csv`` — a file whose
90 drives are *exactly* the 90 drives of the final test split (301 of its 303
rows reproduce the committed test predictions), and which mixes sequence
lengths 1/7/14/30 in a single calibration population.  That made every
reported interval an in-sample statement.

This module is the single source of calibration data.  It:

1. reads the drive-level split assignment from
   ``results/rul_sequences/sequence_length_experiments/seq_30/`` and
   **fails loudly** if the calibration drives intersect the evaluation drives;
2. uses the **deployed** checkpoint and its preprocessing parameters (both
   imported from :mod:`app.ml.component2_rul_predictor`, so calibration can
   never silently drift away from production);
3. runs that checkpoint over the already-preprocessed calibration split and
   verifies that the label vector still lines up with the split metadata
   (row-count, sequence length and feature-count checks) before any residual
   is produced;
4. returns the ``(prediction, actual)`` pairs plus an auditable diagnostics
   block (drives, observations, checkpoint digest, overlap, ...).

The current primary experiment (the deployed model) is the
**sequence-length-30** BiLSTM, so calibration is restricted to that one
configuration.  Mixing sequence lengths is not supported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# The deployed Component 2 model is the sequence-length-30 BiLSTM.
CALIBRATION_SEQUENCE_LENGTH = 30
CALIBRATION_SPLIT = "validation"
EVALUATION_SPLIT = "test"
BATCH_SIZE = 512


class CalibrationLeakageError(RuntimeError):
    """Raised when calibration drives overlap the final evaluation drives."""


class CalibrationSourceError(RuntimeError):
    """Raised when a calibration source cannot be trusted (bad provenance)."""


@dataclass
class CalibrationData:
    """``(prediction, actual)`` calibration pairs plus provenance."""

    pairs: List[Tuple[float, float]] = field(default_factory=list)
    serials: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def residuals(self) -> List[float]:
        return [abs(actual - predicted) for predicted, actual in self.pairs]

    @property
    def n_drives(self) -> int:
        return len(set(self.serials))

    @property
    def n_observations(self) -> int:
        return len(self.pairs)


# ----------------------------------------------------------------------
# Paths — anchored to the *deployed* predictor so calibration cannot drift
# away from production.
# ----------------------------------------------------------------------
def checkpoint_path() -> Path:
    """The deployed BiLSTM checkpoint (single source of truth)."""
    from app.ml.component2_rul_predictor import MODEL_PATH  # noqa: PLC0415

    return Path(MODEL_PATH)


def preprocessing_path() -> Path:
    """The deployed preprocessing parameters (single source of truth)."""
    from app.ml.component2_rul_predictor import (  # noqa: PLC0415
        PREPROCESSING_PATH,
    )

    return Path(PREPROCESSING_PATH)


def preprocessed_dir() -> Path:
    """Directory holding ``X_*.npy`` / ``y_*.npy`` for the deployed model."""
    return preprocessing_path().parent


def sequence_dir() -> Path:
    """The ``seq_30`` experiment directory."""
    return preprocessing_path().parents[1]


def metadata_path() -> Path:
    return sequence_dir() / "sequence_metadata.csv"


# ----------------------------------------------------------------------
# Split bookkeeping
# ----------------------------------------------------------------------
def load_split_metadata() -> pd.DataFrame:
    """Load the drive-level sequence metadata (serial, sequence_end, rul, split)."""
    path = metadata_path()
    if not path.exists():
        raise CalibrationSourceError(f"split metadata not found: {path}")
    frame = pd.read_csv(path)
    missing = {"serial_number", "sequence_end", "failure_date", "rul", "split"}
    absent = missing.difference(frame.columns)
    if absent:
        raise CalibrationSourceError(
            f"{path.name} is missing columns: {sorted(absent)}"
        )
    return frame


def split_drives(metadata: pd.DataFrame) -> Dict[str, Set[str]]:
    """Map each split name to its set of drives."""
    return {
        str(name): set(group["serial_number"].astype(str))
        for name, group in metadata.groupby("split")
    }


def verify_disjoint(
    calibration_drives: Sequence[str],
    evaluation_drives: Sequence[str],
    *,
    context: str = "conformal calibration",
) -> None:
    """Fail loudly when calibration and evaluation drives intersect.

    No serial number may be used both to choose the interval quantile and to
    report interval coverage.
    """
    calibration = {str(s) for s in calibration_drives}
    evaluation = {str(s) for s in evaluation_drives}
    overlap = sorted(calibration & evaluation)
    if overlap:
        preview = ", ".join(overlap[:10])
        suffix = "" if len(overlap) <= 10 else f" (+{len(overlap) - 10} more)"
        raise CalibrationLeakageError(
            f"{context}: {len(overlap)} drives are used for BOTH calibration "
            f"and evaluation: {preview}{suffix}"
        )


def file_digest(path: Path, length: int = 16) -> str:
    """Short sha256 digest of a file (provenance / reproducibility)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def drive_digest(serials: Sequence[str]) -> str:
    """Short reproducible identifier for a set of drives."""
    unique = sorted({str(s) for s in serials})
    return hashlib.sha256(",".join(unique).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def _predict_batch(model, features: np.ndarray) -> np.ndarray:
    """Run the frozen BiLSTM over already-preprocessed sequences."""
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            chunk = features[start : start + BATCH_SIZE]
            tensor = torch.tensor(chunk, dtype=torch.float32).to(device)
            outputs.append(model(tensor).detach().cpu().numpy())
    predictions = np.concatenate(outputs).astype(float).ravel()
    # Mirror Component2RULPredictor.predict(): RUL is clamped at zero.
    return np.maximum(predictions, 0.0)


def build_calibration_data(
    calibration_split: str = CALIBRATION_SPLIT,
    evaluation_split: str = EVALUATION_SPLIT,
    *,
    predictor: Any = None,
) -> CalibrationData:
    """Build the calibration population for the deployed seq_30 BiLSTM.

    Raises :class:`CalibrationLeakageError` on any drive overlap and
    :class:`CalibrationSourceError` when the source cannot be trusted.
    """
    metadata = load_split_metadata()
    drives = split_drives(metadata)

    for split in (calibration_split, evaluation_split):
        if split not in drives:
            raise CalibrationSourceError(
                f"unknown split {split!r}; available: {sorted(drives)}"
            )
        if not drives[split]:
            raise CalibrationSourceError(f"split {split!r} has no drives")

    calibration_rows = (
        metadata[metadata["split"] == calibration_split]
        .reset_index(drop=True)
    )
    evaluation_drives = drives[evaluation_split]
    calibration_drives = set(calibration_rows["serial_number"].astype(str))

    # --- 1. leakage guard ------------------------------------------------
    verify_disjoint(
        calibration_drives,
        evaluation_drives,
        context=f"calibration split {calibration_split!r}",
    )

    # --- 2. checkpoint / preprocessing identity --------------------------
    checkpoint = checkpoint_path()
    preprocessing_file = preprocessing_path()
    if not checkpoint.exists():
        raise CalibrationSourceError(f"checkpoint not found: {checkpoint}")
    if not preprocessing_file.exists():
        raise CalibrationSourceError(
            f"preprocessing parameters not found: {preprocessing_file}"
        )
    with preprocessing_file.open("r", encoding="utf-8") as handle:
        preprocessing = json.load(handle)
    sequence_length = int(preprocessing.get("sequence_length", -1))
    feature_count = int(preprocessing.get("final_feature_count", -1))
    if sequence_length != CALIBRATION_SEQUENCE_LENGTH:
        raise CalibrationSourceError(
            "the deployed preprocessing configuration expects sequence length "
            f"{sequence_length}, calibration supports "
            f"{CALIBRATION_SEQUENCE_LENGTH} only"
        )

    # --- 3. arrays + alignment ------------------------------------------
    features_file = preprocessed_dir() / f"X_{calibration_split}.npy"
    labels_file = preprocessed_dir() / f"y_{calibration_split}.npy"
    if not features_file.exists() or not labels_file.exists():
        raise CalibrationSourceError(
            f"preprocessed calibration arrays missing: {features_file.name}"
        )
    features = np.load(features_file)
    labels = np.load(labels_file).astype(float).ravel()

    if features.ndim != 3:
        raise CalibrationSourceError(
            f"{features_file.name} has unexpected shape {features.shape}"
        )
    if (
        features.shape[0] != len(calibration_rows)
        or labels.size != len(calibration_rows)
    ):
        raise CalibrationSourceError(
            f"{features_file.name} rows ({features.shape[0]}) do not match the "
            f"'{calibration_split}' split rows ({len(calibration_rows)})"
        )
    if (
        features.shape[1] != sequence_length
        or features.shape[2] != feature_count
    ):
        raise CalibrationSourceError(
            f"{features_file.name} shape {features.shape} does not match the "
            f"deployed configuration ({sequence_length}, {feature_count})"
        )
    expected_labels = calibration_rows["rul"].to_numpy(dtype=float)
    if not np.allclose(labels, expected_labels, atol=1e-6, rtol=0.0):
        raise CalibrationSourceError(
            f"{labels_file.name} does not line up with the "
            f"'{calibration_split}' split metadata labels"
        )

    # --- 4. predictions from the unchanged checkpoint --------------------
    if predictor is None:
        from app.ml.component2_rul_predictor import (  # noqa: PLC0415
            Component2RULPredictor,
        )

        predictor = Component2RULPredictor(
            model_path=checkpoint, preprocessing_path=preprocessing_file
        )
    predictions = _predict_batch(predictor.model, features)

    pairs: List[Tuple[float, float]] = [
        (float(pred), float(actual))
        for pred, actual in zip(predictions, expected_labels)
    ]
    serials = calibration_rows["serial_number"].astype(str).tolist()

    train_drives = drives.get("train", set())
    diagnostics: Dict[str, Any] = {
        "generator": "app.ml.calibration_data.build_calibration_data",
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": file_digest(checkpoint),
        "preprocessing_file": preprocessing_file.name,
        "preprocessing_sha256": file_digest(preprocessing_file),
        "sequence_length": sequence_length,
        "feature_count": feature_count,
        "calibration_split": calibration_split,
        "evaluation_split": evaluation_split,
        "metadata_file": metadata_path().name,
        "calibration_drives": len(calibration_drives),
        "calibration_observations": len(pairs),
        "calibration_drive_digest": drive_digest(calibration_drives),
        "evaluation_drives": len(evaluation_drives),
        "evaluation_drive_digest": drive_digest(evaluation_drives),
        "drive_overlap": len(calibration_drives & evaluation_drives),
        "overlap_check": "pass",
        "drives_also_in_training_split": len(
            calibration_drives & set(train_drives)
        ),
        "sequence_lengths_in_population": [sequence_length],
        "source": (
            f"{metadata_path().name} + {features_file.name} via the deployed "
            f"seq_{sequence_length} BiLSTM ({checkpoint.name})"
        ),
    }
    return CalibrationData(pairs=pairs, serials=serials, diagnostics=diagnostics)


_CACHE: Dict[Tuple[str, str], CalibrationData] = {}


def load_calibration(
    calibration_split: str = CALIBRATION_SPLIT,
    evaluation_split: str = EVALUATION_SPLIT,
) -> CalibrationData:
    """Cached :func:`build_calibration_data` (built once per process)."""
    key = (calibration_split, evaluation_split)
    if key not in _CACHE:
        _CACHE[key] = build_calibration_data(calibration_split, evaluation_split)
    return _CACHE[key]


def clear_cache() -> None:
    """Drop the cached calibration population (used by tests)."""
    _CACHE.clear()
