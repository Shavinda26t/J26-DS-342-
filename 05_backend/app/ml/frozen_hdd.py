"""Frozen Notebook 24 inference. No training, labels, or adaptive thresholds."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3] / "01_component_1_behavioral_drift_genai"
RESULTS = "results/single_model_q1q2_2021_v1/"
FEATURES = ["smart_5_raw", "smart_9_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw"]
HIDDEN = [0, 2, 3, 4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaskedTelemetryTransformer(nn.Module):
    """Architecture and masking copied from Notebook 24; weights stay frozen."""

    def __init__(self, n_models, width=64, heads=4, layers=2, feedforward=128, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(15, width)
        self.model_embedding = nn.Embedding(n_models, width)
        self.position = nn.Parameter(torch.zeros(1, 30, width))
        nn.init.normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=layers, enable_nested_tensor=False,
        )
        self.output = nn.Linear(width, 5)

    def forward(self, x, model_id, hide):
        observed = (~hide).to(x.dtype)
        masked = x.masked_fill(hide, 0)
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        delta_visible = observed.clone()
        delta_visible[:, 1:, :] = observed[:, 1:, :] * observed[:, :-1, :]
        delta = delta * delta_visible
        h = self.input(torch.cat([masked, observed, delta], dim=-1))
        h = h + self.model_embedding(model_id)[:, None, :] + self.position
        return self.output(self.encoder(h))


class FrozenHDD:
    def __init__(self, root=ROOT):
        lock_path = root / RESULTS / "q4_blind_candidate_protocol_v1/q4_candidate_lock.json"
        handoff = json.loads((root / RESULTS / "q4_research_review_and_handoff_v1/research_integration_handoff.json").read_text())
        if digest(lock_path) != handoff["candidate_lock_sha256"]:
            raise RuntimeError("Candidate lock hash mismatch")
        self.version = "notebook24-sector-v1:" + digest(lock_path)
        lock = json.loads(lock_path.read_text())
        paths = {}
        for key in ("transformer_checkpoint", "transformer_preprocessing", "transformer_training_settings", "transformer_scoring_settings", "transformer_thresholds", "stream_contract"):
            spec = lock["artifacts"][key]
            path = root.joinpath(*spec["path"].replace("\\", "/").split("/"))
            if digest(path) != spec["sha256"]:
                raise RuntimeError(f"Frozen artifact hash mismatch: {key}")
            paths[key] = path
        self.prep = json.loads(paths["transformer_preprocessing"].read_text())
        self.models = self.prep["models"]
        self.thresholds = json.loads(paths["transformer_thresholds"].read_text())
        if self.prep["features"] != FEATURES or set(self.models) != set(self.thresholds) or len(self.models) != 5:
            raise RuntimeError("Frozen feature/model contract mismatch")
        settings = json.loads(paths["transformer_training_settings"].read_text())
        self.network = MaskedTelemetryTransformer(len(self.models), **{k: settings[k] for k in ("width", "heads", "layers", "feedforward", "dropout")})
        checkpoint = torch.load(paths["transformer_checkpoint"], map_location="cpu", weights_only=True)
        if checkpoint["models"] != self.models or checkpoint["preprocessing_sha256"] != digest(paths["transformer_preprocessing"]):
            raise RuntimeError("Checkpoint metadata mismatch")
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()
        self.checkpoint_hash = digest(paths["transformer_checkpoint"])

    @torch.no_grad()
    def score(self, window, model):
        # Notebook 24 ingests float32 before log1p in float64.
        window = np.asarray(window, dtype=np.float32)
        logged = np.log1p(window.astype(np.float64))
        scaled = ((logged - np.asarray(self.prep["center"][model])) / np.asarray(self.prep["scale"][model])).astype(np.float32)
        if window.shape != (30, 5) or not np.isfinite(scaled).all():
            raise ValueError("Invalid scoring window")
        x = torch.from_numpy(scaled[None])
        hide = torch.zeros_like(x, dtype=torch.bool)
        hide[:, -1, HIDDEN] = True
        residual = (self.network(x, torch.tensor([self.models.index(model)]), hide)[:, -1, HIDDEN] - x[:, -1, HIDDEN]).square()
        errors = residual.numpy()[0].astype(np.float32)
        if not np.isfinite(errors).all():
            raise RuntimeError("Nonfinite frozen score")
        scores = {"temp_error": float(errors[1]), "sector_error": float(errors[[0, 2, 3]].mean()), "joint_error": float(errors.mean())}
        thresholds = self.thresholds[model]
        return {"scores": scores, "thresholds": thresholds,
                "flags": {k: v > thresholds[k] for k, v in scores.items()},
                "anomaly_score": scores["sector_error"], "frozen_threshold": thresholds["sector_error"],
                "candidate_flag": scores["sector_error"] > thresholds["sector_error"],
                "sensor_residuals": dict(zip([FEATURES[i] for i in HIDDEN], map(float, errors))),
                "smart_evidence": {"sector_positive": bool(np.any(window[-1, [0, 3, 4]] > 0)),
                                   "sector_delta": float(np.maximum(window[-1, [0, 3, 4]] - window[-2, [0, 3, 4]], 0).sum())},
                "model_version": self.checkpoint_hash}
