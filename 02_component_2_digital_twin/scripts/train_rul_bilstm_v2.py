
"""
Train an improved BiLSTM RUL model (v2).

Motivation
----------
The v1 model (scripts/train_rul_bilstm.py) regressed strongly toward the
mean RUL: in the 1-7 day range it predicted ~18.5 days on average
(MAE 14.6), so it was least accurate exactly when it mattered most.

Three targeted, independently switchable changes address that bias:

1. Target transform (--target-transform log1p | none)
   log1p compresses the RUL range so small RUL values are not
   dominated by large ones. Note: it also tolerates large absolute
   errors at high RUL, which can hurt aggregate MAE / R2.

2. Loss (--loss huber | mse)
   Huber is less sensitive to the long tail of large-RUL samples
   than MSE.

3. Near-failure sample weighting (--strength)
   Each training sample is weighted by 1 + STRENGTH / (1 + RUL), so
   samples close to failure contribute more to the gradient.
   STRENGTH=0 disables weighting.

Empirical note (seq_30 test split):
  - v1 (MSE, no transform, no weighting):
      test MAE 10.6, R2 0.23
      1-7 day band: MAE 14.6, mean prediction 18.5 for 3.9-day drives
  - log1p + Huber + weighting (strength 2.0):
      1-7 day band MAE 12.0, mean prediction 15.9 (less
      overprediction), but aggregate test MAE ~11.9 and R2 ~0
      because high-RUL absolute errors grow.
  - Use --target-transform none --loss mse --strength 2.0 to keep
    original RUL units while up-weighting near-failure samples.

The architecture, features, splits and optimiser are unchanged from v1
so the models are directly comparable.

Outputs (results/rul_sequences/sequence_length_experiments/seq_30/bilstm_v2/):
  - best_bilstm_v2_model.pt
  - metrics.csv                 train/validation/test MAE, RMSE, R2
  - test_predictions.csv        per-sample actual vs predicted
  - rul_error_by_range.csv      error grouped by actual-RUL band
  - training_history.csv        per-epoch losses

Run from the component root:
  python .\\scripts\\train_rul_bilstm_v2.py
  python .\\scripts\\train_rul_bilstm_v2.py --epochs 15
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

BASE = r".\results\rul_sequences\sequence_length_experiments\seq_30"

INPUT_DIR = os.path.join(BASE, "preprocessed")

V1_DIR = os.path.join(BASE, "bilstm")

SEED = 42

# Weight = 1 + STRENGTH / (1 + RUL). 0.0 disables weighting.
NEAR_FAILURE_STRENGTH = 2.0

# Huber loss transition point (in log1p space).
HUBER_BETA = 1.0

RUL_BINS = [0, 7, 14, 30, 60, np.inf]

RUL_BIN_LABELS = [
    "1-7 days",
    "8-14 days",
    "15-30 days",
    "31-60 days",
    "60+ days",
]


# ============================================================
# Reproducibility
# ============================================================

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Model (identical to v1)
# ============================================================

class RULBiLSTM(nn.Module):

    def __init__(
        self,
        input_size=22,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output).squeeze(1)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(actual, predicted, label):

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "dataset": label,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def predict(model, loader, device, use_log_target):
    """Predict RUL in original day units."""

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)

            raw = model(xb)

            # Invert the log1p target transform when used.
            pred = torch.expm1(raw) if use_log_target else raw

            predictions.extend(pred.cpu().numpy())
            actual.extend(yb.numpy())

    return np.asarray(predictions), np.asarray(actual)


# ============================================================
# Data
# ============================================================

def load_split(name):
    X = np.load(os.path.join(INPUT_DIR, f"X_{name}.npy"))
    y = np.load(os.path.join(INPUT_DIR, f"y_{name}.npy"))
    return X, y


def near_failure_weights(y, strength):
    """Sample weights that emphasise small RUL values."""

    if strength <= 0:
        return np.ones_like(y, dtype=np.float32)

    return (1.0 + strength / (1.0 + y)).astype(np.float32)


# ============================================================
# Error by RUL range
# ============================================================

def error_by_range(actual, predicted):
    """Error grouped by actual-RUL band, matching v1 bins."""

    frame = pd.DataFrame({
        "actual_rul": actual,
        "predicted_rul": predicted,
    })

    frame["RUL_group"] = pd.cut(
        frame["actual_rul"],
        bins=RUL_BINS,
        labels=RUL_BIN_LABELS,
        include_lowest=True,
    )

    rows = []

    for label, group in frame.groupby("RUL_group", observed=True):

        error = group["predicted_rul"] - group["actual_rul"]

        rows.append({
            "RUL_group": label,
            "samples": len(group),
            "MAE": np.mean(np.abs(error)),
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "Mean_Error": np.mean(error),
            "Median_Error": np.median(error),
            "Mean_Actual_RUL": group["actual_rul"].mean(),
            "Mean_Predicted_RUL": group["predicted_rul"].mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# Training
# ============================================================

def train(args):

    set_seeds(SEED)

    output_dir = os.path.join(
        BASE,
        "bilstm_v2" + (f"_{args.tag}" if args.tag else ""),
    )

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("BiLSTM RUL TRAINING (V2 - bias-corrected)")
    print("=" * 70)
    print("Device:", device)
    print(f"Near-failure weight strength: {args.strength}")
    print("Output directory:", output_dir)

    print("\nLoading 30-step dataset...")

    X_train, y_train = load_split("train")
    X_validation, y_validation = load_split("validation")
    X_test, y_test = load_split("test")

    print("Train:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)
    print("RUL range:", y_train.min(), "to", y_train.max())

    use_log_target = args.target_transform == "log1p"

    print(f"Target transform: {'log1p' if use_log_target else 'none'}")
    print(f"Loss: {args.loss}")

    weights = near_failure_weights(y_train, args.strength)

    weight_summary = (
        f"weights min={weights.min():.3f} "
        f"max={weights.max():.3f} mean={weights.mean():.3f}"
    )

    print(weight_summary)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    w_train_t = torch.tensor(weights, dtype=torch.float32)

    X_validation_t = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_t = torch.tensor(y_validation, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t, w_train_t),
        batch_size=args.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        TensorDataset(X_validation_t, y_validation_t),
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = RULBiLSTM().to(device)

    criterion = (
        nn.SmoothL1Loss(beta=HUBER_BETA, reduction="none")
        if args.loss == "huber"
        else nn.MSELoss(reduction="none")
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):

        start = time.time()

        model.train()

        train_losses = []

        for xb, yb, wb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()

            raw_pred = model(xb)
            target = torch.log1p(yb) if use_log_target else yb

            loss = criterion(raw_pred, target)

            # Weighted mean loss.
            loss = (loss * wb).mean()

            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()

        validation_losses = []

        with torch.no_grad():

            for xb, yb in validation_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                raw_pred = model(xb)
                target = torch.log1p(yb) if use_log_target else yb

                loss = criterion(raw_pred, target)

                validation_losses.append(loss.mean().item())

        validation_loss = float(np.mean(validation_losses))

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| Train: {train_loss:.4f} "
            f"| Val: {validation_loss:.4f} "
            f"| {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "seconds": elapsed,
        })

        # Early stopping on validation loss.
        if validation_loss < best_val_loss - 1e-6:

            best_val_loss = validation_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_bilstm_v2_model.pt"),
            )

            print(f"  -> saved best model (val={validation_loss:.4f})")

        else:

            patience_counter += 1

            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best epoch {best_epoch})."
                )
                break

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "training_history.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # Evaluate best model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION (best checkpoint)")
    print("=" * 70)

    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, "best_bilstm_v2_model.pt"),
            map_location=device,
        )
    )

    train_loader_eval = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=False,
    )

    results = []
    predictions = {}

    for label, loader in [
        ("train", train_loader_eval),
        ("validation", validation_loader),
        ("test", test_loader),
    ]:

        predicted, actual = predict(
            model, loader, device, use_log_target
        )

        predictions[label] = (actual, predicted)

        results.append(calculate_metrics(actual, predicted, label))

    metrics = pd.DataFrame(results)

    metrics.to_csv(
        os.path.join(output_dir, "metrics.csv"),
        index=False,
    )

    print("\nModel metrics (original RUL units):")
    print(metrics.to_string(index=False))

    # Per-sample test predictions
    test_actual, test_predicted = predictions["test"]

    pd.DataFrame({
        "actual_rul": test_actual,
        "predicted_rul": test_predicted,
        "error": test_predicted - test_actual,
        "absolute_error": np.abs(test_predicted - test_actual),
    }).to_csv(
        os.path.join(output_dir, "test_predictions.csv"),
        index=False,
    )

    # Error by range
    by_range = error_by_range(test_actual, test_predicted)

    by_range.to_csv(
        os.path.join(output_dir, "rul_error_by_range.csv"),
        index=False,
    )

    print("\nTest error by RUL range (v2):")
    print(by_range.to_string(index=False))

    # --------------------------------------------------------
    # Comparison with v1
    # --------------------------------------------------------

    v1_metrics_path = os.path.join(V1_DIR, "metrics.csv")
    v1_range_path = os.path.join(V1_DIR, "rul_error_by_range.csv")

    if os.path.exists(v1_metrics_path):

        print("\n" + "=" * 70)
        print("COMPARISON WITH V1")
        print("=" * 70)

        v1_metrics = pd.read_csv(v1_metrics_path)
        v1_metrics = v1_metrics.rename(
            columns={"MAE": "v1_MAE", "RMSE": "v1_RMSE", "R2": "v1_R2"}
        )
        v2_metrics = metrics.rename(
            columns={"MAE": "v2_MAE", "RMSE": "v2_RMSE", "R2": "v2_R2"}
        )

        comparison = v2_metrics.merge(
            v1_metrics,
            on="dataset",
            how="left",
        )

        print(
            comparison[
                ["dataset", "v1_MAE", "v2_MAE", "v1_RMSE", "v2_RMSE"]
            ].to_string(index=False)
        )

        comparison.to_csv(
            os.path.join(output_dir, "comparison_with_v1.csv"),
            index=False,
        )

    if os.path.exists(v1_range_path):

        v1_range = pd.read_csv(v1_range_path).rename(
            columns={
                "MAE": "v1_MAE",
                "Mean_Error": "v1_Mean_Error",
                "Mean_Predicted_RUL": "v1_Mean_Predicted_RUL",
            }
        )

        v2_range = by_range.rename(
            columns={
                "MAE": "v2_MAE",
                "Mean_Error": "v2_Mean_Error",
                "Mean_Predicted_RUL": "v2_Mean_Predicted_RUL",
            }
        )

        range_comparison = v1_range.merge(
            v2_range,
            on="RUL_group",
            how="outer",
        )

        print("\nNear-failure error comparison (v1 vs v2):")
        print(
            range_comparison[
                [
                    "RUL_group",
                    "Mean_Actual_RUL",
                    "v1_Mean_Predicted_RUL",
                    "v2_Mean_Predicted_RUL",
                    "v1_MAE",
                    "v2_MAE",
                ]
            ].to_string(index=False)
        )

        range_comparison.to_csv(
            os.path.join(output_dir, "error_by_range_comparison.csv"),
            index=False,
        )

    print("\nOutput directory:", output_dir)
    print("Training complete.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the bias-corrected BiLSTM RUL model (v2)."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50).",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early-stopping patience in epochs (default: 7).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3).",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=NEAR_FAILURE_STRENGTH,
        help=(
            "Near-failure sample-weight strength. "
            f"Default: {NEAR_FAILURE_STRENGTH}. 0 disables weighting."
        ),
    )

    parser.add_argument(
        "--target-transform",
        choices=["log1p", "none"],
        default="log1p",
        help=(
            "Target transform. 'log1p' compresses large RUL values "
            "(better near-failure focus, but tolerates large high-RUL "
            "absolute errors). 'none' keeps original RUL units."
        ),
    )

    parser.add_argument(
        "--loss",
        choices=["huber", "mse"],
        default="huber",
        help="Regression loss (default: huber).",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for the output directory (e.g. s1 -> bilstm_v2_s1).",
    )

    train(parser.parse_args())
