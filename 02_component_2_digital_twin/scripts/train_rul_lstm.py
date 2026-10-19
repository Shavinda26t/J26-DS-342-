import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Configuration
# ============================================================

DATA_DIR = r".\results\rul_sequences\preprocessed"
OUTPUT_DIR = r".\results\rul_lstm"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

MAX_EPOCHS = 50
PATIENCE = 7

NUM_WORKERS = 0


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("LSTM RUL PREDICTION")
print("=" * 80)

print(f"\nPyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
else:
    print("GPU: Not available - using CPU")


# ============================================================
# Load datasets
# ============================================================

print("\nLoading preprocessed datasets...")

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_val = np.load(
    os.path.join(DATA_DIR, "X_validation.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, "y_validation.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)


print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_validation:", X_val.shape)
print("y_validation:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# Convert to PyTorch tensors
# ============================================================

X_train_tensor = torch.from_numpy(
    X_train
).float()

X_val_tensor = torch.from_numpy(
    X_val
).float()

X_test_tensor = torch.from_numpy(
    X_test
).float()

y_train_tensor = torch.from_numpy(
    y_train
).float().unsqueeze(1)

y_val_tensor = torch.from_numpy(
    y_val
).float().unsqueeze(1)

y_test_tensor = torch.from_numpy(
    y_test
).float().unsqueeze(1)


# ============================================================
# DataLoaders
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# LSTM Model
# ============================================================

class RULLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )


    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        # Last time step representation
        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        rul = self.fc(
            last_output
        )

        return rul


# ============================================================
# Create model
# ============================================================

INPUT_SIZE = X_train.shape[2]

model = RULLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)


print("\nModel:")
print(model)

print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Evaluation function
# ============================================================

def evaluate(model, loader):

    model.eval()

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    predictions = []
    targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(
                X_batch
            )

            squared_error = (
                outputs - y_batch
            ) ** 2

            absolute_error = torch.abs(
                outputs - y_batch
            )

            total_squared_error += (
                squared_error.sum().item()
            )

            total_absolute_error += (
                absolute_error.sum().item()
            )

            total_samples += (
                y_batch.size(0)
            )

            predictions.append(
                outputs.cpu().numpy()
            )

            targets.append(
                y_batch.cpu().numpy()
            )

    mse = (
        total_squared_error /
        total_samples
    )

    rmse = np.sqrt(mse)

    mae = (
        total_absolute_error /
        total_samples
    )

    predictions = np.concatenate(
        predictions
    ).flatten()

    targets = np.concatenate(
        targets
    ).flatten()

    return mae, rmse, predictions, targets


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

history = []


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    samples_seen = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            y_batch.size(0)
        )

        running_loss += (
            loss.item() * batch_size
        )

        samples_seen += batch_size


    train_loss = (
        running_loss /
        samples_seen
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            batch_size = (
                y_batch.size(0)
            )

            val_loss += (
                loss.item() * batch_size
            )

            val_samples += batch_size


    val_loss /= val_samples

    val_rmse = np.sqrt(
        val_loss
    )

    history.append({
        "epoch": epoch,
        "train_mse": train_loss,
        "validation_mse": val_loss,
        "validation_rmse": val_rmse
    })


    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train MSE: {train_loss:.4f} | "
        f"Val MSE: {val_loss:.4f} | "
        f"Val RMSE: {val_rmse:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_size":
                    INPUT_SIZE,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "dropout":
                    DROPOUT,

                "epoch":
                    epoch,

                "validation_mse":
                    val_loss
            },
            os.path.join(
                OUTPUT_DIR,
                "best_rul_lstm.pt"
            )
        )

        print(
            "  -> Best model saved."
        )

    else:

        patience_counter += 1

        print(
            f"  -> No improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# Save training history
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    ),
    index=False
)


# ============================================================
# Load best model
# ============================================================

checkpoint = torch.load(
    os.path.join(
        OUTPUT_DIR,
        "best_rul_lstm.pt"
    ),
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print(
    f"Best epoch: {checkpoint['epoch']}"
)

print(
    f"Best validation MSE: "
    f"{checkpoint['validation_mse']:.4f}"
)


# ============================================================
# Final evaluation
# ============================================================

train_mae, train_rmse, _, _ = evaluate(
    model,
    train_loader
)

val_mae, val_rmse, _, _ = evaluate(
    model,
    val_loader
)

test_mae, test_rmse, test_predictions, test_targets = evaluate(
    model,
    test_loader
)


# ============================================================
# R2 calculation
# ============================================================

def calculate_r2(
    targets,
    predictions
):

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - np.mean(targets)) ** 2
    )

    if ss_tot == 0:
        return 0.0

    return 1.0 - (
        ss_res / ss_tot
    )


train_predictions = evaluate(
    model,
    train_loader
)[2]

train_targets = evaluate(
    model,
    train_loader
)[3]

val_predictions = evaluate(
    model,
    val_loader
)[2]

val_targets = evaluate(
    model,
    val_loader
)[3]


train_r2 = calculate_r2(
    train_targets,
    train_predictions
)

val_r2 = calculate_r2(
    val_targets,
    val_predictions
)

test_r2 = calculate_r2(
    test_targets,
    test_predictions
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 80)
print("FINAL RUL RESULTS")
print("=" * 80)

print("\nTRAIN")
print(f"MAE:  {train_mae:.3f} days")
print(f"RMSE: {train_rmse:.3f} days")
print(f"R2:   {train_r2:.4f}")

print("\nVALIDATION")
print(f"MAE:  {val_mae:.3f} days")
print(f"RMSE: {val_rmse:.3f} days")
print(f"R2:   {val_r2:.4f}")

print("\nTEST")
print(f"MAE:  {test_mae:.3f} days")
print(f"RMSE: {test_rmse:.3f} days")
print(f"R2:   {test_r2:.4f}")


# ============================================================
# Save test predictions
# ============================================================

prediction_df = pd.DataFrame({
    "actual_rul_days":
        test_targets,

    "predicted_rul_days":
        test_predictions,

    "absolute_error_days":
        np.abs(
            test_targets -
            test_predictions
        )
})

prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_rul_predictions.csv"
    ),
    index=False
)


# ============================================================
# Save final metrics
# ============================================================

metrics_df = pd.DataFrame({
    "split": [
        "train",
        "validation",
        "test"
    ],

    "MAE_days": [
        train_mae,
        val_mae,
        test_mae
    ],

    "RMSE_days": [
        train_rmse,
        val_rmse,
        test_rmse
    ],

    "R2": [
        train_r2,
        val_r2,
        test_r2
    ]
})

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_metrics.csv"
    ),
    index=False
)


print("\nOutput files:")
print(
    os.path.join(
        OUTPUT_DIR,
        "best_rul_lstm.pt"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "test_rul_predictions.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "final_metrics.csv"
    )
)

print("\nLSTM training completed.")