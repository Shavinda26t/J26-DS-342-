import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Reproducibility
# ============================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Paths
# ============================================================
BASE = r".\results\rul_sequences\preprocessed"
OUT = r".\results\rul_lstm_huber"

os.makedirs(OUT, exist_ok=True)


# ============================================================
# Load preprocessed data
# ============================================================
X_train = np.load(os.path.join(BASE, "X_train.npy"))
y_train = np.load(os.path.join(BASE, "y_train.npy"))

X_val = np.load(os.path.join(BASE, "X_validation.npy"))
y_val = np.load(os.path.join(BASE, "y_validation.npy"))

X_test = np.load(os.path.join(BASE, "X_test.npy"))
y_test = np.load(os.path.join(BASE, "y_test.npy"))


print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# ============================================================
# Convert to PyTorch tensors
# ============================================================
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64,
    shuffle=False
)


# ============================================================
# LSTM Model
# Same architecture as the original baseline
# ============================================================
class RULLSTM(nn.Module):

    def __init__(
        self,
        input_size=22,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Use the final time step
        last_output = output[:, -1, :]

        return self.fc(last_output).squeeze(1)


# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


model = RULLSTM().to(device)


# ============================================================
# Huber Loss
# ============================================================
criterion = nn.HuberLoss(delta=5.0)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)


# ============================================================
# Training
# ============================================================
MAX_EPOCHS = 50
PATIENCE = 7

best_val_loss = float("inf")
patience_counter = 0

history = []


for epoch in range(1, MAX_EPOCHS + 1):

    # -------------------------
    # Training
    # -------------------------
    model.train()

    train_losses = []

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        predictions = model(xb)

        loss = criterion(predictions, yb)

        loss.backward()

        optimizer.step()

        train_losses.append(loss.item())

    train_loss = np.mean(train_losses)


    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    val_losses = []

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            predictions = model(xb)

            loss = criterion(predictions, yb)

            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss
    })


    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} "
        f"| Train Huber: {train_loss:.4f} "
        f"| Val Huber: {val_loss:.4f}"
    )


    # -------------------------
    # Early stopping
    # -------------------------
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            os.path.join(OUT, "best_rul_lstm_huber.pt")
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping.")

            break


# ============================================================
# Save training history
# ============================================================
pd.DataFrame(history).to_csv(
    os.path.join(OUT, "training_history.csv"),
    index=False
)


# ============================================================
# Load best model
# ============================================================
model.load_state_dict(
    torch.load(
        os.path.join(OUT, "best_rul_lstm_huber.pt"),
        map_location=device
    )
)

model.eval()


# ============================================================
# Prediction function
# ============================================================
def predict(loader):

    predictions = []
    actual = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)

            pred = model(xb).cpu().numpy()

            predictions.extend(pred)
            actual.extend(yb.numpy())

    return np.array(predictions), np.array(actual)


# ============================================================
# Evaluation
# ============================================================
def calculate_metrics(actual, predicted):

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - actual.mean()) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    return mae, rmse, r2


# ============================================================
# Train / Validation / Test predictions
# ============================================================
train_pred, train_actual = predict(train_loader)
val_pred, val_actual = predict(val_loader)
test_pred, test_actual = predict(test_loader)


train_mae, train_rmse, train_r2 = calculate_metrics(
    train_actual,
    train_pred
)

val_mae, val_rmse, val_r2 = calculate_metrics(
    val_actual,
    val_pred
)

test_mae, test_rmse, test_r2 = calculate_metrics(
    test_actual,
    test_pred
)


# ============================================================
# Print final results
# ============================================================
print("\n==============================")
print("FINAL HUBER LSTM RESULTS")
print("==============================")

print(
    f"Train      | MAE: {train_mae:.3f} "
    f"| RMSE: {train_rmse:.3f} "
    f"| R2: {train_r2:.3f}"
)

print(
    f"Validation | MAE: {val_mae:.3f} "
    f"| RMSE: {val_rmse:.3f} "
    f"| R2: {val_r2:.3f}"
)

print(
    f"Test       | MAE: {test_mae:.3f} "
    f"| RMSE: {test_rmse:.3f} "
    f"| R2: {test_r2:.3f}"
)


# ============================================================
# Save predictions
# ============================================================
pd.DataFrame({
    "actual_rul": test_actual,
    "predicted_rul": test_pred,
    "error": test_pred - test_actual,
    "absolute_error": np.abs(test_pred - test_actual)
}).to_csv(
    os.path.join(OUT, "test_rul_predictions.csv"),
    index=False
)


# ============================================================
# Save metrics
# ============================================================
pd.DataFrame({
    "dataset": [
        "train",
        "validation",
        "test"
    ],
    "MAE": [
        train_mae,
        val_mae,
        test_mae
    ],
    "RMSE": [
        train_rmse,
        val_rmse,
        test_rmse
    ],
    "R2": [
        train_r2,
        val_r2,
        test_r2
    ]
}).to_csv(
    os.path.join(OUT, "final_metrics.csv"),
    index=False
)

print("\nResults saved to:")
print(OUT)