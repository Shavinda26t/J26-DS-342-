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

BASE = r".\results\rul_sequences\sequence_length_experiments"

SEQUENCE_LENGTHS = [7, 14, 21, 30]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LSTM model
# Same architecture as original baseline
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

        last_output = output[:, -1, :]

        return self.fc(last_output).squeeze(1)


# ============================================================
# Evaluation
# ============================================================

def calculate_metrics(actual, predicted):

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - actual.mean()) ** 2
    )

    r2 = 1 - (
        ss_res / ss_tot
    )

    return mae, rmse, r2


# ============================================================
# Prediction
# ============================================================

def predict(model, loader):

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)

            pred = model(xb)

            predictions.extend(
                pred.cpu().numpy()
            )

            actual.extend(
                yb.numpy()
            )

    return (
        np.asarray(predictions),
        np.asarray(actual)
    )


# ============================================================
# Store final results
# ============================================================

all_results = []


# ============================================================
# Train each sequence length
# ============================================================

for sequence_length in SEQUENCE_LENGTHS:

    print("\n" + "=" * 70)
    print(
        f"SEQUENCE LENGTH: {sequence_length}"
    )
    print("=" * 70)


    input_dir = os.path.join(
        BASE,
        f"seq_{sequence_length}",
        "preprocessed"
    )

    output_dir = os.path.join(
        BASE,
        f"seq_{sequence_length}",
        "lstm"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train = np.load(
        os.path.join(
            input_dir,
            "X_train.npy"
        )
    )

    y_train = np.load(
        os.path.join(
            input_dir,
            "y_train.npy"
        )
    )

    X_validation = np.load(
        os.path.join(
            input_dir,
            "X_validation.npy"
        )
    )

    y_validation = np.load(
        os.path.join(
            input_dir,
            "y_validation.npy"
        )
    )

    X_test = np.load(
        os.path.join(
            input_dir,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            input_dir,
            "y_test.npy"
        )
    )


    print(
        "Train:",
        X_train.shape
    )

    print(
        "Validation:",
        X_validation.shape
    )

    print(
        "Test:",
        X_test.shape
    )


    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_validation = torch.tensor(
        X_validation,
        dtype=torch.float32
    )

    y_validation = torch.tensor(
        y_validation,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train
        ),
        batch_size=64,
        shuffle=True
    )

    validation_loader = DataLoader(
        TensorDataset(
            X_validation,
            y_validation
        ),
        batch_size=64,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(
            X_test,
            y_test
        ),
        batch_size=64,
        shuffle=False
    )


    # --------------------------------------------------------
    # Reset seed before every experiment
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RULLSTM().to(device)


    # --------------------------------------------------------
    # MSE loss
    # --------------------------------------------------------

    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )


    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    MAX_EPOCHS = 50
    PATIENCE = 7

    best_val_loss = float("inf")
    patience_counter = 0

    history = []


    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        train_losses = []


        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            predictions = model(xb)

            loss = criterion(
                predictions,
                yb
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )


        train_loss = np.mean(
            train_losses
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        validation_losses = []


        with torch.no_grad():

            for xb, yb in validation_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                predictions = model(xb)

                loss = criterion(
                    predictions,
                    yb
                )

                validation_losses.append(
                    loss.item()
                )


        validation_loss = np.mean(
            validation_losses
        )


        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": validation_loss
        })


        print(
            f"Epoch {epoch:02d}/50 "
            f"| Train MSE: {train_loss:.3f} "
            f"| Val MSE: {validation_loss:.3f}"
        )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if validation_loss < best_val_loss:

            best_val_loss = validation_loss

            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(
                    output_dir,
                    "best_model.pt"
                )
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping."
                )

                break


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    pd.DataFrame(history).to_csv(
        os.path.join(
            output_dir,
            "training_history.csv"
        ),
        index=False
    )


    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            os.path.join(
                output_dir,
                "best_model.pt"
            ),
            map_location=device
        )
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_pred, train_actual = predict(
        model,
        train_loader
    )

    validation_pred, validation_actual = predict(
        model,
        validation_loader
    )

    test_pred, test_actual = predict(
        model,
        test_loader
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    train_mae, train_rmse, train_r2 = calculate_metrics(
        train_actual,
        train_pred
    )

    validation_mae, validation_rmse, validation_r2 = calculate_metrics(
        validation_actual,
        validation_pred
    )

    test_mae, test_rmse, test_r2 = calculate_metrics(
        test_actual,
        test_pred
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nRESULTS")

    print(
        f"Train      | "
        f"MAE: {train_mae:.3f} | "
        f"RMSE: {train_rmse:.3f} | "
        f"R2: {train_r2:.3f}"
    )

    print(
        f"Validation | "
        f"MAE: {validation_mae:.3f} | "
        f"RMSE: {validation_rmse:.3f} | "
        f"R2: {validation_r2:.3f}"
    )

    print(
        f"Test       | "
        f"MAE: {test_mae:.3f} | "
        f"RMSE: {test_rmse:.3f} | "
        f"R2: {test_r2:.3f}"
    )


    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({
        "actual_rul": test_actual,
        "predicted_rul": test_pred,
        "error": test_pred - test_actual,
        "absolute_error": np.abs(
            test_pred - test_actual
        )
    })

    predictions_df.to_csv(
        os.path.join(
            output_dir,
            "test_predictions.csv"
        ),
        index=False
    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame({
        "dataset": [
            "train",
            "validation",
            "test"
        ],
        "MAE": [
            train_mae,
            validation_mae,
            test_mae
        ],
        "RMSE": [
            train_rmse,
            validation_rmse,
            test_rmse
        ],
        "R2": [
            train_r2,
            validation_r2,
            test_r2
        ]
    })

    metrics_df.to_csv(
        os.path.join(
            output_dir,
            "metrics.csv"
        ),
        index=False
    )


    # --------------------------------------------------------
    # Add test result to overall table
    # --------------------------------------------------------

    all_results.append({
        "sequence_length": sequence_length,
        "train_sequences": len(X_train),
        "validation_sequences": len(X_validation),
        "test_sequences": len(X_test),
        "test_MAE": test_mae,
        "test_RMSE": test_rmse,
        "test_R2": test_r2,
        "best_validation_MSE": best_val_loss
    })


# ============================================================
# Save comparison
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    os.path.join(
        BASE,
        "sequence_length_lstm_comparison.csv"
    ),
    index=False
)


print("\n" + "=" * 70)
print("SEQUENCE LENGTH COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nExperiment completed.")