from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Raw SMART BiLSTM experiment. Run from the component directory:
# python .\scripts\train_raw_smart_bilstm.py

RAW_FILE = Path(r".\results\raw_smart_experiment\raw_smart_temporal_features.csv")
META_FILE = Path(r".\results\rul_sequences\sequence_length_experiments\seq_30\sequence_metadata.csv")
BASELINE_FILE = Path(r".\results\rul_sequences\sequence_length_experiments\seq_30\bilstm\test_predictions.csv")
OUT = Path(r".\results\raw_smart_experiment\bilstm_seq_30")

SEQ_LEN = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.001
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 50
PATIENCE = 7
SEED = 42
CHUNK_SIZE = 500_000

SMART = ["smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw"]
FEATURES = []
for col in SMART:
    FEATURES.extend([
        col, f"{col}_available", f"{col}_previous", f"{col}_delta",
        f"{col}_change_7obs", f"{col}_change_14obs",
        f"{col}_rolling_mean_7obs", f"{col}_rolling_std_7obs"
    ])
USECOLS = ["date", "serial_number", "failure"] + FEATURES
FLAGS = [c for c in FEATURES if c.endswith("_available")]
VALUE_COLS = [c for c in FEATURES if c not in FLAGS]


class RULBiLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True,
            bidirectional=True, dropout=DROPOUT
        )
        self.regressor = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.regressor(output[:, -1, :]).squeeze(-1)


def calc_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }


def predict(model, loader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb.to(device)).cpu().numpy().tolist())
            actuals.extend(yb.numpy().tolist())
    return np.asarray(preds), np.asarray(actuals)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    for path in [RAW_FILE, META_FILE, BASELINE_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.resolve()}")

    print("=" * 70)
    print("RAW SMART BiLSTM — SEQUENCE LENGTH 30")
    print("=" * 70)

    meta = pd.read_csv(META_FILE)
    required = {"serial_number", "sequence_end", "failure_date", "rul", "split"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    meta["serial_number"] = meta["serial_number"].astype(str)
    meta["sequence_end"] = pd.to_datetime(meta["sequence_end"], errors="coerce")
    meta["failure_date"] = pd.to_datetime(meta["failure_date"], errors="coerce")
    meta["rul"] = pd.to_numeric(meta["rul"], errors="coerce")
    meta["split"] = meta["split"].astype(str).str.lower()
    meta = meta.dropna(subset=["sequence_end", "failure_date", "rul", "serial_number"])
    meta = meta[meta["split"].isin(["train", "validation", "test"])].copy()

    # Ensure there is one target/split for each HDD and keep metadata's exact windows.
    hdd_ids = set(meta["serial_number"].unique())
    failure_dates = meta.groupby("serial_number")["failure_date"].min().to_dict()
    print(f"Metadata windows: {len(meta):,}; HDDs: {len(hdd_ids):,}")
    print("Reading raw SMART data in chunks. This scans the large CSV once...")

    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(RAW_FILE, usecols=USECOLS, chunksize=CHUNK_SIZE), start=1
    ):
        chunk["serial_number"] = chunk["serial_number"].astype(str)
        chunk = chunk[chunk["serial_number"].isin(hdd_ids)].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        for col in FEATURES:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=["date", "serial_number"])
        chunks.append(chunk)
        if i % 10 == 0:
            print(f"  scanned {i} chunks")

    if not chunks:
        raise RuntimeError("No raw SMART rows matched the metadata HDDs.")
    raw = pd.concat(chunks, ignore_index=True)
    del chunks
    raw = raw.sort_values(["serial_number", "date"]).drop_duplicates(
        ["serial_number", "date"], keep="last"
    )
    raw["__failure_date"] = raw["serial_number"].map(failure_dates)
    raw = raw[raw["date"] < raw["__failure_date"]].drop(columns="__failure_date").copy()
    print(f"Matched pre-failure rows: {len(raw):,}")

    # Fit imputation and scaling on training HDD observations only.
    train_ids = set(meta.loc[meta["split"] == "train", "serial_number"])
    fit_rows = raw[raw["serial_number"].isin(train_ids)]
    if fit_rows.empty:
        raise RuntimeError("No training observations found.")

    medians = fit_rows[VALUE_COLS].median().fillna(0.0)
    raw[VALUE_COLS] = raw[VALUE_COLS].fillna(medians)
    raw[FLAGS] = raw[FLAGS].fillna(0.0).clip(0, 1)
    means = fit_rows[VALUE_COLS].fillna(medians).mean()
    stds = fit_rows[VALUE_COLS].fillna(medians).std().replace(0, 1).fillna(1)
    raw[VALUE_COLS] = (raw[VALUE_COLS] - means) / stds

    with open(OUT / "preprocessing_parameters.json", "w", encoding="utf-8") as f:
        json.dump({
            "sequence_length": SEQ_LEN,
            "features": FEATURES,
            "availability_flags_not_standardized": FLAGS,
            "medians_fit_on_train_hdds": medians.to_dict(),
            "means_fit_on_train_hdds": means.to_dict(),
            "stds_fit_on_train_hdds": stds.to_dict()
        }, f, indent=2, default=float)

    print("Indexing HDD histories...")
    histories = {}
    for serial, group in raw.groupby("serial_number", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        histories[str(serial)] = (
            group["date"].to_numpy(dtype="datetime64[ns]"),
            group[FEATURES].to_numpy(dtype=np.float32)
        )
    del raw

    xs = {"train": [], "validation": [], "test": []}
    ys = {"train": [], "validation": [], "test": []}
    rows = {"train": [], "validation": [], "test": []}
    skipped = 0
    meta = meta.sort_values(["split", "serial_number", "sequence_end"]).reset_index(drop=True)

    for r in meta.itertuples(index=False):
        split = str(r.split).lower()
        serial = str(r.serial_number)
        if serial not in histories:
            skipped += 1
            continue
        dates, values = histories[serial]
        end = np.datetime64(pd.Timestamp(r.sequence_end), "ns")
        pos = int(np.searchsorted(dates, end, side="right")) - 1
        if pos < 0 or dates[pos] != end or pos + 1 < SEQ_LEN:
            skipped += 1
            continue
        window = values[pos - SEQ_LEN + 1:pos + 1]
        if window.shape != (SEQ_LEN, len(FEATURES)):
            skipped += 1
            continue
        xs[split].append(window)
        ys[split].append(float(r.rul))
        rows[split].append({
            "serial_number": serial,
            "sequence_end": pd.Timestamp(r.sequence_end),
            "failure_date": pd.Timestamp(r.failure_date),
            "actual_rul": float(r.rul)
        })

    for split in ["train", "validation", "test"]:
        if not xs[split]:
            raise RuntimeError(f"No {split} windows were built.")
        xs[split] = np.stack(xs[split]).astype(np.float32)
        ys[split] = np.asarray(ys[split], dtype=np.float32)
        print(f"{split.title():10s}: {len(ys[split]):,} windows")
    print(f"Skipped metadata windows: {skipped:,}")

    def loader(split, shuffle=False):
        ds = TensorDataset(torch.from_numpy(xs[split]), torch.from_numpy(ys[split]))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = loader("train", True)
    train_eval_loader = loader("train")
    val_loader = loader("validation")
    test_loader = loader("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; input features: {len(FEATURES)}")

    model = RULBiLSTM(len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    best_val = float("inf")
    patience_count = 0
    history = []
    model_file = OUT / "best_raw_smart_bilstm_model.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(float(criterion(model(xb), yb).item()))
        train_loss, val_loss = float(np.mean(losses)), float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(model.state_dict(), model_file)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("Early stopping.")
                break

    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
    try:
        state = torch.load(model_file, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)

    train_pred, train_actual = predict(model, train_eval_loader, device)
    val_pred, val_actual = predict(model, val_loader, device)
    test_pred, test_actual = predict(model, test_loader, device)
    result = {
        "train": calc_metrics(train_actual, train_pred),
        "validation": calc_metrics(val_actual, val_pred),
        "test": calc_metrics(test_actual, test_pred),
        "n_features": len(FEATURES),
        "sequence_length": SEQ_LEN,
        "train_windows": len(train_actual),
        "validation_windows": len(val_actual),
        "test_windows": len(test_actual),
        "device": str(device)
    }
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    test_out = pd.DataFrame(rows["test"])
    test_out["predicted_rul"] = test_pred
    test_out["error"] = test_out["predicted_rul"] - test_out["actual_rul"]
    test_out["absolute_error"] = test_out["error"].abs()
    test_out.to_csv(OUT / "test_predictions_with_hdd_dates.csv", index=False)

    baseline = pd.read_csv(BASELINE_FILE)
    baseline_metrics = calc_metrics(baseline["actual_rul"], baseline["predicted_rul"])
    same_count = len(baseline) == len(test_actual)
    same_targets = bool(
        same_count and np.allclose(
            baseline["actual_rul"].to_numpy(dtype=float),
            test_actual.astype(float), rtol=0, atol=1e-5
        )
    )
    comparison = {
        "baseline_test_rows": int(len(baseline)),
        "raw_smart_test_rows": int(len(test_actual)),
        "baseline_test_metrics": baseline_metrics,
        "raw_smart_test_metrics": result["test"],
        "same_test_row_count": same_count,
        "same_actual_rul_sequence": same_targets,
        "warning": "Only treat row-level predictions as directly aligned if same_actual_rul_sequence is true."
    }
    with open(OUT / "baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS")
    for split in ["train", "validation", "test"]:
        m = result[split]
        print(f"{split.title():10s} | MAE {m['MAE']:.3f} | RMSE {m['RMSE']:.3f} | R2 {m['R2']:.3f}")
    print("\nBASELINE vs RAW SMART TEST")
    print(f"Baseline:  MAE {baseline_metrics['MAE']:.3f} | RMSE {baseline_metrics['RMSE']:.3f} | R2 {baseline_metrics['R2']:.3f}")
    print(f"Raw SMART: MAE {result['test']['MAE']:.3f} | RMSE {result['test']['RMSE']:.3f} | R2 {result['test']['R2']:.3f}")
    print(f"Same test row count: {same_count}")
    print(f"Same actual RUL sequence: {same_targets}")
    print(f"Output folder: {OUT.resolve()}")


if __name__ == "__main__":
    main()
