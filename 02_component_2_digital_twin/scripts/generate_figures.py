"""
Generate report figures for Component 2 from the CSVs in results/.

Every figure is read from an existing result file, nothing is retrained.
Output: figures/*.png (300 dpi, ready for the report / slides).

Run from anywhere:
    python 02_component_2_digital_twin/scripts/generate_figures.py
"""

import pandas as pd
import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_paths import COMPONENT_ROOT


# ============================================================
# Paths
# ============================================================

RESULTS = COMPONENT_ROOT / "results"
FIGURES = COMPONENT_ROOT / "figures"
SEQ_DIR = RESULTS / "rul_sequences" / "sequence_length_experiments"
BILSTM_DIR = SEQ_DIR / "seq_30" / "bilstm"
RISK_DIR = RESULTS / "risk_gated_adaptation"
TWIN_DIR = RESULTS / "digital_twin_rul"

FIGURES.mkdir(exist_ok=True)


# ============================================================
# Style (categorical slots in fixed order, recessive grid)
# ============================================================

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_MUTED,
    "axes.edgecolor": GRID,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name):
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("Saved:", path.relative_to(COMPONENT_ROOT))


def label_bars(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        ax.annotate(
            fmt.format(bar.get_height()),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color=TEXT,
        )


# ============================================================
# 1. Sequence length comparison (why seq_30 was chosen)
# ============================================================

df = pd.read_csv(SEQ_DIR / "sequence_length_lstm_comparison.csv")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(df["sequence_length"].astype(str) + " days", df["test_MAE"],
              color=[BLUE] * len(df), width=0.6, edgecolor=SURFACE, linewidth=2)
for bar, length in zip(bars, df["sequence_length"]):
    bar.set_alpha(1.0 if length == df["sequence_length"].max() else 0.45)
label_bars(ax, bars)
ax.set_title("Longer input windows reduce RUL error (LSTM)")
ax.set_ylim(0, df["test_MAE"].max() * 1.15)
ax.set_xlabel("Input sequence length")
ax.set_ylabel("Test MAE (days)")
ax.grid(axis="x", visible=False)
save(fig, "01_sequence_length_comparison.png")


# ============================================================
# 2. BiLSTM (seq_30) training curve
# ============================================================

df = pd.read_csv(BILSTM_DIR / "training_history.csv")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df["epoch"], df["train_mse"], color=BLUE, label="Train")
ax.plot(df["epoch"], df["validation_mse"], color=ORANGE, label="Validation")
ax.set_title("BiLSTM training history (seq_30)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (days²)")
ax.legend()
save(fig, "02_bilstm_training_history.png")


# ============================================================
# 3. Actual vs predicted RUL (held-out test set)
# ============================================================

df = pd.read_csv(BILSTM_DIR / "test_predictions.csv")
metrics = pd.read_csv(BILSTM_DIR / "metrics.csv").set_index("dataset")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(df["actual_rul"], df["predicted_rul"], s=10, alpha=0.25,
           color=BLUE, edgecolors="none")
limit = max(df["actual_rul"].max(), df["predicted_rul"].max()) + 2
ax.plot([0, limit], [0, limit], color=TEXT_MUTED, linewidth=1,
        linestyle="--", label="Perfect prediction")
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_title("Actual vs predicted RUL (test set)")
ax.set_xlabel("Actual RUL (days)")
ax.set_ylabel("Predicted RUL (days)")
ax.text(0.03, 0.97,
        f"MAE {metrics.loc['test', 'MAE']:.1f} d   "
        f"RMSE {metrics.loc['test', 'RMSE']:.1f} d\n"
        f"{len(df):,} test windows",
        transform=ax.transAxes, va="top", fontsize=9, color=TEXT_MUTED)
ax.legend(loc="lower right")
save(fig, "03_actual_vs_predicted_rul.png")


# ============================================================
# 4. Error by RUL range
# ============================================================

df = pd.read_csv(BILSTM_DIR / "rul_error_by_range.csv")

fig, ax = plt.subplots(figsize=(6.5, 4))
x = range(len(df))
width = 0.38
actual = ax.bar([i - width / 2 for i in x], df["Mean_Actual_RUL"], width,
                color=BLUE, label="Mean actual RUL",
                edgecolor=SURFACE, linewidth=2)
predicted = ax.bar([i + width / 2 for i in x], df["Mean_Predicted_RUL"], width,
                   color=ORANGE, label="Mean predicted RUL",
                   edgecolor=SURFACE, linewidth=2)
label_bars(ax, actual)
label_bars(ax, predicted)
ax.set_xticks(list(x))
ax.set_xticklabels(df["RUL_group"])
ax.set_title("Model over-predicts near failure, under-predicts far from it")
ax.set_xlabel("Actual RUL range")
ax.set_ylabel("Days")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left")
save(fig, "04_error_by_rul_range.png")


# ============================================================
# 5. Single-drive RUL trajectory
# ============================================================

df = pd.read_csv(TWIN_DIR / "1050A098F97G_rul_trajectory.csv")
df["sequence_end_date"] = pd.to_datetime(df["sequence_end_date"])
df = df.sort_values("sequence_end_date")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(df["sequence_end_date"], df["actual_rul_days"], color=BLUE,
        marker="o", markersize=7, label="Actual RUL")
ax.plot(df["sequence_end_date"], df["predicted_rul_days"], color=ORANGE,
        marker="o", markersize=7, label="Predicted RUL")
ax.set_title("Digital twin tracking drive 1050A098F97G")
ax.set_xlabel("Observation date")
ax.set_ylabel("Remaining useful life (days)")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
ax.set_ylim(bottom=0)
ax.legend()
save(fig, "05_single_drive_rul_trajectory.png")


# ============================================================
# 6. Risk-gated adaptation: MAE by scenario
# ============================================================

df = pd.read_csv(RISK_DIR / "risk_gated_summary.csv")
methods = [
    ("frozen", "Frozen BiLSTM", TEXT_MUTED),
    ("always_adapt", "Always adapt", ORANGE),
    ("current_twin", "Population adapter", YELLOW),
    ("risk_gated", "Risk-gated (ours)", BLUE),
]
scenarios = list(dict.fromkeys(df["scenario"]))
scenario_labels = [s.split("_", 1)[1].replace("_", " ") for s in scenarios]


def grouped_by_scenario(column, title, ylabel, name):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.2
    for k, (method, label, color) in enumerate(methods):
        values = (df[df["method"] == method]
                  .set_index("scenario").loc[scenarios, column])
        offset = (k - (len(methods) - 1) / 2) * width
        ax.bar([i + offset for i in range(len(scenarios))], values, width,
               color=color, label=label, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenario_labels)
    ax.set_title(title)
    ax.set_xlabel("Stress-test scenario")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, name)


grouped_by_scenario("mae_days", "RUL error under each stress scenario",
                    "MAE (days)", "06_risk_gated_mae_by_scenario.png")


# ============================================================
# 7. Risk-gated adaptation: unsafe over-estimates
# ============================================================

grouped_by_scenario("unsafe_overestimate_rate_pct",
                    "How often RUL is over-estimated (unsafe late warning)",
                    "Unsafe over-estimate rate (%)",
                    "07_risk_gated_unsafe_overestimates.png")


# ============================================================
# 8. Evidence (e-value) build-up before ADAPT fires
# ============================================================

df = pd.read_csv(RISK_DIR / "risk_gated_decisions.csv")
run = df[(df["scenario"] == "S0_real") & (df["method"] == "risk_gated")]
threshold = run["threshold"].iloc[0]
first_adapt = run[run["action"] == "ADAPT"]["step"].min()
shadow = run[run["step"] <= first_adapt]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(shadow["step"], shadow["e_promote"], color=BLUE,
        marker="o", markersize=5, label="Evidence for adapting (E)")
ax.axhline(threshold, color=ORANGE, linewidth=1.5, linestyle="--",
           label=f"ADAPT threshold (1/α = {threshold:.0f})")
ax.set_yscale("log")
ax.set_title("Twin waits until the evidence is strong enough")
ax.set_xlabel("Feedback batch (real held-out stream)")
ax.set_ylabel("E-value (log scale)")
ax.annotate(f"ADAPT at batch {first_adapt}",
            (first_adapt, shadow["e_promote"].iloc[-1]),
            xytext=(-12, 0), textcoords="offset points",
            ha="right", va="center", fontsize=9, color=TEXT)
ax.legend(loc="lower right")
save(fig, "08_evidence_before_adapt.png")


# ============================================================
# 9. Failure warning lead time
# ============================================================

df = pd.read_csv(TWIN_DIR / "heldout_adaptive_lead_time_distribution.csv")
summary = pd.read_csv(TWIN_DIR / "heldout_adaptive_lead_time_summary.csv").iloc[0]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(df["lead_time_band"], df["hdd_count"], color=BLUE, width=0.6,
              edgecolor=SURFACE, linewidth=2)
label_bars(ax, bars, fmt="{:.0f}")
ax.set_title("How early failures were flagged")
ax.set_xlabel("Warning lead time before failure")
ax.set_ylabel("Number of drives")
ax.text(0.03, 0.97,
        f"{int(summary['detected_within_60_days'])} of "
        f"{int(summary['hdds_with_recorded_failure'])} failed drives flagged, "
        f"median {summary['median_lead_time_days']:.0f} days early",
        transform=ax.transAxes, va="top", fontsize=9, color=TEXT_MUTED)
ax.grid(axis="x", visible=False)
ax.set_ylim(0, df["hdd_count"].max() * 1.3)
save(fig, "09_failure_lead_time.png")


# ============================================================
# 10. Which SMART features separate failing drives
# ============================================================

df = pd.read_csv(RESULTS / "failure_feature_effect_size.csv")
top = df.nlargest(10, "absolute_effect_size").iloc[::-1]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(top["feature"].str.replace("_normalized", ""),
        top["absolute_effect_size"], color=BLUE, height=0.6,
        edgecolor=SURFACE, linewidth=2)
ax.set_title("Top 10 SMART features separating failed drives")
ax.set_xlabel("Effect size |Cohen's d| (failed vs normal)")
ax.grid(axis="y", visible=False)
save(fig, "10_top_failure_features.png")


print("\nAll figures written to", FIGURES)
