# ============================================================
# S-Mamba — Minimal Inspection Charts
#
# Reads:
#   inspection/training_history.json
#   inspection/s_mamba_inspection.json
#
# Produces:
#   inspection/charts/
#
# Charts:
#   1. Training vs Test MSE
#   2. Mean Support
#   3. Mean Distance
#   4. Mean Energy
#   5. Mean Pressure
#   6. Release Events
#
# Inspection:
#   Support
#   Distance
#   Energy
#   Pressure
#   Release
#   h norm
# ============================================================

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

INSPECTION_DIR = Path("inspection")

TRAINING_FILE = (
    INSPECTION_DIR / "training_history.json"
)

INSPECTION_FILE = (
    INSPECTION_DIR / "s_mamba_inspection.json"
)

CHART_DIR = (
    INSPECTION_DIR / "charts"
)

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load
# ============================================================

with open(TRAINING_FILE, "r") as f:
    training_history = json.load(f)

with open(INSPECTION_FILE, "r") as f:
    inspection_records = json.load(f)

num_layers = len(
    inspection_records[0]["layers"]
)

print("Samples:", len(inspection_records))
print("Layers :", num_layers)


# ============================================================
# Training / Test MSE
# ============================================================

epochs = [
    x["epoch"]
    for x in training_history
]

train_loss = [
    x["train_loss"]
    for x in training_history
]

test_loss = [
    x["test_loss"]
    for x in training_history
]

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Train MSE"
)

plt.plot(
    epochs,
    test_loss,
    marker="o",
    label="Test MSE"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("S-Mamba — Training vs Test MSE")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CHART_DIR / "training_loss.png",
    dpi=150
)

plt.close()


# ============================================================
# Aggregate inspection
# ============================================================

GROUP_SIZE = 10

metrics = {
    "support": [],
    "distance": [],
    "energy": [],
    "pressure": [],
    "release": [],
    "h_norm": []
}


for layer_index in range(num_layers):

    # --------------------------------------------------------
    # Collect all samples for this layer
    # --------------------------------------------------------

    layer_samples = [
        sample["layers"][layer_index]["timesteps"]
        for sample in inspection_records
    ]

    sequence_length = len(
        layer_samples[0]
    )

    num_groups = (
        sequence_length + GROUP_SIZE - 1
    ) // GROUP_SIZE

    layer_metrics = {
        key: []
        for key in metrics
    }

    # --------------------------------------------------------
    # Aggregate every GROUP_SIZE timesteps
    # --------------------------------------------------------

    for group in range(num_groups):

        start = group * GROUP_SIZE
        end = min(
            start + GROUP_SIZE,
            sequence_length
        )

        values = {
            key: []
            for key in metrics
        }

        for sample in layer_samples:

            for t in range(start, end):

                step = sample[t]

                values["support"].append(
                    step["support_mean"]
                )

                values["distance"].append(
                    step["distance_mean"]
                )

                values["energy"].append(
                    step["energy"]
                )

                values["pressure"].append(
                    step["pressure"]
                )

                values["release"].append(
                    step["release"]
                )

                values["h_norm"].append(
                    step["h_norm"]
                )

        for key in metrics:

            layer_metrics[key].append(
                np.mean(values[key])
            )

    for key in metrics:

        metrics[key].append(
            np.asarray(
                layer_metrics[key]
            )
        )


# ============================================================
# Plot helper
# ============================================================

def save_plot(
    metric,
    ylabel,
    title,
    filename
):

    plt.figure(figsize=(10, 5))

    for layer_index in range(num_layers):

        values = metrics[metric][layer_index]

        plt.plot(
            values,
            label=f"Layer {layer_index}"
        )

    plt.xlabel(
        f"Timestep (every {GROUP_SIZE} steps)"
    )

    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / filename,
        dpi=150
    )

    plt.close()


# ============================================================
# Mechanism charts
# ============================================================

save_plot(
    "support",
    "Mean Support",
    "S-Mamba — Support",
    "support_vs_timestep.png"
)

save_plot(
    "distance",
    "Mean Distance",
    "S-Mamba — Distance",
    "distance_vs_timestep.png"
)

save_plot(
    "energy",
    "Mean Energy",
    "S-Mamba — Energy",
    "energy_vs_timestep.png"
)

save_plot(
    "pressure",
    "Mean Pressure",
    "S-Mamba — Pressure",
    "pressure_vs_timestep.png"
)

save_plot(
    "release",
    "Release Rate",
    "S-Mamba — Release Events",
    "release_vs_timestep.png"
)


# ============================================================
# Small summary
# ============================================================

print()
print("=" * 60)
print("S-MAMBA INSPECTION SUMMARY")
print("=" * 60)

for layer_index in range(num_layers):

    print()
    print(f"Layer {layer_index}")

    for key in [
        "support",
        "distance",
        "energy",
        "pressure",
        "release",
        "h_norm"
    ]:

        values = metrics[key][layer_index]

        print(
            f"  {key:<10}: "
            f"{values.mean():.4f}"
        )


# ============================================================
# Generated files
# ============================================================

print()
print("Charts:")

for path in sorted(
    CHART_DIR.glob("*.png")
):

    print(" ", path)

print()
print("Done.")