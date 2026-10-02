# ============================================
# S-Mamba — Support + Energy Pressure Release
#
# Raw Wave → Mamba → Next Value
#
# Mechanism:
#   Support  → state admission
#   Energy   → relational pressure
#   Pressure → release event
#   Release  → state consolidation
# ============================================


import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from data.simple_generator import generate_dataset
from model.s_mamba import SMamba


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("dataset")

BATCH_SIZE = 40
EPOCHS = 10

LEARNING_RATE = 1e-3

D_MODEL = 64
D_STATE = 16
D_CONV = 4
EXPAND = 2
N_LAYERS = 2

SUPPORT_ALPHA = 1.0

TRAIN_SPLIT = 0.8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# ============================================
# Generate Dataset
# ============================================

DATA_DIR = Path("dataset")

generate_dataset()
print(
    f"\nHard dataset generated in: "
    f"{DATA_DIR}"
)

# ============================================================
# Dataset
# ============================================================

class WaveDataset(Dataset):

    def __init__(self, data_dir):

        self.files = sorted(
            data_dir.glob("*.npz")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npz files found in {data_dir}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        data = np.load(
            self.files[index]
        )

        signal = data["input"].astype(
            np.float32
        )

        target = data["target"].astype(
            np.float32
        )

        signal = torch.from_numpy(
            signal
        ).unsqueeze(-1)

        target = torch.from_numpy(
            target
        ).unsqueeze(-1)

        return signal, target


# ============================================================
# Load dataset
# ============================================================

dataset = WaveDataset(DATA_DIR)

train_size = int(
    TRAIN_SPLIT * len(dataset)
)

test_size = (
    len(dataset) - train_size
)

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Samples:", len(dataset))
print("Train:", len(train_dataset))
print("Test :", len(test_dataset))


# ============================================================
# Model
# ============================================================

model = SMamba(
    d_model=D_MODEL,
    d_state=D_STATE,
    d_conv=D_CONV,
    expand=EXPAND,
    n_layers=N_LAYERS,
    support_alpha=SUPPORT_ALPHA
).to(DEVICE)

print("\nModel:")
print(model)

print(
    "\nParameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

# ============================================================
# Loss / Optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        # Normal forward pass.
        # Internal Support / Energy / Release
        # mechanisms remain active.

        predictions = model(inputs)

        # Temporary scalar prediction head.
        predictions = predictions[..., :1]

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()
        optimizer.step()

        train_loss += (
            loss.item() * inputs.size(0)
        )

    train_loss /= len(train_dataset)


    # ========================================================
    # Evaluation
    # ========================================================

    model.eval()

    test_loss = 0.0

    with torch.no_grad():

        for inputs, targets in test_loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(inputs)

            predictions = predictions[..., :1]

            loss = criterion(
                predictions,
                targets
            )

            test_loss += (
                loss.item() * inputs.size(0)
            )

    test_loss /= len(test_dataset)


    # ========================================================
    # Save epoch result
    # ========================================================

    print(
        f"Epoch {epoch:02d} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Test MSE: {test_loss:.6f}"
    )


# ============================================================
# Done
# ============================================================

print("S-Mamba training complete.")