# ============================================
# S-Mamba Adv1 — Support + Gate
#
# Raw Wave → Mamba → Next Value
#
# Mechanism:
#   Support → State admission
#   Energy  → Relational energy
#   SLayer  → Representation + energy
#   Gate    → Selective information flow
#
# - Support-based state admission
# - Scalar relational energy
# - Energy-conditioned admission gate
# - Recurrent state update
# ============================================


import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from data.hard_generator import generate_dataset
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


# ============================================================
# Generate Dataset
# ============================================================

generate_dataset()

print(f"\nDataset generated in: {DATA_DIR}")


# ============================================================
# Dataset
# ============================================================

class WaveDataset(Dataset):

    def __init__(self, data_dir: Path):
        self.files = sorted(data_dir.glob("*.npz"))

        if not self.files:
            raise RuntimeError(
                f"No .npz files found in {data_dir}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as data:
            signal = data["input"].astype(np.float32)
            target = data["target"].astype(np.float32)

        signal = torch.from_numpy(signal).unsqueeze(-1)
        target = torch.from_numpy(target).unsqueeze(-1)

        return signal, target


# ============================================================
# Load Dataset
# ============================================================

dataset = WaveDataset(DATA_DIR)

train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size

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
print("Test:", len(test_dataset))


# ============================================================
# Initialize Model
# ============================================================

model = SMamba(
    d_model=D_MODEL,
    d_state=D_STATE,
    d_conv=D_CONV,
    expand=EXPAND,
    n_layers=N_LAYERS,
    support_alpha=SUPPORT_ALPHA
).to(DEVICE)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel:")
print(model)
print(f"\nParameters: {parameter_count:,}")


# ============================================================
# Loss and Optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training Loop
# ============================================================

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(inputs)

        # Scalar next-value prediction.
        predictions = predictions[..., :1]

        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

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

            loss = criterion(predictions, targets)

            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_dataset)


    # ========================================================
    # Epoch Results
    # ========================================================

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Test MSE: {test_loss:.6f}"
    )


# ============================================================
# Training Complete
# ============================================================

print("\n✅ S-Mamba Adv1 training complete.")

# Tested on Google Colab & Local PC for different sequence length (because of PC RAM crash for longer sequence)